"""Run inference + detection on the final-test slice; produce report + per-query payload.

Runs all three calibrated variants:
  v1 — per-node OR + per-node τ_v + σ-floor
  v2 — max-of-V + validation-max τ
  v3 — max-of-V + 400-threshold paper-style F1-search τ

Reports F1 / precision / recall under each. The per-query payload (parquet/CSV)
records the **v3 alarm** as the canonical alarm for triage downstream (paper-protocol
F1 is the headline number); v1 and v2 alarms are also stored alongside.

Outputs:
  - results/swat_ensemble/<datestr>/queries.{parquet,csv}
  - results/swat_ensemble/<datestr>/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _select_inference_backend(manifest_path: Path):
    """Dispatch on manifest['model']: returns (backend_module, model_name)."""
    with manifest_path.open() as f:
        manifest = json.load(f)
    model_name = manifest.get('model', 'gdn_uq')
    if model_name == 'gdn_uq':
        import inference as backend  # noqa: PLC0415
    elif model_name == 'dualstgf_uq':
        import inference_dualstgf as backend  # noqa: PLC0415
    else:
        raise ValueError(f"unknown manifest['model']={model_name!r}")
    return backend, model_name


from inference import (  # noqa: E402  -- helpers shared across both backends
    apply_lambda,
    apply_sigma_floor,
    sma_smooth,
    standardised_residual,
)
from util.uq_decomposition import information_decomposition  # noqa: E402
from util.ood import MahalanobisFit, score_mahalanobis  # noqa: E402
from util.graph_sensitivity import ensemble_sensitivity  # noqa: E402


def _build_subset_test_csv(test_csv: Path, row_range: tuple[int, int], out_csv: Path):
    df = pd.read_csv(test_csv, sep=',', index_col=0)
    sub = df.iloc[row_range[0]:row_range[1]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv)


def _f1(alarms: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(((alarms == 1) & (labels == 1)).sum())
    fp = int(((alarms == 1) & (labels == 0)).sum())
    fn = int(((alarms == 0) & (labels == 1)).sum())
    tn = int(((alarms == 0) & (labels == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn,
                alarm_rate=float(alarms.mean()))


def _triage_query(omega: float, omega_thresh: float,
                  rho_e: float, theta_e: float,
                  alarm: bool, attack_label: int) -> str:
    is_ood = omega > omega_thresh
    is_epi = rho_e > theta_e
    if alarm and attack_label == 0:
        return 'false_positive_candidate'
    if not alarm and attack_label == 1:
        return 'false_negative_candidate'
    if is_ood:
        return 'ood_dominant'
    if is_epi:
        return 'epistemic_dominant'
    return 'aleatoric_dominant'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=str,
                        default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'manifest.json'))
    parser.add_argument('--bundle-dir', type=str,
                        default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'calibration_bundle'))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n-samples-mi', type=int, default=100)
    parser.add_argument('--max-flagged-sensitivity', type=int, default=200)
    parser.add_argument('--out-dir', type=str,
                        default=str(REPO_ROOT / 'results' / 'swat_ensemble'))
    parser.add_argument('--variant', type=str, default='A', choices=['A', 'B'],
                        help='Which decomposition to use for triage rho_e (variance / entropy-MI).')
    parser.add_argument('--canonical-alarm', type=str, default='v3',
                        choices=['v1', 'v2', 'v3'],
                        help='Which variant\'s alarm to use for triage table.')
    args = parser.parse_args()

    # Dispatch on manifest['model'] to pick GDN vs DualSTGF inference backend.
    backend, model_name = _select_inference_backend(Path(args.manifest))
    load_ensemble = backend.load_ensemble
    build_dataset_from_csv = backend.build_dataset_from_csv
    run_inference = backend.run_inference

    # Auto-derive bundle dir from manifest's parent if user didn't override it.
    if args.bundle_dir == str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'calibration_bundle'):
        bundle_dir = Path(args.manifest).parent / 'calibration_bundle'
    else:
        bundle_dir = Path(args.bundle_dir)
    # Likewise, default out_dir tracks the model name.
    if args.out_dir == str(REPO_ROOT / 'results' / 'swat_ensemble'):
        out_dir_root = REPO_ROOT / 'results' / Path(args.manifest).parent.name
    else:
        out_dir_root = Path(args.out_dir)

    if not (bundle_dir / 'bundle.json').is_file():
        print(f'[detect] bundle missing at {bundle_dir}; run scripts/calibrate.py first',
              file=sys.stderr)
        sys.exit(1)
    print(f'[detect] backend={model_name}  bundle_dir={bundle_dir}  out_dir_root={out_dir_root}')
    with (bundle_dir / 'bundle.json').open() as f:
        bundle = json.load(f)

    taus_npz = np.load(bundle_dir / 'taus.npz')
    taus = taus_npz['taus']
    sigma_floor_v = taus_npz['sigma_floor_v']
    # Per-sensor λ_v (RESULTS.md future-work #3) lands in taus.npz under
    # `lam_v`. Older bundles only have a scalar in bundle['lambda']; fall
    # back gracefully so this detect.py keeps working with previous runs.
    if 'lam_v' in taus_npz.files:
        lam_for_apply = taus_npz['lam_v']
    else:
        lam_for_apply = float(bundle['lambda'])
    maha = np.load(bundle_dir / 'mahalanobis.npz')
    fit = MahalanobisFit(
        mean=maha['mean'], inv_cov=maha['inv_cov'], log_det_cov=maha['log_det_cov'],
    )
    sigma_A = np.load(bundle_dir / 'adjacency_cov.npz')['sigma_A']

    lam = bundle['lambda']
    sma_window = bundle['sma_window']
    omega_thresh = bundle['omega_thresh_per_query']
    theta_e = bundle['theta_e_A'] if args.variant == 'A' else bundle['theta_e_B']
    v2_threshold = bundle['variant2_maxv_validation_max']['threshold']
    v3_threshold = bundle['variant3_maxv_paper_sweep']['threshold']
    # Variant 4 — sustained-window. Older bundles don't have it; we then skip
    # the v4 row in the scoreboard.
    v4_cfg = bundle.get('variant4_sustained_window')

    print(f'[detect] bundle: λ={lam:.4f}  σ_floor median={np.median(sigma_floor_v):.4e}  '
          f'sma={sma_window}')
    print(f'[detect]   v1 best α (labeled-val): {bundle["variant1_pernode_or"]["best_alpha"]}')
    print(f'[detect]   v2 threshold (val-max)  : {v2_threshold:.4f}')
    print(f'[detect]   v3 threshold (paper sweep): {v3_threshold:.4f}')
    if v4_cfg is not None:
        print(f'[detect]   v4 sustained: τ={v4_cfg["threshold"]:.4f} '
              f'W={v4_cfg["W"]} K_w={v4_cfg["K_w"]}')

    ensemble = load_ensemble(args.manifest, device=args.device, repo_root=REPO_ROOT)
    print(f'[detect] ensemble: M={ensemble.cfg.M} V={ensemble.cfg.node_num} '
          f'device={ensemble.device}')

    split = bundle['split']
    final_range = tuple(split['final_test_range'])
    test_csv = REPO_ROOT / 'data' / 'swat' / 'test.csv'
    sub_csv = bundle_dir / '_finaltest_subset.csv'
    _build_subset_test_csv(test_csv, final_range, sub_csv)
    ds = build_dataset_from_csv(
        sub_csv, ensemble.feature_map, ensemble.fc_edge_index,
        slide_win=ensemble.cfg.slide_win, slide_stride=1, mode='test',
    )
    print(f'[detect] final-test windows: {len(ds):,} '
          f'(rows [{final_range[0]:,}, {final_range[1]:,}))')

    t0 = time.time()
    out = run_inference(ensemble, ds, batch_size=ensemble.cfg.batch)
    print(f'[detect] inference: {time.time()-t0:.1f}s')

    # ------------------------------------------------------------------
    # Apply σ-floor + λ → standardised residuals → max-of-V scores
    # ------------------------------------------------------------------
    sigma2_cal = apply_lambda(apply_sigma_floor(out.sigma2_total, sigma_floor_v), lam_for_apply)
    r = standardised_residual(out.ground_truth, out.mu_bar, sigma2_cal)
    A = r.max(axis=1)
    A_sm = sma_smooth(A, sma_window)
    flagged_node_per_t = r.argmax(axis=1)

    # Variant 1 — per-node OR
    alarms_v1_node = r > taus[None, :]
    alarms_v1 = alarms_v1_node.any(axis=1)
    m1 = _f1(alarms_v1.astype(int), out.attack_label)

    # Variant 2 — max-of-V, validation-max
    alarms_v2 = (A_sm > v2_threshold).astype(int)
    m2 = _f1(alarms_v2, out.attack_label)

    # Variant 3 — max-of-V, paper-sweep (uses the threshold picked on labeled-val).
    alarms_v3 = (A_sm > v3_threshold).astype(int)
    m3 = _f1(alarms_v3, out.attack_label)

    # Variant 4 — sustained-window post-process on v3 alarms.
    alarms_v4 = None
    m4 = None
    if v4_cfg is not None:
        W = int(v4_cfg['W'])
        K_w = int(v4_cfg['K_w'])
        rolling = np.zeros_like(alarms_v3)
        for t in range(len(alarms_v3)):
            rolling[t] = alarms_v3[max(0, t - W + 1):t + 1].sum()
        alarms_v4 = (rolling >= K_w).astype(int)
        m4 = _f1(alarms_v4, out.attack_label)

    print(f'[detect] FINAL-TEST RESULTS:')
    print(f'   v1 per-node OR     : P={m1["precision"]:.4f} R={m1["recall"]:.4f} '
          f'F1={m1["f1"]:.4f}  TP={m1["tp"]} FP={m1["fp"]} FN={m1["fn"]} TN={m1["tn"]}')
    print(f'   v2 max-of-V val-max: P={m2["precision"]:.4f} R={m2["recall"]:.4f} '
          f'F1={m2["f1"]:.4f}  TP={m2["tp"]} FP={m2["fp"]} FN={m2["fn"]} TN={m2["tn"]}')
    print(f'   v3 max-of-V paper  : P={m3["precision"]:.4f} R={m3["recall"]:.4f} '
          f'F1={m3["f1"]:.4f}  TP={m3["tp"]} FP={m3["fp"]} FN={m3["fn"]} TN={m3["tn"]}')
    if m4 is not None:
        print(f'   v4 sustained (W={int(v4_cfg["W"])} K_w={int(v4_cfg["K_w"])}): '
              f'P={m4["precision"]:.4f} R={m4["recall"]:.4f} '
              f'F1={m4["f1"]:.4f}  TP={m4["tp"]} FP={m4["fp"]} FN={m4["fn"]} TN={m4["tn"]}')

    # Pick canonical alarm for triage
    canonical = {'v1': alarms_v1.astype(int), 'v2': alarms_v2, 'v3': alarms_v3}[args.canonical_alarm]

    # OOD score
    ood_member_index = int(np.array(maha['ood_member_index']))
    # Older bundles don't have ood_mode; default to 'single' for back-compat.
    ood_mode = str(np.array(maha.get('ood_mode', 'single'))) if 'ood_mode' in maha.files else 'single'
    if ood_mode == 'ensemble_avg':
        ood_source = ensemble.members
    else:
        ood_source = ensemble.members[ood_member_index]
    print(f'[detect] OOD: mode={ood_mode} member={"all" if ood_mode == "ensemble_avg" else ood_member_index}')
    omega = score_mahalanobis(
        ood_source, ds, fit, ensemble.device, ood_mode=ood_mode,
        batch_size=ensemble.cfg.batch,
    )
    omega_max = omega.max(axis=1)

    # rho_e for triage
    if args.variant == 'B':
        info = information_decomposition(
            out.mu_per_member, out.logvar_per_member, n_samples=args.n_samples_mi,
        )
        rho_e_per_tv = info['MI'] / np.maximum(np.abs(info['H_tot']), 1e-9)
        rho_e_per_tv = np.minimum(rho_e_per_tv, 1.0)
    else:
        rho_e_per_tv = out.sigma2_epistemic / np.maximum(out.sigma2_total, 1e-9)
    rho_e_per_t = rho_e_per_tv.max(axis=1)

    # Triage
    triage_labels = []
    for t in range(len(ds)):
        triage_labels.append(_triage_query(
            omega=float(omega_max[t]),
            omega_thresh=omega_thresh,
            rho_e=float(rho_e_per_t[t]),
            theta_e=theta_e,
            alarm=bool(canonical[t]),
            attack_label=int(out.attack_label[t]),
        ))
    print(f'[detect] triage label distribution (canonical={args.canonical_alarm}): '
          f'{pd.Series(triage_labels).value_counts().to_dict()}')

    # ------------------------------------------------------------------
    # Lazy graph sensitivity for flagged queries (Step 11)
    # ------------------------------------------------------------------
    flagged_idx = np.where(canonical)[0]
    if flagged_idx.size > args.max_flagged_sensitivity:
        rng = np.random.default_rng(0)
        flagged_idx = rng.choice(flagged_idx, size=args.max_flagged_sensitivity, replace=False)
        flagged_idx.sort()
    print(f'[detect] computing graph sensitivity for {len(flagged_idx)} flagged queries '
          f'(cap={args.max_flagged_sensitivity}, total flagged={int(canonical.sum())})')
    sensitivity_weighted = np.full(len(ds), np.nan, dtype=np.float64)
    sensitivity_unweighted = np.full(len(ds), np.nan, dtype=np.float64)
    t0 = time.time()
    for t_idx in flagged_idx:
        x_window, _y, _lab, _ei = ds[int(t_idx)]
        x_window = x_window.unsqueeze(0).to(ensemble.device).float()
        v_q = int(flagged_node_per_t[t_idx])
        try:
            sens = ensemble_sensitivity(ensemble.members, x_window, v_q, sigma_A)
            sensitivity_weighted[t_idx] = sens['weighted']
            sensitivity_unweighted[t_idx] = sens['unweighted']
        except Exception as e:
            print(f'[detect]   sensitivity failed at t={t_idx}: {e}')
    if flagged_idx.size:
        wmed = float(np.nanmedian(sensitivity_weighted))
        print(f'[detect] graph sensitivity: {time.time()-t0:.1f}s  weighted_median={wmed:.4e}')

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    out_dir = out_dir_root / datestr
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_cols = {
        'window_index': np.arange(len(ds)),
        'attack_label': out.attack_label.astype(np.int8),
        'alarm_v1_pernode': alarms_v1.astype(np.int8),
        'alarm_v2_maxv_valmax': alarms_v2.astype(np.int8),
        'alarm_v3_maxv_paper': alarms_v3.astype(np.int8),
    }
    if alarms_v4 is not None:
        payload_cols['alarm_v4_sustained'] = alarms_v4.astype(np.int8)
    payload_cols.update({
        'flagged_node': flagged_node_per_t.astype(np.int32),
        'A_sm': A_sm.astype(np.float32),
        'omega_max': omega_max.astype(np.float32),
        'rho_e_max': rho_e_per_t.astype(np.float32),
        'triage_label': triage_labels,
        'sensitivity_weighted': sensitivity_weighted,
        'sensitivity_unweighted': sensitivity_unweighted,
    })
    payload = pd.DataFrame(payload_cols)
    parquet_path = out_dir / 'queries.parquet'
    try:
        payload.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f'[detect] parquet write failed ({e}); falling back to CSV')
        parquet_path = out_dir / 'queries.csv'
        payload.to_csv(parquet_path, index=False)

    report = {
        'datestr': datestr,
        'manifest': args.manifest,
        'bundle': str(bundle_dir),
        'variant': args.variant,
        'canonical_alarm': args.canonical_alarm,
        'final_test_range': list(final_range),
        'n_windows': len(ds),
        'metrics_final_test': {
            'v1_pernode_or': m1,
            'v2_maxv_valmax': m2,
            'v3_maxv_paper': m3,
            **({'v4_sustained': m4} if m4 is not None else {}),
        },
        'lambda_used': lam,
        'sigma_floor': {
            'min': float(sigma_floor_v.min()),
            'median': float(np.median(sigma_floor_v)),
            'max': float(sigma_floor_v.max()),
        },
        'taus_summary': {
            'min': float(taus.min()), 'max': float(taus.max()),
            'mean': float(taus.mean()), 'median': float(np.median(taus)),
        },
        'omega_thresh': omega_thresh,
        'theta_e': theta_e,
        'triage_distribution': pd.Series(triage_labels).value_counts().to_dict(),
        'queries_path': str(parquet_path),
    }
    with (out_dir / 'report.json').open('w') as f:
        json.dump(report, f, indent=2)
    print(f'[detect] wrote {out_dir}/report.json + {parquet_path.name}')

    try:
        sub_csv.unlink()
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    main()
