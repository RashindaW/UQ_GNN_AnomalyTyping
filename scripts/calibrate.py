"""Build the calibration bundle for the trained GDN_UQ ensemble.

Implements outline Steps 6 (lambda), 9.2-9.4 (per-node taus + alpha sweep),
5 (Mahalanobis OOD fit), 10 (theta_e), 11 (sensitivity threshold + adjacency
covariance), plus a reliability diagram diagnostic.

The first run found a sigma-collapse failure mode (some sensors have log_var
saturated at -10 -> sigma ~ 0.0067 -> standardised residuals explode and the
per-node OR alarm rule trips on every timestep). This version adds:

  - **sigma-floor** per sensor, derived from the empirical residual std on 𝒞.
    Floors sigma_total before residual computation. Keeps the variance head's
    estimate where it's healthy and prevents collapse.
  - **Three detection variants**, all calibrated and persisted, all reported:
        v1 — per-node OR + per-node tau_v (outline Step 9), with sigma-floor.
        v2 — max-of-V aggregation with single global threshold = max(A_smooth on 𝒞)
             (paper §3.6 canonical detection rule).
        v3 — max-of-V aggregation + 400-threshold F1 sweep on labeled-val
             (paper §4.3 evaluation protocol).
    detect.py reports F1 under each.

Run once after training. Output goes to
`pretrained/swat_ensemble/calibration_bundle/`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _select_inference_backend(manifest_path: Path):
    """Dispatch on manifest['model']: returns (backend_module, model_name).

    Both backends expose the same public API: load_ensemble, run_inference,
    build_dataset_from_csv, plus the post-process helpers re-exported from
    `inference` (apply_lambda, apply_sigma_floor, sma_smooth, standardised_residual).
    """
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
from util.ood import fit_mahalanobis, score_mahalanobis  # noqa: E402
from util.graph_sensitivity import (  # noqa: E402
    empirical_adjacency_covariance,
    ensemble_sensitivity,
)


Z_95 = 1.959963984540054


def _load_split(indices_path: Path) -> dict:
    if not indices_path.is_file():
        print(f"[calibrate] missing {indices_path}; running scripts/build_test_split.py first")
        os.system(f"{sys.executable} {REPO_ROOT / 'scripts' / 'build_test_split.py'}")
    with indices_path.open() as f:
        return json.load(f)


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


def _reliability_diagram(residuals_flat: np.ndarray,
                         levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    from scipy.stats import norm
    rows = []
    for q in levels:
        z = norm.ppf(0.5 + q / 2)
        emp = float((residuals_flat <= z).mean())
        rows.append({'nominal': q, 'empirical': emp, 'gap': emp - q, 'z': z})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=str,
                        default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'manifest.json'))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--bundle-dir', type=str,
                        default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'calibration_bundle'))
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument('--n-thresholds-paper', type=int, default=400,
                        help='Number of thresholds in the paper-style F1 sweep (variant 3).')
    parser.add_argument('--sma-window', type=int, default=4,
                        help='SMA window for max-of-V variants (matches evaluate.py:53-56).')
    parser.add_argument('--n-samples-mi', type=int, default=100)
    parser.add_argument('--ood-member-index', type=int, default=0,
                        help='Index of the ensemble member used as the OOD '
                             'representation source when --ood-mode=single.')
    parser.add_argument('--ood-mode', type=str, default='single',
                        choices=['single', 'ensemble_avg'],
                        help='single: use one member; ensemble_avg: average '
                             'penultimate reps across all M members (RESULTS.md '
                             'future-work #6).')
    parser.add_argument('--n-sensitivity-samples', type=int, default=100)
    args = parser.parse_args()

    # Dispatch on manifest['model'] before touching the bundle path.
    backend, model_name = _select_inference_backend(Path(args.manifest))
    load_ensemble = backend.load_ensemble
    build_dataset_from_csv = backend.build_dataset_from_csv
    run_inference = backend.run_inference

    # Auto-derive bundle directory from the manifest's parent if the user
    # didn't override --bundle-dir.
    if args.bundle_dir == str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'calibration_bundle'):
        bundle_dir = Path(args.manifest).parent / 'calibration_bundle'
    else:
        bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    print(f'[calibrate] backend={model_name}  bundle_dir={bundle_dir}')

    indices_path = bundle_dir / 'calibration_set_indices.json'
    if not indices_path.is_file():
        # Reuse the canonical calibration-set indices computed for the GDN
        # ensemble (same test.csv → same row ranges → apples-to-apples).
        canonical = REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'calibration_bundle' / 'calibration_set_indices.json'
        if canonical.is_file():
            print(f'[calibrate] copying canonical indices from {canonical}')
            import shutil
            shutil.copy(canonical, indices_path)
    split = _load_split(indices_path)
    test_csv = REPO_ROOT / 'data' / 'swat' / 'test.csv'
    train_csv = REPO_ROOT / 'data' / 'swat' / 'train.csv'

    print(f'[calibrate] loading ensemble from {args.manifest}')
    ensemble = load_ensemble(args.manifest, device=args.device, repo_root=REPO_ROOT)
    print(f'[calibrate]   M={ensemble.cfg.M} V={ensemble.cfg.node_num} '
          f'slide_win={ensemble.cfg.slide_win} batch={ensemble.cfg.batch} '
          f'device={ensemble.device}')

    # ------------------------------------------------------------------
    # 1. 𝒞 inference (Steps 1-3)
    # ------------------------------------------------------------------
    c_range = tuple(split['C_row_range'])
    print(f'[calibrate] preparing 𝒞 from test.csv rows {c_range}, '
          f'attack-zero rows={len(split["C_attack_zero_indices"])}')
    c_csv = bundle_dir / '_C_subset.csv'
    _build_subset_test_csv(test_csv, c_range, c_csv)
    df_C = pd.read_csv(c_csv, sep=',', index_col=0)
    df_C = df_C[df_C['attack'] == 0]
    df_C.to_csv(c_csv)

    ds_C = build_dataset_from_csv(
        c_csv, ensemble.feature_map, ensemble.fc_edge_index,
        slide_win=ensemble.cfg.slide_win, slide_stride=1, mode='test',
    )
    print(f'[calibrate] |𝒞 windows| = {len(ds_C):,}')

    t0 = time.time()
    out_C = run_inference(ensemble, ds_C, batch_size=ensemble.cfg.batch)
    print(f'[calibrate] 𝒞 inference: {time.time()-t0:.1f}s')

    # ------------------------------------------------------------------
    # 1.5 SIGMA HEALTH DIAGNOSTIC + per-sensor floor
    # ------------------------------------------------------------------
    median_logvar_per_v = np.median(out_C.logvar_per_member, axis=(0, 1))   # (V,)
    sat_low_per_v = (out_C.logvar_per_member <= -9.9).mean(axis=(0, 1))
    sat_high_per_v = (out_C.logvar_per_member >= 9.9).mean(axis=(0, 1))
    print(f'[calibrate] σ-health on 𝒞 (per sensor across all members):')
    print(f'   median(log_var) range: [{median_logvar_per_v.min():.2f}, '
          f'{median_logvar_per_v.max():.2f}]; '
          f'< -3: {(median_logvar_per_v < -3).sum()} / {ensemble.cfg.node_num} sensors')
    print(f'   sat_low (≤-9.9) per-sensor max: {sat_low_per_v.max():.4f}')
    print(f'   sat_high (≥9.9) per-sensor max: {sat_high_per_v.max():.4f}')

    # σ-floor: use the empirical std of residuals on 𝒞 per sensor.
    raw_residuals_C = out_C.ground_truth - out_C.mu_bar
    sigma_floor_v = raw_residuals_C.std(axis=0).astype(np.float32)
    # Numerical safety: avoid pathologically tiny floors.
    sigma_floor_v = np.maximum(sigma_floor_v, 1e-4)
    print(f'[calibrate] σ-floor per sensor: '
          f'min={sigma_floor_v.min():.4e} median={np.median(sigma_floor_v):.4e} '
          f'max={sigma_floor_v.max():.4e}')

    # Apply floor to σ̂_total before λ.
    sigma2_floored_C = apply_sigma_floor(out_C.sigma2_total, sigma_floor_v)
    print(f'[calibrate] floor changed {(sigma2_floored_C > out_C.sigma2_total + 1e-12).mean():.4f} '
          f'of (t,v) cells')

    # ------------------------------------------------------------------
    # 2. Step 6 — λ on FLOORED residuals
    # ------------------------------------------------------------------
    r_C_uncal = standardised_residual(out_C.ground_truth, out_C.mu_bar, sigma2_floored_C)

    # PER-SENSOR λ_v (RESULTS.md future-work #3): each sensor gets its own
    # multiplicative correction so over- AND under-confidence are corrected
    # independently. A single global λ couldn't compensate for both directions
    # at once.
    q95_per_v = np.quantile(np.abs(r_C_uncal), 0.95, axis=0).astype(np.float32)   # (V,)
    lam_v_raw = q95_per_v / Z_95
    # Per-sensor identity-clamp: any sensor whose raw λ_v lands in [0.95, 1.05]
    # is treated as already calibrated (set to 1.0) to match the existing
    # global-λ heuristic.
    lam_v = np.where((lam_v_raw >= 0.95) & (lam_v_raw <= 1.05), 1.0, lam_v_raw).astype(np.float32)

    # Scalar summary kept for backward-compat in bundle.json and for log lines.
    lam_raw = float(np.median(lam_v_raw))
    lam = float(np.median(lam_v))
    print(f'[calibrate] λ (median per-sensor) raw={lam_raw:.4f}  used={lam:.4f}  '
          f'per-sensor range=[{lam_v.min():.4f}, {lam_v.max():.4f}]; '
          f'sensors clamped-to-1: {int((lam_v == 1.0).sum())} / {ensemble.cfg.node_num}')

    # σ̂_total_cal = floor + λ_v (broadcast over T).
    sigma2_cal_C = apply_lambda(sigma2_floored_C, lam_v)
    r_C = standardised_residual(out_C.ground_truth, out_C.mu_bar, sigma2_cal_C)

    # ------------------------------------------------------------------
    # 3. Reliability diagram (after floor + λ)
    # ------------------------------------------------------------------
    rel_df = _reliability_diagram(r_C.flatten())
    rel_df.to_csv(bundle_dir / 'reliability.csv', index=False)
    print('[calibrate] reliability: '
          + ', '.join(f'{r.nominal:.2f}->{r.empirical:.4f}' for _, r in rel_df.iterrows()))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], 'k--', label='perfect calibration')
        plt.plot(rel_df['nominal'], rel_df['empirical'], 'o-', label='empirical (floor+λ)')
        plt.xlabel('nominal coverage')
        plt.ylabel('empirical coverage on cal set')
        plt.title(f'GDN_UQ reliability  λ={lam:.3f}  floor active')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(bundle_dir / 'reliability.png', dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'[calibrate] reliability png skipped: {e}')

    # ------------------------------------------------------------------
    # 4. Per-node τ table (Variant 1, outline Step 9.3)
    # ------------------------------------------------------------------
    R_per_node = np.sort(r_C, axis=0)
    print(f'[calibrate] R_v matrix: shape={R_per_node.shape}')

    # ------------------------------------------------------------------
    # 5. Variant 2/3 — max-of-V scores on 𝒞 + threshold from max
    # ------------------------------------------------------------------
    A_C = r_C.max(axis=1)
    A_C_sm = sma_smooth(A_C, args.sma_window)
    v2_threshold = float(A_C_sm.max())  # paper canonical rule (validation-max)
    print(f'[calibrate] Variant 2 (max-of-V) threshold from max(A_sm on 𝒞): {v2_threshold:.4f}')

    # ------------------------------------------------------------------
    # 6. Inference on labeled-val
    # ------------------------------------------------------------------
    lv_range = tuple(split['labeled_val_range'])
    lv_csv = bundle_dir / '_labeledval_subset.csv'
    _build_subset_test_csv(test_csv, lv_range, lv_csv)
    ds_LV = build_dataset_from_csv(
        lv_csv, ensemble.feature_map, ensemble.fc_edge_index,
        slide_win=ensemble.cfg.slide_win, slide_stride=1, mode='test',
    )
    t0 = time.time()
    out_LV = run_inference(ensemble, ds_LV, batch_size=ensemble.cfg.batch)
    print(f'[calibrate] labeled-val inference: {time.time()-t0:.1f}s  windows={len(ds_LV):,}')

    sigma2_cal_LV = apply_lambda(apply_sigma_floor(out_LV.sigma2_total, sigma_floor_v), lam_v)
    r_LV = standardised_residual(out_LV.ground_truth, out_LV.mu_bar, sigma2_cal_LV)
    A_LV = r_LV.max(axis=1)
    A_LV_sm = sma_smooth(A_LV, args.sma_window)

    # ------------------------------------------------------------------
    # 7. Variant 1 — α sweep with per-node OR
    # ------------------------------------------------------------------
    print('[calibrate] Variant 1 (per-node OR) α sweep on labeled-val:')
    v1_sweep = []
    v1_best = None
    for alpha in args.alphas:
        idx = int(round((1 - alpha) * (R_per_node.shape[0] - 1)))
        taus = R_per_node[idx, :]
        alarms_per_node = r_LV > taus[None, :]
        alarms = alarms_per_node.any(axis=1)
        m = _f1(alarms.astype(int), out_LV.attack_label)
        v1_sweep.append({'alpha': alpha, **m})
        print(f'   α={alpha:6.4f}: P={m["precision"]:.4f} R={m["recall"]:.4f} '
              f'F1={m["f1"]:.4f}  alarms={int(alarms.sum())}')
        if v1_best is None or m['f1'] > v1_best['f1']:
            v1_best = {'alpha': alpha, 'taus': taus.copy(), **m}
    print(f'[calibrate] Variant 1 best: α={v1_best["alpha"]} F1={v1_best["f1"]:.4f}')

    # ------------------------------------------------------------------
    # 8. Variant 2 — single global τ from max(A_sm on 𝒞)
    # ------------------------------------------------------------------
    alarms_v2 = (A_LV_sm > v2_threshold).astype(int)
    v2_metrics = _f1(alarms_v2, out_LV.attack_label)
    print(f'[calibrate] Variant 2 (max-of-V, validation-max τ={v2_threshold:.4f}): '
          f'P={v2_metrics["precision"]:.4f} R={v2_metrics["recall"]:.4f} '
          f'F1={v2_metrics["f1"]:.4f}  alarms={v2_metrics["tp"]+v2_metrics["fp"]}')

    # ------------------------------------------------------------------
    # 9. Variant 3 — paper-style 400-threshold F1 sweep on labeled-val
    # ------------------------------------------------------------------
    th_grid = np.linspace(A_LV_sm.min(), A_LV_sm.max(), args.n_thresholds_paper)
    v3_best = {'f1': -1}
    for th in th_grid:
        alarms = (A_LV_sm > th).astype(int)
        m = _f1(alarms, out_LV.attack_label)
        if m['f1'] > v3_best['f1']:
            v3_best = {'threshold': float(th), **m}
    print(f'[calibrate] Variant 3 (max-of-V, 400-threshold F1 sweep): '
          f'τ={v3_best["threshold"]:.4f}  P={v3_best["precision"]:.4f} '
          f'R={v3_best["recall"]:.4f} F1={v3_best["f1"]:.4f}')

    # ------------------------------------------------------------------
    # 9b. Variant 4 — sustained-window detection rule (RESULTS.md FW#5)
    # Two-pass: keep v3's chosen τ fixed, then sweep (W, K_w) on labeled-val.
    # alarms_v4(t) = 1 iff sum(alarms_v3[t-W+1 : t+1]) >= K_w.
    # ------------------------------------------------------------------
    v3_alarms_lv = (A_LV_sm > v3_best['threshold']).astype(np.int32)
    w_grid = [3, 5, 10, 20]
    v4_best = {'f1': -1}
    for W in w_grid:
        # Cumulative-sum trick: rolling sum of size W over a binary series.
        cs = np.concatenate([[0], np.cumsum(v3_alarms_lv)])
        # rolling[t] = sum(alarms_v3[max(0, t-W+1) : t+1])
        rolling = cs[1:] - cs[:-len(cs) + 1]   # placeholder; replaced below
        rolling = np.zeros_like(v3_alarms_lv)
        for t in range(len(v3_alarms_lv)):
            lo = max(0, t - W + 1)
            rolling[t] = v3_alarms_lv[lo:t + 1].sum()
        for K_w in range(1, W + 1):
            alarms_v4 = (rolling >= K_w).astype(int)
            m = _f1(alarms_v4, out_LV.attack_label)
            if m['f1'] > v4_best['f1']:
                v4_best = {
                    'threshold': float(v3_best['threshold']),
                    'W': int(W), 'K_w': int(K_w), **m,
                }
    print(f'[calibrate] Variant 4 (sustained-window after v3): '
          f'τ={v4_best["threshold"]:.4f} W={v4_best["W"]} K_w={v4_best["K_w"]}  '
          f'P={v4_best["precision"]:.4f} R={v4_best["recall"]:.4f} F1={v4_best["f1"]:.4f}')

    # ------------------------------------------------------------------
    # 10. Mahalanobis OOD (Step 5)
    # ------------------------------------------------------------------
    print(f'[calibrate] fitting Mahalanobis OOD on training set '
          f'(mode={args.ood_mode}, member={args.ood_member_index if args.ood_mode == "single" else "all"})')
    ds_train = build_dataset_from_csv(
        train_csv, ensemble.feature_map, ensemble.fc_edge_index,
        slide_win=ensemble.cfg.slide_win, slide_stride=ensemble.cfg.slide_stride,
        mode='train',
    )
    # Pick the OOD source: a single member, or the full ensemble for averaging.
    if args.ood_mode == 'ensemble_avg':
        ood_source = ensemble.members
    else:
        ood_source = ensemble.members[args.ood_member_index]

    t0 = time.time()
    fit = fit_mahalanobis(
        ood_source, ds_train, ensemble.device,
        batch_size=ensemble.cfg.batch, ood_mode=args.ood_mode,
    )
    print(f'[calibrate] Mahalanobis fit: {time.time()-t0:.1f}s  '
          f'mean shape={fit.mean.shape}  inv_cov shape={fit.inv_cov.shape}')

    omega_C = score_mahalanobis(
        ood_source, ds_C, fit, ensemble.device,
        batch_size=ensemble.cfg.batch, ood_mode=args.ood_mode,
    )
    omega_thresh = float(np.quantile(omega_C, 0.99))
    omega_C_per_query = omega_C.max(axis=1)
    omega_thresh_per_query = float(np.quantile(omega_C_per_query, 0.99))
    print(f'[calibrate] Ω 99th-pct per-(t,v): {omega_thresh:.4f}  '
          f'per-timestep: {omega_thresh_per_query:.4f}')

    # ------------------------------------------------------------------
    # 11. Information decomposition + θ_e (Step 4 + Step 10)
    # ------------------------------------------------------------------
    print(f'[calibrate] computing information decomposition on 𝒞 (n_samples={args.n_samples_mi})')
    t0 = time.time()
    info = information_decomposition(
        out_C.mu_per_member, out_C.logvar_per_member,
        n_samples=args.n_samples_mi,
    )
    print(f'[calibrate] information decomposition: {time.time()-t0:.1f}s')

    eps = 1e-9
    rho_e_B = info['MI'] / np.maximum(np.abs(info['H_tot']), eps)
    rho_e_B_per_query = rho_e_B.max(axis=1)
    # Cap rho_e at 1 to absorb the pathological H_tot ≈ 0 cases.
    rho_e_B_per_query = np.minimum(rho_e_B_per_query, 1.0)
    theta_e_B = float(np.median(rho_e_B_per_query))

    rho_e_A = out_C.sigma2_epistemic / np.maximum(out_C.sigma2_total, eps)
    rho_e_A_per_query = rho_e_A.max(axis=1)
    theta_e_A = float(np.median(rho_e_A_per_query))
    print(f'[calibrate] θ_e (Variant A) = {theta_e_A:.6f}')
    print(f'[calibrate] θ_e (Variant B, capped at 1) = {theta_e_B:.6f}')

    # ------------------------------------------------------------------
    # 12. Step 11 — adjacency covariance + sensitivity threshold
    # ------------------------------------------------------------------
    sigma_A = empirical_adjacency_covariance(ensemble.members)
    print(f'[calibrate] Σ̂_A shape={sigma_A.shape}  trace={np.trace(sigma_A):.4f}')

    print(f'[calibrate] sampling {args.n_sensitivity_samples} nominal queries for sensitivity')
    rng = np.random.default_rng(0)
    n_sample = min(args.n_sensitivity_samples, len(ds_C))
    sample_idx = rng.choice(len(ds_C), size=n_sample, replace=False)
    weighted_vals = []
    t0 = time.time()
    for s in sample_idx:
        x_window, _y, _label, _ei = ds_C[int(s)]
        x_window = x_window.unsqueeze(0).to(ensemble.device).float()
        v_query = int(rng.integers(0, ensemble.cfg.node_num))
        try:
            sens = ensemble_sensitivity(ensemble.members, x_window, v_query, sigma_A)
            weighted_vals.append(sens['weighted'])
        except Exception as e:
            pass
    weighted_vals = np.array(weighted_vals)
    sens_thresh = float(np.median(weighted_vals)) if weighted_vals.size > 0 else 0.0
    print(f'[calibrate] sensitivity median (weighted): {sens_thresh:.4e}  '
          f'(n={len(weighted_vals)}, {time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # 13. Persist bundle
    # ------------------------------------------------------------------
    np.savez_compressed(
        bundle_dir / 'taus.npz',
        taus=v1_best['taus'].astype(np.float32),
        alpha=v1_best['alpha'],
        sigma_floor_v=sigma_floor_v.astype(np.float32),
        lam_v=lam_v.astype(np.float32),
    )
    np.savez_compressed(
        bundle_dir / 'mahalanobis.npz',
        mean=fit.mean,
        inv_cov=fit.inv_cov,
        log_det_cov=fit.log_det_cov,
        ood_member_index=np.array(args.ood_member_index, dtype=np.int32),
        ood_mode=np.array(args.ood_mode, dtype=np.dtype('U16')),
    )
    np.savez_compressed(
        bundle_dir / 'adjacency_cov.npz',
        sigma_A=sigma_A.astype(np.float32),
    )
    bundle = {
        'manifest': args.manifest,
        'lambda': lam,                     # median(λ_v) — back-compat scalar
        'lambda_raw': lam_raw,             # median(λ_v_raw) before identity-clamp
        'lambda_v_summary': {              # per-sensor stats for diagnostics
            'min': float(lam_v.min()), 'max': float(lam_v.max()),
            'mean': float(lam_v.mean()), 'median': float(np.median(lam_v)),
            'n_clamped_to_1': int((lam_v == 1.0).sum()),
        },
        'sma_window': args.sma_window,
        # Variant 1 — per-node OR
        'variant1_pernode_or': {
            'best_alpha': v1_best['alpha'],
            'metrics_labeled_val': {k: v1_best[k] for k in
                                    ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn', 'alarm_rate')},
            'sweep': v1_sweep,
        },
        # Variant 2 — max-of-V, validation-max τ
        'variant2_maxv_validation_max': {
            'threshold': v2_threshold,
            'metrics_labeled_val': v2_metrics,
        },
        # Variant 3 — max-of-V, 400-threshold F1 sweep
        'variant3_maxv_paper_sweep': {
            'threshold': v3_best['threshold'],
            'n_thresholds': args.n_thresholds_paper,
            'metrics_labeled_val': {k: v3_best[k] for k in
                                    ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn', 'alarm_rate')},
        },
        # Variant 4 — sustained-window after v3
        'variant4_sustained_window': {
            'threshold': v4_best['threshold'],
            'W': v4_best['W'],
            'K_w': v4_best['K_w'],
            'metrics_labeled_val': {k: v4_best[k] for k in
                                    ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn', 'alarm_rate')},
        },
        # Diagnostics
        'sigma_health': {
            'median_logvar_per_v_min': float(median_logvar_per_v.min()),
            'median_logvar_per_v_max': float(median_logvar_per_v.max()),
            'sat_low_max': float(sat_low_per_v.max()),
            'sat_high_max': float(sat_high_per_v.max()),
            'sigma_floor_min': float(sigma_floor_v.min()),
            'sigma_floor_median': float(np.median(sigma_floor_v)),
            'sigma_floor_max': float(sigma_floor_v.max()),
        },
        'omega_thresh': omega_thresh,
        'omega_thresh_per_query': omega_thresh_per_query,
        'theta_e_A': theta_e_A,
        'theta_e_B': theta_e_B,
        'sensitivity_threshold': sens_thresh,
        'reliability_csv': str(bundle_dir / 'reliability.csv'),
        'split': split,
        'hyperparameters': {
            'M': ensemble.cfg.M,
            'V': ensemble.cfg.node_num,
            'slide_win': ensemble.cfg.slide_win,
            'topk': ensemble.cfg.topk,
            'dim': ensemble.cfg.dim,
            'n_samples_mi': args.n_samples_mi,
            'n_sensitivity_samples': args.n_sensitivity_samples,
            'ood_member_index': args.ood_member_index,
        },
    }
    with (bundle_dir / 'bundle.json').open('w') as f:
        json.dump(bundle, f, indent=2)
    print(f'[calibrate] wrote bundle to {bundle_dir}')

    for tmp in (c_csv, lv_csv):
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    main()
