"""Emit deep-ensemble epistemic arrays.npz for the ensemble-vs-anchor comparison.

Runs the 10-member heteroscedastic GDN_UQ ensemble over the EXACT same test +
val target-row streams the G-DeltaUQ anchor used, so the emitted arrays.npz flows
through competitors/common/eval_from_arrays.py identically to the anchor's.

Anchor stream (CONFIG A) was built with slide_win=60:
  - test : full test.csv, TimeDataset mode='test' -> target rows 60..44775 (44716 windows)
  - val  : train.csv, mode='test', subset to val_rows=[34499,39427]
           -> target rows 34499..39426 (4928 windows)

The ensemble members were trained with slide_win=5 (input_dim=5). We therefore run
the ensemble at slide_win=5 (correct model input) over the full streams, then SLICE
to the anchor's exact target rows. Because both use the same CSVs and same target
rows, test_ground_truth / test_attack_label / val_ground_truth come out IDENTICAL to
the anchor's arrays -- which we assert as a correctness gate.

For each M in {5, 10} we decompose the first M members via
util.uq_decomposition.variance_decomposition and write one arrays.npz per config.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from inference import load_ensemble, build_dataset_from_csv, run_inference  # noqa: E402
from util.uq_decomposition import variance_decomposition  # noqa: E402


def run_members(ensemble, dataset, batch_size):
    """Run all M members once, return (mu_per_member (M,T,V), logvar_per_member (M,T,V),
    ground_truth (T,V), attack_label (T,)). Mirrors inference.run_inference but keeps the
    full per-member stacks (we re-decompose for M=5 and M=10)."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = ensemble.device
    M = ensemble.cfg.M
    V = ensemble.cfg.node_num
    T = len(dataset)
    mu_buf = np.empty((M, T, V), dtype=np.float32)
    lv_buf = np.empty((M, T, V), dtype=np.float32)
    y_buf = np.empty((T, V), dtype=np.float32)
    label_buf = np.empty((T,), dtype=np.int8)
    pos = 0
    for batch in loader:
        x, y, label, edge_index = batch
        x = x.to(device).float()
        edge_index = edge_index.to(device).long()
        b = x.shape[0]
        with torch.no_grad():
            for m_idx, member in enumerate(ensemble.members):
                mu, log_var = member(x, edge_index)
                mu_buf[m_idx, pos:pos + b, :] = mu.detach().cpu().numpy()
                lv_buf[m_idx, pos:pos + b, :] = log_var.detach().cpu().numpy()
        y_buf[pos:pos + b, :] = y.cpu().numpy()
        label_buf[pos:pos + b] = label.cpu().numpy().astype(np.int8)
        pos += b
    assert pos == T, f"buffer underflow: {pos}/{T}"
    return mu_buf, lv_buf, y_buf, label_buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=str(REPO / 'pretrained/swat_ensemble/manifest.json'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--anchor', default=str(REPO / 'results/gdn/ref_seed42/arrays.npz'))
    ap.add_argument('--split', default=str(REPO / 'data/swat/gdeltauq_split.json'))
    ap.add_argument('--anchor-slide-win', type=int, default=60)
    ap.add_argument('--out-dir', default=str(REPO / 'results/paper/ensemble'))
    ap.add_argument('--m-list', default='5,10')
    ap.add_argument('--timing-m', type=int, default=10,
                    help='Which M to time/peak-mem for the compute row (the strong ref).')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_list = [int(x) for x in args.m_list.split(',')]

    anchor = np.load(args.anchor)
    a_test_gt = anchor['test_ground_truth'].astype(np.float32)
    a_test_lab = anchor['test_attack_label'].astype(np.int8)
    a_val_gt = anchor['val_ground_truth'].astype(np.float32)
    T_test = a_test_gt.shape[0]
    T_val = a_val_gt.shape[0]
    print(f'[emit] anchor test stream T={T_test} val stream T={T_val}', flush=True)

    with open(args.split) as f:
        split = json.load(f)
    val_rows = split['val_rows']          # [34499, 39427] target-row range (inclusive-exclusive)
    asw = args.anchor_slide_win

    ensemble = load_ensemble(args.manifest, device=args.device, repo_root=REPO)
    sw = ensemble.cfg.slide_win
    print(f'[emit] ensemble M={ensemble.cfg.M} V={ensemble.cfg.node_num} '
          f'slide_win={sw} seeds={ensemble.seeds}', flush=True)

    # ------------------------------------------------------------------ TEST
    # Full test.csv at slide_win=sw -> windows predict target rows sw..(N-1).
    # Anchor used slide_win=asw -> target rows asw..(N-1) == last T_test windows.
    test_csv = REPO / 'data' / 'swat' / 'test.csv'
    ds_test = build_dataset_from_csv(test_csv, ensemble.feature_map, ensemble.fc_edge_index,
                                     slide_win=sw, slide_stride=1, mode='test')
    print(f'[emit] full test windows (sw={sw}): {len(ds_test)}', flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mu_te, lv_te, y_te, lab_te = run_members(ensemble, ds_test, ensemble.cfg.batch)
    test_wall = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else float('nan')
    print(f'[emit] test inference (M={ensemble.cfg.M}): {test_wall:.1f}s peak_mem={peak_mb:.1f}MB', flush=True)

    # Slice last T_test windows to align target rows asw..(N-1) with the anchor.
    if len(ds_test) < T_test:
        raise SystemExit(f'test windows {len(ds_test)} < anchor T_test {T_test}')
    off_te = len(ds_test) - T_test
    mu_te = mu_te[:, off_te:, :]
    lv_te = lv_te[:, off_te:, :]
    y_te = y_te[off_te:, :]
    lab_te = lab_te[off_te:]
    print(f'[emit] test sliced off first {off_te} windows -> T={y_te.shape[0]}', flush=True)

    # CORRECTNESS GATE: ground truth + labels must match the anchor exactly.
    gt_ok = np.allclose(y_te, a_test_gt, atol=1e-4)
    lab_ok = bool((lab_te == a_test_lab).all())
    gt_maxabs = float(np.max(np.abs(y_te - a_test_gt)))
    print(f'[emit] GATE test_gt match={gt_ok} (max|diff|={gt_maxabs:.3e})  '
          f'test_label match={lab_ok}', flush=True)
    if not (gt_ok and lab_ok):
        raise SystemExit('[emit] FATAL: test stream misaligned with anchor')

    # ------------------------------------------------------------------ VAL
    # Anchor val came from train.csv windows (slide_win=asw) subset to val_rows.
    # Build train.csv at sw, keep windows whose target row is in [val_rows[0], val_rows[1]).
    train_csv = REPO / 'data' / 'swat' / 'train.csv'
    ds_val_full = build_dataset_from_csv(train_csv, ensemble.feature_map, ensemble.fc_edge_index,
                                         slide_win=sw, slide_stride=1, mode='test')
    # window k -> target row (sw + k). Keep target rows in [val_rows[0], val_rows[1]).
    vr0, vr1 = val_rows
    k0 = vr0 - sw
    k1 = vr1 - sw
    print(f'[emit] full train windows (sw={sw}): {len(ds_val_full)}; val window slice [{k0},{k1}) '
          f'-> {k1 - k0}', flush=True)
    from torch.utils.data import Subset
    ds_val = Subset(ds_val_full, list(range(k0, k1)))
    mu_va, lv_va, y_va, lab_va = run_members(ensemble, ds_val, ensemble.cfg.batch)
    val_gt_ok = np.allclose(y_va, a_val_gt, atol=1e-4)
    val_gt_maxabs = float(np.max(np.abs(y_va - a_val_gt)))
    print(f'[emit] GATE val_gt match={val_gt_ok} (max|diff|={val_gt_maxabs:.3e}) T_val={y_va.shape[0]}',
          flush=True)
    if not val_gt_ok:
        raise SystemExit('[emit] FATAL: val stream misaligned with anchor')

    # ------------------------------------------------------------------ EMIT per M
    summary = {
        'manifest': args.manifest, 'seeds': ensemble.seeds,
        'slide_win': sw, 'anchor_slide_win': asw,
        'T_test': int(T_test), 'T_val': int(T_val),
        'test_inference_wall_s_M10': float(test_wall),
        'test_peak_mem_MB_M10': float(peak_mb),
        'configs': {},
    }
    for M in m_list:
        dec = variance_decomposition(mu_te[:M], lv_te[:M])      # (T,V) each
        dec_val = variance_decomposition(mu_va[:M], lv_va[:M])
        out_path = out_dir / f'arrays_ensemble{M}.npz'
        np.savez(
            out_path,
            test_mu_bar=dec['mu_bar'].astype(np.float32),
            test_ground_truth=a_test_gt,                # use anchor's (identical, asserted)
            test_attack_label=a_test_lab,
            test_U_par=dec['sigma2_epistemic'].astype(np.float32),
            test_sigma2_ale=dec['sigma2_aleatoric'].astype(np.float32),
            test_sigma2_total=dec['sigma2_total'].astype(np.float32),
            val_mu_bar=dec_val['mu_bar'].astype(np.float32),
            val_ground_truth=a_val_gt,
        )
        summary['configs'][f'ensemble{M}'] = {
            'M': M, 'arrays': str(out_path),
            'sig2_epi_mean': float(dec['sigma2_epistemic'].mean()),
            'sig2_epi_max': float(dec['sigma2_epistemic'].max()),
            'sig2_ale_mean': float(dec['sigma2_aleatoric'].mean()),
            'sig2_tot_mean': float(dec['sigma2_total'].mean()),
        }
        print(f'[emit] wrote {out_path} (M={M}) '
              f'sig2_epi.mean={dec["sigma2_epistemic"].mean():.4g}', flush=True)

    (out_dir / 'emit_summary.json').write_text(json.dumps(summary, indent=2))
    print('[emit] DONE; summary -> ' + str(out_dir / 'emit_summary.json'), flush=True)


if __name__ == '__main__':
    main()
