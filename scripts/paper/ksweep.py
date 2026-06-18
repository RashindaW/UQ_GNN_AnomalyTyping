#!/usr/bin/env python3
"""Ideal-K sweep driver for G-DeltaUQ anchors.

For each anchor count K, this script:
  1. Runs K-anchor G-DeltaUQ inference (the SAME path as
     scripts/eval_paper_protocol_gdeltauq.py: build val + test TimeDatasets,
     load checkpoint + the K-anchor anchor_pool + aleatoric head from
     calibration_bundle_K{K}, run_inference on val and test). Wall-clock and
     peak GPU memory (torch.cuda.max_memory_allocated) are recorded.
  2. Computes DETECTION metrics (M0 residual-only top-1 aggregate): F1 and
     PA%K-AUC, using exactly the competitor eval primitives
     (fusion_sweep_K100_full.eval_score_full + pa_k_metric.f1_pa_k_auc) on the
     SAME eval split/bundle as competitors/common/eval_from_arrays.py.
  3. Computes EPISTEMIC QUALITY metrics directly from the produced arrays:
       - attack_AUROC_Upar : roc_auc_score(test_attack_label,
                             U_par.mean(axis=1))   (epistemic anomaly detection)
       - ause_sigtot_norm  : AUSE / sparsification of sigma_tot =
                             sqrt(sigma2_ale + U_par) over NOMINAL timesteps,
                             normalized 0..1 against the oracle error-sort.
       - mean_Upar, std_Upar : stability (Monte-Carlo K-anchor mean variance).
  4. Writes one CSV row per K and (optionally) deletes the large arrays.npz.

Reproducible: re-running with the same K grid reproduces the CSV. Inference
cost scales ~linearly in K (K forward passes per window).

Usage:
  python scripts/paper/ksweep.py --ks 20,30,50,70,100,140,200 \
      --device cuda:0 --out_csv results/paper/ksweep/ksweep.csv \
      --work_dir results/ksweep --keep_arrays 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from sklearn.metrics import roc_auc_score

# ---- inference building blocks (same as eval_paper_protocol_gdeltauq.py) ----
from datasets.TimeDataset import TimeDataset
from inference_gdeltauq import LoadedGDeltaUQ, run_inference
from models.GDN_GDeltaUQ import GDN_GDeltaUQ
from models.aleatoric_head import AleatoricHead
from util.env import set_device, get_device
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data
import pandas as pd

# ---- detection eval primitives (same path as eval_from_arrays.py) ----
from fusion_sweep_K100_full import setup_context, eval_score_full
from pa_k_metric import f1_pa_k_auc


# ===================== inference dataset builders ============================
def _build_train_dataset(dataset_name, slide_win):
    train_csv = pd.read_csv(f'./data/{dataset_name}/train.csv', sep=',', index_col=0)
    if 'attack' in train_csv.columns:
        train_csv = train_csv.drop(columns=['attack'])
    feature_map = get_feature_map(dataset_name)
    fc_struc = get_fc_graph_struc(dataset_name)
    fc_edge_index = build_loc_net(fc_struc, list(train_csv.columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    indata = construct_data(train_csv, feature_map, labels=0)
    cfg = {'slide_win': slide_win, 'slide_stride': 1}
    ds = TimeDataset(indata, fc_edge_index, mode='test', config=cfg)
    return ds, feature_map, fc_edge_index


def _build_test_dataset(dataset_name, slide_win):
    test_csv = pd.read_csv(f'./data/{dataset_name}/test.csv', sep=',', index_col=0)
    feature_map = get_feature_map(dataset_name)
    fc_struc = get_fc_graph_struc(dataset_name)
    cols_no_attack = [c for c in test_csv.columns if c != 'attack']
    fc_edge_index = build_loc_net(fc_struc, cols_no_attack, feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    attack_col = test_csv['attack'].tolist() if 'attack' in test_csv.columns else 0
    indata = construct_data(test_csv, feature_map, labels=attack_col)
    cfg = {'slide_win': slide_win, 'slide_stride': 1}
    ds = TimeDataset(indata, fc_edge_index, mode='test', config=cfg)
    return ds, feature_map, fc_edge_index


def _row_range_to_window_range(row_range, total_windows, slide_win):
    r0, r1 = row_range
    return (max(0, r0 - slide_win), min(total_windows, r1 - slide_win))


def build_model(hp, fc_edge_index, V, device):
    causal_mask_tensor = None
    cm_path = hp.get('causal_mask', '')
    if cm_path:
        from models.causal_mask import load_causal_mask
        causal_mask_tensor = load_causal_mask(cm_path, None)
    causal_restrict_tensor = None
    cr_path = hp.get('causal_restrict', '')
    if cr_path:
        from models.causal_mask import load_causal_mask
        causal_restrict_tensor = load_causal_mask(cr_path, None)
    model = GDN_GDeltaUQ(
        [fc_edge_index], V,
        dim=int(hp['dim']),
        input_dim=int(hp['slide_win']),
        out_layer_num=int(hp['out_layer_num']),
        out_layer_inter_dim=int(hp['out_layer_inter_dim']),
        topk=int(hp['topk']),
        n_gnn_layers=int(hp['n_gnn_layers']),
        causal_mask=causal_mask_tensor,
        causal_mask_keep_self=bool(hp.get('causal_mask_keep_self', 1)),
        use_learnable_adj=bool(hp.get('use_learnable_adj', 0)),
        lsa_tau=float(hp.get('lsa_tau', 1.0)),
        causal_restrict=causal_restrict_tensor,
        causal_restrict_mode=hp.get('causal_restrict_mode', 'pure'),
        causal_restrict_keep_self=bool(hp.get('causal_restrict_keep_self', 1)),
    ).to(device)
    return model


# ============================ epistemic metrics =============================
def compute_ause_norm(err, sigma, n_bins=20):
    """Normalized AUSE (sparsification error) for a 1-D error / uncertainty pair.

    err   : per-sample scalar forecast error (>=0), shape (N,)
    sigma : per-sample predicted predictive std, shape (N,)
    We drop the most-uncertain fraction (by predicted sigma) in steps and track
    the mean error of the remaining samples (sparsification curve). The oracle
    curve sorts by the TRUE error. AUSE = area between the predicted-sort curve
    and the oracle curve. We normalize by the area between a random-removal
    baseline (the flat mean-error line) and the oracle, giving a 0..1 score
    where 0 = oracle-perfect ranking and 1 = no better than random.
    Lower is better.
    """
    err = np.asarray(err, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    N = err.shape[0]
    if N < n_bins + 1:
        return float('nan')
    fracs = np.linspace(0.0, 1.0, n_bins + 1)[:-1]  # remove 0%, 5%, ... <100%

    order_pred = np.argsort(-sigma)   # most uncertain first
    order_true = np.argsort(-err)     # largest error first
    err_pred_sorted = err[order_pred]
    err_true_sorted = err[order_true]

    def curve(err_sorted):
        out = np.empty(fracs.shape[0])
        for i, fr in enumerate(fracs):
            k = int(np.floor(fr * N))
            remaining = err_sorted[k:]
            out[i] = remaining.mean() if remaining.size else 0.0
        return out

    spars_pred = curve(err_pred_sorted)   # drop most-uncertain by predicted sigma
    spars_oracle = curve(err_true_sorted)  # drop largest-error (oracle)
    random_line = np.full(fracs.shape[0], err.mean())  # random removal ~ flat mean

    area_pred = np.trapz(spars_pred - spars_oracle, fracs)
    area_rand = np.trapz(random_line - spars_oracle, fracs)
    if area_rand <= 1e-12:
        return 0.0
    return float(np.clip(area_pred / area_rand, 0.0, None))


def epistemic_metrics(arrays_path):
    d = np.load(arrays_path)
    U_par = d['test_U_par'].astype(np.float64)            # (T, V)
    sigma2_ale = d['test_sigma2_ale'].astype(np.float64)  # (T, V)
    mu = d['test_mu_bar'].astype(np.float64)              # (T, V)
    gt = d['test_ground_truth'].astype(np.float64)        # (T, V)
    label = d['test_attack_label'].astype(np.int8)        # (T,)

    upar_mean_v = U_par.mean(axis=1)                       # (T,)

    # (a) attack-AUROC of the epistemic channel
    if label.min() == label.max():
        attack_auroc = float('nan')
    else:
        attack_auroc = float(roc_auc_score(label, upar_mean_v))

    # (b) AUSE of predictive sigma_tot on NOMINAL timesteps.
    sigma_tot_v = np.sqrt(np.clip(sigma2_ale + U_par, 0, None))  # (T, V)
    abs_err_v = np.abs(gt - mu)                                  # (T, V)
    # Reduce over sensors: mean over V (a robust per-window predictive summary).
    sigma_tot = sigma_tot_v.mean(axis=1)
    abs_err = abs_err_v.mean(axis=1)
    nominal = (label == 0)
    ause_norm = compute_ause_norm(abs_err[nominal], sigma_tot[nominal])

    # (c) stability
    mean_upar = float(U_par.mean())
    std_upar = float(upar_mean_v.std())  # across-time std of the per-window mean

    return {
        'attack_AUROC_Upar': attack_auroc,
        'ause_sigtot_norm': ause_norm,
        'mean_Upar': mean_upar,
        'std_Upar_time': std_upar,
    }


# ============================ detection metrics =============================
def detection_m0(arrays_path, split_path, bundle_path, slide_win, seed):
    """M0 residual-only: F1 + PA%K-AUC, identical primitives to eval_from_arrays."""
    ctx_args = argparse.Namespace(
        arrays=str(arrays_path), split=str(split_path), bundle=str(bundle_path),
        slide_win=slide_win, seed=seed,
    )
    ctx = setup_context(ctx_args)
    label = ctx['label']
    agg = ctx['agg']
    m0 = eval_score_full(agg, label)
    pa = f1_pa_k_auc(agg, label, K_grid=np.arange(0, 101, 1), n_thresholds=400)
    return {
        'M0_F1': float(m0['F1']),
        'M0_P': float(m0['P']),
        'M0_R': float(m0['R']),
        'M0_PAK_AUC': float(pa['PA_K_AUC']),
        'M0_PA_K0': float(pa['F1_PA_K0']),
        'M0_PA_K100': float(pa['F1_PA_K100']),
    }


# ================================ per-K run ================================
def run_one_k(K, hp, ctx_objs, args):
    device = ctx_objs['device']
    bundle_dir = REPO_ROOT / args.bundle_root / f'calibration_bundle_K{K}'
    anchor_pool = torch.load(bundle_dir / 'anchor_pool.pt', map_location='cpu')
    K_actual = int(anchor_pool.shape[0])
    print(f'[K={K}] anchor_pool.shape={tuple(anchor_pool.shape)} -> K_actual={K_actual}',
          flush=True)

    aleatoric_head = AleatoricHead(
        hidden_dim=int(hp['dim']), num_sensors=ctx_objs['V'],
        sensor_embed_dim=16, mlp_hidden=64,
    )
    aleatoric_head.load_state_dict(torch.load(bundle_dir / 'aleatoric_head.pt',
                                              map_location='cpu'))
    aleatoric_head.to(device).eval()

    loaded = LoadedGDeltaUQ(
        model=ctx_objs['model'], aleatoric_head=aleatoric_head,
        anchor_pool=anchor_pool, q_v=None, u_bar_norm={},
        feature_map=ctx_objs['feature_map'], cfg=hp, device=device,
    )

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    val_out = run_inference(loaded, ctx_objs['val_subset'], batch_size=int(hp['batch']))
    test_out = run_inference(loaded, ctx_objs['test_ds'], batch_size=int(hp['batch']))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall_s = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
               if device.type == 'cuda' else float('nan'))
    print(f'[K={K}] inference wall={wall_s:.2f}s peak_gpu={peak_mb:.1f}MB '
          f'test_mu_bar={test_out.mu_bar.shape}', flush=True)

    # write arrays.npz (minimal channels the eval + epistemic metrics need)
    run_dir = REPO_ROOT / args.work_dir / f'K{K}' / datetime.now().strftime('%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = run_dir / 'arrays.npz'
    np.savez(
        arrays_path,
        test_mu_bar=test_out.mu_bar,
        test_ground_truth=test_out.ground_truth,
        test_attack_label=test_out.attack_label,
        test_U_par=test_out.U_par,
        test_U_str=test_out.U_str,
        test_U_dist=test_out.U_dist,
        test_sigma2_ale=test_out.sigma2_ale,
        val_mu_bar=val_out.mu_bar,
        val_ground_truth=val_out.ground_truth,
    )
    # free GPU tensors before eval
    del val_out, test_out, loaded, aleatoric_head, anchor_pool
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    det = detection_m0(arrays_path, args.eval_split, args.eval_bundle,
                       args.slide_win, args.seed)
    epi = epistemic_metrics(arrays_path)

    row = {
        'K': K_actual, 'wall_s': round(wall_s, 3),
        'peak_gpu_mb': round(peak_mb, 1),
        'M0_F1': det['M0_F1'], 'M0_PAK_AUC': det['M0_PAK_AUC'],
        'M0_P': det['M0_P'], 'M0_R': det['M0_R'],
        'M0_PA_K0': det['M0_PA_K0'], 'M0_PA_K100': det['M0_PA_K100'],
        'attack_AUROC_Upar': epi['attack_AUROC_Upar'],
        'ause_sigtot_norm': epi['ause_sigtot_norm'],
        'mean_Upar': epi['mean_Upar'], 'std_Upar_time': epi['std_Upar_time'],
        'arrays': str(arrays_path),
    }
    # save per-K metrics json (kept even if arrays deleted)
    (run_dir / 'metrics.json').write_text(json.dumps(row, indent=2))
    print(f'[K={K}] M0_F1={row["M0_F1"]:.4f} PAK_AUC={row["M0_PAK_AUC"]:.4f} '
          f'attackAUROC={row["attack_AUROC_Upar"]:.4f} '
          f'AUSE={row["ause_sigtot_norm"]:.4f} meanUpar={row["mean_Upar"]:.5f}',
          flush=True)

    if not args.keep_arrays:
        try:
            arrays_path.unlink()
            print(f'[K={K}] deleted {arrays_path} (kept metrics.json)', flush=True)
        except OSError:
            pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', default='20,30,50,70,100,140,200')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--checkpoint',
                    default='pretrained/swat_gdeltauq_sw60/best_0513-211014.pt')
    ap.add_argument('--hyperparameters',
                    default='pretrained/swat_gdeltauq_sw60/hyperparameters_0513-211014.json')
    ap.add_argument('--bundle_root', default='pretrained/swat_gdeltauq_sw60')
    ap.add_argument('--split_path', default='data/swat/gdeltauq_split.json',
                    help='inference val/aleatoric split')
    ap.add_argument('--eval_split',
                    default='pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json')
    ap.add_argument('--eval_bundle',
                    default='pretrained/swat_ensemble/calibration_bundle')
    ap.add_argument('--slide_win', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--work_dir', default='results/ksweep')
    ap.add_argument('--out_csv', default='results/paper/ksweep/ksweep.csv')
    ap.add_argument('--keep_arrays', type=int, default=0)
    ap.add_argument('--resume', action='store_true',
                    help='reuse existing per-K metrics.json under work_dir and '
                         'only compute the remaining K (crash-resilient).')
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    with open(REPO_ROOT / args.hyperparameters) as f:
        hp = json.load(f)

    set_device(args.device)
    device = get_device()
    # get_device() may return a plain string (e.g. 'cuda:0'); normalize so that
    # device.type works for the cuda timing/peak-mem guards below.
    device = torch.device(device) if not isinstance(device, torch.device) else device
    dataset_name = hp['dataset']
    slide_win = int(hp['slide_win'])

    # build datasets + model ONCE (shared across all K)
    train_full_ds, feature_map, fc_edge_index = _build_train_dataset(dataset_name, slide_win)
    V = len(feature_map)
    with open(REPO_ROOT / args.split_path) as f:
        split = json.load(f)
    val_range = _row_range_to_window_range(split['val_rows'], len(train_full_ds), slide_win)
    val_subset = Subset(train_full_ds, list(range(*val_range)))
    test_ds, _, _ = _build_test_dataset(dataset_name, slide_win)
    print(f'val windows={len(val_subset)} test windows={len(test_ds)} V={V}', flush=True)

    model = build_model(hp, fc_edge_index, V, device)
    state = torch.load(REPO_ROOT / args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f'loaded checkpoint {args.checkpoint}', flush=True)

    ctx_objs = {
        'device': device, 'model': model, 'feature_map': feature_map,
        'val_subset': val_subset, 'test_ds': test_ds, 'V': V,
    }

    out_csv = REPO_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['K', 'wall_s', 'peak_gpu_mb', 'M0_F1', 'M0_PAK_AUC',
                  'M0_P', 'M0_R', 'M0_PA_K0', 'M0_PA_K100',
                  'attack_AUROC_Upar', 'ause_sigtot_norm',
                  'mean_Upar', 'std_Upar_time', 'arrays']

    def write_csv_atomic(all_rows):
        # Write to a temp file then os.replace so a kill mid-write can never
        # truncate the real CSV (the failure mode we hit on the shared GPU).
        tmp = out_csv.with_suffix('.csv.tmp')
        with tmp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            for r in sorted(all_rows, key=lambda x: x['K']):
                w.writerow({k: r.get(k, '') for k in fieldnames})
        os.replace(tmp, out_csv)

    rows = []
    if args.resume:
        # Reuse any already-computed per-K metrics.json under work_dir so a
        # crashed run can be continued without recomputing finished K.
        for mj in sorted((REPO_ROOT / args.work_dir).glob('K*/*/metrics.json')):
            try:
                r = json.loads(mj.read_text())
                rows.append(r)
                print(f'[resume] loaded {mj} -> K={r["K"]}', flush=True)
            except Exception as e:  # noqa: BLE001
                print(f'[resume] skip {mj}: {e}', flush=True)
        done_ks = {int(r['K']) for r in rows}
        ks = [k for k in ks if k not in done_ks]
        if rows:
            write_csv_atomic(rows)
        print(f'[resume] already done: {sorted(done_ks)}; remaining: {ks}',
              flush=True)

    for K in ks:
        row = run_one_k(K, hp, ctx_objs, args)
        rows.append(row)
        write_csv_atomic(rows)   # atomic, survives a kill mid-write
        print(f'=== wrote {out_csv} ({len(rows)} rows) ===', flush=True)

    print('DONE_KSWEEP', flush=True)


if __name__ == '__main__':
    main()
