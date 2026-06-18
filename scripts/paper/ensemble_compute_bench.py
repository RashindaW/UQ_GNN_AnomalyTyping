"""Fair compute micro-benchmark: ANCHOR (K=100 forward_anchored of ONE model) vs
ENSEMBLE-10 (10 forward passes of 10 stored models), on the SAME GPU, SAME window
count. Times the epistemic-inference inner loop only (load excluded), measures peak
GPU mem, then extrapolates wall to the full 44771-window test stream.

Both share V=51, dim=64, topk=15. The anchor runs at slide_win=60 (input_dim=60);
the ensemble at slide_win=5. We time per-window throughput on a fixed N_BENCH window
budget for each and report per-window cost, full-stream extrapolation, and peak mem.
Model on-disk storage is reported too (1 anchor ckpt vs 10 member ckpts).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.TimeDataset import TimeDataset  # noqa: E402
from util.net_struct import get_feature_map, get_fc_graph_struc  # noqa: E402
from util.preprocess import build_loc_net, construct_data  # noqa: E402
from util.env import set_device  # noqa: E402


def build_test_ds(slide_win, n_rows):
    df = pd.read_csv(REPO / 'data/swat/test.csv', index_col=0)
    df = df.iloc[:n_rows]
    fm = get_feature_map('swat')
    fc = get_fc_graph_struc('swat')
    cols = [c for c in df.columns if c != 'attack']
    ei = torch.tensor(build_loc_net(fc, cols, feature_map=fm), dtype=torch.long)
    attack = df['attack'].tolist() if 'attack' in df.columns else 0
    indata = construct_data(df, fm, labels=attack)
    ds = TimeDataset(indata, ei, mode='test', config={'slide_win': slide_win, 'slide_stride': 1})
    return ds, fm, ei


def bench_ensemble(device, n_bench_rows, full_T):
    from inference import load_ensemble, run_inference
    ens = load_ensemble(REPO / 'pretrained/swat_ensemble/manifest.json', device=device, repo_root=REPO)
    sw = ens.cfg.slide_win
    ds, _, _ = build_test_ds(sw, n_bench_rows)
    n = len(ds)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time()
    _ = run_inference(ens, ds, batch_size=ens.cfg.batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else float('nan')
    per_win = dt / n
    ckpts = glob.glob(str(REPO / 'pretrained/swat_ensemble/member_*/best_*.pt'))
    # newest per member dir
    size_mb = 0.0
    for d in sorted(glob.glob(str(REPO / 'pretrained/swat_ensemble/member_*'))):
        fs = sorted(glob.glob(d + '/best_*.pt'), key=lambda p: Path(p).stat().st_mtime)
        if fs:
            size_mb += Path(fs[-1]).stat().st_size / 1e6
    return dict(n_windows=n, wall_s=dt, per_window_ms=per_win * 1e3,
                full_stream_extrap_s=per_win * full_T, peak_mem_MB=peak,
                n_models=10, fwd_passes=10, storage_MB=size_mb)


def bench_anchor(device, n_bench_rows, full_T):
    from inference_gdeltauq import LoadedGDeltaUQ, run_inference
    from models.GDN_GDeltaUQ import GDN_GDeltaUQ
    from models.aleatoric_head import AleatoricHead
    set_device(device)
    dev = torch.device(device)
    hp_path = sorted(glob.glob(str(REPO / 'pretrained/swat_gdeltauq_sw60/hyperparameters*.json')))[0]
    hp = json.load(open(hp_path))
    ckpt = str(REPO / 'pretrained/swat_gdeltauq_sw60/best_0513-211014.pt')
    bundle = REPO / 'pretrained/swat_gdeltauq_sw60/calibration_bundle_K100'
    sw = int(hp['slide_win'])
    ds, fm, ei = build_test_ds(sw, n_bench_rows)
    V = len(fm)
    model = GDN_GDeltaUQ(
        [ei.to(dev)], V, dim=int(hp['dim']), input_dim=sw,
        out_layer_num=int(hp['out_layer_num']), out_layer_inter_dim=int(hp['out_layer_inter_dim']),
        topk=int(hp['topk']), n_gnn_layers=int(hp['n_gnn_layers']),
    ).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()
    anchor_pool = torch.load(bundle / 'anchor_pool.pt', map_location='cpu')  # (100,V,64)
    aleatoric = AleatoricHead(hidden_dim=int(hp['dim']), num_sensors=V,
                              sensor_embed_dim=16, mlp_hidden=64)
    aleatoric.load_state_dict(torch.load(bundle / 'aleatoric_head.pt', map_location='cpu'))
    aleatoric.to(dev).eval()
    loaded = LoadedGDeltaUQ(model=model, aleatoric_head=aleatoric, anchor_pool=anchor_pool,
                            q_v=None, u_bar_norm={}, feature_map=fm, cfg=hp, device=dev)
    K = anchor_pool.shape[0]
    n = len(ds)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time()
    _ = run_inference(loaded, ds, batch_size=int(hp['batch']))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else float('nan')
    per_win = dt / n
    storage = Path(ckpt).stat().st_size / 1e6
    return dict(n_windows=n, wall_s=dt, per_window_ms=per_win * 1e3,
                full_stream_extrap_s=per_win * full_T, peak_mem_MB=peak,
                n_models=1, fwd_passes=int(K), storage_MB=storage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-bench-rows', type=int, default=2060)   # ~2000 windows after sw
    ap.add_argument('--full-T', type=int, default=44771)        # full test windows (sw=5 stream)
    ap.add_argument('--out', default=str(REPO / 'results/paper/ensemble/compute_bench.json'))
    args = ap.parse_args()

    print('[bench] ENSEMBLE-10 ...', flush=True)
    ens = bench_ensemble(args.device, args.n_bench_rows, args.full_T)
    print(f'  ens: {ens["n_windows"]} win  {ens["wall_s"]:.2f}s  '
          f'{ens["per_window_ms"]:.3f} ms/win  extrap_full={ens["full_stream_extrap_s"]:.1f}s  '
          f'peak={ens["peak_mem_MB"]:.1f}MB storage={ens["storage_MB"]:.1f}MB', flush=True)

    print('[bench] ANCHOR K=100 ...', flush=True)
    anc = bench_anchor(args.device, args.n_bench_rows, args.full_T)
    print(f'  anc: {anc["n_windows"]} win  {anc["wall_s"]:.2f}s  '
          f'{anc["per_window_ms"]:.3f} ms/win  extrap_full={anc["full_stream_extrap_s"]:.1f}s  '
          f'peak={anc["peak_mem_MB"]:.1f}MB storage={anc["storage_MB"]:.1f}MB', flush=True)

    ratio_wall = anc['per_window_ms'] / ens['per_window_ms']
    ratio_fwd = anc['fwd_passes'] / ens['fwd_passes']
    ratio_storage = ens['storage_MB'] / anc['storage_MB']
    out = {
        'ensemble10': ens, 'anchor_K100': anc,
        'ratios': {
            'anchor_over_ensemble_wall_per_window': ratio_wall,
            'anchor_over_ensemble_fwd_passes': ratio_fwd,
            'ensemble_over_anchor_storage': ratio_storage,
        },
        'full_T': args.full_T,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2)
    print(f'[bench] anchor/ensemble per-window wall ratio = {ratio_wall:.2f}x  '
          f'fwd-pass ratio = {ratio_fwd:.1f}x  ensemble/anchor storage = {ratio_storage:.1f}x', flush=True)
    print(f'[bench] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
