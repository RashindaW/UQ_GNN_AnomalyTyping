"""Detection on the SWaT test window using a calibrated GDN_GDeltaUQ.

For each timestep:
  1. K-anchor inference -> mu_bar, U_par, U_str, U_dist, sigma2_ale.
  2. First-stage gate: |y_v - mu_bar_v| > q_v * sqrt(sigma2_ale_v) for each
     sensor. System alarm if any sensor flagged (plan Phase 4.2).
  3. If alarm:
        rho_par  = max_v U_par_v / U_bar_par
        rho_str  = max_e U_str_e / U_bar_str
        rho_dist = U_dist     / U_bar_dist
     Type rule (plan Phase 4.3):
        - max(rho_*) < 2     -> NONE
        - elif 2 runner-ups > 0.7 * rho*  -> MIXED
        - else argmax type   -> PAR / STR / DIST
  4. Persist a per-timestep record + report.json (F1/P/R, type distribution).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.TimeDataset import TimeDataset
from inference_gdeltauq import (
    LoadedGDeltaUQ, run_inference, top_k_edges_per_timestep, top_k_sensors_by_upar,
)
from models.GDN_GDeltaUQ import GDN_GDeltaUQ
from models.aleatoric_head import AleatoricHead
from util.env import set_device, get_device
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data


PAR, STR, DIST, MIXED, NONE = 'PARAMETRIC', 'STRUCTURAL', 'DISTRIBUTIONAL', 'MIXED', 'NONE'


def assign_type(rho_par, rho_str, rho_dist, mixed_threshold=0.7, none_threshold=2.0):
    rhos = np.array([rho_par, rho_str, rho_dist])
    names = np.array([PAR, STR, DIST])
    if rhos.max() < none_threshold:
        return NONE
    star = rhos.argmax()
    rho_star = rhos[star]
    others = np.delete(rhos, star)
    if (others > mixed_threshold * rho_star).all():
        return MIXED
    return str(names[star])


def _build_test_dataset(dataset_name, slide_win, test_row_range=None):
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
    if test_row_range is not None:
        r0, r1 = test_row_range
        w0 = max(0, r0 - slide_win)
        w1 = min(len(ds), r1 - slide_win)
        ds = Subset(ds, list(range(w0, w1)))
    return ds, feature_map, fc_edge_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint', type=str, required=True)
    parser.add_argument('-hyperparameters', type=str, required=True)
    parser.add_argument('-bundle_dir', type=str,
                        default='pretrained/swat_gdeltauq/calibration_bundle')
    parser.add_argument('-test_row_start', type=int, default=None,
                        help='optional row index to start the test slice at')
    parser.add_argument('-test_row_end', type=int, default=None,
                        help='optional row index to end the test slice at (exclusive)')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-top_k_sensors', type=int, default=3)
    parser.add_argument('-top_k_edges', type=int, default=3)
    parser.add_argument('-results_dir', type=str, default='results/swat_gdeltauq')
    args = parser.parse_args()

    with open(args.hyperparameters) as f:
        hp = json.load(f)
    bundle_dir = Path(args.bundle_dir)
    with (bundle_dir / 'bundle.json').open() as f:
        bundle = json.load(f)

    set_device(args.device)
    device = get_device()

    test_range = None
    if args.test_row_start is not None and args.test_row_end is not None:
        test_range = (args.test_row_start, args.test_row_end)
    dataset_name = hp['dataset']
    slide_win = int(hp['slide_win'])
    test_ds, feature_map, fc_edge_index = _build_test_dataset(
        dataset_name, slide_win, test_row_range=test_range,
    )
    V = len(feature_map)
    print(f'test windows: {len(test_ds)}', flush=True)

    # Build model.
    model = GDN_GDeltaUQ(
        [fc_edge_index], V,
        dim=int(hp['dim']),
        input_dim=int(hp['slide_win']),
        out_layer_num=int(hp['out_layer_num']),
        out_layer_inter_dim=int(hp['out_layer_inter_dim']),
        topk=int(hp['topk']),
        n_gnn_layers=int(hp['n_gnn_layers']),
        use_learnable_adj=bool(hp.get('use_learnable_adj', 0)),
        lsa_tau=float(hp.get('lsa_tau', 1.0)),
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Aleatoric head + anchor pool.
    anchor_pool = torch.load(bundle_dir / 'anchor_pool.pt', map_location='cpu')
    aleatoric_head = AleatoricHead(
        hidden_dim=int(hp['dim']), num_sensors=V,
        sensor_embed_dim=16, mlp_hidden=64,
    )
    aleatoric_head.load_state_dict(torch.load(bundle_dir / 'aleatoric_head.pt',
                                              map_location='cpu'))
    aleatoric_head.to(device).eval()

    q_v = np.load(bundle_dir / 'q_v.npz')['q_v']
    u_norm = np.load(bundle_dir / 'u_bar_norm.npz')
    u_bar = {
        'U_par': float(u_norm['U_par']),
        'U_str': float(u_norm['U_str']),
        'U_dist': float(u_norm['U_dist']),
    }

    loaded = LoadedGDeltaUQ(
        model=model, aleatoric_head=aleatoric_head, anchor_pool=anchor_pool,
        q_v=q_v, u_bar_norm=u_bar,
        feature_map=feature_map, cfg=hp, device=device,
    )

    out = run_inference(loaded, test_ds, batch_size=int(hp['batch']))

    # Per-timestep gate + typing.
    sigma_ale = np.sqrt(np.maximum(out.sigma2_ale, 1e-12))
    residual = out.ground_truth - out.mu_bar
    abs_r = np.abs(residual)
    threshold = q_v[None, :] * sigma_ale                              # (T, V)
    sensor_flag = abs_r > threshold                                    # (T, V)
    alarm = sensor_flag.any(axis=1)                                    # (T,)

    rho_par = out.U_par.max(axis=1) / max(u_bar['U_par'], 1e-12)        # (T,)
    rho_str = out.U_str.max(axis=1) / max(u_bar['U_str'], 1e-12)        # (T,)
    rho_dist = out.U_dist / max(u_bar['U_dist'], 1e-12)                 # (T,)

    types = []
    for t in range(len(test_ds)):
        if not alarm[t]:
            types.append('NORMAL')
        else:
            types.append(assign_type(rho_par[t], rho_str[t], rho_dist[t]))

    top_sensors = top_k_sensors_by_upar(out.U_par, top_k=args.top_k_sensors)
    top_edges = top_k_edges_per_timestep(
        out.U_str, out.edge_index_sample, top_k=args.top_k_edges,
    )

    # Build per-timestep records.
    rec = pd.DataFrame({
        't': np.arange(len(test_ds), dtype=np.int64),
        'alarm': alarm.astype(np.int8),
        'type': types,
        'rho_par': rho_par.astype(np.float32),
        'rho_str': rho_str.astype(np.float32),
        'rho_dist': rho_dist.astype(np.float32),
        'U_dist': out.U_dist.astype(np.float32),
        'attack_label': out.attack_label.astype(np.int8),
        'top_par_sensors': [list(map(int, top_sensors[t])) for t in range(len(test_ds))],
        'top_str_edges': [
            [(int(e[0]), int(e[1])) for e in top_edges[t]]
            for t in range(len(test_ds))
        ],
    })

    # Metrics: F1/P/R using alarm vs attack_label.
    y_true = out.attack_label.astype(np.int8)
    y_pred = alarm.astype(np.int8)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Type distribution + type-vs-attack confusion.
    type_counts = rec['type'].value_counts().to_dict()
    type_attack = rec[rec['alarm'] == 1].groupby(['type', 'attack_label']).size().to_dict()
    type_attack_json = {f'{k[0]}|attack={k[1]}': int(v) for k, v in type_attack.items()}

    results_dir = Path(args.results_dir)
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    run_dir = results_dir / datestr
    run_dir.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(run_dir / 'queries.parquet')

    report = {
        'datestr': datestr,
        'checkpoint': str(args.checkpoint),
        'hyperparameters': str(args.hyperparameters),
        'bundle_dir': str(bundle_dir),
        'test_row_range': test_range,
        'n_windows': int(len(test_ds)),
        'metrics': {
            'F1': f1, 'precision': precision, 'recall': recall,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        },
        'type_counts': {k: int(v) for k, v in type_counts.items()},
        'type_vs_attack': type_attack_json,
        'u_bar_norm': u_bar,
    }
    with (run_dir / 'report.json').open('w') as f:
        json.dump(report, f, indent=2)

    print(f'F1={f1:.4f} precision={precision:.4f} recall={recall:.4f} '
          f'TP={tp} FP={fp} FN={fn} TN={tn}', flush=True)
    print(f'type counts: {type_counts}', flush=True)
    print(f'results saved to {run_dir}', flush=True)


if __name__ == '__main__':
    main()
