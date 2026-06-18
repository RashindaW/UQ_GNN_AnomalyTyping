"""Evaluate GDN_GDeltaUQ using the GDN paper's per-feature IQR-normalized,
smoothed, threshold-sweep protocol on top of K-anchor mean predictions
(mu_bar).

Reports:
  1. PAPER F1: feeds (val_mu_bar, val_y, test_mu_bar, test_y) into the same
     evaluate.get_full_err_scores + get_best_performance_data pipeline that
     the original GDN / GDN_UQ uses. Apples-to-apples with the F1=0.85
     baseline reported in RESULTS.md.
  2. CONFORMAL F1 (if -conformal_report is given): re-loads the metrics from
     a previous detect_gdeltauq.py run for side-by-side comparison.

K-anchor inference is run on the val (10%) slice and the full test set; the
val mu_bar is what the paper protocol uses for IQR normalization.
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
from evaluate import get_full_err_scores, get_best_performance_data
from inference_gdeltauq import LoadedGDeltaUQ, run_inference
from models.GDN_GDeltaUQ import GDN_GDeltaUQ
from models.aleatoric_head import AleatoricHead
from util.env import set_device, get_device
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint', required=True)
    parser.add_argument('-hyperparameters', required=True)
    parser.add_argument('-bundle_dir',
                        default='pretrained/swat_gdeltauq/calibration_bundle_bonferroni',
                        help='Calibration bundle (anchor pool + aleatoric head). The '
                             'aleatoric head is loaded but not used by the paper '
                             'protocol; only mu_bar from K-anchor inference matters.')
    parser.add_argument('-split_path', default='data/swat/gdeltauq_split.json')
    parser.add_argument('-conformal_report', default=None,
                        help='Optional path to a detect_gdeltauq report.json — its '
                             'metrics are echoed alongside the paper-protocol F1.')
    parser.add_argument('-topk', type=int, default=1,
                        help='Top-k feature aggregation for the paper protocol. '
                             'Original GDN paper uses 1.')
    parser.add_argument('-device', default='cuda:0')
    parser.add_argument('-results_dir', default='results/swat_gdeltauq_paper_protocol')
    args = parser.parse_args()

    with open(args.hyperparameters) as f:
        hp = json.load(f)
    bundle_dir = Path(args.bundle_dir)

    set_device(args.device)
    device = get_device()

    dataset_name = hp['dataset']
    slide_win = int(hp['slide_win'])

    train_full_ds, feature_map, fc_edge_index = _build_train_dataset(dataset_name, slide_win)
    V = len(feature_map)
    with open(args.split_path) as f:
        split = json.load(f)
    val_range = _row_range_to_window_range(split['val_rows'], len(train_full_ds), slide_win)
    val_subset = Subset(train_full_ds, list(range(*val_range)))

    test_ds, _, _ = _build_test_dataset(dataset_name, slide_win)
    print(f'val windows: {len(val_subset)}; test windows: {len(test_ds)}', flush=True)

    causal_mask_tensor = None
    cm_path = hp.get('causal_mask', '')
    if cm_path:
        from models.causal_mask import load_causal_mask
        causal_mask_tensor = load_causal_mask(cm_path, feature_map)
        print(f'  causal_mask: {cm_path} edges={int(causal_mask_tensor.sum())} '
              f'keep_self={bool(hp.get("causal_mask_keep_self", 1))}', flush=True)

    causal_restrict_tensor = None
    cr_path = hp.get('causal_restrict', '')
    if cr_path:
        from models.causal_mask import load_causal_mask
        causal_restrict_tensor = load_causal_mask(cr_path, feature_map)
        print(f'  causal_restrict: {cr_path} edges={int(causal_restrict_tensor.sum())} '
              f'mode={hp.get("causal_restrict_mode", "pure")} '
              f'keep_self={bool(hp.get("causal_restrict_keep_self", 1))}', flush=True)

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
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f'loaded checkpoint {args.checkpoint}', flush=True)

    anchor_pool = torch.load(bundle_dir / 'anchor_pool.pt', map_location='cpu')
    aleatoric_head = AleatoricHead(
        hidden_dim=int(hp['dim']), num_sensors=V,
        sensor_embed_dim=16, mlp_hidden=64,
    )
    aleatoric_head.load_state_dict(torch.load(bundle_dir / 'aleatoric_head.pt',
                                              map_location='cpu'))
    aleatoric_head.to(device).eval()

    loaded = LoadedGDeltaUQ(
        model=model, aleatoric_head=aleatoric_head, anchor_pool=anchor_pool,
        q_v=None, u_bar_norm={},
        feature_map=feature_map, cfg=hp, device=device,
    )

    print('running K-anchor inference on val ...', flush=True)
    val_out = run_inference(loaded, val_subset, batch_size=int(hp['batch']))
    print(f'  val mu_bar shape={val_out.mu_bar.shape}', flush=True)

    print('running K-anchor inference on test ...', flush=True)
    test_out = run_inference(loaded, test_ds, batch_size=int(hp['batch']))
    print(f'  test mu_bar shape={test_out.mu_bar.shape}', flush=True)

    # evaluate.get_full_err_scores does np.array([pred, gt, labels]) and indexes
    # [:2, :, i] per feature, so labels must be (T, V) too — tile from (T,).
    val_labels_tile = np.tile(val_out.attack_label[:, None], (1, V))
    test_labels_tile = np.tile(test_out.attack_label[:, None], (1, V))
    val_result = [
        val_out.mu_bar.tolist(),
        val_out.ground_truth.tolist(),
        val_labels_tile.tolist(),
    ]
    test_result = [
        test_out.mu_bar.tolist(),
        test_out.ground_truth.tolist(),
        test_labels_tile.tolist(),
    ]

    print('computing per-feature err scores ...', flush=True)
    full_scores, _ = get_full_err_scores(test_result, val_result)
    print(f'  full_scores shape={full_scores.shape}', flush=True)

    print('sweeping 400 thresholds ...', flush=True)
    f1, pre, rec, auc, thr = get_best_performance_data(
        full_scores, list(test_out.attack_label), topk=args.topk,
    )

    print('', flush=True)
    print(f'PAPER     F1={f1:.4f}  P={pre:.4f}  R={rec:.4f}  AUC={auc:.4f}  '
          f'thr={thr:.6f}  topk={args.topk}', flush=True)

    conformal_metrics = None
    if args.conformal_report is not None:
        with open(args.conformal_report) as f:
            cr = json.load(f)
        conformal_metrics = cr['metrics']
        m = conformal_metrics
        print(f'CONFORMAL F1={m["F1"]:.4f}  P={m["precision"]:.4f}  R={m["recall"]:.4f}  '
              f'TP={m["TP"]}  FP={m["FP"]}  FN={m["FN"]}  TN={m["TN"]}', flush=True)

    results_dir = Path(args.results_dir)
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    run_dir = results_dir / datestr
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        run_dir / 'arrays.npz',
        test_mu_bar=test_out.mu_bar,
        test_ground_truth=test_out.ground_truth,
        test_attack_label=test_out.attack_label,
        test_U_par=test_out.U_par,
        test_U_str=test_out.U_str,
        test_U_dist=test_out.U_dist,
        test_sigma2_ale=test_out.sigma2_ale,
        val_mu_bar=val_out.mu_bar,
        val_ground_truth=val_out.ground_truth,
        full_scores=full_scores.astype(np.float32),
    )

    report = {
        'datestr': datestr,
        'checkpoint': str(args.checkpoint),
        'hyperparameters': str(args.hyperparameters),
        'bundle_dir': str(bundle_dir),
        'topk': int(args.topk),
        'paper_protocol': {
            'F1': float(f1), 'precision': float(pre), 'recall': float(rec),
            'AUC': float(auc), 'threshold': float(thr),
        },
        'conformal_gate': conformal_metrics,
    }
    with (run_dir / 'report.json').open('w') as f:
        json.dump(report, f, indent=2)
    print(f'results saved to {run_dir}', flush=True)


if __name__ == '__main__':
    main()
