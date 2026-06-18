"""Calibrate a trained GDN_GDeltaUQ checkpoint.

Steps (plan Phase 3 steps 4-7):
  1. Load frozen GDN_GDeltaUQ.
  2. Run model.forward_split on the 10% val slice to collect per-sample layer-1
     outputs h_pre.  Sample K=10 with a fixed seed to form the anchor pool.
  3. Run K-anchor inference on the 10% val slice with that pool to compute
     U_bar_par, U_bar_str, U_bar_dist (normalizers for rho).
  4. Run K-anchor inference on the 20% aleatoric slice; train AleatoricHead.
  5. Re-run K-anchor inference on the 10% val slice with the trained head to
     get sigma_ale_v(t), then compute per-sensor q_v as the (1-alpha)-quantile
     of |y_v - mu_bar_v| / sigma_ale_v.
  6. Persist all artifacts under pretrained/{save_dir}/calibration_bundle/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Repo root on sys.path so this script can be invoked as
# `python scripts/calibrate_gdeltauq.py ...` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.TimeDataset import TimeDataset
from inference_gdeltauq import LoadedGDeltaUQ, run_inference
from models.GDN_GDeltaUQ import GDN_GDeltaUQ
from models.aleatoric_head import AleatoricHead
from train_aleatoric_gdeltauq import precompute_frozen_predictions, train_aleatoric
from util.env import set_device, get_device
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data


def _build_full_train_dataset(dataset, slide_win):
    train_csv = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
    if 'attack' in train_csv.columns:
        train_csv = train_csv.drop(columns=['attack'])
    feature_map = get_feature_map(dataset)
    fc_struc = get_fc_graph_struc(dataset)
    fc_edge_index = build_loc_net(fc_struc, list(train_csv.columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    indata = construct_data(train_csv, feature_map, labels=0)
    cfg = {'slide_win': slide_win, 'slide_stride': 1}
    ds = TimeDataset(indata, fc_edge_index, mode='test', config=cfg)
    return ds, feature_map, fc_edge_index


def _row_range_to_window_range(row_range, total_windows, slide_win):
    r0, r1 = row_range
    return (max(0, r0 - slide_win), min(total_windows, r1 - slide_win))


def _diverse_select(pool, K, seed=0):
    """Greedy max-min: pick K anchors that maximize the minimum pairwise L2
    distance in the flattened (V * d_in) latent space. First anchor is the
    pool element farthest from the pool mean (deterministic seed-independent
    starting point), then each subsequent anchor maximizes its minimum
    distance to the already-selected set.

    `seed` only affects ties (none expected in practice on float32 vectors).
    """
    flat = torch.stack([p.flatten() for p in pool])      # (N, V*d_in)
    N = flat.shape[0]
    mean = flat.mean(dim=0, keepdim=True)
    dist_to_mean = torch.cdist(flat, mean).squeeze(-1)   # (N,)
    first = int(torch.argmax(dist_to_mean).item())
    selected = [first]
    min_dist = torch.cdist(flat, flat[first:first + 1]).squeeze(-1)  # (N,)

    while len(selected) < K:
        masked = min_dist.clone()
        masked[selected] = -float('inf')
        nxt = int(torch.argmax(masked).item())
        selected.append(nxt)
        new_d = torch.cdist(flat, flat[nxt:nxt + 1]).squeeze(-1)
        min_dist = torch.minimum(min_dist, new_d)
    return selected


def _collect_anchor_pool(model, dataset, K, device, seed=0, strategy='random'):
    """Run forward_split, collect per-sample (V, d) tensors, select K anchors.

    strategy = 'random' (default): rng.choice(K) with the given seed.
    strategy = 'diverse': greedy max-min in flattened latent space (Q2-2).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    pool = []
    with torch.no_grad():
        for x, _, _, edge_index in loader:
            x = x.float().to(device)
            edge_index = edge_index.float().to(device)
            h_pre = model.forward_split(x, edge_index)  # (B, V, d_in)
            for b in range(h_pre.shape[0]):
                pool.append(h_pre[b].detach().cpu())
    if len(pool) < K:
        raise ValueError(f'val slice has {len(pool)} windows, need at least K={K}')

    if strategy == 'random':
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pool), size=K, replace=False).tolist()
    elif strategy == 'diverse':
        idx = _diverse_select(pool, K, seed=seed)
    else:
        raise ValueError(f'unknown anchor strategy {strategy!r}')

    anchors = torch.stack([pool[int(i)] for i in idx], dim=0)  # (K, V, d_in)
    return anchors, list(idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint', type=str, required=True,
                        help='path to a trained GDN_GDeltaUQ .pt file')
    parser.add_argument('-hyperparameters', type=str, required=True,
                        help='hyperparameters.json saved alongside the checkpoint')
    parser.add_argument('-split_path', type=str, default='data/swat/gdeltauq_split.json')
    parser.add_argument('-K_anchors', type=int, default=10)
    parser.add_argument('-anchor_seed', type=int, default=0)
    parser.add_argument('-anchor_strategy', type=str, default='random',
                        choices=['random', 'diverse'],
                        help='random = rng.choice; diverse = greedy max-min in '
                             'flattened forward_split latent space (Q2-2).')
    parser.add_argument('-conformal_alpha', type=float, default=0.01,
                        help='Per-sensor target miscoverage. The first-stage '
                             'gate alarms if ANY of V sensors exceeds its q_v, '
                             'so the joint nominal FP rate is ~1 - (1-alpha)^V. '
                             'At V=51 and alpha=0.01 this is ~40%%. Use '
                             '-bonferroni to divide alpha by V.')
    parser.add_argument('-bonferroni', action='store_true',
                        help='If set, apply alpha := alpha / V (Bonferroni). '
                             'Recommended for V >> 1 to keep joint FP rate '
                             'near alpha.')
    parser.add_argument('-aleatoric_epochs', type=int, default=5)
    parser.add_argument('-aleatoric_batch', type=int, default=32)
    parser.add_argument('-pretrained_head_path', type=str, default='',
                        help='If set, load this aleatoric_head .pt and skip '
                             'phase-2 training (used for jointly-trained head '
                             'from train_gdeltauq_jointnll_main.py).')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-save_dir', type=str, default='pretrained/swat_gdeltauq/calibration_bundle')
    args = parser.parse_args()

    with open(args.hyperparameters) as f:
        hp = json.load(f)

    set_device(args.device)
    device = get_device()

    dataset_name = hp['dataset']
    slide_win = int(hp['slide_win'])

    full_ds, feature_map, fc_edge_index = _build_full_train_dataset(dataset_name, slide_win)
    total_windows = len(full_ds)
    V = len(feature_map)

    with open(args.split_path) as f:
        split = json.load(f)
    val_range = _row_range_to_window_range(split['val_rows'], total_windows, slide_win)
    ale_range = _row_range_to_window_range(split['aleatoric_rows'], total_windows, slide_win)
    val_subset = Subset(full_ds, list(range(*val_range)))
    ale_subset = Subset(full_ds, list(range(*ale_range)))
    print(f'val windows: {len(val_subset)}; aleatoric windows: {len(ale_subset)}', flush=True)

    edge_index_sets = [fc_edge_index]

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
        edge_index_sets, V,
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

    # 1. Anchor pool from val.
    anchor_pool, anchor_indices = _collect_anchor_pool(
        model, val_subset, K=args.K_anchors, device=device,
        seed=args.anchor_seed, strategy=args.anchor_strategy,
    )
    print(f'anchor strategy: {args.anchor_strategy}', flush=True)
    print(f'anchor pool shape {anchor_pool.shape}; sample indices {anchor_indices}',
          flush=True)

    # 2. U_bar normalizers on val (no aleatoric head yet -> use placeholder).
    loaded_no_head = LoadedGDeltaUQ(
        model=model, aleatoric_head=None, anchor_pool=anchor_pool,
        q_v=None, u_bar_norm={},
        feature_map=feature_map, cfg=hp, device=device,
    )
    val_out = run_inference(loaded_no_head, val_subset, batch_size=int(hp['batch']))
    u_bar_par = float(val_out.U_par.mean())
    u_bar_str = float(val_out.U_str.mean())
    u_bar_dist = float(val_out.U_dist.mean())
    print(f'U_bar_par={u_bar_par:.6e} U_bar_str={u_bar_str:.6e} U_bar_dist={u_bar_dist:.6e}',
          flush=True)

    # 3. Aleatoric head: either load a pre-trained one (joint-NLL pipeline)
    # or run phase-2 training on the 20% slice using frozen mu_bar/h_bar.
    aleatoric_head = AleatoricHead(
        hidden_dim=int(hp['dim']), num_sensors=V,
        sensor_embed_dim=int(hp.get('sensor_embed_dim', 16)),
        mlp_hidden=int(hp.get('aleatoric_mlp_hidden', 64)),
    )
    if args.pretrained_head_path:
        head_state = torch.load(args.pretrained_head_path, map_location='cpu')
        aleatoric_head.load_state_dict(head_state)
        aleatoric_head = aleatoric_head.to(device)
        aleatoric_head.eval()
        print(f'aleatoric head LOADED (joint-NLL) from '
              f'{args.pretrained_head_path}', flush=True)
    else:
        h_bar_ale, y_ale, mu_bar_ale = precompute_frozen_predictions(
            loaded_no_head, ale_subset, batch_size=int(hp['batch']),
        )
        aleatoric_head = train_aleatoric(
            aleatoric_head, h_bar_ale, y_ale, mu_bar_ale,
            device=device, epochs=args.aleatoric_epochs,
            batch_size=args.aleatoric_batch, lr=1e-3,
        )
        aleatoric_head.eval()
        print('aleatoric head trained (phase-2 post-hoc)', flush=True)

    # 4. Re-run inference on val with trained head -> per-sensor q_v.
    loaded_with_head = LoadedGDeltaUQ(
        model=model, aleatoric_head=aleatoric_head, anchor_pool=anchor_pool,
        q_v=None, u_bar_norm={'U_par': u_bar_par, 'U_str': u_bar_str, 'U_dist': u_bar_dist},
        feature_map=feature_map, cfg=hp, device=device,
    )
    val_out2 = run_inference(loaded_with_head, val_subset, batch_size=int(hp['batch']))
    sigma_ale = np.sqrt(np.maximum(val_out2.sigma2_ale, 1e-12))
    residual = np.abs(val_out2.ground_truth - val_out2.mu_bar)
    s_v = residual / sigma_ale                                # (T, V)
    effective_alpha = args.conformal_alpha / V if args.bonferroni else args.conformal_alpha
    if args.bonferroni:
        print(f'BONFERRONI: applying alpha := alpha / V = '
              f'{args.conformal_alpha} / {V} = {effective_alpha:.6f}',
              flush=True)
    else:
        joint_fp_rate = 1.0 - (1.0 - args.conformal_alpha) ** V
        print(f'WARNING: uncorrected alpha={args.conformal_alpha} over V={V} '
              f'sensors implies joint nominal FP rate ~ {joint_fp_rate:.3f}. '
              f'Use -bonferroni to keep joint FP near alpha.', flush=True)
    q_v = np.quantile(s_v, 1.0 - effective_alpha, axis=0)  # (V,)
    print(f'q_v summary: median={float(np.median(q_v)):.4f} '
          f'min={float(q_v.min()):.4f} max={float(q_v.max()):.4f}', flush=True)

    # Coverage check.
    coverage = float((s_v <= q_v[None, :]).all(axis=1).mean())
    print(f'val coverage (all-sensors-within-q): {coverage:.4f} '
          f'(target ~ {1.0 - args.conformal_alpha:.4f})', flush=True)

    # 5. Persist artifacts.
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(anchor_pool, save_dir / 'anchor_pool.pt')
    torch.save(aleatoric_head.state_dict(), save_dir / 'aleatoric_head.pt')
    np.savez(save_dir / 'q_v.npz', q_v=q_v)
    np.savez(save_dir / 'u_bar_norm.npz',
             U_par=u_bar_par, U_str=u_bar_str, U_dist=u_bar_dist)
    np.savez(save_dir / 'edge_index_sample.npz',
             edge_index_sample=val_out2.edge_index_sample)

    bundle = {
        'checkpoint': str(args.checkpoint),
        'hyperparameters': str(args.hyperparameters),
        'K_anchors': int(args.K_anchors),
        'anchor_seed': int(args.anchor_seed),
        'anchor_indices_in_val_subset': anchor_indices,
        'conformal_alpha': float(args.conformal_alpha),
        'conformal_alpha_effective': float(effective_alpha),
        'bonferroni': bool(args.bonferroni),
        'val_window_range': val_range,
        'aleatoric_window_range': ale_range,
        'u_bar_norm': {
            'U_par': u_bar_par, 'U_str': u_bar_str, 'U_dist': u_bar_dist,
        },
        'val_coverage_all_sensors': coverage,
        'q_v_summary': {
            'median': float(np.median(q_v)),
            'min': float(q_v.min()),
            'max': float(q_v.max()),
        },
    }
    with (save_dir / 'bundle.json').open('w') as f:
        json.dump(bundle, f, indent=2)
    print(f'bundle written to {save_dir}', flush=True)


if __name__ == '__main__':
    main()
