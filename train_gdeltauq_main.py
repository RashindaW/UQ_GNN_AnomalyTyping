"""Entry point for training the GDN_GDeltaUQ variant.

Reads a three-way split (70 / 10 / 20) from a JSON file produced by
scripts/build_gdeltauq_split.py and runs the G-DeltaUQ training loop on the
70% slice with early-stop on the 10% slice. The 20% aleatoric slice is not
touched here — it is consumed later by scripts/calibrate_gdeltauq.py.
"""
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from datasets.TimeDataset import TimeDataset
from models.GDN_GDeltaUQ import GDN_GDeltaUQ
from models.causal_mask import load_causal_mask
from train_gdeltauq import train_gdeltauq
from util.env import set_device, get_device
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def _row_to_window_indices(row_ranges, total_windows, slide_win, total_rows):
    """Convert row index ranges [start_row, end_row) into window index ranges.

    A window at index w predicts row `slide_win + w`. So a row range
    [r0, r1) corresponds to window range [max(0, r0 - slide_win),
    min(total_windows, r1 - slide_win)). We use the predicted-row criterion
    so that no window's *target* leaks across the slice boundary; the input
    window for a target on the boundary may overlap the previous slice by
    up to slide_win rows, which is acceptable for early-stop and aleatoric
    training (we never train and evaluate at the same row).
    """
    out = {}
    for name, (r0, r1) in row_ranges.items():
        w0 = max(0, r0 - slide_win)
        w1 = min(total_windows, r1 - slide_win)
        out[name] = (w0, w1)
    return out


def build_main_for_training(args):
    _seed_everything(args.random_seed)
    set_device(args.device)
    device = get_device()

    dataset = args.dataset
    train_csv = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
    if 'attack' in train_csv.columns:
        train_csv = train_csv.drop(columns=['attack'])

    feature_map = get_feature_map(dataset)
    fc_struc = get_fc_graph_struc(dataset)
    fc_edge_index = build_loc_net(fc_struc, list(train_csv.columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

    train_indata = construct_data(train_csv, feature_map, labels=0)
    cfg = {'slide_win': args.slide_win, 'slide_stride': args.slide_stride}

    # We construct a single TimeDataset over the full training window and
    # then subset by window indices.  Using slide_stride=1 here gives us the
    # densest window indexing so that the row-range -> window-range mapping
    # is straightforward; the train slice uses train_slide_stride only at
    # DataLoader-sampling time (via Subset indices).
    ts_cfg = {'slide_win': args.slide_win, 'slide_stride': 1}
    full_ds = TimeDataset(train_indata, fc_edge_index, mode='test', config=ts_cfg)

    if not os.path.exists(args.split_path):
        raise FileNotFoundError(
            f"split file not found: {args.split_path}\n"
            f"Run `python scripts/build_gdeltauq_split.py -dataset {args.dataset} "
            f"-out_path {args.split_path}` first, or use scripts/run_gdeltauq_pipeline.sh."
        )
    with open(args.split_path) as f:
        split = json.load(f)
    row_ranges = {
        'train': tuple(split['train_rows']),
        'val': tuple(split['val_rows']),
        'aleatoric': tuple(split['aleatoric_rows']),
    }
    total_rows = split['total_rows']
    total_windows = len(full_ds)
    win_ranges = _row_to_window_indices(row_ranges, total_windows, args.slide_win, total_rows)
    print(f'split row ranges: {row_ranges}', flush=True)
    print(f'split window ranges: {win_ranges}', flush=True)

    # Apply train_slide_stride by subsampling the train slice indices.
    tw0, tw1 = win_ranges['train']
    vw0, vw1 = win_ranges['val']
    train_indices = list(range(tw0, tw1, args.slide_stride))
    val_indices = list(range(vw0, vw1))

    train_subset = Subset(full_ds, train_indices)
    val_subset = Subset(full_ds, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch, shuffle=False, num_workers=0)

    print(
        f'train windows: {len(train_indices)}; val windows: {len(val_indices)}',
        flush=True,
    )

    edge_index_sets = [fc_edge_index]

    causal_mask = None
    causal_mask_path = getattr(args, 'causal_mask', '')
    if causal_mask_path:
        causal_mask = load_causal_mask(causal_mask_path, feature_map)
        n_edges = int(causal_mask.sum())
        print(
            f'causal_mask: loaded {causal_mask_path}  '
            f'V={causal_mask.shape[0]}  edges={n_edges}  '
            f'keep_self={bool(getattr(args, "causal_mask_keep_self", 1))}',
            flush=True,
        )
        print(
            'WARNING: --causal_mask produces variable in-degree per node. '
            'inference_gdeltauq.py assumes a fixed (topk-1)*V layout and '
            'will need an update before it can consume checkpoints trained '
            'with this flag.',
            flush=True,
        )

    causal_restrict = None
    causal_restrict_path = getattr(args, 'causal_restrict', '')
    if causal_restrict_path:
        causal_restrict = load_causal_mask(causal_restrict_path, feature_map)
        print(
            f'causal_restrict: loaded {causal_restrict_path}  '
            f'V={causal_restrict.shape[0]}  edges={int(causal_restrict.sum())}  '
            f'mode={getattr(args, "causal_restrict_mode", "pure")}  '
            f'keep_self={bool(getattr(args, "causal_restrict_keep_self", 1))}',
            flush=True,
        )

    # LSA flags are owned by train_gdeltauq_jointnll_main.py's argparse; this
    # build path defaults them off so the legacy entrypoints (this file,
    # main.py for the original GDN) keep behaving as before.
    use_learnable_adj = bool(getattr(args, 'use_learnable_adj', False))
    lsa_tau = float(getattr(args, 'lsa_tau', 1.0))

    model = GDN_GDeltaUQ(
        edge_index_sets, len(feature_map),
        dim=args.dim,
        input_dim=args.slide_win,
        out_layer_num=args.out_layer_num,
        out_layer_inter_dim=args.out_layer_inter_dim,
        topk=args.topk,
        n_gnn_layers=args.n_gnn_layers,
        causal_mask=causal_mask,
        causal_mask_keep_self=bool(getattr(args, 'causal_mask_keep_self', 1)),
        use_learnable_adj=use_learnable_adj,
        lsa_tau=lsa_tau,
        causal_restrict=causal_restrict,
        causal_restrict_mode=getattr(args, 'causal_restrict_mode', 'pure'),
        causal_restrict_keep_self=bool(getattr(args, 'causal_restrict_keep_self', 1)),
    ).to(device)

    return model, train_loader, val_loader, device, feature_map, fc_edge_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', type=int, default=128)
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-slide_win', type=int, default=5)
    parser.add_argument('-slide_stride', type=int, default=1)
    parser.add_argument('-dim', type=int, default=64)
    parser.add_argument('-out_layer_num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', type=int, default=128)
    parser.add_argument('-topk', type=int, default=15)
    parser.add_argument('-n_gnn_layers', type=int, default=2)
    parser.add_argument('-K_anchors', type=int, default=10,
                        help='K anchors for inference (not used during training; '
                             'persisted in hyperparameters.json for downstream use).')
    parser.add_argument('-decay', type=float, default=0.0)
    parser.add_argument('-dataset', type=str, default='swat')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-random_seed', type=int, default=42)
    parser.add_argument('-split_path', type=str, default='data/swat/gdeltauq_split.json')
    parser.add_argument('-save_path_pattern', type=str, default='swat_gdeltauq')
    parser.add_argument('-comment', type=str, default='')
    parser.add_argument(
        '-causal_mask', type=str, default='',
        help='Path to a binary causal adjacency .npy (src->tgt convention). '
             'If set, the per-forward-pass learned top-K graph is AND-ed '
             'with this mask before message-passing. Expects a sibling '
             'features sidecar JSON for safe reindexing. '
             'Typical: data/swat/causal_adjacency.npy.',
    )
    parser.add_argument(
        '-causal_mask_keep_self', type=int, default=1, choices=[0, 1],
        help='When --causal_mask is set, force the diagonal to 1 before '
             'AND-ing so every node keeps its self-edge (default 1).',
    )
    parser.add_argument(
        '-causal_restrict', type=str, default='',
        help='Path to a binary causal scaffold .npy (src->tgt). If set, the '
             'cosine matrix is RESTRICTED to causally-allowed parent pairs '
             'BEFORE top-K (injects the prior; cf. --causal_mask which AND-s '
             'AFTER top-K). Mutually exclusive with --causal_mask. '
             'Typical: data/swat/causal_scaffold_dag.npy.',
    )
    parser.add_argument(
        '-causal_restrict_mode', type=str, default='pure',
        choices=['pure', 'augment'],
        help="'pure' keeps only allowed parents (<=K, variable in-degree); "
             "'augment' guarantees allowed parents then fills to K with best "
             "cosine (preserves detection).",
    )
    parser.add_argument(
        '-causal_restrict_keep_self', type=int, default=1, choices=[0, 1],
        help='Force each node self-edge in the restricted candidate set.',
    )

    args = parser.parse_args()

    model, train_loader, val_loader, device, feature_map, _ = build_main_for_training(args)

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    save_dir = Path(f'./pretrained/{args.save_path_pattern}')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'best_{datestr}.pt'
    print(f'CHECKPOINT_PATH={save_path}', flush=True)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'slide_stride': args.slide_stride,
        'dim': args.dim,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'topk': args.topk,
        'n_gnn_layers': args.n_gnn_layers,
        'K_anchors': args.K_anchors,
        'decay': args.decay,
        'seed': args.random_seed,
        'model': 'gdn_gdeltauq',
        'dataset': args.dataset,
        'comment': args.comment,
        'causal_mask': args.causal_mask,
        'causal_mask_keep_self': args.causal_mask_keep_self,
        'causal_restrict': args.causal_restrict,
        'causal_restrict_mode': args.causal_restrict_mode,
        'causal_restrict_keep_self': args.causal_restrict_keep_self,
    }
    with (save_dir / f'hyperparameters_{datestr}.json').open('w') as f:
        json.dump(train_config, f, indent=2)

    train_gdeltauq(
        model=model,
        save_path=str(save_path),
        config={'epoch': args.epoch, 'decay': args.decay},
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
    )

    print(f'TRAINING_DONE checkpoint={save_path}', flush=True)


if __name__ == '__main__':
    main()
