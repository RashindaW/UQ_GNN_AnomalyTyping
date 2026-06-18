"""Train one DualSTGF_UQ member on SWaT.

Mirrors main.py for the GDN_UQ ensemble: configures cfg, builds dataloaders via
the SWaT adapter, instantiates DualSTAGE_UQ, trains under Gaussian NLL on
reconstruction residuals with a per-epoch sigma-health diagnostic.

Used by scripts/train_dualstgf_ensemble.sh, which runs 5 seeds across 4 GPUs.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make the vendored DualSTGF subtree importable.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'dualstgf'))
sys.path.insert(0, str(REPO_ROOT / 'dualstgf' / 'dualstage'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-dataset', type=str, default='swat',
                        choices=['swat', 'pronto', 'ashrae'])
    parser.add_argument('-data_dir', type=str, default=None,
                        help='Override default_data_dir from the adapter.')
    parser.add_argument('-window_size', type=int, default=60)
    parser.add_argument('-train_stride', type=int, default=1)
    parser.add_argument('-val_stride', type=int, default=5)
    parser.add_argument('-batch', type=int, default=32)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-weight_decay', type=float, default=1e-3)
    parser.add_argument('-early_stop_patience', type=int, default=15)
    parser.add_argument('-random_seed', type=int, default=0)
    parser.add_argument('-save_path_pattern', type=str, default='dualstgf_smoke',
                        help='Subdirectory under pretrained/dualstgf_ensemble/.')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-gnn_embed_dim', type=int, default=16,
                        help='Per the upstream config; default 16.')
    parser.add_argument('-temp_node_embed_dim', type=int, default=16)
    parser.add_argument('-recon_hidden_dim', type=int, default=10)
    parser.add_argument('-topk', type=int, default=15)
    parser.add_argument('-num_gnn_layers', type=int, default=1)
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-aug_control', action='store_true',
                        help='SWaT has no separate control vars; default False.')
    parser.add_argument('-use_spectral_view', action='store_true')
    parser.add_argument('-lambda_div', type=float, default=0.0)
    parser.add_argument('-anomaly_weight', type=float, default=0.0)
    parser.add_argument('-with_variance_head', type=int, default=1,
                        help='1 to enable heteroscedastic UQ (DualSTGF_UQ); '
                             '0 collapses to deterministic upstream behaviour.')
    args = parser.parse_args()

    # --- reproducibility (mirrors main.py:206-213) ---
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    device = torch.device(args.device)

    # --- adapter + dataloaders ---
    from datasets import get_adapter
    from src.config import cfg

    adapter = get_adapter(args.dataset)
    n_nodes = adapter.measurement_count()
    ocvar_dim = len(adapter.get_control_variables(args.data_dir or adapter.default_data_dir or ''))
    cfg.set_dataset_params(n_nodes=n_nodes, window_size=args.window_size, ocvar_dim=ocvar_dim)
    cfg.device = str(device)
    cfg.validate()

    train_loader, val_loader, _ = adapter.create_dataloaders(
        window_size=args.window_size,
        batch_size=args.batch,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        test_stride=args.val_stride,
        data_dir=args.data_dir or adapter.default_data_dir,
        num_workers=args.num_workers,
    )
    print(f'[train_dualstgf] adapter={args.dataset}  n_nodes={n_nodes}  ocvar_dim={ocvar_dim}  '
          f'window={args.window_size}  '
          f'train_windows={len(train_loader.dataset)}  val_windows={len(val_loader.dataset)}',
          flush=True)

    # --- model ---
    from src.model.dualstage_uq import DualSTAGE_UQ
    model = DualSTAGE_UQ(
        feat_input_node=1,
        feat_target_node=1,
        feat_input_edge=1,
        aug_control=bool(args.aug_control),
        use_spectral_view=bool(args.use_spectral_view),
        gnn_embed_dim=args.gnn_embed_dim,
        temp_node_embed_dim=args.temp_node_embed_dim,
        recon_hidden_dim=args.recon_hidden_dim,
        topk=args.topk,
        num_gnn_layers=args.num_gnn_layers,
        with_variance_head=bool(args.with_variance_head),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[train_dualstgf] model={type(model).__name__}  params={n_params:,}', flush=True)

    # --- save path & manifest hook ---
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    save_dir = REPO_ROOT / 'pretrained' / 'dualstgf_ensemble' / args.save_path_pattern
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'best_{datestr}.pt'
    print(f'CHECKPOINT_PATH={save_path}', flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- training loop ---
    min_val_loss = float('inf')
    stop_count = 0
    train_loss_history = []

    for i_epoch in range(args.epoch):
        model.train()
        acu_loss = 0.0
        t_epoch = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            mu, log_var = model(batch)
            loss = F.gaussian_nll_loss(mu, batch.x, log_var.exp(),
                                        reduction='mean', eps=1e-6)
            loss.backward()
            optimizer.step()
            acu_loss += loss.item()
            train_loss_history.append(loss.item())
        epoch_train_loss = acu_loss / max(1, len(train_loader))
        print(f'epoch ({i_epoch} / {args.epoch}) train_NLL={epoch_train_loss:.6f}  '
              f'wall={time.time()-t_epoch:.1f}s', flush=True)

        # --- validation + sigma-health diagnostic ---
        model.eval()
        val_acu = 0.0
        log_var_chunks = []
        with torch.no_grad():
            for vbatch in val_loader:
                vbatch = vbatch.to(device)
                vmu, vlv = model(vbatch)
                val_acu += F.gaussian_nll_loss(vmu, vbatch.x, vlv.exp(),
                                               reduction='mean', eps=1e-6).item()
                log_var_chunks.append(vlv.detach().cpu())
        val_loss = val_acu / max(1, len(val_loader))

        if log_var_chunks:
            lv = torch.cat(log_var_chunks).flatten()
            sat = float(((lv <= -9.9) | (lv >= 9.9)).float().mean())
            print(f'epoch ({i_epoch} / {args.epoch}) val_NLL={val_loss:.6f}  '
                  f'sigma_health: mean(log_var)={float(lv.mean()):.4f} '
                  f'median(log_var)={float(lv.median()):.4f} clamp_saturation={sat:.4f}',
                  flush=True)
        else:
            print(f'epoch ({i_epoch} / {args.epoch}) val_NLL={val_loss:.6f}', flush=True)

        if val_loss < min_val_loss:
            torch.save(model.state_dict(), save_path)
            min_val_loss = val_loss
            stop_count = 0
        else:
            stop_count += 1
            if stop_count >= args.early_stop_patience:
                print(f'[train_dualstgf] early stopping at epoch {i_epoch}', flush=True)
                break

    print(f'[train_dualstgf] best val_NLL={min_val_loss:.6f}  ckpt={save_path}', flush=True)


if __name__ == '__main__':
    main()
