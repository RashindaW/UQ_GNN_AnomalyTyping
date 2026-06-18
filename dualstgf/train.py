"""
Minimal training script for DualSTAGE.

Usage:
    python train.py --dataset-key pronto --data-dir /path/to/data \
                    --epochs 50 --batch-size 32 --window-size 60 --lr 1e-3
"""

import argparse
import os
import sys

import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dualstage"))

from datasets import get_adapter, list_adapter_keys
from src.config import cfg
from src.model.dualstage import DualSTAGE


def parse_args() -> argparse.Namespace:
    available = list_adapter_keys()
    p = argparse.ArgumentParser(description="Train DualSTAGE")
    p.add_argument("--dataset-key", required=True, choices=available,
                    help="Dataset adapter to use.")
    p.add_argument("--data-dir", required=True, help="Path to dataset files.")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window-size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True, help="Learning rate.")
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--val-stride", type=int, default=5)
    p.add_argument("--device", type=str, default="auto",
                    help="Device: cpu, cuda, or auto.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # --- dataset ---
    adapter = get_adapter(args.dataset_key)
    adapter.ensure("training")

    n_nodes = adapter.measurement_count()
    ocvar_dim = len(adapter.get_control_variables(args.data_dir))

    cfg.set_dataset_params(
        n_nodes=n_nodes,
        window_size=args.window_size,
        ocvar_dim=ocvar_dim,
    )
    cfg.device = str(device)
    cfg.validate()

    train_loader, val_loader, _ = adapter.create_dataloaders(
        window_size=args.window_size,
        batch_size=args.batch_size,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        test_stride=args.val_stride,
        data_dir=args.data_dir,
        num_workers=0,
    )

    # --- model ---
    model = DualSTAGE(
        feat_input_node=1,
        feat_target_node=1,
        feat_input_edge=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # --- training loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out.x_hat, batch.x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                val_loss += criterion(out.x_hat, batch.x).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")


if __name__ == "__main__":
    main()
