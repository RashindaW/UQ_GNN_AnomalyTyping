#!/usr/bin/env python3
"""Train the anchored CSTGL_GDeltaUQ (our ONE UQ method on the CST-GL backbone).

Same backbone hyper-parameters / data / masked-MSE loss as the CST-GL baseline,
but the model is CSTGL_GDeltaUQ (G-DeltaUQ anchoring at x=relu(skip), the end-conv
head input). Trained-in anchoring via batch-shuffle (so the method matches
GDN/TopoGDN). num_split=1 (full graph) -> no node-subset bookkeeping. The existing
Trainer works unchanged (model(input, idx) returns mu in train mode).

  python v1v2_cstgl_gdeltauq_train.py --variant V1 --seed 0 --device cuda:0
Runs in the cstgl conda env. V1/V2 train fractions are baked into swat_canon_V{1,2}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CST = os.path.join(ROOT, "competitors", "CST-GL")
sys.path.insert(0, CST)
os.chdir(CST)

from util import load_dataset            # noqa: E402
from trainer import Trainer             # noqa: E402
from stgnn_gdeltauq import CSTGL_GDeltaUQ  # noqa: E402

# CST-GL backbone hp (match the baseline: run.py defaults + baseline runner args)
HP = dict(num_nodes=51, subgraph_size=15, seq_in_len=60, in_dim=1, seq_out_len=1,
          layers=2, conv_channels=16, residual_channels=16, skip_channels=32,
          end_channels=64, node_dim=256, gcn_depth=2, dilation_exponential=1,
          propalpha=0.1, tanhalpha=20, dropout=0.1, batch=32,
          lr=3e-4, weight_decay=1e-4, clip=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="swat",
                    help="dataset tag: canon dir = data/<dataset>_canon_<variant>, "
                         "results under results/uq_v1v2/cstgl (swat) or "
                         "results/uq_wadi_v2/cstgl (wadi)")
    ap.add_argument("--num-nodes", type=int, default=None,
                    help="override HP num_nodes (default 51 for swat, 123 for wadi)")
    ap.add_argument("--subgraph-size", type=int, default=None,
                    help="override HP subgraph_size (default 15 for swat, 30 for wadi)")
    args = ap.parse_args()

    if args.num_nodes is None:
        args.num_nodes = 123 if args.dataset == "wadi" else 51
    if args.subgraph_size is None:
        args.subgraph_size = 30 if args.dataset == "wadi" else 15
    HP["num_nodes"], HP["subgraph_size"] = args.num_nodes, args.subgraph_size

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)

    data_dir = os.path.join(CST, f"data/{args.dataset}_canon_{args.variant}")
    dl = load_dataset(data_dir, HP["batch"], HP["batch"], HP["batch"], scaling_required=False)
    scaler = dl["scaler"]

    model = CSTGL_GDeltaUQ(True, True, HP["gcn_depth"], HP["num_nodes"], device,
                           predefined_A=None, dropout=HP["dropout"],
                           subgraph_size=HP["subgraph_size"], node_dim=HP["node_dim"],
                           dilation_exponential=HP["dilation_exponential"],
                           conv_channels=HP["conv_channels"], residual_channels=HP["residual_channels"],
                           skip_channels=HP["skip_channels"], end_channels=HP["end_channels"],
                           seq_length=HP["seq_in_len"], in_dim=HP["in_dim"], out_dim=HP["seq_out_len"],
                           layers=HP["layers"], propalpha=HP["propalpha"], tanhalpha=HP["tanhalpha"],
                           layer_norm_affline=True)
    engine = Trainer(model, HP["lr"], HP["weight_decay"], HP["clip"], 2500,
                     HP["seq_out_len"], scaler, device, scaling_required=False)

    res_root = "results/uq_v1v2/cstgl" if args.dataset == "swat" else f"results/uq_{args.dataset}_v2/cstgl"
    OUT = os.path.join(ROOT, f"{res_root}/{args.variant}/seed{args.seed}")
    os.makedirs(OUT, exist_ok=True)
    save_path = os.path.join(OUT, "best.pt")
    print(f"[cstgl-uq {args.variant} s{args.seed}] params={sum(p.nelement() for p in model.parameters())}", flush=True)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); dl["train_loader"].shuffle(); tr = []
        for x, y in dl["train_loader"].get_iterator():
            tx = torch.Tensor(x).to(device).transpose(1, 3)      # (B, in_dim, V, seq_len)
            ty = torch.Tensor(y).to(device).transpose(1, 3)      # (B, in_dim, V, horizon)
            loss, _, _ = engine.train(tx, ty[:, 0, :, :], None)  # idx=None -> full graph + anchored
            tr.append(loss)
        va = []
        for x, y in dl["val_loader"].get_iterator():
            tx = torch.Tensor(x).to(device).transpose(1, 3)
            ty = torch.Tensor(y).to(device).transpose(1, 3)
            va.append(engine.eval(tx, ty[:, 0, :, :])[0])
        vmean = float(np.mean(va))
        print(f"  ep {ep}/{args.epochs} train_mse={np.mean(tr):.4f} val_mse={vmean:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vmean < best:
            best = vmean
            torch.save(engine.model.state_dict(), save_path)

    hp = dict(HP); hp.update(variant=args.variant, seed=args.seed, best_val_mse=best,
                             dataset=args.dataset,
                             data_dir=f"data/{args.dataset}_canon_{args.variant}")
    with open(os.path.join(OUT, "hyperparameters.json"), "w") as f:
        json.dump(hp, f, indent=2)
    print(f"[cstgl-uq {args.variant} s{args.seed}] best_val_mse={best:.4f} -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
