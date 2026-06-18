#!/usr/bin/env python3
"""Train the anchored TopoGDN_GDeltaUQ (our ONE UQ method on the TopoGDN backbone).

Same data / window / V1-V2 splits / stabilizers as the real TopoGDN baseline, but
the model is the 2-layer anchored TopoGDN_GDeltaUQ (topology in the pre-anchor
layer, G-DeltaUQ stochastic anchoring at the final graph layer). Trained with MSE
on the batch-shuffle-anchored mean prediction. Saves best.pt (min val MSE) +
hyperparameters.json for the downstream calibrate/extract step.

  V1: train windows [0,70%), val [85,100%)   V2: train [0,85%), val [85,100%)
Runs in the topogdn conda env (persistent homology). One (variant, seed) per call:
  python v1v2_topogdn_gdeltauq_train.py --variant V1 --seed 0 --epoch 50 --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOPO = os.path.join(ROOT, "competitors", "TopoGDN")
sys.path.insert(0, TOPO)
os.chdir(TOPO)

from models.TopoGDN_GDeltaUQ import TopoGDN_GDeltaUQ   # noqa: E402
from util.net_struct import get_feature_map, get_fc_graph_struc  # noqa: E402
from util.preprocess import build_loc_net, construct_data  # noqa: E402
from datasets.TimeDataset import TimeDataset  # noqa: E402
from util.env import set_device, get_device  # noqa: E402

SW = 60
DIM = 128
TOPK = 15
N_GNN = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="swat",
                    help="swat -> results/uq_v1v2/topogdn; wadi -> results/uq_wadi_v2/topogdn")
    ap.add_argument("--topk", type=int, default=None,
                    help="learned-graph top-k (default 15 for swat, 30 for wadi)")
    args = ap.parse_args()
    if args.topk is None:
        args.topk = 30 if args.dataset == "wadi" else 15
    global TOPK
    TOPK = args.topk
    set_device(args.device)
    dev = get_device()

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # stabilizers (same as the baseline; tame raw-scale grad inflation / seed collapse)
    grad_clip = float(os.environ.get("TOPO_GRAD_CLIP", "1000"))
    warmup_iters = int(os.environ.get("TOPO_LR_WARMUP_ITERS", "500"))

    feature_map = get_feature_map(args.dataset)
    fc = get_fc_graph_struc(args.dataset)
    train_df = pd.read_csv(f"data/{args.dataset}/train.csv", index_col=0)
    test_df = pd.read_csv(f"data/{args.dataset}/test.csv", index_col=0)
    cols = [c for c in train_df.columns if c != "attack"]
    fc_ei = torch.tensor(build_loc_net(fc, cols, feature_map=feature_map), dtype=torch.long)

    tr_indata = construct_data(train_df, feature_map, labels=0)
    cfg = {"slide_win": SW, "slide_stride": 1}
    full_train_ds = TimeDataset(tr_indata, fc_ei, mode="train", config=cfg)

    N = len(full_train_ds)
    p70, p85 = int(0.70 * N), int(0.85 * N)
    tr_idx = list(range(0, p70 if args.variant == "V1" else p85))
    val_idx = list(range(p85, N))                       # last 15%, both variants
    train_loader = DataLoader(Subset(full_train_ds, tr_idx), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(Subset(full_train_ds, val_idx), batch_size=args.batch, shuffle=False)
    print(f"[topo-uq {args.variant} s{args.seed}] N={N} train={len(tr_idx)} val={len(val_idx)} "
          f"clip={grad_clip} warmup={warmup_iters}", flush=True)

    V = len(feature_map)
    model = TopoGDN_GDeltaUQ([fc_ei], V, dim=DIM, out_layer_inter_dim=DIM, input_dim=SW,
                             out_layer_num=1, topk=TOPK, n_gnn_layers=N_GNN,
                             use_topo=True, use_msconv=True).to(dev)

    res_root = "results/uq_v1v2/topogdn" if args.dataset == "swat" \
        else f"results/uq_{args.dataset}_v2/topogdn"
    OUT = os.path.join(ROOT, f"{res_root}/{args.variant}/seed{args.seed}")
    os.makedirs(OUT, exist_ok=True)
    save_path = os.path.join(OUT, "best.pt")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    base_lr = opt.param_groups[0]["lr"]
    early_stop_win = 15
    best_val = float("inf"); stop_ct = 0; gi = 0

    for ep in range(args.epoch):
        model.train(); acu = 0.0; t0 = time.time()
        for x, y, lbl, ei in train_loader:
            x = x.float().to(dev); y = y.float().to(dev)
            if warmup_iters > 0 and gi < warmup_iters:
                for g in opt.param_groups:
                    g["lr"] = base_lr * float(gi + 1) / warmup_iters
            opt.zero_grad()
            mu, _, _ = model(x)                          # anchor=None -> batch shuffle
            loss = F.mse_loss(mu, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            acu += loss.item(); gi += 1
        sched.step()

        # validation MSE (eval mode; batch-shuffle anchor) for checkpoint selection
        model.eval(); vacu = 0.0; vn = 0
        with torch.no_grad():
            for x, y, lbl, ei in val_loader:
                x = x.float().to(dev); y = y.float().to(dev)
                mu, _, _ = model(x)
                vacu += F.mse_loss(mu, y).item(); vn += 1
        val_mse = vacu / max(vn, 1)
        print(f"  ep {ep+1}/{args.epoch} train_mse={acu/len(train_loader):.4f} "
              f"val_mse={val_mse:.4f} ({time.time()-t0:.0f}s)", flush=True)

        if val_mse < best_val:
            best_val = val_mse; stop_ct = 0
            torch.save(model.state_dict(), save_path)
        else:
            stop_ct += 1
            if stop_ct >= early_stop_win:
                print(f"  early stop at ep {ep+1}", flush=True); break

    hp = {"dataset": args.dataset, "slide_win": SW, "dim": DIM, "topk": TOPK,
          "n_gnn_layers": N_GNN, "out_layer_num": 1, "out_layer_inter_dim": DIM,
          "batch": args.batch, "variant": args.variant, "seed": args.seed,
          "use_topo": True, "use_msconv": True, "best_val_mse": best_val}
    with open(os.path.join(OUT, "hyperparameters.json"), "w") as f:
        json.dump(hp, f, indent=2)
    print(f"[topo-uq {args.variant} s{args.seed}] best_val_mse={best_val:.4f} -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
