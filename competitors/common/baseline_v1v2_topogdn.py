#!/usr/bin/env python3
"""Baseline (no-uncertainty) TopoGDN forecaster under V1/V2 contiguous splits.

V1: train windows [0,70%), val [85,100%); V2: train [0,85%), val [85,100%).
Validates on the LAST 15% (contiguous), NOT TopoGDN's default random val.
Trains the forecaster, runs it on the test set, emits a baseline arrays.npz
(mu + ground_truth + label) and scores M0 via the shared harness.

Runs in the topogdn conda env. One (variant, seed) per invocation:
  python baseline_v1v2_topogdn.py --variant V1 --seed 0 --device cuda:0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOPO = os.path.join(ROOT, "competitors", "TopoGDN")
sys.path.insert(0, TOPO)
os.chdir(TOPO)

from models.GDN import GDN  # noqa: E402
from models.MSTCN import TCN1d  # noqa: E402
from util.net_struct import get_feature_map, get_fc_graph_struc  # noqa: E402
from util.preprocess import build_loc_net, construct_data  # noqa: E402
from datasets.TimeDataset import TimeDataset  # noqa: E402
from train import train  # noqa: E402
from util.env import set_device, get_device  # noqa: E402

SW = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="swat",
                    help="swat -> results/baseline_v1v2/topogdn; wadi -> results/baseline_wadi_v2/topogdn")
    ap.add_argument("--topk", type=int, default=None,
                    help="learned-graph top-k (default 15 for swat, 30 for wadi)")
    args = ap.parse_args()
    if args.topk is None:
        args.topk = 30 if args.dataset == "wadi" else 15
    set_device(args.device)            # set TopoGDN's global device (GDN.forward reads get_device())
    dev = get_device()

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    feature_map = get_feature_map(args.dataset)
    fc = get_fc_graph_struc(args.dataset)
    train_df = pd.read_csv(f"data/{args.dataset}/train.csv", index_col=0)
    test_df = pd.read_csv(f"data/{args.dataset}/test.csv", index_col=0)
    cols = [c for c in train_df.columns if c != "attack"]
    fc_ei = torch.tensor(build_loc_net(fc, cols, feature_map=feature_map), dtype=torch.long)

    tr_indata = construct_data(train_df, feature_map, labels=0)
    te_indata = construct_data(test_df, feature_map, labels=test_df.attack.tolist())
    cfg = {"slide_win": SW, "slide_stride": 1}
    full_train_ds = TimeDataset(tr_indata, fc_ei, mode="train", config=cfg)
    test_ds = TimeDataset(te_indata, fc_ei, mode="test", config=cfg)

    # contiguous V1/V2 split over the all-normal train windows
    N = len(full_train_ds)
    p70, p85 = int(0.70 * N), int(0.85 * N)
    if args.variant == "V1":
        tr_idx = list(range(0, p70))             # first 70%
    else:
        tr_idx = list(range(0, p85))             # first 85%
    val_idx = list(range(p85, N))                # last 15% (contiguous), both variants
    train_loader = DataLoader(Subset(full_train_ds, tr_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(full_train_ds, val_idx), batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    print(f"[topo {args.variant} s{args.seed}] N={N} train={len(tr_idx)} val={len(val_idx)} test={len(test_ds)}", flush=True)

    V = len(feature_map)
    # REAL TopoGDN: enable the persistent-homology TopologyLayer (use_topo=True) AND
    # the multi-scale temporal convolution (MSConv=TCN1d over the V sensor channels).
    # Both were silently OFF in the prior baseline -> it was plain GDN. TCN1d maps the
    # (B, V, W) window to (B, V, W) (depthwise multi-scale temporal conv, kernels 3/5/7).
    msconv = TCN1d(feature_num=V)
    model = GDN([fc_ei], V, dim=128, input_dim=SW, out_layer_num=1,
                out_layer_inter_dim=128, topk=args.topk,
                MSConv=msconv, use_topo=True).to(dev)

    res_root = "results/baseline_v1v2/topogdn" if args.dataset == "swat" \
        else f"results/baseline_{args.dataset}_v2/topogdn"
    OUT = os.path.join(ROOT, f"{res_root}/{args.variant}/seed{args.seed}")
    os.makedirs(OUT, exist_ok=True)
    save_path = os.path.join(OUT, "best.pt")
    train(model=model, save_path=save_path, config={"epoch": args.epoch, "decay": 0, "seed": args.seed},
          train_dataloader=train_loader, val_dataloader=val_loader, feature_map=feature_map,
          test_dataloader=test_loader, test_dataset=test_ds, dataset_name=args.dataset,
          train_dataset=Subset(full_train_ds, tr_idx))

    # load best, run on test -> mu, build baseline arrays.npz
    model.load_state_dict(torch.load(save_path, map_location=dev)); model.eval()
    mus, ys, labs = [], [], []
    with torch.no_grad():
        for x, y, lbl, ei in test_loader:
            x = x.float().to(dev)
            out, _ = model(x)
            mus.append(out.cpu().numpy()); ys.append(y.numpy()); labs.append(lbl.numpy())
    mu = np.concatenate(mus, 0); gt = np.concatenate(ys, 0); lab = np.concatenate(labs, 0).astype(np.int8)
    # val forecasts for the harness (val slice of train)
    vmus, vys = [], []
    with torch.no_grad():
        for x, y, lbl, ei in val_loader:
            x = x.float().to(dev); out, _ = model(x)
            vmus.append(out.cpu().numpy()); vys.append(y.numpy())
    vmu = np.concatenate(vmus, 0); vgt = np.concatenate(vys, 0)
    arr = os.path.join(OUT, "arrays.npz")
    np.savez_compressed(arr, test_mu_bar=mu.astype(np.float32), test_ground_truth=gt.astype(np.float32),
                        test_attack_label=lab, val_mu_bar=vmu.astype(np.float32),
                        val_ground_truth=vgt.astype(np.float32))
    print(f"[topo {args.variant} s{args.seed}] wrote {arr} test_mu={mu.shape}", flush=True)


if __name__ == "__main__":
    main()
