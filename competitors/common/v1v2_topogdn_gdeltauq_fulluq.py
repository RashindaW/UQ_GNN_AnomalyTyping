#!/usr/bin/env python3
"""Calibrate + extract full UQ for ONE anchored TopoGDN_GDeltaUQ checkpoint.

Loads results/uq_v1v2/topogdn/<V>/seed<S>/best.pt (trained by
v1v2_topogdn_gdeltauq_train.py) and produces arrays_full.npz with the SAME schema
the fusion stage consumes (test-side channels):

  test_mu_bar, test_ground_truth, test_attack_label,
  test_U_par (epistemic, T,V), test_U_str (structural, T,(topk-1)*V),
  test_U_dist (placeholder = U_par mean, kept for compat),
  test_sigma2_ale (aleatoric, T,V),
  test_U_dist_maha_mean / _pernode (Mahalanobis Omega),
  val_mu_bar, val_ground_truth (for the paper-protocol IQR normaliser).

Pipeline (the ONE method, identical to GDN, only the backbone differs):
  1. anchor pool: K=100 forward_split reps sampled from the val slice (last 15%).
  2. K-anchor inference -> mu_bar=mean_K, U_par=var_K, U_str=var_K(anchored-layer
     attention, non-self block), h_bar=mean_K(penultimate).
  3. aleatoric head: Gaussian-NLL on the held-out slice (h_bar, mu_bar, y).
  4. Omega: per-node Mahalanobis fit on a train-slice subsample of h_bar, scored on test.

Runs in the topogdn conda env. One (variant, seed) per call:
  python v1v2_topogdn_gdeltauq_fulluq.py --variant V1 --seed 0 --device cuda:0
"""
import argparse
import json
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
sys.path.insert(0, HERE)
os.chdir(TOPO)

import build_full_topogdn as B          # noqa: E402  train_ale, fit_maha, score_maha, auroc, DEV
from models.TopoGDN_GDeltaUQ import TopoGDN_GDeltaUQ   # noqa: E402
from util.net_struct import get_feature_map, get_fc_graph_struc  # noqa: E402
from util.preprocess import build_loc_net, construct_data  # noqa: E402
from datasets.TimeDataset import TimeDataset  # noqa: E402
from util.env import set_device, get_device  # noqa: E402

SW = 60
OMEGA_TRAIN_STRIDE = 8     # subsample the train slice for the Mahalanobis fit


def collect_anchor_pool(model, subset, K, dev, seed=0):
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    pool = []
    with torch.no_grad():
        for x, _, _, _ in loader:
            h_pre = model.forward_split(x.float().to(dev))   # (B,V,d)
            for b in range(h_pre.shape[0]):
                pool.append(h_pre[b].detach().cpu())
    if len(pool) < K:
        raise ValueError(f"val slice has {len(pool)} windows < K={K}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=K, replace=False)
    return torch.stack([pool[int(i)] for i in idx], 0).to(dev)   # (K,V,d)


def run_kanchor(model, subset, anchor_pool, dev, want_hbar=True, want_str=True):
    """K-anchor inference with memory-light running moments.
    Returns dict: mu_bar (T,V), U_par (T,V), U_str (T,E) or None, h_bar (T,V,d) or None,
    gt (T,V), label (T,)."""
    K = anchor_pool.shape[0]
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    MU, UPAR, USTR, HBAR, GT, LAB = [], [], [], [], [], []
    bi = 0
    with torch.no_grad():
        for x, y, lbl, _ in loader:
            bi += 1
            if torch.cuda.is_available() and bi % 20 == 0:
                torch.cuda.empty_cache()      # allocator-balloon fix (dualstage campaign)
            x = x.float().to(dev); Bn, V = x.shape[0], x.shape[1]
            h_pre = model.forward_split(x)                    # (B,V,d)
            d = h_pre.shape[-1]
            mu_sum = torch.zeros(Bn, V, device=dev); mu_sq = torch.zeros(Bn, V, device=dev)
            h_sum = torch.zeros(Bn, V, d, device=dev) if want_hbar else None
            att_sum = att_sq = None; nonself = None
            for k in range(K):
                mu_k, h_k, att_k = model.forward_anchored(h_pre, anchor_pool[k])
                mu_sum += mu_k; mu_sq += mu_k * mu_k
                if want_hbar:
                    h_sum += h_k
                if want_str:
                    if nonself is None:
                        eps = att_k.shape[0] // Bn; nonself = eps - V
                        att_sum = torch.zeros(Bn, nonself, device=dev)
                        att_sq = torch.zeros(Bn, nonself, device=dev)
                    a = att_k.view(-1)[:Bn * nonself].view(Bn, nonself)
                    att_sum += a; att_sq += a * a
            mu_bar = mu_sum / K
            U_par = ((mu_sq / K - mu_bar * mu_bar) * (K / (K - 1))).clamp_min(0.0)
            MU.append(mu_bar.cpu().numpy()); UPAR.append(U_par.cpu().numpy())
            if want_str:
                a_bar = att_sum / K
                U_str = ((att_sq / K - a_bar * a_bar) * (K / (K - 1))).clamp_min(0.0)
                USTR.append(U_str.cpu().numpy())
            if want_hbar:
                HBAR.append((h_sum / K).cpu().numpy())
            GT.append(y.numpy()); LAB.append(lbl.numpy())
    out = dict(mu_bar=np.concatenate(MU, 0), U_par=np.concatenate(UPAR, 0),
               gt=np.concatenate(GT, 0), label=np.concatenate(LAB, 0).astype(np.int8))
    out["U_str"] = np.concatenate(USTR, 0) if want_str else None
    out["h_bar"] = np.concatenate(HBAR, 0) if want_hbar else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K_anchors", type=int, default=100)
    ap.add_argument("--anchor_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="swat",
                    help="swat -> results/uq_v1v2/topogdn; wadi -> results/uq_wadi_v2/topogdn")
    args = ap.parse_args()
    # allocator hygiene (the fulluq balloon fix from the dualstage campaign):
    # fragmentation across the K-anchor loop can fill the card on big graphs.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    set_device(args.device); dev = get_device()

    res_root = "results/uq_v1v2/topogdn" if args.dataset == "swat" \
        else f"results/uq_{args.dataset}_v2/topogdn"
    OUT = os.path.join(ROOT, f"{res_root}/{args.variant}/seed{args.seed}")
    ckpt = os.path.join(OUT, "best.pt")
    if not os.path.exists(ckpt):
        print(f"[topo-uq {args.variant} s{args.seed}] missing {ckpt}", flush=True); sys.exit(11)
    with open(os.path.join(OUT, "hyperparameters.json")) as f:
        hp = json.load(f)

    feature_map = get_feature_map(args.dataset); fc = get_fc_graph_struc(args.dataset)
    train_df = pd.read_csv(f"data/{args.dataset}/train.csv", index_col=0)
    test_df = pd.read_csv(f"data/{args.dataset}/test.csv", index_col=0)
    cols = [c for c in train_df.columns if c != "attack"]
    fc_ei = torch.tensor(build_loc_net(fc, cols, feature_map=feature_map), dtype=torch.long)
    V = len(feature_map)

    tr_in = construct_data(train_df, feature_map, labels=0)
    te_in = construct_data(test_df, feature_map, labels=test_df.attack.tolist())
    cfg = {"slide_win": SW, "slide_stride": 1}
    full_train = TimeDataset(tr_in, fc_ei, mode="train", config=cfg)
    test_ds = TimeDataset(te_in, fc_ei, mode="test", config=cfg)

    N = len(full_train)
    p70, p85 = int(0.70 * N), int(0.85 * N)
    tr_idx = list(range(0, p70 if args.variant == "V1" else p85))
    val_idx = list(range(p85, N))                          # last 15% = anchor pool + aleatoric
    val_sub = Subset(full_train, val_idx)
    om_sub = Subset(full_train, tr_idx[::OMEGA_TRAIN_STRIDE])

    model = TopoGDN_GDeltaUQ([fc_ei], V, dim=int(hp["dim"]), out_layer_inter_dim=int(hp["out_layer_inter_dim"]),
                             input_dim=int(hp["slide_win"]), out_layer_num=int(hp["out_layer_num"]),
                             topk=int(hp["topk"]), n_gnn_layers=int(hp["n_gnn_layers"]),
                             use_topo=bool(hp["use_topo"]), use_msconv=bool(hp["use_msconv"])).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[topo-uq {args.variant} s{args.seed}] N={N} train={len(tr_idx)} val={len(val_idx)} "
          f"om_sub={len(om_sub)} K={args.K_anchors}", flush=True)

    # 1. anchor pool from the val slice
    pool = collect_anchor_pool(model, val_sub, args.K_anchors, dev, seed=args.anchor_seed)
    print(f"  anchor pool {tuple(pool.shape)}", flush=True)

    # 2. K-anchor inference on val -> aleatoric training data + val_mu_bar
    vo = run_kanchor(model, val_sub, pool, dev, want_hbar=True, want_str=False)
    head = B.train_ale(vo["h_bar"], vo["gt"], vo["mu_bar"])       # Gaussian NLL on held-out slice
    print("  aleatoric head trained on held-out slice", flush=True)

    # 3. K-anchor inference on test -> all channels
    to = run_kanchor(model, test_ds, pool, dev, want_hbar=True, want_str=True)
    with torch.no_grad():
        sig2 = head(torch.Tensor(to["h_bar"]).to(dev)).exp().cpu().numpy()
    print(f"  test mu_bar {to['mu_bar'].shape} U_par {to['U_par'].shape} U_str {to['U_str'].shape}", flush=True)

    # 4. Omega: Mahalanobis fit on a train-slice subsample of h_bar, scored on test
    oo = run_kanchor(model, om_sub, pool, dev, want_hbar=True, want_str=False)
    mean, inv = B.fit_maha(oo["h_bar"])
    omega_pn = B.score_maha(to["h_bar"], mean, inv); omega_mean = omega_pn.mean(1)

    lab = to["label"].astype(int)
    out = dict(
        test_mu_bar=to["mu_bar"].astype(np.float32),
        test_ground_truth=to["gt"].astype(np.float32),
        test_attack_label=to["label"].astype(np.int8),
        test_U_par=to["U_par"].astype(np.float32),
        test_U_str=to["U_str"].astype(np.float32),
        test_U_dist=to["U_par"].mean(1).astype(np.float32),       # placeholder (promoted in fusion)
        test_sigma2_ale=sig2.astype(np.float32),
        test_U_dist_maha_mean=omega_mean.astype(np.float32),
        test_U_dist_maha_pernode=omega_pn.astype(np.float32),
        val_mu_bar=vo["mu_bar"].astype(np.float32),
        val_ground_truth=vo["gt"].astype(np.float32),
    )
    outp = os.path.join(OUT, "arrays_full.npz")
    np.savez_compressed(outp, **out)
    torch.save(pool.cpu(), os.path.join(OUT, "anchor_pool.pt"))
    torch.save(head.state_dict(), os.path.join(OUT, "aleatoric_head.pt"))

    a_om = B.auroc(omega_mean, lab); a_epi = B.auroc(to["U_par"].mean(1), lab)
    a_str = B.auroc(to["U_str"].mean(1), lab)
    print(f"[topo-uq {args.variant} s{args.seed}] Omega AUROC={a_om:.4f} epi={a_epi:.4f} "
          f"str={a_str:.4f} sig2_real={sig2.std()>1e-9} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
