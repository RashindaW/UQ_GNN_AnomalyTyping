#!/usr/bin/env python3
"""Plain DualSTAGE (dual-view, paper config) V1/V2 baseline on SWaT.

The paper method (DualSTAGE_edited_RW.pdf): temporal branch (IDCNN edge input,
GRU node encoder, GATv2+EdgeGRU dynamic graph) + spectral branch (rFFT, linear
band mixing, GATv2 graph) -> 2x WeightedGIN per view -> gated fusion ->
reverse-GRU reconstruction decoder. Loss = MSE + omega_anom * s_topo +
lambda_div * JS(P||Q) with the PRONTO-transfer weights (0.5 / 0.05).

Campaign conventions: W=60 stride 1; V1=0.70 / V2=0.85 window fractions,
val = last 15 percent of train windows; z-score stats fit ONLY on the
variant's train rows; arrays de-normalized to raw scale; per-timestep
alignment = last-step slice of the reconstruction (one row per window,
label at the window's last timestep, the SWaTDataset convention).

Output: results/baseline_v1v2/dualstage/{V}/seed{S}/{arrays.npz, best.pt,
hyperparameters.json}. Runs in the topogdn conda env.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "dualstgf", "dualstage"))
sys.path.insert(0, os.path.join(ROOT, "dualstgf"))

from src.config import cfg  # noqa: E402

# Dataset selection via UQ_DATASET (swat default, or wadi). Only the data, the
# node count, and the data directory change; the model and method are identical.
_DS = os.environ.get("UQ_DATASET", "swat").lower()
N_NODES = 123 if _DS == "wadi" else 51
DATA_DIR = "data/wadi" if _DS == "wadi" else "data/swat"
cfg.set_dataset_params(n_nodes=N_NODES, window_size=60, ocvar_dim=0,
                       pred_horizon=0, task="reconstruction")
from src.model.dualstage import DualSTAGE  # noqa: E402
if _DS == "wadi":
    from src.data.wadi_dataset import WADIDataset as SeriesDataset  # noqa: E402
    from src.data.wadi_column_config import MEASUREMENT_VARS  # noqa: E402
else:
    from src.data.swat_dataset import SWaTDataset as SeriesDataset  # noqa: E402
    from src.data.swat_column_config import MEASUREMENT_VARS  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

SW = 60
W_ANOM = 0.5     # omega_anom, PRONTO transfer
W_DIV = 0.05     # lambda_div, PRONTO transfer

HP = dict(
    temp_node_embed_dim=16, gnn_embed_dim=40, num_gnn_layers=2, topk=20,
    time_dim=5, temp_edge_hid_dim=100, feat_edge_hid_dim=128,
    sub_window_size=1, recon_hidden_dim=16, num_recon_layers=1,
    dropout=0.15, feat_dropout=0.0, gnn_type="gin",
    encoder_norm_type="layer", gnn_norm_type="layer", decoder_norm_type="layer",
    aug_control=False, use_spectral_view=True, freq_node_embed_dim=16,
    freq_use_log=True, freq_use_spectral_features=False,
    fuse_mode="gated", divergence_type="js", task="reconstruction",
)
OPT = dict(lr=5e-4, weight_decay=1e-4, clip=1.0, batch=64,
           plateau_patience=10, plateau_factor=0.5, early_stop=15,
           min_delta=1e-5, w_anom=W_ANOM, w_div=W_DIV)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device):
    model = DualSTAGE(feat_input_node=1, feat_target_node=1, feat_input_edge=1, **HP)
    return model.to(device)


def load_data(variant, smoke=0):
    frac = 0.70 if variant == "V1" else 0.85
    train_csv = os.path.join(ROOT, DATA_DIR, "train.csv")
    test_csv = os.path.join(ROOT, DATA_DIR, "test.csv")
    df = pd.read_csv(train_csv, sep=",", index_col=0)
    feats = df[MEASUREMENT_VARS].astype(np.float32).to_numpy()
    cut = int(frac * len(feats))
    stats = (feats[:cut].mean(0), feats[:cut].std(0) + 1e-8)

    full_train = SeriesDataset(train_csv, SW, 1, normalize=True, normalization_stats=stats)
    test_ds = SeriesDataset(test_csv, SW, 1, normalize=True, normalization_stats=stats)
    n_w = full_train.len()
    p = int(frac * n_w)
    v0 = int(0.85 * n_w)
    tr_idx = list(range(0, p))
    va_idx = list(range(v0, n_w))
    te_idx = list(range(test_ds.len()))
    if smoke:
        tr_idx = tr_idx[:smoke]
        va_idx = va_idx[:max(200, smoke // 10)]
        te_idx = te_idx[:smoke]
    return full_train, test_ds, tr_idx, va_idx, te_idx, stats


def composite_loss(model, batch, mse, device):
    recon, adj_t, attn_t, aux = model(batch, return_graph=True)
    target = batch.x
    rl = mse(recon, target)
    st = model.compute_topology_aware_anomaly_score(target, recon, adj_t, attn_t)
    dv = aux.get("divergence_loss", torch.zeros((), device=device))
    if not torch.isfinite(st):
        st = torch.zeros((), device=device)
    if not torch.isfinite(dv):
        dv = torch.zeros((), device=device)
    return rl + W_ANOM * st + W_DIV * dv, rl, st, dv


@torch.no_grad()
def collect_last_step(model, loader, device, stats):
    """Run the model, slice the chronological last timestep, de-normalize."""
    model.eval()
    mean = torch.from_numpy(np.asarray(stats[0])).to(device)
    std = torch.from_numpy(np.asarray(stats[1])).to(device)
    mus, gts, labs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        recon = model(batch, return_graph=False)          # [B*N_NODES, 60]
        b = recon.shape[0] // N_NODES
        mu_z = recon.view(b, N_NODES, SW)[:, :, -1]
        gt_z = batch.x.view(b, N_NODES, SW)[:, :, -1]
        mus.append((mu_z * std + mean).cpu().numpy())
        gts.append((gt_z * std + mean).cpu().numpy())
        labs.append(batch.y.view(-1).cpu().numpy())
    return (np.concatenate(mus), np.concatenate(gts),
            np.concatenate(labs).astype(np.int8))


def orientation_check(model, loader, device):
    """recon[:, -1] must track x[:, :, -1] better than recon[:, 0] does."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        recon = model(batch, return_graph=False)
        b = recon.shape[0] // N_NODES
        r = recon.view(b, N_NODES, SW)
        x = batch.x.view(b, N_NODES, SW)
        def corr(a, c):
            a = a.flatten() - a.mean()
            c = c.flatten() - c.mean()
            return float((a * c).sum() / (a.norm() * c.norm() + 1e-12))
        c_last = corr(r[:, :, -1], x[:, :, -1])
        c_first = corr(r[:, :, 0], x[:, :, -1])
        c_id0 = corr(r[:, :, 0], x[:, :, 0])
    return c_last, c_first, c_id0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=OPT["batch"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", type=int, default=0, help="subset windows for a smoke run")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "results/baseline_v1v2/dualstage"))
    args = ap.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)
    out_dir = os.path.join(args.out_root, args.variant, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    full_train, test_ds, tr_idx, va_idx, te_idx, stats = load_data(args.variant, args.smoke)
    tr_loader = DataLoader(full_train[tr_idx], batch_size=args.batch, shuffle=True, num_workers=2)
    va_loader = DataLoader(full_train[va_idx], batch_size=args.batch, shuffle=False, num_workers=2)
    te_loader = DataLoader(test_ds[te_idx], batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"[{args.variant} s{args.seed}] windows train={len(tr_idx)} val={len(va_idx)} "
          f"test={len(te_idx)}", flush=True)

    model = build_model(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_par}", flush=True)
    mse = torch.nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=OPT["lr"], weight_decay=OPT["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=OPT["plateau_factor"], patience=OPT["plateau_patience"])

    best_val, best_state, bad, best_ep = float("inf"), None, 0, -1
    for ep in range(args.epochs):
        model.train()
        t0, tl, trl, tst, tdv, nb = time.time(), 0.0, 0.0, 0.0, 0.0, 0
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            loss, rl, st, dv = composite_loss(model, batch, mse, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=OPT["clip"])
            opt.step()
            tl += loss.item(); trl += rl.item(); tst += st.item(); tdv += dv.item(); nb += 1
        model.eval()
        vl, vb = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                loss, _, _, _ = composite_loss(model, batch, mse, device)
                vl += loss.item(); vb += 1
        vl /= max(vb, 1)
        sched.step(vl)
        print(f"ep{ep:02d} train={tl/max(nb,1):.5f} (recon={trl/max(nb,1):.5f} "
              f"topo={tst/max(nb,1):.5f} div={tdv/max(nb,1):.5f}) val={vl:.5f} "
              f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
        if vl < best_val - OPT["min_delta"]:
            best_val, bad, best_ep = vl, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= OPT["early_stop"]:
                print(f"early stop at ep{ep} (best ep{best_ep} val={best_val:.5f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    c_last, c_first, c_id0 = orientation_check(model, te_loader, device)
    print(f"orientation: corr(recon[-1], x[-1])={c_last:.3f}  "
          f"corr(recon[0], x[-1])={c_first:.3f}  corr(recon[0], x[0])={c_id0:.3f}", flush=True)

    test_mu, test_gt, test_lab = collect_last_step(model, te_loader, device, stats)
    val_mu, val_gt, _ = collect_last_step(model, va_loader, device, stats)
    np.savez_compressed(
        os.path.join(out_dir, "arrays.npz"),
        test_mu_bar=test_mu, test_ground_truth=test_gt, test_attack_label=test_lab,
        val_mu_bar=val_mu, val_ground_truth=val_gt)
    with open(os.path.join(out_dir, "hyperparameters.json"), "w") as f:
        json.dump(dict(HP, **OPT, variant=args.variant, seed=args.seed,
                       epochs=args.epochs, best_epoch=best_ep, best_val=best_val,
                       params=n_par, smoke=args.smoke,
                       align="last_step_slice", scale="raw(de-normalized)",
                       orientation=dict(c_last=c_last, c_first=c_first, c_id0=c_id0)),
                  f, indent=2, default=float)
    print(f"wrote {out_dir}/arrays.npz  T={len(test_lab)} attack_rate={test_lab.mean():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
