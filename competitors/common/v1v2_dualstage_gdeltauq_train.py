#!/usr/bin/env python3
"""Train ONE anchored DualSTAGE_GDeltaUQ (V, seed) on SWaT.

The ONE canonical method: trained-in G-DeltaUQ batch-shuffle anchoring at
z_fused (post gated-fusion, post decoder_norm), anchored GRU decoder tail.
Data path, split conventions, paper hyperparameters and the composite loss
(MSE + 0.5 s_topo + 0.05 JS divergence) are imported from the plain baseline
driver so the two tiers differ ONLY by anchoring.

Saves best.pt + hyperparameters.json to results/uq_v1v2/dualstage/{V}/seed{S}/.
Runs in the topogdn conda env.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "competitors", "common"))

from baseline_v1v2_dualstage import (HP, OPT, SW, W_ANOM, W_DIV,  # noqa: E402
                                     set_seed, load_data, composite_loss)
from src.model.dualstage_gdeltauq import DualSTAGE_GDeltaUQ  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=OPT["batch"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "results/uq_v1v2/dualstage"))
    args = ap.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)
    out_dir = os.path.join(args.out_root, args.variant, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    full_train, _, tr_idx, va_idx, _, stats = load_data(args.variant, args.smoke)
    tr_loader = DataLoader(full_train[tr_idx], batch_size=args.batch, shuffle=True, num_workers=2)
    va_loader = DataLoader(full_train[va_idx], batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"[uq {args.variant} s{args.seed}] anchored train windows={len(tr_idx)} "
          f"val={len(va_idx)}", flush=True)

    model = DualSTAGE_GDeltaUQ(feat_input_node=1, feat_target_node=1,
                               feat_input_edge=1, **HP).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_par} (anchored decoder in_dim={2 * model.anchor_dim})", flush=True)
    mse = torch.nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=OPT["lr"], weight_decay=OPT["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=OPT["plateau_factor"], patience=OPT["plateau_patience"])

    best_val, best_state, bad, best_ep = float("inf"), None, 0, -1
    for ep in range(args.epochs):
        model.train()
        t0, tl, nb = time.time(), 0.0, 0
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            loss, rl, st, dv = composite_loss(model, batch, mse, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=OPT["clip"])
            opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        vl, vb = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                loss, _, _, _ = composite_loss(model, batch, mse, device)
                vl += loss.item(); vb += 1
        vl /= max(vb, 1)
        sched.step(vl)
        print(f"ep{ep:02d} train={tl/max(nb,1):.5f} val={vl:.5f} "
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
    with open(os.path.join(out_dir, "hyperparameters.json"), "w") as f:
        json.dump(dict(HP, **OPT, variant=args.variant, seed=args.seed,
                       epochs=args.epochs, best_epoch=best_ep, best_val=best_val,
                       params=n_par, smoke=args.smoke, anchored=True,
                       anchor_cut="z_fused_post_norm"),
                  f, indent=2, default=float)
    print(f"[uq {args.variant} s{args.seed}] wrote {out_dir}/best.pt "
          f"(best ep{best_ep} val={best_val:.5f})", flush=True)


if __name__ == "__main__":
    main()
