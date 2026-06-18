#!/usr/bin/env python3
"""Full UQ extraction for ONE TopoGDN V1/V2 baseline model.

Loads results/baseline_v1v2/topogdn/<V>/seed<S>/best.pt, runs:
  - K MC-dropout passes -> epistemic U_par (T,V) + structural U_str_mean (T,)
  - aleatoric head (Gaussian NLL) trained on the contiguous V1/V2 TRAIN slice
  - Mahalanobis Omega on the penultimate (fit on V1/V2 train, scored on test)
Splices into a NEW arrays_full.npz alongside the baseline arrays.npz (which has
mu/gt/label). Reuses the validated helpers from build_full_topogdn.py.

  python v1v2_fulluq_topogdn.py --variant V1 --seed 0 --device cuda:0
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
# reuse the validated builder helpers
import build_full_topogdn as B  # noqa: E402  (build_model, make_loader, extract, AleHead, train_ale, fit_maha, score_maha, auroc, TOPO, DEV)
from torch.utils.data import DataLoader, Subset  # noqa: E402

SW = 60
K_MC = 15   # epistemic var stabilizes by ~15 MC passes; keeps 36-run batch tractable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    V, S = args.variant, args.seed

    rdir = os.path.join(ROOT, f"results/baseline_v1v2/topogdn/{V}/seed{S}")
    ckpt = os.path.join(rdir, "best.pt")
    base = os.path.join(rdir, "arrays.npz")
    if not (os.path.exists(ckpt) and os.path.exists(base)):
        print(f"[topo {V} s{S}] missing ckpt/arrays", flush=True); sys.exit(11)

    model, fmap, fc_ei = B.build_model(ckpt)
    # full all-normal train windows; cut contiguous V1/V2 train slice for fit
    tr_loader_full = B.make_loader(os.path.join(B.TOPO, "data/swat/train.csv"), fmap, fc_ei, False)
    te_loader = B.make_loader(os.path.join(B.TOPO, "data/swat/test.csv"), fmap, fc_ei, True)
    cap = B.Cap(); h = model.dp.register_forward_hook(cap)

    mu_tr, phi_tr, _ = B.extract(model, tr_loader_full, cap)
    Ntr = phi_tr.shape[0]
    tr_end = int((0.70 if V == "V1" else 0.85) * Ntr)
    phi_fit = phi_tr[:tr_end]                      # contiguous V1/V2 train slice
    # targets y for aleatoric on the fit slice
    ys = []
    for _x, y, _l, _e in tr_loader_full:
        ys.append(y.numpy())
    y_tr = np.concatenate(ys, 0)[:tr_end]
    mu_fit = mu_tr[:tr_end]

    mu_te, phi_te, _ = B.extract(model, te_loader, cap)
    # K MC passes for epistemic + structural
    mu_stack = []; att_stack = []
    for _k in range(K_MC):
        m_k, _p, att = B.extract(model, te_loader, cap, want_attention=True, mc=True)
        mu_stack.append(m_k); att_stack.append(att)
    model.eval(); h.remove()
    mu_stack = np.stack(mu_stack, 0)               # (K,T,V)
    U_par = mu_stack.var(0)                         # (T,V) epistemic
    ustr = np.var(np.stack(att_stack, 0), axis=0)  # (T,) structural

    ref = np.load(base); T = ref["test_attack_label"].shape[0]
    mu_te, phi_te = mu_te[:T], phi_te[:T]; U_par = U_par[:T]
    ustr = ustr[:T] if ustr.shape[0] >= T else np.pad(ustr, (0, T - ustr.shape[0]))
    lab = ref["test_attack_label"].astype(int)

    nval = min(8000, phi_fit.shape[0] // 4)
    head = B.train_ale(phi_fit[-nval:], y_tr[-nval:], mu_fit[-nval:])
    with torch.no_grad():
        sig2 = head(torch.Tensor(phi_te).to(B.DEV)).exp().cpu().numpy()
    mean, inv = B.fit_maha(phi_fit[::4])
    omega_pn = B.score_maha(phi_te, mean, inv); omega_mean = omega_pn.mean(1)

    out = {k: ref[k] for k in ref.files}
    out["test_U_par"] = U_par.astype(np.float32)
    out["test_U_dist"] = U_par.mean(1).astype(np.float32)          # placeholder kept for compat
    out["test_sigma2_ale"] = sig2.astype(np.float32)
    out["test_U_str"] = ustr.astype(np.float32)
    out["test_U_dist_maha_mean"] = omega_mean.astype(np.float32)
    out["test_U_dist_maha_pernode"] = omega_pn.astype(np.float32)
    outp = os.path.join(rdir, "arrays_full.npz")
    np.savez_compressed(outp, **out)
    a_om = B.auroc(omega_mean, lab); a_epi = B.auroc(U_par.mean(1), lab)
    print(f"[topo {V} s{S}] Omega AUROC={a_om:.4f} epi AUROC={a_epi:.4f} sig2_real={sig2.std()>1e-9} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
