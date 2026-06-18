#!/usr/bin/env python3
"""Build the real ALEATORIC + Mahalanobis OMEGA channels for CST-GL (all 5 seeds).

Runs in the cstgl conda env (torch 1.13). Self-contained (no repo-internal imports
beyond CST-GL's own stgnn/util, which are importable from competitors/CST-GL/).

Per seed:
  1. load checkpoint, hook the penultimate F.relu(end_conv_1(x)) -> (B,64,V,1)->(B,V,64)
  2. extract penultimate phi on TRAIN (normal) and TEST (deterministic, dropout off)
  3. train a small Gaussian-NLL aleatoric head (phi -> per-sensor log-sigma2) on a
     held-out NORMAL slice (tail of train); emit sigma2_ale (T,V) on test
  4. fit per-node Gaussian on TRAIN phi; Omega_v = sqrt Mahalanobis on TEST phi;
     emit maha_mean/max/pernode + kNN(k=10)
  5. splice into NEW results/competitors/cstgl/seed{S}_full_arrays.npz (copy mc keys,
     overwrite sigma2_ale with the real one, add omega keys). Originals untouched.
  6. print per-seed AUROC (vs attack label) + distinctness for the report.

Usage (from repo root, cstgl env):
  CUDA_VISIBLE_DEVICES=<g> /home/rashinda/.conda/envs/cstgl/bin/python \
      competitors/common/build_full_cstgl.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CSTGL = os.path.join(ROOT, "competitors", "CST-GL")
sys.path.insert(0, CSTGL)
os.chdir(CSTGL)  # CST-GL code uses relative data paths

from stgnn import stgnn  # noqa: E402

SEEDS = [1, 2, 3, 42, 100]
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model():
    return stgnn(True, True, 2, 51, DEV, predefined_A=None, dropout=0.1,
                 subgraph_size=15, node_dim=256, dilation_exponential=1,
                 conv_channels=16, residual_channels=16, skip_channels=32,
                 end_channels=64, seq_length=60, in_dim=1, out_dim=1,
                 layers=2, propalpha=0.1, tanhalpha=20, layer_norm_affline=True).to(DEV)


class Penult:
    """Forward hook capturing relu(end_conv_1) output -> (B,V,64)."""
    def __init__(self):
        self.val = None

    def __call__(self, module, inp, out):
        # out of end_conv_1: (B, end_channels=64, V, 1); apply relu (as in forward)
        z = torch.relu(out).squeeze(-1)        # (B,64,V)
        self.val = z.permute(0, 2, 1).contiguous()  # (B,V,64)


def run_forward(model, x_np, cap, bs=64):
    """Deterministic forward over windows x_np (N,60,V,1). Returns mu (N,V), phi (N,V,64)."""
    model.eval()
    N = x_np.shape[0]
    mus, phis = [], []
    with torch.no_grad():
        for i in range(0, N, bs):
            xb = torch.Tensor(x_np[i:i + bs]).to(DEV).transpose(1, 3)  # (B,1,V,60)
            cap.val = None
            out = model(xb)                                            # (B,1,V,1) or (x,adp)
            if isinstance(out, tuple):
                out = out[0]
            mus.append(out.squeeze(-1).squeeze(1).cpu().numpy())       # (B,V)
            phis.append(cap.val.cpu().numpy())                         # (B,V,64)
    return np.concatenate(mus, 0), np.concatenate(phis, 0)


class AleatoricHead(nn.Module):
    def __init__(self, d, V, emb=16, h=64):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.mlp = nn.Sequential(nn.Linear(d + emb, h), nn.ReLU(), nn.Linear(h, 1))
        self.V = V

    def forward(self, phi):  # (B,V,d)
        B, V, _ = phi.shape
        e = self.emb(torch.arange(V, device=phi.device)).unsqueeze(0).expand(B, -1, -1)
        return self.mlp(torch.cat([phi, e], -1)).squeeze(-1).clamp(-10, 10)  # (B,V) log-var


def train_aleatoric(phi, y, mu, epochs=8, bs=256, beta=0.5):
    """Gaussian (beta-)NLL head on frozen (phi, y, mu). All numpy in; returns head."""
    d = phi.shape[-1]; V = phi.shape[1]
    head = AleatoricHead(d, V).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    P = torch.Tensor(phi).to(DEV); Y = torch.Tensor(y).to(DEV); M = torch.Tensor(mu).to(DEV)
    N = P.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEV)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            lv = head(P[idx]); s2 = lv.exp().clamp_min(1e-6)
            per = ((Y[idx] - M[idx]) ** 2) / (2 * s2) + 0.5 * lv
            if beta > 0:
                per = per * (s2.detach() ** beta)
            loss = per.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def fit_mahalanobis(phi_tr, eps=1e-3):
    """Per-node Gaussian. phi_tr (N,V,d) -> mean (V,d), inv_cov (V,d,d)."""
    V, d = phi_tr.shape[1], phi_tr.shape[2]
    mean = phi_tr.mean(0)
    inv = np.empty((V, d, d), np.float64)
    eye = np.eye(d)
    for v in range(V):
        c = np.cov(phi_tr[:, v, :].T) + eps * eye
        inv[v] = np.linalg.inv(c)
    return mean, inv


def score_mahalanobis(phi, mean, inv):
    cen = phi - mean[None]                                 # (T,V,d)
    tmp = np.einsum("tvi,vij->tvj", cen, inv)
    quad = np.maximum(np.einsum("tvj,tvj->tv", tmp, cen), 0.0)
    return np.sqrt(quad)                                   # (T,V)


def knn_omega(phi_tr, phi_te, k=10, maxn=5000):
    Xtr = phi_tr.reshape(phi_tr.shape[0], -1)
    Xte = phi_te.reshape(phi_te.shape[0], -1)
    if Xtr.shape[0] > maxn:
        idx = np.linspace(0, Xtr.shape[0] - 1, maxn).astype(int)
        Xtr = Xtr[idx]
    from sklearn.neighbors import NearestNeighbors
    nn_ = NearestNeighbors(n_neighbors=k).fit(Xtr)
    dist, _ = nn_.kneighbors(Xte)
    return dist[:, -1]


def auroc(score, label):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(label, score))
    except Exception:
        return float("nan")


def main():
    data = "data/swat_canon"
    tr = np.load(f"{data}/train.npz"); te = np.load(f"{data}/test.npz")
    xtr, ytr = tr["x"], tr["y"][:, 0, :, 0]            # (Ntr,60,V,1), (Ntr,V)
    xte = te["x"]
    print(f"[cstgl] train {xtr.shape} test {xte.shape}", flush=True)

    report = []
    for s in SEEDS:
        ckpt = os.path.join(CSTGL, "save", f"expswat5_{s}.pth")
        if not os.path.exists(ckpt):
            print(f"[cstgl s{s}] missing ckpt {ckpt}", flush=True); continue
        model = build_model()
        sd = torch.load(ckpt, map_location=DEV)
        model.load_state_dict(sd)
        cap = Penult()
        h = model.end_conv_1.register_forward_hook(cap)

        mu_tr, phi_tr = run_forward(model, xtr, cap)
        mu_te, phi_te = run_forward(model, xte, cap)
        h.remove()

        # align to cached arrays (tail slice to T)
        ref = np.load(os.path.join(ROOT, f"results/competitors/cstgl/seed{s}_mc_arrays.npz"))
        T = ref["test_attack_label"].shape[0]
        mu_te, phi_te = mu_te[-T:], phi_te[-T:]
        lab = ref["test_attack_label"].astype(int)

        # aleatoric head on a held-out NORMAL tail of train
        nval = min(8000, xtr.shape[0] // 4)
        head = train_aleatoric(phi_tr[-nval:], ytr[-nval:], mu_tr[-nval:])
        with torch.no_grad():
            sig2 = head(torch.Tensor(phi_te).to(DEV)).exp().cpu().numpy()  # (T,V)

        # omega
        mean, inv = fit_mahalanobis(phi_tr[::4])  # subsample train for cov speed
        omega_pn = score_mahalanobis(phi_te, mean, inv)
        omega_mean = omega_pn.mean(1); omega_max = omega_pn.max(1)
        try:
            omega_knn = knn_omega(phi_tr[::4], phi_te)
        except Exception as e:
            print(f"  knn fail s{s}: {e}", flush=True); omega_knn = omega_mean.copy()

        # splice
        out = {k: ref[k] for k in ref.files}
        out["test_sigma2_ale_real"] = sig2.astype(np.float32)
        out["test_sigma2_ale"] = sig2.astype(np.float32)
        out["test_U_dist_maha_mean"] = omega_mean.astype(np.float32)
        out["test_U_dist_maha_max"] = omega_max.astype(np.float32)
        out["test_U_dist_maha_pernode"] = omega_pn.astype(np.float32)
        out["test_U_dist_knn"] = omega_knn.astype(np.float32)
        outp = os.path.join(ROOT, f"results/competitors/cstgl/seed{s}_full_arrays.npz")
        np.savez_compressed(outp, **out)

        upar = ref["test_U_par"].mean(1)
        a_pl = auroc(ref["test_U_dist"], lab); a_om = auroc(omega_mean, lab)
        try:
            from scipy.stats import spearmanr
            rho = float(spearmanr(omega_mean, upar).correlation)
        except Exception:
            rho = float("nan")
        rec = dict(seed=s, omega_auroc=round(a_om, 4), placeholder_auroc=round(a_pl, 4),
                   distinct_rho=round(rho, 3), sig2_real=bool(sig2.std() > 1e-9))
        report.append(rec)
        print(f"[cstgl s{s}] Omega AUROC={a_om:.4f} (placeholder {a_pl:.4f}) rho={rho:.3f} "
              f"sig2_real={rec['sig2_real']} -> {outp}", flush=True)

    # report
    rp = os.path.join(ROOT, "results/competitors/cstgl/cstgl_full_report.md")
    with open(rp, "w") as f:
        f.write("# CST-GL real aleatoric + Omega (5 seeds)\n\n")
        f.write("| seed | Omega AUROC | placeholder AUROC | distinct rho | sigma2_ale real |\n")
        f.write("|------|-------------|-------------------|--------------|------------------|\n")
        for r in report:
            f.write(f"| {r['seed']} | {r['omega_auroc']} | {r['placeholder_auroc']} | "
                    f"{r['distinct_rho']} | {r['sig2_real']} |\n")
    print(f"\nwrote {rp}", flush=True)
    if report:
        oa = np.mean([r["omega_auroc"] for r in report]); pa = np.mean([r["placeholder_auroc"] for r in report])
        print(f"MEAN Omega AUROC={oa:.4f} vs placeholder {pa:.4f}  (beats: {oa > pa})", flush=True)


if __name__ == "__main__":
    main()
