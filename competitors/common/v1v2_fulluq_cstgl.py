#!/usr/bin/env python3
"""Full UQ extraction for ONE CST-GL V1/V2 baseline model.

Loads competitors/CST-GL/save/expV{1,2}base_{S}.pth, reads the re-sliced
swat_canon_{V1,V2} data, and produces epistemic (MC-dropout) + aleatoric +
Mahalanobis Omega, spliced into results/baseline_v1v2/cstgl/<V>/seed<S>/
arrays_full.npz (which already has mu/gt/label from the baseline run).
Self-contained (CST-GL's build_full module chdir's at import, so we don't import it).

  python v1v2_fulluq_cstgl.py --variant V1 --seed 0 --device cuda:0
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CSTGL = os.path.join(ROOT, "competitors", "CST-GL")
sys.path.insert(0, CSTGL)
os.chdir(CSTGL)
from stgnn import stgnn  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
K_MC = 30   # epistemic MC passes; reduced from 100 to keep the 12-run batch tractable


def build_model():
    return stgnn(True, True, 2, 51, DEV, predefined_A=None, dropout=0.1,
                 subgraph_size=15, node_dim=256, dilation_exponential=1,
                 conv_channels=16, residual_channels=16, skip_channels=32,
                 end_channels=64, seq_length=60, in_dim=1, out_dim=1,
                 layers=2, propalpha=0.1, tanhalpha=20, layer_norm_affline=True).to(DEV)


class Penult:
    def __init__(self): self.val = None
    def __call__(self, m, i, o):
        z = torch.relu(o).squeeze(-1)            # (B,64,V)
        self.val = z.permute(0, 2, 1).contiguous()  # (B,V,64)


def forward(model, x_np, cap, mc=False, bs=64):
    if mc:
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.train()
        model.training = True
    else:
        model.eval()
    N = x_np.shape[0]; mus, phis = [], []
    with torch.no_grad():
        for i in range(0, N, bs):
            xb = torch.Tensor(x_np[i:i+bs]).to(DEV).transpose(1, 3)
            cap.val = None; out = model(xb)
            if isinstance(out, tuple): out = out[0]
            mus.append(out.squeeze(-1).squeeze(1).cpu().numpy())
            phis.append(cap.val.cpu().numpy())
    return np.concatenate(mus, 0), np.concatenate(phis, 0)


class AleHead(nn.Module):
    def __init__(self, d, V, emb=16, h=64):
        super().__init__(); self.emb = nn.Embedding(V, emb)
        self.mlp = nn.Sequential(nn.Linear(d+emb, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, phi):
        B, V, _ = phi.shape
        e = self.emb(torch.arange(V, device=phi.device)).unsqueeze(0).expand(B, -1, -1)
        return self.mlp(torch.cat([phi, e], -1)).squeeze(-1).clamp(-10, 10)


def train_ale(phi, y, mu, epochs=8, bs=256, beta=0.5):
    head = AleHead(phi.shape[-1], phi.shape[1]).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    P, Y, M = (torch.Tensor(a).to(DEV) for a in (phi, y, mu)); N = P.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=DEV)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; lv = head(P[idx]); s2 = lv.exp().clamp_min(1e-6)
            per = ((Y[idx]-M[idx])**2)/(2*s2) + 0.5*lv; per = per*(s2.detach()**beta)
            opt.zero_grad(); per.mean().backward(); opt.step()
    head.eval(); return head


def fit_maha(phi_tr, eps=1e-3):
    V, d = phi_tr.shape[1], phi_tr.shape[2]; mean = phi_tr.mean(0)
    inv = np.empty((V, d, d)); eye = np.eye(d)
    for v in range(V): inv[v] = np.linalg.inv(np.cov(phi_tr[:, v, :].T) + eps*eye)
    return mean, inv


def score_maha(phi, mean, inv):
    cen = phi - mean[None]; tmp = np.einsum("tvi,vij->tvj", cen, inv)
    return np.sqrt(np.maximum(np.einsum("tvj,tvj->tv", tmp, cen), 0.0))


def auroc(s, l):
    from sklearn.metrics import roc_auc_score
    try: return float(roc_auc_score(l, s))
    except Exception: return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(); V, S = args.variant, args.seed

    ckpt = os.path.join(CSTGL, "save", f"expV{V[-1]}base_{S}.pth")
    rdir = os.path.join(ROOT, f"results/baseline_v1v2/cstgl/{V}/seed{S}")
    base = os.path.join(rdir, "arrays.npz")
    if not (os.path.exists(ckpt) and os.path.exists(base)):
        print(f"[cstgl {V} s{S}] missing ckpt={ckpt} or arrays", flush=True); sys.exit(11)

    data = os.path.join(CSTGL, "data", f"swat_canon_{V}")
    tr = np.load(os.path.join(data, "train.npz")); te = np.load(os.path.join(data, "test.npz"))
    xtr, ytr = tr["x"], tr["y"][:, 0, :, 0]; xte = te["x"]

    model = build_model(); model.load_state_dict(torch.load(ckpt, map_location=DEV))
    cap = Penult(); h = model.end_conv_1.register_forward_hook(cap)
    mu_tr, phi_tr = forward(model, xtr, cap)
    mu_te, phi_te = forward(model, xte, cap)
    # MC epistemic on test
    sums = np.zeros_like(mu_te, dtype=np.float64); sumsq = sums.copy()
    for _k in range(K_MC):
        m_k, _ = forward(model, xte, cap, mc=True); sums += m_k; sumsq += m_k**2
    model.eval(); h.remove()
    U_par = np.maximum(sumsq/K_MC - (sums/K_MC)**2, 0.0)

    ref = np.load(base); T = ref["test_attack_label"].shape[0]
    mu_te, phi_te, U_par = mu_te[-T:], phi_te[-T:], U_par[-T:]
    lab = ref["test_attack_label"].astype(int)

    nval = min(8000, phi_tr.shape[0] // 4)
    head = train_ale(phi_tr[-nval:], ytr[-nval:], mu_tr[-nval:])
    with torch.no_grad():
        sig2 = head(torch.Tensor(phi_te).to(DEV)).exp().cpu().numpy()
    mean, inv = fit_maha(phi_tr[::4]); omega_pn = score_maha(phi_te, mean, inv); omega_mean = omega_pn.mean(1)

    out = {k: ref[k] for k in ref.files}
    out["test_U_par"] = U_par.astype(np.float32)
    out["test_U_dist"] = U_par.mean(1).astype(np.float32)
    out["test_sigma2_ale"] = sig2.astype(np.float32)
    out["test_U_dist_maha_mean"] = omega_mean.astype(np.float32)
    out["test_U_dist_maha_pernode"] = omega_pn.astype(np.float32)
    outp = os.path.join(rdir, "arrays_full.npz")
    np.savez_compressed(outp, **out)
    print(f"[cstgl {V} s{S}] Omega AUROC={auroc(omega_mean,lab):.4f} epi={auroc(U_par.mean(1),lab):.4f} sig2_real={sig2.std()>1e-9} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
