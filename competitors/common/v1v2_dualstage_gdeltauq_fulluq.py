#!/usr/bin/env python3
"""Calibrate + extract full UQ for ONE anchored DualSTAGE_GDeltaUQ checkpoint.

3 channels (epistemic + aleatoric + distributional Omega), no structural: the
GATv2 attention is computed in forward_split and is anchor-invariant (CST-GL
precedent). Mirrors v1v2_cstgl_gdeltauq_fulluq.py:

  anchor pool (K=100 forward_split reps from the val slice) -> K-anchor
  inference with running moments (mu_bar, U_par=var_K, h_bar=mean penultimate)
  -> post-hoc Gaussian-NLL aleatoric head on val -> per-node Mahalanobis Omega
  fit on a strided train subsample, scored on test.

Per-timestep alignment: mu = the reconstruction's chronological-last step
(de-normalized to raw scale, matching the baseline arrays). Attack labels and
T are sourced from results/baseline_v1v2/dualstage/{V}/seed{S}/arrays.npz.
Writes arrays_full.npz + anchor_pool.pt + aleatoric_head.pt to
results/uq_v1v2/dualstage/{V}/seed{S}/. Runs in the topogdn conda env.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # max_split_size worsens eval fragmentation

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "competitors", "common"))

from baseline_v1v2_dualstage import HP, SW, set_seed, load_data  # noqa: E402
from src.model.dualstage_gdeltauq import DualSTAGE_GDeltaUQ  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

OMEGA_TRAIN_MAX = 8000
OMEGA_TRAIN_STRIDE = 8


# ---- aleatoric + omega helpers (verbatim from the CST-GL extractor) ----
class AleHead(nn.Module):
    def __init__(self, d, V, emb=16, h=64):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.mlp = nn.Sequential(nn.Linear(d + emb, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, phi):
        B, V, _ = phi.shape
        e = self.emb(torch.arange(V, device=phi.device)).unsqueeze(0).expand(B, -1, -1)
        return self.mlp(torch.cat([phi, e], -1)).squeeze(-1).clamp(-10, 10)


def train_ale(phi, y, mu, dev, epochs=8, bs=256, beta=0.5):
    head = AleHead(phi.shape[-1], phi.shape[1]).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    P, Y, M = (torch.Tensor(a).to(dev) for a in (phi, y, mu))
    N = P.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            lv = head(P[idx]); s2 = lv.exp().clamp_min(1e-6)
            per = ((Y[idx] - M[idx]) ** 2) / (2 * s2) + 0.5 * lv
            loss = (per * (s2.detach() ** beta)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def fit_maha(phi_tr, eps=1e-3):
    V, d = phi_tr.shape[1], phi_tr.shape[2]
    mean = phi_tr.mean(0); inv = np.empty((V, d, d)); eye = np.eye(d)
    for v in range(V):
        inv[v] = np.linalg.inv(np.cov(phi_tr[:, v, :].T) + eps * eye)
    return mean, inv


def score_maha(phi, mean, inv):
    cen = phi - mean[None]
    tmp = np.einsum("tvi,vij->tvj", cen, inv)
    return np.sqrt(np.maximum(np.einsum("tvj,tvj->tv", tmp, cen), 0.0))


def auroc(s, l):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(l, s))
    except Exception:
        return float("nan")


def collect_pool(model, loader, K, dev, seed=0):
    pool = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi % 20 == 0:
                torch.cuda.empty_cache()   # cap allocator fragmentation balloon
            batch = batch.to(dev)
            h_pre, _, _, _ = model.forward_split(batch)          # (B,N,D)
            for b in range(h_pre.shape[0]):
                pool.append(h_pre[b].detach().cpu())             # (N,D)
    if len(pool) < K:
        raise ValueError(f"val slice has {len(pool)} windows < K={K}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=K, replace=False)
    return torch.stack([pool[int(i)] for i in idx], 0).to(dev)   # (K,N,D)


def run_kanchor(model, loader, pool, dev, mean, std):
    """K-anchor inference. Returns mu_bar (T,V) raw, U_par (T,V), h_bar (T,V,d),
    gt (T,V) raw."""
    K = pool.shape[0]
    MU, UPAR, HBAR, GT = [], [], [], []
    m_t = torch.from_numpy(np.asarray(mean)).to(dev)
    s_t = torch.from_numpy(np.asarray(std)).to(dev)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi % 20 == 0:
                torch.cuda.empty_cache()   # cap allocator fragmentation balloon
            batch = batch.to(dev)
            h_pre, _, _, _ = model.forward_split(batch)          # (B,N,D)
            B, N, _ = h_pre.shape
            mu_sum = mu_sq = h_sum = None
            for k in range(K):
                anchor = pool[k].unsqueeze(0).expand(B, -1, -1)
                recon_k, h_k = model.forward_anchored(h_pre, anchor)
                mu2 = recon_k.view(B, N, SW)[:, :, -1]            # (B,V) z-scale
                if mu_sum is None:
                    mu_sum = torch.zeros_like(mu2); mu_sq = torch.zeros_like(mu2)
                    h_sum = torch.zeros_like(h_k)
                mu_sum += mu2; mu_sq += mu2 * mu2; h_sum += h_k
            mu_bar = mu_sum / K
            U_par = ((mu_sq / K - mu_bar * mu_bar) * (K / (K - 1))).clamp_min(0.0)
            # de-normalize mu/gt to raw scale; U_par scales by std^2
            mu_raw = mu_bar * s_t + m_t
            gt_raw = batch.x.view(B, N, SW)[:, :, -1] * s_t + m_t
            MU.append(mu_raw.cpu().numpy())
            UPAR.append((U_par * s_t * s_t).cpu().numpy())
            HBAR.append((h_sum / K).cpu().numpy())
            GT.append(gt_raw.cpu().numpy())
    return (np.concatenate(MU, 0), np.concatenate(UPAR, 0),
            np.concatenate(HBAR, 0), np.concatenate(GT, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K_anchors", type=int, default=100)
    ap.add_argument("--anchor_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--uq-root", default=os.path.join(ROOT, "results/uq_v1v2/dualstage"))
    ap.add_argument("--base-arrays", default="")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.set_num_threads(4)
    set_seed(args.seed)

    OUT = os.path.join(args.uq_root, args.variant, f"seed{args.seed}")
    ckpt = os.path.join(OUT, "best.pt")
    if not os.path.exists(ckpt):
        print(f"[ds-uq {args.variant} s{args.seed}] missing {ckpt}", flush=True); sys.exit(11)
    base = args.base_arrays or os.path.join(
        ROOT, f"results/baseline_v1v2/dualstage/{args.variant}/seed{args.seed}/arrays.npz")
    if not os.path.exists(base):
        print(f"[ds-uq {args.variant} s{args.seed}] missing baseline arrays: {base}",
              flush=True); sys.exit(12)
    ref = np.load(base)
    lab = ref["test_attack_label"].astype(np.int8)
    T = lab.shape[0]

    full_train, test_ds, tr_idx, va_idx, te_idx, stats = load_data(args.variant, args.smoke)
    va_loader = DataLoader(full_train[va_idx], batch_size=args.batch, shuffle=False, num_workers=2)
    te_loader = DataLoader(test_ds[te_idx], batch_size=args.batch, shuffle=False, num_workers=2)
    om_idx = tr_idx[::OMEGA_TRAIN_STRIDE][:OMEGA_TRAIN_MAX]
    om_loader = DataLoader(full_train[om_idx], batch_size=args.batch, shuffle=False, num_workers=2)

    model = DualSTAGE_GDeltaUQ(feat_input_node=1, feat_target_node=1,
                               feat_input_edge=1, **HP).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[ds-uq {args.variant} s{args.seed}] loaded; T(test)={T} K={args.K_anchors}",
          flush=True)

    pool = collect_pool(model, va_loader, args.K_anchors, dev, seed=args.anchor_seed)
    print(f"  anchor pool {tuple(pool.shape)}", flush=True)

    v_mu, v_up, v_hbar, v_gt = run_kanchor(model, va_loader, pool, dev, *stats)
    head = train_ale(v_hbar, v_gt, v_mu, dev)

    t_mu, t_up, t_hbar, t_gt = run_kanchor(model, te_loader, pool, dev, *stats)
    if t_mu.shape[0] < T:
        print(f"  WARNING: test windows {t_mu.shape[0]} < label T {T}", flush=True)
        T = t_mu.shape[0]; lab = lab[:T]
    t_mu, t_up, t_hbar, t_gt = t_mu[:T], t_up[:T], t_hbar[:T], t_gt[:T]
    with torch.no_grad():
        sig2 = head(torch.Tensor(t_hbar).to(dev)).exp().cpu().numpy()

    o_mu, o_up, o_hbar, o_gt = run_kanchor(model, om_loader, pool, dev, *stats)
    mean, inv = fit_maha(o_hbar)
    omega_pn = score_maha(t_hbar, mean, inv)
    omega_mean = omega_pn.mean(1)

    out = dict(
        test_mu_bar=t_mu.astype(np.float32),
        test_ground_truth=t_gt.astype(np.float32),
        test_attack_label=lab,
        test_U_par=t_up.astype(np.float32),
        test_U_dist=t_up.mean(1).astype(np.float32),    # placeholder (promoted in fusion)
        test_sigma2_ale=sig2.astype(np.float32),
        test_U_dist_maha_mean=omega_mean.astype(np.float32),
        test_U_dist_maha_pernode=omega_pn.astype(np.float32),
        val_mu_bar=v_mu.astype(np.float32),
        val_ground_truth=v_gt.astype(np.float32),
    )
    outp = os.path.join(OUT, "arrays_full.npz")
    np.savez_compressed(outp, **out)
    torch.save(pool.cpu(), os.path.join(OUT, "anchor_pool.pt"))
    torch.save(head.state_dict(), os.path.join(OUT, "aleatoric_head.pt"))

    a_om = auroc(omega_mean, lab.astype(int))
    a_epi = auroc(t_up.mean(1), lab.astype(int))
    print(f"[ds-uq {args.variant} s{args.seed}] 3-channel Omega AUROC={a_om:.4f} "
          f"epi={a_epi:.4f} U_par_real={t_up.std()>1e-12} sig2_real={sig2.std()>1e-9} "
          f"-> {outp}", flush=True)


if __name__ == "__main__":
    main()
