#!/usr/bin/env python3
"""Calibrate + extract full UQ for ONE anchored CSTGL_GDeltaUQ checkpoint.

CST-GL has no attention -> 3 channels (epistemic + aleatoric + distributional), no
structural. Loads results/uq_v1v2/cstgl/<V>/seed<S>/best.pt and writes arrays_full.npz:

  test_mu_bar, test_ground_truth, test_attack_label,
  test_U_par (epistemic, T,V), test_U_dist (placeholder = U_par mean),
  test_sigma2_ale (aleatoric, T,V),
  test_U_dist_maha_mean / _pernode (Mahalanobis Omega),
  val_mu_bar, val_ground_truth.

Pipeline: anchor pool (K=100 forward_split reps from val) -> K-anchor inference
(mu_bar=mean_K, U_par=var_K, h_bar=mean_K penultimate) -> aleatoric Gaussian-NLL on
val -> Mahalanobis Omega fit on a train subsample, scored on test. Attack labels are
reused from the baseline arrays.npz (same test set). Runs in the cstgl conda env.

  python v1v2_cstgl_gdeltauq_fulluq.py --variant V1 --seed 0 --device cuda:0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CST = os.path.join(ROOT, "competitors", "CST-GL")
sys.path.insert(0, CST)
os.chdir(CST)

from util import load_dataset                # noqa: E402
from stgnn_gdeltauq import CSTGL_GDeltaUQ    # noqa: E402

OMEGA_TRAIN_MAX = 8000   # cap windows used for the Mahalanobis fit


# ---- aleatoric + omega helpers (cstgl env; same form as build_full_topogdn) ----
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
        for x, y in loader.get_iterator():
            inp = torch.Tensor(x).to(dev).transpose(1, 3)        # (B,in_dim,V,seq)
            h_pre, _ = model.forward_split(inp)                  # (B,C,V,1)
            for b in range(h_pre.shape[0]):
                pool.append(h_pre[b].detach().cpu())             # (C,V,1)
    if len(pool) < K:
        raise ValueError(f"val slice has {len(pool)} windows < K={K}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=K, replace=False)
    return torch.stack([pool[int(i)] for i in idx], 0).to(dev)   # (K,C,V,1)


def run_kanchor(model, loader, pool, dev, max_windows=None):
    """K-anchor inference. Returns mu_bar (T,V), U_par (T,V), h_bar (T,V,d), gt (T,V)."""
    K = pool.shape[0]
    MU, UPAR, HBAR, GT = [], [], [], []
    seen = 0
    with torch.no_grad():
        for x, y in loader.get_iterator():
            inp = torch.Tensor(x).to(dev).transpose(1, 3)        # (B,in_dim,V,seq)
            B = inp.shape[0]
            h_pre, _ = model.forward_split(inp)                  # (B,C,V,1)
            mu_sum = mu_sq = h_sum = None
            for k in range(K):
                mu_k, h_k = model.forward_anchored(h_pre, pool[k])
                mu2 = mu_k[:, 0, :, 0]                            # (B,V)
                h2 = h_k[:, :, :, 0].permute(0, 2, 1)            # (B,V,end_ch)
                if mu_sum is None:
                    mu_sum = torch.zeros_like(mu2); mu_sq = torch.zeros_like(mu2)
                    h_sum = torch.zeros_like(h2)
                mu_sum += mu2; mu_sq += mu2 * mu2; h_sum += h2
            mu_bar = mu_sum / K
            U_par = ((mu_sq / K - mu_bar * mu_bar) * (K / (K - 1))).clamp_min(0.0)
            MU.append(mu_bar.cpu().numpy()); UPAR.append(U_par.cpu().numpy())
            HBAR.append((h_sum / K).cpu().numpy())
            gt = torch.Tensor(y).transpose(1, 3)[:, 0, :, :].squeeze(-1).numpy()  # (B,V)
            GT.append(gt)
            seen += B
            if max_windows is not None and seen >= max_windows:
                break
    return (np.concatenate(MU, 0), np.concatenate(UPAR, 0),
            np.concatenate(HBAR, 0), np.concatenate(GT, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K_anchors", type=int, default=100)
    ap.add_argument("--anchor_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="swat",
                    help="swat -> results/{uq,baseline}_v1v2/cstgl; "
                         "wadi -> results/uq_wadi_v2 + results/baseline_wadi_v2")
    args = ap.parse_args()
    dev = torch.device(args.device); torch.set_num_threads(4)

    uq_root = "results/uq_v1v2/cstgl" if args.dataset == "swat" else f"results/uq_{args.dataset}_v2/cstgl"
    base_root = "results/baseline_v1v2/cstgl" if args.dataset == "swat" else f"results/baseline_{args.dataset}_v2/cstgl"
    OUT = os.path.join(ROOT, f"{uq_root}/{args.variant}/seed{args.seed}")
    ckpt = os.path.join(OUT, "best.pt")
    if not os.path.exists(ckpt):
        print(f"[cstgl-uq {args.variant} s{args.seed}] missing {ckpt}", flush=True); sys.exit(11)
    with open(os.path.join(OUT, "hyperparameters.json")) as f:
        hp = json.load(f)
    base = os.path.join(ROOT, f"{base_root}/{args.variant}/seed{args.seed}/arrays.npz")
    if not os.path.exists(base):
        print(f"[cstgl-uq {args.variant} s{args.seed}] missing baseline arrays for labels: {base}", flush=True); sys.exit(12)
    ref = np.load(base); lab = ref["test_attack_label"].astype(np.int8); T = lab.shape[0]

    dl = load_dataset(os.path.join(CST, hp["data_dir"]), hp["batch"], hp["batch"], hp["batch"], scaling_required=False)

    model = CSTGL_GDeltaUQ(True, True, hp["gcn_depth"], hp["num_nodes"], dev,
                           predefined_A=None, dropout=hp["dropout"],
                           subgraph_size=hp["subgraph_size"], node_dim=hp["node_dim"],
                           dilation_exponential=hp["dilation_exponential"],
                           conv_channels=hp["conv_channels"], residual_channels=hp["residual_channels"],
                           skip_channels=hp["skip_channels"], end_channels=hp["end_channels"],
                           seq_length=hp["seq_in_len"], in_dim=hp["in_dim"], out_dim=hp["seq_out_len"],
                           layers=hp["layers"], propalpha=hp["propalpha"], tanhalpha=hp["tanhalpha"],
                           layer_norm_affline=True).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[cstgl-uq {args.variant} s{args.seed}] loaded; T(test)={T} K={args.K_anchors}", flush=True)

    pool = collect_pool(model, dl["val_loader"], args.K_anchors, dev, seed=args.anchor_seed)
    print(f"  anchor pool {tuple(pool.shape)}", flush=True)

    v_mu, v_up, v_hbar, v_gt = run_kanchor(model, dl["val_loader"], pool, dev)
    head = train_ale(v_hbar, v_gt, v_mu, dev)

    t_mu, t_up, t_hbar, t_gt = run_kanchor(model, dl["test_loader"], pool, dev)
    t_mu, t_up, t_hbar, t_gt = t_mu[:T], t_up[:T], t_hbar[:T], t_gt[:T]
    with torch.no_grad():
        sig2 = head(torch.Tensor(t_hbar).to(dev)).exp().cpu().numpy()

    o_mu, o_up, o_hbar, o_gt = run_kanchor(model, dl["train_loader"], pool, dev, max_windows=OMEGA_TRAIN_MAX)
    mean, inv = fit_maha(o_hbar)
    omega_pn = score_maha(t_hbar, mean, inv); omega_mean = omega_pn.mean(1)

    out = dict(
        test_mu_bar=t_mu.astype(np.float32),
        test_ground_truth=t_gt.astype(np.float32),
        test_attack_label=lab,
        test_U_par=t_up.astype(np.float32),
        test_U_dist=t_up.mean(1).astype(np.float32),              # placeholder (promoted in fusion)
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

    a_om = auroc(omega_mean, lab.astype(int)); a_epi = auroc(t_up.mean(1), lab.astype(int))
    print(f"[cstgl-uq {args.variant} s{args.seed}] 3-channel (no structural) Omega AUROC={a_om:.4f} "
          f"epi={a_epi:.4f} sig2_real={sig2.std()>1e-9} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
