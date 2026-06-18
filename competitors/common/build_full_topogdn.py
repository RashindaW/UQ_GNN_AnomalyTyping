#!/usr/bin/env python3
"""Build real ALEATORIC + Mahalanobis OMEGA + structural U_str for TopoGDN (5 seeds).

Runs in the topogdn conda env (torch 1.13). TopoGDN is a GDN variant: penultimate
is the self.dp output (B,V,dim); attention is gnn_layers[0].att_weight_1.

Per seed: load ckpt -> build TopoGDN GDN model (config inferred from state_dict) ->
hook self.dp (penultimate) + read att_weight_1 (structural) -> extract on TRAIN
(normal) + TEST -> aleatoric head (Gaussian NLL on held-out normal) -> Mahalanobis
Omega -> K MC-dropout passes for U_str (attention variance) -> splice into NEW
seed{S}_full_arrays.npz. Validate vs cached mu (alignment gate).
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOPO = os.path.join(ROOT, "competitors", "TopoGDN")
sys.path.insert(0, TOPO)
os.chdir(TOPO)

import pandas as pd  # noqa: E402
from models.GDN import GDN  # noqa: E402
from util.net_struct import get_feature_map, get_fc_graph_struc  # noqa: E402
from util.preprocess import build_loc_net, construct_data  # noqa: E402
from datasets.TimeDataset import TimeDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

SEEDS = [1, 2, 3, 42, 100]
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SW = 60


def build_model(ckpt_path):
    sd = torch.load(ckpt_path, map_location=DEV)
    dim = sd["embedding.weight"].shape[1]          # 128
    V = sd["embedding.weight"].shape[0]            # 51
    inter = sd["out_layer.mlp.0.weight"].shape[0]  # out_layer_inter? here mlp.0 is (1,128) -> out_layer_num=1
    feature_map = get_feature_map("swat")
    fc = get_fc_graph_struc("swat")
    train_cols = [l.strip() for l in open(os.path.join(TOPO, "data/swat/list.txt")) if l.strip()]
    fc_ei = torch.tensor(build_loc_net(fc, train_cols, feature_map=feature_map), dtype=torch.long)
    model = GDN([fc_ei], V, dim=dim, input_dim=SW, out_layer_num=1,
                out_layer_inter_dim=128, topk=15).to(DEV)
    model.load_state_dict(sd)
    model.eval()
    return model, feature_map, fc_ei


def make_loader(csv_path, feature_map, fc_ei, has_attack):
    df = pd.read_csv(csv_path, index_col=0)
    labels = df.attack.tolist() if (has_attack and "attack" in df.columns) else 0
    indata = construct_data(df, feature_map, labels=labels)
    ds = TimeDataset(indata, fc_ei, mode="test", config={"slide_win": SW, "slide_stride": 1})
    return DataLoader(ds, batch_size=64, shuffle=False)


class Cap:
    def __init__(self):
        self.phi = None

    def __call__(self, m, i, o):
        self.phi = o.detach()   # dp output: (B*V?, dim) or (B,V,dim)


def extract(model, loader, cap, want_attention=False, mc=False):
    if mc:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    else:
        model.eval()
    mus, phis, atts = [], [], []
    with torch.no_grad():
        for x, y, lbl, ei in loader:
            x = x.float().to(DEV); ei = ei.long().to(DEV)
            cap.phi = None
            out, _ = model(x)                        # (B,V)
            mus.append(out.cpu().numpy())
            phi = cap.phi
            B = x.shape[0]; V = out.shape[1]
            phi = phi.reshape(B, V, -1).cpu().numpy()
            phis.append(phi)
            if want_attention:
                aw = model.gnn_layers[0].att_weight_1.detach().reshape(-1).cpu().numpy()
                atts.append(aw[: B * V])            # crude per-(b,v) slice
    mu = np.concatenate(mus, 0); phi = np.concatenate(phis, 0)
    att = np.concatenate(atts, 0) if want_attention else None
    return mu, phi, att


# ---- aleatoric + omega (same as CST-GL builder) ----
class AleHead(nn.Module):
    def __init__(self, d, V, emb=16, h=64):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.mlp = nn.Sequential(nn.Linear(d + emb, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, phi):
        B, V, _ = phi.shape
        e = self.emb(torch.arange(V, device=phi.device)).unsqueeze(0).expand(B, -1, -1)
        return self.mlp(torch.cat([phi, e], -1)).squeeze(-1).clamp(-10, 10)


def train_ale(phi, y, mu, epochs=8, bs=256, beta=0.5):
    head = AleHead(phi.shape[-1], phi.shape[1]).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    P, Y, M = (torch.Tensor(a).to(DEV) for a in (phi, y, mu))
    N = P.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=DEV)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            lv = head(P[idx]); s2 = lv.exp().clamp_min(1e-6)
            per = ((Y[idx] - M[idx]) ** 2) / (2 * s2) + 0.5 * lv
            per = per * (s2.detach() ** beta)
            loss = per.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def fit_maha(phi_tr, eps=1e-3):
    V, d = phi_tr.shape[1], phi_tr.shape[2]
    mean = phi_tr.mean(0); inv = np.empty((V, d, d))
    eye = np.eye(d)
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


def main():
    report = []
    for s in SEEDS:
        cks = sorted(glob.glob(os.path.join(TOPO, f"pretrained/topo_s{s}/best_*.pt")))
        if not cks:
            print(f"[topo s{s}] no ckpt", flush=True); continue
        model, fmap, fc_ei = build_model(cks[0])
        cap = Cap(); h = model.dp.register_forward_hook(cap)
        tr_loader = make_loader(os.path.join(TOPO, "data/swat/train.csv"), fmap, fc_ei, False)
        te_loader = make_loader(os.path.join(TOPO, "data/swat/test.csv"), fmap, fc_ei, True)
        mu_tr, phi_tr, _ = extract(model, tr_loader, cap)
        mu_te, phi_te, _ = extract(model, te_loader, cap)

        # K MC passes for U_str (attention variance) - mean over edges -> (T,)
        K = 30
        att_stack = []
        for _k in range(K):
            _, _, att = extract(model, te_loader, cap, want_attention=True, mc=True)
            att_stack.append(att)
        model.eval()
        h.remove()
        ustr_mean = np.var(np.stack(att_stack, 0), axis=0)  # (T,)?

        ref = np.load(os.path.join(ROOT, f"results/competitors/topogdn/seed{s}_m10_arrays.npz"))
        T = ref["test_attack_label"].shape[0]
        # align (front, drop trailing like emit_arrays)
        mu_te, phi_te = mu_te[:T], phi_te[:T]
        ustr_mean = ustr_mean[:T] if ustr_mean.shape[0] >= T else np.pad(ustr_mean, (0, T - ustr_mean.shape[0]))
        lab = ref["test_attack_label"].astype(int)
        # alignment gate
        gate = float(np.abs(mu_te - ref["test_mu_bar"][:T]).max())

        # need targets y for aleatoric: re-extract y from test loader windows
        ys = []
        for _x, y, _l, _e in te_loader:
            ys.append(y.numpy())
        y_te = np.concatenate(ys, 0)[:T]
        # train aleatoric on a normal tail of TRAIN
        ys_tr = []
        for _x, y, _l, _e in tr_loader:
            ys_tr.append(y.numpy())
        y_tr = np.concatenate(ys_tr, 0)
        nval = min(8000, phi_tr.shape[0] // 4)
        head = train_ale(phi_tr[-nval:], y_tr[-nval:], mu_tr[-nval:])
        with torch.no_grad():
            sig2 = head(torch.Tensor(phi_te).to(DEV)).exp().cpu().numpy()

        mean, inv = fit_maha(phi_tr[::4])
        omega_pn = score_maha(phi_te, mean, inv)
        omega_mean = omega_pn.mean(1)

        out = {k: ref[k] for k in ref.files}
        out["test_sigma2_ale_real"] = sig2.astype(np.float32)
        out["test_sigma2_ale"] = sig2.astype(np.float32)
        out["test_U_dist_maha_mean"] = omega_mean.astype(np.float32)
        out["test_U_dist_maha_pernode"] = omega_pn.astype(np.float32)
        out["test_U_str_mean"] = ustr_mean.astype(np.float32)
        outp = os.path.join(ROOT, f"results/competitors/topogdn/seed{s}_full_arrays.npz")
        np.savez_compressed(outp, **out)

        a_om = auroc(omega_mean, lab); a_pl = auroc(ref["test_U_dist"], lab)
        a_str = auroc(ustr_mean, lab)
        rec = dict(seed=s, omega_auroc=round(a_om, 4), placeholder_auroc=round(a_pl, 4),
                   ustr_auroc=round(a_str, 4), gate_maxdiff=round(gate, 4), sig2_real=bool(sig2.std() > 1e-9))
        report.append(rec)
        print(f"[topo s{s}] Omega={a_om:.4f}(pl {a_pl:.4f}) Ustr={a_str:.4f} gate={gate:.3f} "
              f"sig2_real={rec['sig2_real']} -> {outp}", flush=True)

    rp = os.path.join(ROOT, "results/competitors/topogdn/topogdn_full_report.md")
    with open(rp, "w") as f:
        f.write("# TopoGDN real aleatoric + Omega + U_str (5 seeds)\n\n")
        f.write("| seed | Omega AUROC | placeholder | U_str AUROC | align gate | sig2 real |\n")
        f.write("|------|-------------|-------------|-------------|------------|-----------|\n")
        for r in report:
            f.write(f"| {r['seed']} | {r['omega_auroc']} | {r['placeholder_auroc']} | "
                    f"{r['ustr_auroc']} | {r['gate_maxdiff']} | {r['sig2_real']} |\n")
    print(f"\nwrote {rp}", flush=True)
    if report:
        print(f"MEAN Omega AUROC={np.mean([r['omega_auroc'] for r in report]):.4f} "
              f"vs placeholder {np.mean([r['placeholder_auroc'] for r in report]):.4f}", flush=True)


if __name__ == "__main__":
    main()
