#!/usr/bin/env python3
"""Phase 3 -- THE FUSION STUDY: which way of fusing forecast residual with the
uncertainty channels is best, across all four GNN backbones?

Compares, on every backbone x 5 seeds, primary metric PA%K-AUC (threshold-robust)
plus best-F1 and SEED STABILITY (std + worst-seed), the following fusion methods:

  M0   residual baseline (per-sensor smoothed err, top-1 aggregate)
  L1   standardized residual:   z = |y-mu| / sqrt(sig_ale + U_par), top-k aggregate
  L2   Gaussian NLPD (strictly proper): 0.5 log(2pi s2) + 0.5 r^2/s2, sum over V
  L3   predictive Mahalanobis (diag): r^2 / s2, sum top-2
  V1   variance-weighted residual: r * (1 + lambda * z_Upar), top-1   (lambda=0.5)
  S1   logistic stacker on standardized named channels  [interpretable weights]
  S2   GBM stacker (the incumbent M10; the unstable black box)

Selection criterion: maximise mean PA%K-AUC subject to LOW seed variance and
interpretability. The study is designed to show the GBM (S2) is unstable on some
backbones while an interpretable calibrated score is strong AND stable.

Reuses competitors/common/eval machinery via setup_context (per-backbone arrays).
All thresholds: PA%K-AUC is threshold-free; F1 uses the vectorized oracle sweep
(labelled oracle) for a consistent ceiling comparison across methods.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))

import argparse
from fusion_sweep_K100_full import setup_context, POST_W, POST_G
from compute_M10_PAK import fit_M10_score
from pa_k_metric import f1_pa_k_auc
from fusion_likelihood import fast_oracle_eval, _postproc_fast
from sklearn.linear_model import LogisticRegression

SEEDS = [1, 2, 3, 42, 100]
SPLIT = os.path.join(ROOT, "pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json")
BUNDLE = os.path.join(ROOT, "pretrained/swat_ensemble/calibration_bundle")
EPS = 1e-6

# arrays path per (backbone, seed): use the UQ arrays (mc/m10) so channels exist
def arr_path(bb, s):
    if bb == "gdn":
        sub = "ref_seed42" if s == 42 else f"seed{s}"
        return os.path.join(ROOT, f"results/gdn/{sub}/arrays.npz")
    suff = "mc" if bb == "cstgl" else "m10"
    return os.path.join(ROOT, f"results/competitors/{bb}/seed{s}_{suff}_arrays.npz")


def topk_sum(tv, k):
    return np.sort(tv, axis=1)[:, -k:].sum(axis=1)

def topk_mean(tv, k):
    return np.sort(tv, axis=1)[:, -k:].mean(axis=1) if k > 1 else tv.max(axis=1)


def build_scores(ctx):
    """Return {method: score(T,)} from a backbone's ctx (channels already loaded)."""
    mu = np.asarray(ctx["test_mu"], np.float64)
    gt = np.asarray(ctx["test_gt"], np.float64)
    r = np.abs(gt - mu)                                   # (T,V)
    sa = np.asarray(ctx["test_sigma2_ale"], np.float64)   # ones for competitors
    up = np.asarray(ctx["test_U_par"], np.float64)
    s2 = sa + up + EPS                                    # predictive variance
    agg = np.asarray(ctx["agg"], np.float64)              # M0 baseline score
    z_up = np.asarray(ctx["z_U_par_TxV"], np.float64) if "z_U_par_TxV" in ctx else (up - up.mean(0)) / (up.std(0) + EPS)
    out = {}
    out["M0_residual"] = agg
    out["L1_stdres"] = topk_mean(r / np.sqrt(s2), 2)
    nlpd = 0.5 * np.log(2 * np.pi * s2) + 0.5 * r ** 2 / s2
    out["L2_NLPD"] = nlpd.sum(axis=1)
    out["L3_Maha"] = topk_sum(r ** 2 / s2, 2)
    out["V1_varweighted"] = topk_mean(r * (1.0 + 0.5 * z_up), 1)
    return out


def logistic_stacker(ctx):
    """Interpretable logistic regression on standardized named aggregate features,
    trained on the val slice (same data the GBM uses). Returns score + weights."""
    mu = np.asarray(ctx["test_mu"], np.float64); gt = np.asarray(ctx["test_gt"], np.float64)
    r = np.abs(gt - mu); sa = np.asarray(ctx["test_sigma2_ale"], np.float64)
    up = np.asarray(ctx["test_U_par"], np.float64); ud = np.asarray(ctx["test_U_dist"], np.float64)
    feats = np.column_stack([
        np.asarray(ctx["agg"], np.float64),               # residual aggregate
        (r / np.sqrt(sa + up + EPS)).max(1),              # standardized residual
        up.mean(1),                                       # epistemic
        sa.mean(1),                                       # aleatoric
        ud,                                               # distributional
    ])
    names = ["resid", "stdres", "epistemic", "aleatoric", "distributional"]
    # standardize features on the C-slice nominal
    c = np.asarray(ctx["c_idx"]); mu_f = feats[c].mean(0); sd_f = feats[c].std(0) + EPS
    Z = (feats - mu_f) / sd_f
    vi = np.asarray(ctx["val_idx"]); y = np.asarray(ctx["label"])[vi]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Z[vi], y)
    score = clf.decision_function(Z)
    return score, dict(zip(names, [round(float(w), 3) for w in clf.coef_[0]]))


def eval_method(score, label, K_grid=None):
    if K_grid is None:
        K_grid = np.linspace(0, 100, 11)
    pak = f1_pa_k_auc(score, label, K_grid=K_grid, n_thresholds=200)
    orc = fast_oracle_eval(score, label, 400)
    return float(pak["PA_K_AUC"]), float(orc["F1"])


def main():
    rows = []
    logistic_weights = {}
    for bb in ["gdn", "cstgl", "gta", "topogdn"]:
        for s in SEEDS:
            ap = arr_path(bb, s)
            if not os.path.exists(ap):
                print(f"[skip] {bb} seed{s} missing {ap}", flush=True); continue
            ctx = setup_context(argparse.Namespace(arrays=ap, split=SPLIT, bundle=BUNDLE,
                                                    slide_win=60, seed=s))
            label = np.asarray(ctx["label"])
            scores = build_scores(ctx)
            # S2 GBM
            try:
                gbm, _ = fit_M10_score(ctx); scores["S2_GBM"] = gbm
            except Exception as e:
                print(f"  GBM fail {bb} s{s}: {e}", flush=True)
            # S1 logistic
            try:
                lg, w = logistic_stacker(ctx); scores["S1_logistic"] = lg
                logistic_weights[f"{bb}_s{s}"] = w
            except Exception as e:
                print(f"  logistic fail {bb} s{s}: {e}", flush=True)
            for m, sc in scores.items():
                pak, f1 = eval_method(sc, label)
                rows.append(dict(backbone=bb, seed=s, method=m, PA_K_AUC=round(pak, 4), F1=round(f1, 4)))
                print(f"[{bb} s{s} {m}] PA%K={pak:.4f} F1={f1:.4f}", flush=True)

    outdir = os.path.join(ROOT, "results/paper/fusion_study"); os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "fusion_study.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "seed", "method", "PA_K_AUC", "F1"]); w.writeheader()
        for r in rows: w.writerow(r)
    json.dump(logistic_weights, open(os.path.join(outdir, "logistic_weights.json"), "w"), indent=2)

    # ---- summary: per (backbone,method) mean+-std + worst seed ----
    methods = ["M0_residual", "L1_stdres", "L2_NLPD", "L3_Maha", "V1_varweighted", "S1_logistic", "S2_GBM"]
    print("\n" + "=" * 78)
    print("PA%K-AUC by backbone x method (mean +- std [worst-seed])")
    print("=" * 78)
    summ = {}
    for bb in ["gdn", "cstgl", "gta", "topogdn"]:
        line = f"{bb:8}"
        for m in methods:
            v = [r["PA_K_AUC"] for r in rows if r["backbone"] == bb and r["method"] == m]
            if v:
                summ[(bb, m)] = (np.mean(v), np.std(v), min(v))
                line += f"  {m[:4]}:{np.mean(v):.3f}+-{np.std(v):.3f}[{min(v):.3f}]"
        print(line)
    # stability ranking: which method has best mean and lowest worst-case drop
    print("\nSTABILITY (mean PA%K over all backbones&seeds, std, min):")
    for m in methods:
        v = [r["PA_K_AUC"] for r in rows if r["method"] == m]
        if v:
            print(f"  {m:16} mean={np.mean(v):.4f}  std={np.std(v):.4f}  min={min(v):.4f}")
    json.dump({f"{bb}|{m}": summ[(bb, m)] for (bb, m) in summ}, open(os.path.join(outdir, "summary.json"), "w"),
              indent=2, default=float)
    print(f"\nwrote {outdir}/fusion_study.csv + summary.json + logistic_weights.json")


if __name__ == "__main__":
    main()
