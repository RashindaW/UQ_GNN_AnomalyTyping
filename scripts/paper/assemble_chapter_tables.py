#!/usr/bin/env python3
"""Assemble the final chapter tables from all verified results.

Produces results/paper/chapter/:
  T1_detection.csv       -- per backbone: M0 vs best-UQ-fusion (S1 logistic), 5-seed
                            mean+-std PA%K + F1 + Wilcoxon p
  T2_uq_quality.csv      -- per backbone: epistemic attack-AUROC, real-Omega AUROC vs
                            placeholder, distinctness rho, aleatoric-real
  T3_fusion_methods.csv  -- the fusion-study method comparison (mean PA%K, std, worst)
  chapter_numbers.json   -- all headline numbers for the writeup
Pure analysis on cached arrays; no GPU.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
import argparse
from fusion_sweep_K100_full import setup_context
from pa_k_metric import f1_pa_k_auc
from fusion_likelihood import fast_oracle_eval
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, spearmanr

SEEDS = [1, 2, 3, 42, 100]
SPLIT = os.path.join(ROOT, "pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json")
BUNDLE = os.path.join(ROOT, "pretrained/swat_ensemble/calibration_bundle")
EPS = 1e-6
OUT = os.path.join(ROOT, "results/paper/chapter"); os.makedirs(OUT, exist_ok=True)


def arr_path(bb, s, full=False):
    if bb == "gdn":
        sub = "ref_seed42" if s == 42 else f"seed{s}"
        base = os.path.join(ROOT, f"results/gdn/{sub}/arrays.npz")
        omg = os.path.join(ROOT, f"results/gdn/{sub}/arrays_omega.npz")
        return omg if (full and os.path.exists(omg)) else base
    suff = "mc" if bb == "cstgl" else "m10"
    base = os.path.join(ROOT, f"results/competitors/{bb}/seed{s}_{suff}_arrays.npz")
    fp = os.path.join(ROOT, f"results/competitors/{bb}/seed{s}_full_arrays.npz")
    return fp if (full and os.path.exists(fp)) else base


def s1_logistic_score(ctx):
    mu = np.asarray(ctx["test_mu"], float); gt = np.asarray(ctx["test_gt"], float)
    r = np.abs(gt - mu); sa = np.asarray(ctx["test_sigma2_ale"], float)
    up = np.asarray(ctx["test_U_par"], float); ud = np.asarray(ctx["test_U_dist"], float)
    feats = np.column_stack([np.asarray(ctx["agg"], float), (r / np.sqrt(sa + up + EPS)).max(1),
                             up.mean(1), sa.mean(1), ud])
    c = np.asarray(ctx["c_idx"]); mf = feats[c].mean(0); sf = feats[c].std(0) + EPS
    Z = (feats - mf) / sf
    vi = np.asarray(ctx["val_idx"]); y = np.asarray(ctx["label"])[vi]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Z[vi], y)
    return clf.decision_function(Z)


def main():
    Kg = np.linspace(0, 100, 11)
    det = {}   # bb -> {m0:[...], s1:[...], m0f:[...], s1f:[...]}
    uq = {}    # bb -> per-seed channel quality
    for bb in ["gdn", "cstgl", "gta", "topogdn"]:
        det[bb] = dict(m0=[], s1=[], m0f=[], s1f=[]); uq[bb] = dict(epi=[], om=[], pl=[], rho=[], ale_real=[])
        for s in SEEDS:
            ap = arr_path(bb, s, full=True)
            if not os.path.exists(ap):
                continue
            ctx = setup_context(argparse.Namespace(arrays=ap, split=SPLIT, bundle=BUNDLE, slide_win=60, seed=s))
            lab = np.asarray(ctx["label"])
            agg = np.asarray(ctx["agg"], float)
            s1 = s1_logistic_score(ctx)
            det[bb]["m0"].append(f1_pa_k_auc(agg, lab, K_grid=Kg, n_thresholds=200)["PA_K_AUC"])
            det[bb]["s1"].append(f1_pa_k_auc(s1, lab, K_grid=Kg, n_thresholds=200)["PA_K_AUC"])
            det[bb]["m0f"].append(fast_oracle_eval(agg, lab, 400)["F1"])
            det[bb]["s1f"].append(fast_oracle_eval(s1, lab, 400)["F1"])
            # uq quality (from full arrays directly)
            d = np.load(ap)
            up = d["test_U_par"].mean(1) if "test_U_par" in d.files else None
            if up is not None:
                uq[bb]["epi"].append(roc_auc_score(lab, up))
            for k in ["test_U_dist_maha_mean"]:
                if k in d.files:
                    uq[bb]["om"].append(roc_auc_score(lab, d[k]))
                    uq[bb]["rho"].append(float(spearmanr(d[k], up).correlation) if up is not None else float("nan"))
            if "test_U_dist" in d.files:
                uq[bb]["pl"].append(roc_auc_score(lab, d["test_U_dist"]))
            if "test_sigma2_ale" in d.files:
                uq[bb]["ale_real"].append(bool(d["test_sigma2_ale"].std() > 1e-9 and not np.allclose(d["test_sigma2_ale"], 1)))
            print(f"[{bb} s{s}] M0 PA%K={det[bb]['m0'][-1]:.4f} S1 PA%K={det[bb]['s1'][-1]:.4f}", flush=True)

    # ---- T1 detection ----
    def ms(v): return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))
    rows1 = []
    for bb in ["gdn", "cstgl", "gta", "topogdn"]:
        m0m, m0s = ms(det[bb]["m0"]); s1m, s1s = ms(det[bb]["s1"])
        f0m, f0s = ms(det[bb]["m0f"]); f1m, f1s = ms(det[bb]["s1f"])
        try:
            wp = float(wilcoxon(det[bb]["s1"], det[bb]["m0"]).pvalue) if len(det[bb]["m0"]) >= 2 else float("nan")
        except Exception:
            wp = float("nan")
        rows1.append(dict(backbone=bb, n=len(det[bb]["m0"]),
                          M0_PAK=f"{m0m:.4f}+-{m0s:.4f}", S1_PAK=f"{s1m:.4f}+-{s1s:.4f}",
                          gain_PAK=f"{s1m-m0m:+.4f}", M0_F1=f"{f0m:.4f}", S1_F1=f"{f1m:.4f}",
                          gain_F1=f"{f1m-f0m:+.4f}", wilcoxon_p=f"{wp:.3f}"))
    with open(os.path.join(OUT, "T1_detection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows1[0].keys())); w.writeheader(); [w.writerow(r) for r in rows1]

    # ---- T2 uq quality ----
    rows2 = []
    for bb in ["gdn", "cstgl", "gta", "topogdn"]:
        em, _ = ms(uq[bb]["epi"]); om, _ = ms(uq[bb]["om"]); pl, _ = ms(uq[bb]["pl"]); rh, _ = ms(uq[bb]["rho"])
        rows2.append(dict(backbone=bb, epistemic_attackAUROC=f"{em:.4f}",
                          omega_AUROC=f"{om:.4f}" if uq[bb]["om"] else "n/a",
                          placeholder_AUROC=f"{pl:.4f}",
                          omega_distinct_rho=f"{rh:.3f}" if uq[bb]["om"] else "n/a",
                          aleatoric_real=str(all(uq[bb]["ale_real"]) if uq[bb]["ale_real"] else False)))
    with open(os.path.join(OUT, "T2_uq_quality.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows2[0].keys())); w.writeheader(); [w.writerow(r) for r in rows2]

    # ---- T3 fusion methods (from the study) ----
    fs = list(csv.DictReader(open(os.path.join(ROOT, "results/paper/fusion_study/fusion_study.csv"))))
    methods = ["M0_residual", "L1_stdres", "L2_NLPD", "L3_Maha", "V1_varweighted", "S1_logistic", "S2_GBM"]
    rows3 = []
    for m in methods:
        v = [float(r["PA_K_AUC"]) for r in fs if r["method"] == m]
        rows3.append(dict(method=m, mean_PAK=f"{np.mean(v):.4f}", std=f"{np.std(v):.4f}", worst_seed=f"{min(v):.4f}"))
    with open(os.path.join(OUT, "T3_fusion_methods.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows3[0].keys())); w.writeheader(); [w.writerow(r) for r in rows3]

    json.dump(dict(detection=rows1, uq_quality=rows2, fusion_methods=rows3),
              open(os.path.join(OUT, "chapter_numbers.json"), "w"), indent=2)

    print("\n=== T1 DETECTION (M0 vs S1 interpretable UQ fusion) ===")
    for r in rows1:
        print(f"  {r['backbone']:8} M0 PA%K {r['M0_PAK']}  S1 PA%K {r['S1_PAK']}  gain {r['gain_PAK']}  (F1 {r['M0_F1']}->{r['S1_F1']})")
    print("\n=== T2 UQ QUALITY (real Omega vs placeholder) ===")
    for r in rows2:
        print(f"  {r['backbone']:8} epi-AUROC {r['epistemic_attackAUROC']}  Omega {r['omega_AUROC']} (pl {r['placeholder_AUROC']})  rho {r['omega_distinct_rho']}  ale_real {r['aleatoric_real']}")
    print(f"\nwrote {OUT}/T1_detection.csv T2_uq_quality.csv T3_fusion_methods.csv chapter_numbers.json")


if __name__ == "__main__":
    main()
