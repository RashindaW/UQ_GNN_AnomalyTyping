#!/usr/bin/env python3
"""Compute PTaPR-AUC (Kang et al. 2026) for the M0 residual baseline across all
backbones x 5 seeds, completing the metric panel.

Apples-to-apples with the rest of the panel: the M0 score is the canonical
per-sensor smoothed residual top-1 aggregate (build_full_err_scores +
topk_aggregate, smoothing=5), thresholded to a binary alarm with the SAME Fix-A
oracle best-F1 threshold + post-proc used elsewhere. PTaPR needs a binary
prediction (not a continuous score), so the alarm is the right input.

Output: results/paper/metrics/ptapr_5seed.csv  (backbone,seed,F1_oracle,
PTaPR_AUC,PTaPR_F1_theta0,PTaPR_F1_theta1).
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))

import ptapr_metric as P
from sweep_eval_gdeltauq import build_full_err_scores, topk_aggregate
from fusion_sweep_K100_full import POST_W, POST_G
from fusion_likelihood import fast_oracle_eval, _postproc_fast

SEEDS = [1, 2, 3, 42, 100]

# arrays.npz path per (backbone, seed). GDN lives on its native path; competitors
# use *_baseline_arrays.npz (the M0 residual-only arrays, no UQ needed for M0).
def arrays_path(backbone, seed):
    if backbone == "gdn":
        sub = "ref_seed42" if seed == 42 else f"seed{seed}"
        return os.path.join(ROOT, f"results/gdn/{sub}/arrays.npz")
    return os.path.join(ROOT,
        f"results/competitors/{backbone}/seed{seed}_baseline_arrays.npz")


def m0_alarm(arr_path):
    """Canonical M0 residual score -> Fix-A oracle best-F1 binary alarm."""
    d = np.load(arr_path)
    full = build_full_err_scores(
        d["test_mu_bar"].astype(np.float64), d["test_ground_truth"].astype(np.float64),
        d["val_mu_bar"].astype(np.float64), d["val_ground_truth"].astype(np.float64), 5)
    agg = topk_aggregate(full, 1).astype(np.float64)
    lab = d["test_attack_label"].astype(int)
    res = fast_oracle_eval(agg, lab, 400)
    alarm = _postproc_fast((agg >= res["tau"]).astype(np.int8), POST_W, POST_G)
    return lab, alarm.astype(int), res["F1"]


def main():
    out_rows = []
    for backbone in ["gdn", "cstgl", "gta", "topogdn"]:
        for seed in SEEDS:
            ap = arrays_path(backbone, seed)
            if not os.path.exists(ap):
                print(f"[skip] {backbone} seed{seed}: missing {ap}", flush=True)
                continue
            lab, alarm, f1 = m0_alarm(ap)
            pp = P.ptapr_auc(lab, alarm)
            row = {"backbone": backbone, "seed": seed,
                   "F1_oracle": round(f1, 4),
                   "PTaPR_AUC": round(float(pp["auc"]), 4),
                   "PTaPR_F1_theta0": round(float(pp["F1_0"]), 4),
                   "PTaPR_F1_theta1": round(float(pp["F1_1"]), 4)}
            out_rows.append(row)
            print(f"[{backbone} seed{seed}] F1={row['F1_oracle']:.4f} "
                  f"PTaPR-AUC={row['PTaPR_AUC']:.4f}", flush=True)

    out_csv = os.path.join(ROOT, "results/paper/metrics/ptapr_5seed.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "seed", "F1_oracle",
                                          "PTaPR_AUC", "PTaPR_F1_theta0", "PTaPR_F1_theta1"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {out_csv} ({len(out_rows)} rows)", flush=True)

    # per-backbone mean +- std
    print("\nPer-backbone PTaPR-AUC (mean +- std over seeds):")
    for b in ["gdn", "cstgl", "gta", "topogdn"]:
        vals = [r["PTaPR_AUC"] for r in out_rows if r["backbone"] == b]
        if vals:
            print(f"  {b:8s} {np.mean(vals):.4f} +- {np.std(vals):.4f}  (n={len(vals)})")
    print("\nGATE: gdn seed42 F1 should be ~0.8109; PTaPR-AUC ~0.27 (research note).")


if __name__ == "__main__":
    main()
