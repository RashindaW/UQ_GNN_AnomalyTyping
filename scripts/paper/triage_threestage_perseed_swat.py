#!/usr/bin/env python3
"""Per-seed (mean +/- std) three-stage triage at BOTH operating points, SWaT held-out.

For each backbone and each detector operating point:
  ORACLE     = best-F1 sweep threshold (fast_oracle_eval), the chapter detection convention.
  DEPLOYABLE = Q0.995 on C-slice nominal of the score (label-free), the realistic point.
report, as 6-seed mean +/- std:
  Stage 1 M0 F1, Stage 2 fusion F1, Stage 3 fusion+triage F1 (band rule),
  %TP in fault bins {R1,R2,R3,R4,R4b}, %FP in benign bins {R5,R6,quiet} (fusion stream).

Oracle gate (pooled-mean oracle F1 vs the chapter) is asserted so we know it is the same
pipeline. CPU only. Writes results/typing_v1v2/triage_threestage_perseed_swat.csv.
"""
import argparse
import csv
import os
import sys

os.environ.setdefault("UQ_DATASET", "swat")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))

from sweep_eval_gdeltauq import build_full_err_scores, topk_aggregate  # noqa: E402
from typing_rules_v1v2 import (load_combo, c_slice_thresholds, type_step,  # noqa: E402
                               load_attack_table, VAL_SLICE, Q_HIGH, C_END)
from analyze_multistage_attacks import estimate_offset  # noqa: E402
import fusion_study as FS  # noqa: E402
from fusion_sweep_K100_full import setup_context  # noqa: E402
from fusion_v1v2 import promote_omega, DIRMAP_SWAT, SPLIT, BUNDLE  # noqa: E402
from fusion_likelihood import fast_oracle_eval, _postproc_fast, POST_W, POST_G  # noqa: E402

SEEDS = [0, 1, 2, 3, 4, 42]
H0 = VAL_SLICE[1]
SHORT = {"R1_high_confidence": "R1", "R2_noisy_sensor": "R2", "R3_borderline": "R3",
         "R4_ood_suspect": "R4", "R4b_ood_rescue": "R4b", "R5_benign_noise": "R5",
         "R6_data_gap": "R6", "normal_quiet": "quiet"}
ALLV = ["R1", "R2", "R3", "R4", "R4b", "R5", "R6", "quiet"]
FAULT = {"R1", "R2", "R3", "R4", "R4b"}
BEN = {"R5", "R6", "quiet"}
DISM = {"R5", "R6", "quiet"}
SELECT = {"gdn": "L1_stdres", "topogdn": "S2_GBM", "cstgl": "S1_logistic", "dualstage": "L1_stdres"}
CHAPTER = {"gdn": (0.810, 0.821), "topogdn": (0.800, 0.823),
           "cstgl": (0.613, 0.820), "dualstage": (0.721, 0.835)}
NAME = {"gdn": "GDN", "topogdn": "TopoGDN", "cstgl": "CST-GL", "dualstage": "DualSTGF"}


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def fusion_score(bb, seed, ctxf):
    m = SELECT[bb]
    if m == "S2_GBM":
        return np.load(os.path.join(ROOT, "results", "typing_v1v2", "gbm",
                                    f"{bb}_V2_s{seed}_cache.npz"))["score"].astype(np.float64)
    if m == "S1_logistic":
        s, _ = FS.logistic_stacker(ctxf)
        return np.asarray(s, np.float64)
    return np.asarray(FS.build_scores(ctxf)[m], np.float64)


def alarms(score_h, thr, mode, lab_h):
    tau = fast_oracle_eval(score_h, lab_h.astype(int))["tau"] if mode == "oracle" else thr
    return _postproc_fast((score_h >= tau).astype(np.int8), POST_W, POST_G).astype(bool)


def basket_split(a1, vh, labh):
    """%TP in fault bins, %FP in benign bins for the fusion alarm set a1."""
    tp = int((a1 & labh).sum()) or 1
    fp = int((a1 & ~labh).sum()) or 1
    tpf = sum(int((a1 & labh & (vh == v)).sum()) for v in FAULT)
    fpb = sum(int((a1 & ~labh & (vh == v)).sum()) for v in BEN)
    return 100.0 * tpf / tp, 100.0 * fpb / fp


def run_backbone(bb, atts):
    # per-seed metric lists, keyed (threshold_mode, metric)
    acc = {(mode, k): [] for mode in ("oracle", "deploy")
           for k in ("m0F1", "fusF1", "triF1", "tpFault", "fpBen")}
    orc_m0, orc_fus = [], []
    for seed in SEEDS:
        z = np.load(os.path.join(ROOT, "results", DIRMAP_SWAT[bb], "V2", f"seed{seed}", "arrays_full.npz"))
        tmu, tgt = z["test_mu_bar"].astype(float), z["test_ground_truth"].astype(float)
        vmu, vgt = z["val_mu_bar"].astype(float), z["val_ground_truth"].astype(float)
        lab = z["test_attack_label"].astype(int); T = len(lab)
        m0 = topk_aggregate(build_full_err_scores(tmu, tgt, vmu, vgt, 5), 1)
        ctx = load_combo(bb, "V2", seed)
        doff = estimate_offset(lab, atts); thr = c_slice_thresholds(ctx, doff)
        bits = np.stack([ctx["R"] > thr["R"], ctx["A"] > thr["A"],
                         ctx["E"] > thr["E"], ctx["O"] > thr["O"]], 1)
        vd = np.array([SHORT[type_step(*b)] for b in bits])
        arr = os.path.join(ROOT, "results", DIRMAP_SWAT[bb], "V2", f"seed{seed}", "arrays_full.npz")
        promote_omega(arr)
        ctxf = setup_context(argparse.Namespace(arrays=arr, split=SPLIT, bundle=BUNDLE,
                                                slide_win=60, seed=seed))
        fus = fusion_score(bb, seed, ctxf)
        assert len(fus) == T and len(m0) == T
        H = slice(H0, T); labh = lab[H].astype(bool); vh = vd[H]
        # deployable detector thresholds (Q0.995 on C-slice nominal)
        cmsk = ctx["nominal"].copy(); cmsk[min(T, C_END + max(0, doff)):] = False
        thr_m0 = float(np.quantile(m0[cmsk], Q_HIGH)); thr_fus = float(np.quantile(fus[cmsk], Q_HIGH))
        thrO_band = float(np.quantile(ctx["O"][cmsk], 0.99))
        ohit = (ctx["O"] > thrO_band)[H0:T]
        orc_m0.append(fast_oracle_eval(m0[H], labh.astype(int))["F1"])
        orc_fus.append(fast_oracle_eval(fus[H], labh.astype(int))["F1"])
        for mode, tm0, tfus in (("oracle", None, None), ("deploy", thr_m0, thr_fus)):
            a0 = alarms(m0[H], tm0, mode, labh)
            a1 = alarms(fus[H], tfus, mode, labh)
            dism = np.isin(vh, list(DISM)) & ~ohit
            a2 = a1 & ~dism
            acc[(mode, "m0F1")].append(f1(int((a0 & labh).sum()), int((a0 & ~labh).sum()), int((~a0 & labh).sum())))
            acc[(mode, "fusF1")].append(f1(int((a1 & labh).sum()), int((a1 & ~labh).sum()), int((~a1 & labh).sum())))
            acc[(mode, "triF1")].append(f1(int((a2 & labh).sum()), int((a2 & ~labh).sum()), int((~a2 & labh).sum())))
            tpf, fpb = basket_split(a1, vh, labh)
            acc[(mode, "tpFault")].append(tpf); acc[(mode, "fpBen")].append(fpb)
        print(f"  [{bb} s{seed}] done", flush=True)
    return acc, float(np.mean(orc_m0)), float(np.mean(orc_fus))


def ms(xs):
    return float(np.mean(xs)), float(np.std(xs))


def main():
    atts = load_attack_table()
    rows = []
    for bb in ("gdn", "topogdn", "dualstage", "cstgl"):
        print(f"\n##### {NAME[bb]} (fusion={SELECT[bb]}) #####", flush=True)
        acc, m0bar, fusbar = run_backbone(bb, atts)
        cm0, cf = CHAPTER[bb]
        g0 = "OK" if abs(m0bar - cm0) <= 0.015 else "MISMATCH"
        g1 = "OK" if abs(fusbar - cf) <= 0.015 else "MISMATCH"
        print(f"=== {NAME[bb]} === oracle gate M0 {m0bar:.3f}/{cm0} [{g0}]  fus {fusbar:.3f}/{cf} [{g1}]")
        for mode in ("oracle", "deploy"):
            vals = {k: ms(acc[(mode, k)]) for k in ("m0F1", "fusF1", "triF1", "tpFault", "fpBen")}
            print(f"  [{mode:7s}] "
                  f"M0 {vals['m0F1'][0]:.3f}+/-{vals['m0F1'][1]:.3f}  "
                  f"fus {vals['fusF1'][0]:.3f}+/-{vals['fusF1'][1]:.3f}  "
                  f"+triage {vals['triF1'][0]:.3f}+/-{vals['triF1'][1]:.3f}  | "
                  f"%TP-fault {vals['tpFault'][0]:.1f}+/-{vals['tpFault'][1]:.1f}  "
                  f"%FP-benign {vals['fpBen'][0]:.1f}+/-{vals['fpBen'][1]:.1f}")
            for k, (mn, sd) in vals.items():
                rows.append([bb, mode, k, round(mn, 4), round(sd, 4)])
    out = os.path.join(ROOT, "results", "typing_v1v2", "triage_threestage_perseed_swat.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["backbone", "threshold", "metric", "mean", "std"]); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
