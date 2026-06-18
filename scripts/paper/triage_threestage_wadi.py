#!/usr/bin/env python3
"""Three-stage triage study for WADI (Payoff-2 tables), mirroring
triage_threestage_swat.py. Held-out region, 6 seeds, oracle detector thresholds,
Fix-A postproc, dual-quantile Omega band on the dismiss tier.

Per backbone (gdn, topogdn, cstgl, dualstage):
  Stage 1: M0 chapter detection score; Stage 2: best WADI fusion (Table tab:fusion-wadi);
  Stage 3: fusion alarms minus the dismiss baskets (R5/R6/quiet) with the Omega band.
  Basket table on fusion alarms: n, true, P(true|basket).

WADI best-fusion per backbone (from the held-out ladder):
  gdn=S2_GBM, topogdn=V1_varweighted, cstgl=S1_logistic, dualstage=S1_logistic.
GBM is computed on the fly (no WADI cache) via compute_M10_PAK.fit_M10_score.
CPU only. Writes results/typing_wadi_v2/triage_threestage_wadi.csv.
"""
import argparse
import csv
import os
import sys

os.environ["UQ_DATASET"] = "wadi"   # MUST precede the imports below
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))

from sweep_eval_gdeltauq import build_full_err_scores, topk_aggregate  # noqa: E402
from typing_rules_v1v2 import (load_combo, c_slice_thresholds, type_step,  # noqa: E402
                               load_attack_table, VAL_SLICE, C_END)
from analyze_multistage_attacks import estimate_offset  # noqa: E402
import fusion_study as FS  # noqa: E402
from compute_M10_PAK import fit_M10_score  # noqa: E402
from fusion_sweep_K100_full import setup_context  # noqa: E402
from fusion_v1v2 import promote_omega, DIRMAP_WADI, SPLIT, BUNDLE  # noqa: E402
from fusion_likelihood import fast_oracle_eval, _postproc_fast, POST_W, POST_G  # noqa: E402

SEEDS = [0, 1, 2, 3, 4, 42]
H0 = VAL_SLICE[1]
SHORT = {"R1_high_confidence": "R1", "R2_noisy_sensor": "R2", "R3_borderline": "R3",
         "R4_ood_suspect": "R4", "R4b_ood_rescue": "R4b", "R5_benign_noise": "R5",
         "R6_data_gap": "R6", "normal_quiet": "quiet"}
ALLV = ["R1", "R2", "R3", "R4", "R4b", "R5", "R6", "quiet"]
DISM = {"R5", "R6", "quiet"}
SELECT = {"gdn": "S2_GBM", "topogdn": "V1_varweighted", "cstgl": "S1_logistic", "dualstage": "S1_logistic"}
CHAPTER = {"gdn": (0.387, 0.429), "topogdn": (0.114, 0.321),
           "cstgl": (0.095, 0.430), "dualstage": (0.094, 0.480)}
NAME = {"gdn": "GDN", "topogdn": "TopoGDN", "cstgl": "CST-GL", "dualstage": "DualSTGF"}


def f1pr(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def fusion_score(bb, ctxf):
    m = SELECT[bb]
    if m == "S2_GBM":
        return np.asarray(fit_M10_score(ctxf)[0], np.float64)
    if m == "S1_logistic":
        return np.asarray(FS.logistic_stacker(ctxf)[0], np.float64)
    return np.asarray(FS.build_scores(ctxf)[m], np.float64)


def run_backbone(bb, atts):
    cms = {s: dict(TP=0, FP=0, FN=0, TN=0) for s in ("m0", "fus", "tri")}
    basket = {v: [0, 0] for v in ALLV}
    orc_m0, orc_fus = [], []
    for seed in SEEDS:
        arr = os.path.join(ROOT, "results", DIRMAP_WADI[bb], "V2", f"seed{seed}", "arrays_full.npz")
        z = np.load(arr)
        tmu, tgt = z["test_mu_bar"].astype(float), z["test_ground_truth"].astype(float)
        vmu, vgt = z["val_mu_bar"].astype(float), z["val_ground_truth"].astype(float)
        lab = z["test_attack_label"].astype(int); T = len(lab)
        m0 = topk_aggregate(build_full_err_scores(tmu, tgt, vmu, vgt, 5), 1)
        ctx = load_combo(bb, "V2", seed)
        doff = estimate_offset(lab, atts); thr = c_slice_thresholds(ctx, doff)
        bits = np.stack([ctx["R"] > thr["R"], ctx["A"] > thr["A"],
                         ctx["E"] > thr["E"], ctx["O"] > thr["O"]], 1)
        vd = np.array([SHORT[type_step(*b)] for b in bits])
        promote_omega(arr)
        ctxf = setup_context(argparse.Namespace(arrays=arr, split=SPLIT, bundle=BUNDLE,
                                                slide_win=60, seed=seed))
        fus = fusion_score(bb, ctxf)
        assert len(fus) == T and len(m0) == T, (len(fus), len(m0), T)
        H = slice(H0, T); labh = lab[H].astype(bool); vh = vd[H]
        r0 = fast_oracle_eval(m0[H], labh.astype(int)); r1 = fast_oracle_eval(fus[H], labh.astype(int))
        orc_m0.append(r0["F1"]); orc_fus.append(r1["F1"])
        a0 = _postproc_fast((m0[H] >= r0["tau"]).astype(np.int8), POST_W, POST_G).astype(bool)
        a1 = _postproc_fast((fus[H] >= r1["tau"]).astype(np.int8), POST_W, POST_G).astype(bool)
        cmsk = ctx["nominal"].copy(); cmsk[min(T, C_END + max(0, doff)):] = False
        thrO_band = float(np.quantile(ctx["O"][cmsk], 0.99))
        dism = np.isin(vh, list(DISM)) & ~((ctx["O"] > thrO_band)[H0:T])
        a2 = a1 & ~dism
        for tag, al in (("m0", a0), ("fus", a1), ("tri", a2)):
            cms[tag]["TP"] += int((al & labh).sum()); cms[tag]["FP"] += int((al & ~labh).sum())
            cms[tag]["FN"] += int((~al & labh).sum()); cms[tag]["TN"] += int((~al & ~labh).sum())
        for v in ALLV:
            mm = a1 & (vh == v)
            basket[v][0] += int(mm.sum()); basket[v][1] += int((mm & labh).sum())
        print(f"  [{bb} s{seed}] m0F1={r0['F1']:.3f} fusF1={r1['F1']:.3f}", flush=True)
    return cms, basket, float(np.mean(orc_m0)), float(np.mean(orc_fus))


def main():
    atts = load_attack_table()
    out_rows = []
    for bb in ("gdn", "topogdn", "cstgl", "dualstage"):
        print(f"\n##### {NAME[bb]} (fusion = {SELECT[bb]}) #####", flush=True)
        cms, basket, m0bar, fusbar = run_backbone(bb, atts)
        cm0, cf = CHAPTER[bb]
        g0 = "OK" if abs(m0bar - cm0) <= 0.03 else "CHECK"
        g1 = "OK" if abs(fusbar - cf) <= 0.03 else "CHECK"
        print(f"\n=== {NAME[bb]} === M0 {m0bar:.3f} vs ladder {cm0:.3f} [{g0}]; "
              f"fusion {fusbar:.3f} vs {cf:.3f} [{g1}]")
        for tag, label in (("m0", "Stage1 M0"), ("fus", f"Stage2 fusion({SELECT[bb]})"),
                           ("tri", "Stage3 +triage")):
            d = cms[tag]; f, p, r = f1pr(d["TP"], d["FP"], d["FN"])
            print(f"{label:26s} TP {d['TP']:6d} FP {d['FP']:6d} FN {d['FN']:6d} TN {d['TN']:7d}"
                  f"  P={p:.3f} R={r:.3f} F1={f:.3f}")
            out_rows.append([bb, tag, d["TP"], d["FP"], d["FN"], d["TN"],
                             round(p, 4), round(r, 4), round(f, 4)])
        # TP-retained / FP-removed (band rule, fusion->triage)
        tp2, tp3 = cms["fus"]["TP"], cms["tri"]["TP"]; fp2, fp3 = cms["fus"]["FP"], cms["tri"]["FP"]
        print(f"  TP retained {100*tp3/tp2:.1f}%  FP removed {100*(fp2-fp3)/fp2:.1f}%")
        out_rows.append([bb, "retain_remove", tp2, tp3, fp2, fp3,
                         round(100*tp3/tp2, 2), round(100*(fp2-fp3)/fp2, 2), ""])
        print("baskets: " + "  ".join(
            f"{v}:{basket[v][0]}({basket[v][1]}T)" for v in ALLV if basket[v][0]))
        for v in ALLV:
            n, t = basket[v]
            out_rows.append([bb, f"basket_{v}", n, t, "", "",
                             round(t / n, 3) if n else "", "", ""])
    out = os.path.join(ROOT, "results", "typing_wadi_v2", "triage_threestage_wadi.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["backbone", "row", "a", "b", "c", "d", "x", "y", "z"]); w.writerows(out_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
