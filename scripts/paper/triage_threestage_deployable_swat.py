#!/usr/bin/env python3
"""Three-stage triage at the DEPLOYABLE detector threshold (A10 rebuttal).

Identical to triage_threestage_swat.py EXCEPT the detector operating point: instead of
the oracle best-F1 threshold (fast_oracle_eval tau), both M0 and the fusion score are cut
at their Q0.995 quantile on the C-SLICE NOMINAL rows -- the same leakage-free, label-free
convention the verdict rule and alarm_triage_v1v2 use. This removes the oracle dependence
the examiner flagged as fatal to the "deployable" framing. The honest result (including any
weak-baseline collapse at this threshold) is the reportable finding.

Oracle F1 is still printed per seed for reference / the chapter-parity sanity, but the
confusion matrices and baskets are at the deployable threshold. CPU only.
Writes results/typing_v1v2/triage_threestage_deployable_swat.csv.
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
DISM = {"R5", "R6", "quiet"}
SELECT = {"gdn": "L1_stdres", "topogdn": "S2_GBM", "cstgl": "S1_logistic", "dualstage": "L1_stdres"}
CHAPTER = {"gdn": (0.810, 0.821), "topogdn": (0.800, 0.823),
           "cstgl": (0.613, 0.820), "dualstage": (0.721, 0.835)}
NAME = {"gdn": "GDN", "topogdn": "TopoGDN", "cstgl": "CST-GL", "dualstage": "DualSTGF"}


def f1pr(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def fusion_score(bb, seed, ctxf):
    m = SELECT[bb]
    if m == "S2_GBM":
        cache = os.path.join(ROOT, "results", "typing_v1v2", "gbm", f"{bb}_V2_s{seed}_cache.npz")
        return np.load(cache)["score"].astype(np.float64)
    if m == "S1_logistic":
        s, _ = FS.logistic_stacker(ctxf)
        return np.asarray(s, np.float64)
    return np.asarray(FS.build_scores(ctxf)[m], np.float64)


def deploy_thr(score, ctx, doff, T):
    """Q0.995 on C-slice nominal rows of this detector score (label-free, leakage-free)."""
    cmsk = ctx["nominal"].copy()
    cmsk[min(T, C_END + max(0, doff)):] = False
    return float(np.quantile(score[cmsk], Q_HIGH)), cmsk


def run_backbone(bb, atts):
    cms = {s: dict(TP=0, FP=0, FN=0, TN=0) for s in ("m0", "fus", "tri")}
    basket = {v: [0, 0] for v in ALLV}
    decomp = {st: {c: {v: 0 for v in ALLV} for c in ("TP", "FP")} for st in ("m0", "fus")}
    orc_m0, orc_fus, dep_m0F1, dep_fusF1 = [], [], [], []
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
        # oracle (reference only) + deployable thresholds
        r0 = fast_oracle_eval(m0[H], labh.astype(int)); r1 = fast_oracle_eval(fus[H], labh.astype(int))
        orc_m0.append(r0["F1"]); orc_fus.append(r1["F1"])
        thr_m0, _ = deploy_thr(m0, ctx, doff, T)
        thr_fus, _ = deploy_thr(fus, ctx, doff, T)
        a0 = _postproc_fast((m0[H] >= thr_m0).astype(np.int8), POST_W, POST_G).astype(bool)
        a1 = _postproc_fast((fus[H] >= thr_fus).astype(np.int8), POST_W, POST_G).astype(bool)
        # band rule stage 3 (unchanged): dismiss R5/R6/quiet unless Omega over its 0.99 band
        cmsk = ctx["nominal"].copy()
        cmsk[min(T, C_END + max(0, doff)):] = False
        thrO_band = float(np.quantile(ctx["O"][cmsk], 0.99))
        dism = np.isin(vh, list(DISM)) & ~((ctx["O"] > thrO_band)[H0:T])
        a2 = a1 & ~dism
        f0, _, _ = f1pr(int((a0 & labh).sum()), int((a0 & ~labh).sum()), int((~a0 & labh).sum()))
        f1d, _, _ = f1pr(int((a1 & labh).sum()), int((a1 & ~labh).sum()), int((~a1 & labh).sum()))
        dep_m0F1.append(f0); dep_fusF1.append(f1d)
        for tag, al in (("m0", a0), ("fus", a1), ("tri", a2)):
            cms[tag]["TP"] += int((al & labh).sum()); cms[tag]["FP"] += int((al & ~labh).sum())
            cms[tag]["FN"] += int((~al & labh).sum()); cms[tag]["TN"] += int((~al & ~labh).sum())
        for v in ALLV:
            m = a1 & (vh == v)
            basket[v][0] += int(m.sum()); basket[v][1] += int((m & labh).sum())
            for st, al in (("m0", a0), ("fus", a1)):
                decomp[st]["TP"][v] += int((al & labh & (vh == v)).sum())
                decomp[st]["FP"][v] += int((al & ~labh & (vh == v)).sum())
        print(f"  [{bb} s{seed}] oracleF1 m0={r0['F1']:.3f} fus={r1['F1']:.3f} | "
              f"deployF1 m0={f0:.3f} fus={f1d:.3f}", flush=True)
    return (cms, basket, decomp, float(np.mean(orc_m0)), float(np.mean(orc_fus)),
            float(np.mean(dep_m0F1)), float(np.mean(dep_fusF1)))


def main():
    atts = load_attack_table()
    out_rows = []
    for bb in ("gdn", "topogdn", "dualstage", "cstgl"):
        print(f"\n##### {NAME[bb]} (fusion = {SELECT[bb]}) #####", flush=True)
        cms, basket, decomp, m0bar, fusbar, depm0, depfus = run_backbone(bb, atts)
        cm0, cf = CHAPTER[bb]
        g0 = "OK" if abs(m0bar - cm0) <= 0.015 else "MISMATCH"
        g1 = "OK" if abs(fusbar - cf) <= 0.015 else "MISMATCH"
        print(f"\n=== {NAME[bb]} === oracle gate: M0 {m0bar:.3f} vs chapter {cm0:.3f} [{g0}]; "
              f"fusion {fusbar:.3f} vs {cf:.3f} [{g1}]")
        print(f"    DEPLOYABLE pooled-mean F1: M0 {depm0:.3f}  fusion {depfus:.3f}")
        for tag, label in (("m0", "Stage 1  M0 residual"), ("fus", f"Stage 2  fusion ({SELECT[bb]})"),
                           ("tri", "Stage 3  fusion + triage")):
            d = cms[tag]; f, p, r = f1pr(d["TP"], d["FP"], d["FN"])
            print(f"{label:28s} TP {d['TP']:6d}  FP {d['FP']:6d}  FN {d['FN']:6d}  TN {d['TN']:7d}"
                  f"   P={p:.3f} R={r:.3f} F1={f:.3f}")
            out_rows.append([bb, tag, d["TP"], d["FP"], d["FN"], d["TN"],
                             round(p, 4), round(r, 4), round(f, 4)])
        print("baskets on fusion alarms:  " + "  ".join(
            f"{v}:{basket[v][0]}({basket[v][1]}T,P={basket[v][1]/basket[v][0]:.2f})"
            for v in ALLV if basket[v][0]))
        for st in ("m0", "fus"):
            ttp = sum(decomp[st]["TP"].values()) or 1; tfp = sum(decomp[st]["FP"].values()) or 1
            dtp = sum(decomp[st]["TP"][v] for v in DISM); dfp = sum(decomp[st]["FP"][v] for v in DISM)
            print(f"{st:>4} stream: benefit {dfp}/{tfp} = {100*dfp/tfp:.1f}%   "
                  f"collateral {dtp}/{ttp} = {100*dtp/ttp:.1f}%")
            out_rows.append([bb, f"{st}_benefit_collateral", dfp, tfp, dtp, ttp,
                             round(100 * dfp / tfp, 2), round(100 * dtp / ttp, 2), ""])
        for v in ALLV:
            out_rows.append([bb, f"basket_{v}", basket[v][0], basket[v][1], "", "",
                             round(basket[v][1] / basket[v][0], 3) if basket[v][0] else "", "", ""])
        out_rows.append([bb, "deployable_F1", round(depm0, 4), round(depfus, 4), "", "", "", "", ""])
    out = os.path.join(ROOT, "results", "typing_v1v2", "triage_threestage_deployable_swat.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "row", "a", "b", "c", "d", "x", "y", "z"])
        w.writerows(out_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
