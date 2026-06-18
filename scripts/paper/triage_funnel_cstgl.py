#!/usr/bin/env python3
"""CST-GL SWaT false-alarm funnel + typing concordance, 6 seeds, held-out, oracle threshold.

Pipeline per seed (held-out, oracle best-F1 operating point, Fix-A postproc):
  M0 = chapter residual detection score (build_full_err_scores smooth=5, top-1).
  Fusion = S1 logistic stacker (fusion_study.logistic_stacker).
  Verdict = Table-7.1 type_step on the robust typing channels (deployable Q0.995 thr).

Reports:
  1. Confusion matrices (M0, fusion) pooled + per seed.
  2. False-alarm funnel: M0 FP -> fusion FP -> fusion FP that the rule still ESCALATES
     (R1/R3/R4); the dismissible remainder (R5/R6/quiet) is what the rule resolves.
  3. Verdict breakdown of fusion's FP cell.
  4. Concordance on M0's FP: do fusion-suppression and rule-dismissal agree?

CPU only. Writes results/typing_v1v2/triage_funnel_cstgl.csv.
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
                               load_attack_table, VAL_SLICE)
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
FIRES = {"R1", "R2", "R3", "R4"}      # residual-fires -> escalate-type
DISM = {"R5", "R6", "quiet"}          # residual-silent low-priority -> dismissible


def f1pr(d):
    tp, fp, fn = d["TP"], d["FP"], d["FN"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main():
    atts = load_attack_table()
    rows = []
    pool = dict(m0=dict(TP=0, FP=0, FN=0, TN=0), fus=dict(TP=0, FP=0, FN=0, TN=0))
    fusfp_v = {v: 0 for v in ALLV}
    funnel = dict(m0_fp=0, fus_fp=0, fus_fp_escalate=0, fus_fp_dismiss=0)
    conc = dict(both_kill=0, fus_only=0, rule_only=0, both_keep=0, m0_fp=0)
    print(f"postproc W={POST_W} G={POST_G};  held-out start={H0}\n")
    print(f"{'seed':>4} | {'M0 TP/FP/FN':>16} {'F1':>5} | {'FUS TP/FP/FN':>16} {'F1':>5} | "
          f"{'funnel M0fp>FUSfp>esc':>22}")
    for seed in SEEDS:
        z = np.load(os.path.join(ROOT, "results", DIRMAP_SWAT["cstgl"], "V2",
                                 f"seed{seed}", "arrays_full.npz"))
        tmu, tgt = z["test_mu_bar"].astype(float), z["test_ground_truth"].astype(float)
        vmu, vgt = z["val_mu_bar"].astype(float), z["val_ground_truth"].astype(float)
        lab = z["test_attack_label"].astype(int); T = len(lab)
        m0 = topk_aggregate(build_full_err_scores(tmu, tgt, vmu, vgt, 5), 1)
        ctx = load_combo("cstgl", "V2", seed)
        doff = estimate_offset(lab, atts); thr = c_slice_thresholds(ctx, doff)
        bits = np.stack([ctx["R"] > thr["R"], ctx["A"] > thr["A"],
                         ctx["E"] > thr["E"], ctx["O"] > thr["O"]], 1)
        verdict = np.array([SHORT[type_step(*b)] for b in bits])
        arr = os.path.join(ROOT, "results", DIRMAP_SWAT["cstgl"], "V2", f"seed{seed}", "arrays_full.npz")
        promote_omega(arr)
        ctxf = setup_context(argparse.Namespace(arrays=arr, split=SPLIT, bundle=BUNDLE,
                                                slide_win=60, seed=seed))
        s1, _ = FS.logistic_stacker(ctxf)
        tau_m0 = fast_oracle_eval(m0[H0:], lab[H0:])["tau"]
        tau_s1 = fast_oracle_eval(s1[H0:], lab[H0:])["tau"]
        H = slice(H0, T); labh = lab[H].astype(bool); vh = verdict[H]
        pm0 = _postproc_fast((m0[H] >= tau_m0).astype(np.int8), POST_W, POST_G).astype(bool)
        ps1 = _postproc_fast((s1[H] >= tau_s1).astype(np.int8), POST_W, POST_G).astype(bool)

        def cmat(pred):
            return dict(TP=int((pred & labh).sum()), FP=int((pred & ~labh).sum()),
                        FN=int((~pred & labh).sum()), TN=int((~pred & ~labh).sum()))
        dm0, dfus = cmat(pm0), cmat(ps1)
        m0_fp = pm0 & ~labh
        fus_fp = ps1 & ~labh
        esc = int((fus_fp & np.isin(vh, list(FIRES))).sum())
        dis = int((fus_fp & np.isin(vh, list(DISM))).sum())
        # concordance on M0's false alarms
        rule_dism = np.isin(vh, list(DISM))
        bk = int((m0_fp & ~ps1 & rule_dism).sum())      # both kill
        fo = int((m0_fp & ~ps1 & ~rule_dism).sum())     # fusion only
        ro = int((m0_fp & ps1 & rule_dism).sum())       # rule only
        bp = int((m0_fp & ps1 & ~rule_dism).sum())      # both keep
        for k in ("TP", "FP", "FN", "TN"):
            pool["m0"][k] += dm0[k]; pool["fus"][k] += dfus[k]
        for v in ALLV:
            fusfp_v[v] += int((fus_fp & (vh == v)).sum())
        funnel["m0_fp"] += dm0["FP"]; funnel["fus_fp"] += dfus["FP"]
        funnel["fus_fp_escalate"] += esc; funnel["fus_fp_dismiss"] += dis
        conc["both_kill"] += bk; conc["fus_only"] += fo; conc["rule_only"] += ro
        conc["both_keep"] += bp; conc["m0_fp"] += int(m0_fp.sum())
        fm0 = f1pr(dm0)[0]; ffu = f1pr(dfus)[0]
        rows.append(dict(seed=seed, m0=dm0, fus=dfus, esc=esc, dis=dis))
        print(f"{seed:>4} | {dm0['TP']:5d}/{dm0['FP']:5d}/{dm0['FN']:4d} {fm0:>5.3f} | "
              f"{dfus['TP']:5d}/{dfus['FP']:4d}/{dfus['FN']:4d} {ffu:>5.3f} | "
              f"{dm0['FP']:6d} > {dfus['FP']:4d} > {esc:3d}", flush=True)

    print("\n" + "=" * 80)
    print("POOLED over 6 seeds (held-out, oracle threshold, postproc)")
    print("=" * 80)
    fm0, pm, rm = f1pr(pool["m0"]); ffu, pf, rf = f1pr(pool["fus"])
    print(f"M0  : TP {pool['m0']['TP']}  FP {pool['m0']['FP']}  FN {pool['m0']['FN']}  TN {pool['m0']['TN']}"
          f"   P={pm:.3f} R={rm:.3f} F1={fm0:.3f}")
    print(f"FUS : TP {pool['fus']['TP']}  FP {pool['fus']['FP']}  FN {pool['fus']['FN']}  TN {pool['fus']['TN']}"
          f"   P={pf:.3f} R={rf:.3f} F1={ffu:.3f}")
    print(f"\nFALSE-ALARM FUNNEL (pooled FP windows):")
    print(f"  M0 false alarms            {funnel['m0_fp']:6d}")
    print(f"  -> after fusion            {funnel['fus_fp']:6d}  "
          f"({100*(1-funnel['fus_fp']/max(1,funnel['m0_fp'])):.0f}% cut)")
    print(f"  -> rule escalates (R1/R3/R4){funnel['fus_fp_escalate']:5d}  "
          f"(rule dismisses {funnel['fus_fp_dismiss']} = "
          f"{100*funnel['fus_fp_dismiss']/max(1,funnel['fus_fp']):.0f}% of fusion FP)")
    print(f"  net recall preserved: M0 R={rm:.3f} -> fusion R={rf:.3f}")
    print(f"\nFUSION FP cell by verdict (pooled): "
          + "  ".join(f"{v}={fusfp_v[v]}" for v in ALLV if fusfp_v[v]))
    esc_tot = sum(fusfp_v[v] for v in FIRES); dis_tot = sum(fusfp_v[v] for v in DISM)
    tot = sum(fusfp_v.values()) or 1
    print(f"   escalate-type {esc_tot} ({100*esc_tot/tot:.0f}%)  |  dismissible {dis_tot} ({100*dis_tot/tot:.0f}%)")
    print(f"\nCONCORDANCE on M0's {conc['m0_fp']} false alarms (fusion-suppress vs rule-dismiss):")
    print(f"  both kill={conc['both_kill']}  fusion-only={conc['fus_only']}  "
          f"rule-only={conc['rule_only']}  both-keep={conc['both_keep']}")
    agree = (conc["both_kill"] + conc["both_keep"]) / max(1, conc["m0_fp"])
    print(f"  agreement = {100*agree:.1f}%")

    out = os.path.join(ROOT, "results", "typing_v1v2", "triage_funnel_cstgl.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["section", "key", "value"])
        for tag in ("m0", "fus"):
            for k, v in pool[tag].items():
                w.writerow([f"pooled_{tag}", k, v])
        for k, v in funnel.items():
            w.writerow(["funnel", k, v])
        for v in ALLV:
            w.writerow(["fusion_fp_verdict", v, fusfp_v[v]])
        for k, v in conc.items():
            w.writerow(["concordance", k, v])
        for r in rows:
            w.writerow(["per_seed", f"seed{r['seed']}",
                        f"M0={r['m0']} FUS={r['fus']} esc={r['esc']} dis={r['dis']}"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
