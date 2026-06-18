#!/usr/bin/env python3
"""Deeper anomaly triage: the typing rule as a confusion-matrix transformer.

CST-GL on SWaT, held-out region, 6 seeds. Builds two confusion matrices --
residual-only (M0 = residual channel at the deployable Q0.995 threshold, the
verdict-coherent residual) and best fusion (S1 logistic stacker) -- then overlays
the Table-7.1 verdicts on the error cells:

  * M0 false-negative cell  -> {R4b rescue | R5 | R6 | quiet}; R4b = parameter-free rescue.
  * M0 false-positive cell  -> {R1 | R2 | R3 | R4}; R2 = dismissible sensor-health.
  * fusion false-positive cell -> full verdict set; R2/R5/R6 = dismissible flood.

Strict partition (type_step): residual fires -> {R1,R2,R3,R4};
residual silent -> {R4b,R5,R6,quiet}. Verified as a built-in assertion.

Window-level (large N) AND event-level (the honest yardstick). Deployable Q0.995
is primary; oracle best-F1 reported as a sensitivity line.

CPU only, ~2 min. Writes results/typing_v1v2/triage_confusion_cstgl.{csv,pdf,png}.
"""
import argparse
import csv
import os
import sys

os.environ.setdefault("UQ_DATASET", "swat")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))

from typing_rules_v1v2 import (load_combo, c_slice_thresholds, type_step,  # noqa: E402
                               load_attack_table, VAL_SLICE, Q_HIGH)
from analyze_multistage_attacks import estimate_offset  # noqa: E402
import fusion_study as FS  # noqa: E402
from fusion_sweep_K100_full import setup_context  # noqa: E402
from fusion_v1v2 import promote_omega, DIRMAP_SWAT, SPLIT, BUNDLE  # noqa: E402
from fusion_likelihood import fast_oracle_eval  # noqa: E402

SEEDS = [0, 1, 2, 3, 4, 42]
HELD0 = VAL_SLICE[1]            # 24530
BB = "cstgl"
FIRES = {"R1_high_confidence", "R2_noisy_sensor", "R3_borderline", "R4_ood_suspect"}
SILENT = {"R4b_ood_rescue", "R5_benign_noise", "R6_data_gap", "normal_quiet"}
DISMISSIBLE = {"R2_noisy_sensor", "R5_benign_noise", "R6_data_gap"}   # operator can deprioritise
SHORT = {"R1_high_confidence": "R1", "R2_noisy_sensor": "R2", "R3_borderline": "R3",
         "R4_ood_suspect": "R4", "R4b_ood_rescue": "R4b", "R5_benign_noise": "R5",
         "R6_data_gap": "R6", "normal_quiet": "quiet"}


def cm(pred, lab):
    pred = pred.astype(bool); lab = lab.astype(bool)
    return dict(TP=int((pred & lab).sum()), FP=int((pred & ~lab).sum()),
                FN=int((~pred & lab).sum()), TN=int((~pred & ~lab).sum()))


def f1(d):
    p = d["TP"] / (d["TP"] + d["FP"]) if d["TP"] + d["FP"] else 0.0
    r = d["TP"] / (d["TP"] + d["FN"]) if d["TP"] + d["FN"] else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0, p, r


def vcount(verdicts):
    u, c = np.unique(verdicts, return_counts=True)
    return {SHORT.get(k, k): int(v) for k, v in zip(u, c)}


def runs(atts, doff, T):
    out = []
    for x in atts:
        s, e = max(0, x["s"] + doff), min(T, x["e"] + doff)
        if e > s:
            out.append((x["aid"], s, e))
    return out


def analyse():
    atts = load_attack_table()
    per_seed, agg = [], dict(m0=dict(TP=0, FP=0, FN=0, TN=0), fus=dict(TP=0, FP=0, FN=0, TN=0))
    fn_verd, m0fp_verd, fusfp_verd = {}, {}, {}
    ev = dict(n=0, det_m0=0, rescued=0, det_fus=0)
    f1rows = []
    for seed in SEEDS:
        ctx = load_combo(BB, "V2", seed)
        lab = ctx["lab"]; T = ctx["T"]
        doff = estimate_offset(lab, atts)
        thr = c_slice_thresholds(ctx, doff)
        R, A, E, O = ctx["R"], ctx["A"], ctx["E"], ctx["O"]
        bits = np.stack([R > thr["R"], A > thr["A"], E > thr["E"], O > thr["O"]], 1)
        verdict = np.array([type_step(*b) for b in bits])
        # built-in partition assertion
        rfire = bits[:, 0]
        assert set(np.unique(verdict[rfire])) <= FIRES, "residual-fires leaked a silent verdict"
        assert set(np.unique(verdict[~rfire])) <= SILENT, "residual-silent leaked a fires verdict"

        # fusion S1 logistic
        arr = os.path.join(ROOT, "results", DIRMAP_SWAT[BB], "V2", f"seed{seed}", "arrays_full.npz")
        promote_omega(arr)
        ctxf = setup_context(argparse.Namespace(arrays=arr, split=SPLIT, bundle=BUNDLE,
                                                slide_win=60, seed=seed))
        s1, _ = FS.logistic_stacker(ctxf)
        thr_s1 = float(np.quantile(s1[ctxf["c_mask"]], Q_HIGH))
        assert len(s1) == T, (len(s1), T)

        H = slice(HELD0, T)
        labh = lab[H]; vh = verdict[H]
        pm0 = (R > thr["R"])[H]
        pfus = (s1 > thr_s1)[H]
        d_m0, d_fus = cm(pm0, labh), cm(pfus, labh)

        # verdict decomposition of error cells (held-out, window level)
        fn_cell = vh[(labh == 1) & (~pm0)]            # M0 false negatives
        m0fp_cell = vh[(labh == 0) & (pm0)]           # M0 false positives
        fusfp_cell = vh[(labh == 0) & (pfus)]         # fusion false positives
        rescued_w = int((fn_cell == "R4b_ood_rescue").sum())

        # oracle (sensitivity) vs deployable F1
        orc_m0 = fast_oracle_eval(R[H], labh)["F1"]
        orc_fus = fast_oracle_eval(s1[H], labh)["F1"]
        dep_m0 = f1(d_m0); dep_fus = f1(d_fus)

        # event level (held-out events)
        evs = [(aid, s, e) for (aid, s, e) in runs(atts, doff, T) if s >= HELD0]
        for aid, s, e in evs:
            ev["n"] += 1
            det = bool((R[s:e] > thr["R"]).any())
            ev["det_m0"] += int(det)
            if not det and (verdict[s:e] == "R4b_ood_rescue").any():
                ev["rescued"] += 1
            ev["det_fus"] += int((s1[s:e] > thr_s1).any())

        for k in ("TP", "FP", "FN", "TN"):
            agg["m0"][k] += d_m0[k]; agg["fus"][k] += d_fus[k]
        for src, cell in [(fn_verd, fn_cell), (m0fp_verd, m0fp_cell), (fusfp_verd, fusfp_cell)]:
            for kk, vv in vcount(cell).items():
                src[kk] = src.get(kk, 0) + vv

        per_seed.append(dict(seed=seed, doff=doff,
                             m0=d_m0, fus=d_fus, rescued_w=rescued_w,
                             dep_f1_m0=round(dep_m0[0], 3), dep_f1_fus=round(dep_fus[0], 3),
                             orc_f1_m0=round(orc_m0, 3), orc_f1_fus=round(orc_fus, 3)))
        f1rows.append((seed, dep_m0, dep_fus, orc_m0, orc_fus))
        print(f"[seed {seed}] doff={doff}  M0 {d_m0}  depF1={dep_m0[0]:.3f} orcF1={orc_m0:.3f}  |  "
              f"FUS {d_fus}  depF1={dep_fus[0]:.3f} orcF1={orc_fus:.3f}  | R4b-rescued windows={rescued_w}",
              flush=True)
    return per_seed, agg, fn_verd, m0fp_verd, fusfp_verd, ev, f1rows


def pct(d):
    tot = sum(d.values()) or 1
    return {k: (v, 100 * v / tot) for k, v in sorted(d.items(), key=lambda x: -x[1])}


def main():
    per_seed, agg, fn_verd, m0fp_verd, fusfp_verd, ev, f1rows = analyse()

    print("\n" + "=" * 78)
    print("POOLED over 6 seeds (held-out, window level, deployable Q0.995)")
    print("=" * 78)
    fm0, pm0, rm0 = f1(agg["m0"]); ffu, pfu, rfu = f1(agg["fus"])
    print(f"M0 residual     {agg['m0']}  P={pm0:.2f} R={rm0:.2f} F1={fm0:.2f}")
    print(f"S1 fusion       {agg['fus']}  P={pfu:.2f} R={rfu:.2f} F1={ffu:.2f}")
    resc = fn_verd.get("R4b", 0)
    tp2, fn2 = agg["m0"]["TP"] + resc, agg["m0"]["FN"] - resc
    f_post = 2 * (tp2 / (tp2 + agg["m0"]["FP"])) * (tp2 / (tp2 + fn2)) / \
        ((tp2 / (tp2 + agg["m0"]["FP"])) + (tp2 / (tp2 + fn2))) if tp2 else 0.0
    print(f"M0 + R4b rescue TP={tp2} FP={agg['m0']['FP']} FN={fn2} TN={agg['m0']['TN']}  F1={f_post:.2f}")
    print(f"\nM0 FN cell (missed-attack windows) by verdict: {pct(fn_verd)}")
    print(f"   -> R4b rescues {resc} of {sum(fn_verd.values())} missed windows "
          f"({100*resc/max(1,sum(fn_verd.values())):.0f}%)")
    print(f"M0 FP cell (false-alarm windows) by verdict:   {pct(m0fp_verd)}")
    dm0 = sum(v for k, v in m0fp_verd.items() if k in {'R2'})
    print(f"   -> dismissible (R2) {dm0} of {sum(m0fp_verd.values())} "
          f"({100*dm0/max(1,sum(m0fp_verd.values())):.0f}%)")
    print(f"FUSION FP cell by verdict:                     {pct(fusfp_verd)}")
    dfu = sum(v for k, v in fusfp_verd.items() if k in {'R2', 'R5', 'R6', 'quiet'})
    print(f"   -> dismissible (R2/R5/R6/quiet) {dfu} of {sum(fusfp_verd.values())} "
          f"({100*dfu/max(1,sum(fusfp_verd.values())):.0f}%)")
    print(f"\nEVENT level (held-out attack events, pooled): n={ev['n']}  "
          f"M0 detects {ev['det_m0']} ({100*ev['det_m0']/max(1,ev['n']):.0f}%); "
          f"R4b rescues {ev['rescued']} of the {ev['n']-ev['det_m0']} missed; "
          f"M0+rescue {ev['det_m0']+ev['rescued']} ({100*(ev['det_m0']+ev['rescued'])/max(1,ev['n']):.0f}%); "
          f"fusion detects {ev['det_fus']} ({100*ev['det_fus']/max(1,ev['n']):.0f}%)")

    # ---- CSV ----
    out = os.path.join(ROOT, "results", "typing_v1v2", "triage_confusion_cstgl.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        for d, tag in [(agg["m0"], "m0"), (agg["fus"], "fus")]:
            for k, v in d.items():
                w.writerow([f"pooled_{tag}", k, v])
        for tag, dd in [("fn_verd", fn_verd), ("m0fp_verd", m0fp_verd), ("fusfp_verd", fusfp_verd)]:
            for k, v in dd.items():
                w.writerow([tag, k, v])
        for k, v in ev.items():
            w.writerow(["events", k, v])
        for r in per_seed:
            w.writerow(["per_seed", f"seed{r['seed']}",
                        f"M0={r['m0']} dep_f1={r['dep_f1_m0']} orc_f1={r['orc_f1_m0']} "
                        f"FUS={r['fus']} dep_f1={r['dep_f1_fus']} orc_f1={r['orc_f1_fus']} "
                        f"rescued_w={r['rescued_w']}"])
    print(f"\nwrote {out}")

    make_figure(agg, fn_verd, m0fp_verd, fusfp_verd, ev)


def make_figure(agg, fn_verd, m0fp_verd, fusfp_verd, ev):
    rcParams["font.family"] = "serif"; rcParams["mathtext.fontset"] = "cm"
    rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, d, title, fpv, fnv in [
        (axes[0], agg["m0"], "Residual baseline (M0)", m0fp_verd, fn_verd),
        (axes[1], agg["fus"], "Best fusion (logistic stacker)", fusfp_verd, None)]:
        grid = np.array([[d["TP"], d["FN"]], [d["FP"], d["TN"]]], float)
        ax.imshow(np.array([[1, 0], [0, 1]]), cmap="Greens", alpha=0.18, vmin=0, vmax=1)
        f, p, r = f1(d)
        ax.set_title(f"{title}\nP={p:.2f}  R={r:.2f}  F1={f:.2f}", fontsize=12)
        labels = [["TP", "FN"], ["FP", "TN"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i - 0.13, f"{labels[i][j]}", ha="center", va="center",
                        fontsize=12, fontweight="bold")
                ax.text(j, i + 0.10, f"{int(grid[i][j]):,}", ha="center", va="center", fontsize=13)
        # annotate FN cell (R4b rescue) for M0, FP cell (dismissible) for both
        if fnv is not None:
            resc = fnv.get("R4b", 0); tot = sum(fnv.values()) or 1
            ax.text(1, 0.34, f"R4b rescues {resc} ({100*resc/tot:.0f}%)",
                    ha="center", va="center", fontsize=8.5, color="#1a7a1a")
        dis = sum(v for k, v in fpv.items() if k in DISMISSIBLE)
        tot = sum(fpv.values()) or 1
        ax.text(0, 1.34, f"dismissible {dis} ({100*dis/tot:.0f}%)",
                ha="center", va="center", fontsize=8.5, color="#b25400")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["attack", "nominal"], fontsize=10)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["alarm", "no alarm"], fontsize=10)
        ax.set_xlabel("ground truth", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("detector decision", fontsize=11)
    fig.suptitle(
        f"CST-GL SWaT held-out: the typing rule resolves the confusion matrix "
        f"(event recall {ev['det_m0']}/{ev['n']} M0 -> {ev['det_m0']+ev['rescued']}/{ev['n']} with R4b rescue)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fp = os.path.join(ROOT, "results", "typing_v1v2", f"triage_confusion_cstgl.{ext}")
        fig.savefig(fp, dpi=200)
    plt.close(fig)
    print(f"wrote results/typing_v1v2/triage_confusion_cstgl.pdf")


if __name__ == "__main__":
    main()
