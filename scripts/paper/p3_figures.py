#!/usr/bin/env python3
"""P3: thesis figures for Part 1 (detection) and Part 2 (typing), post-A1
scope (GDN/TopoGDN/CST-GL). Data-ready figures only; the PA%K curve figure
is generated separately once the curve fleet lands.

Outputs: results/thesis_part1/figs/{F1_ladder,F2_heldout_deltas}.png
         results/thesis_part2/figs/{G1_verdict_grid,G2_triage_pareto,
                                     G3_deployment_table,G4_fingerprints}.png
         + captions.md in each figs dir (claims-ladder-compliant wording).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F1DIR = os.path.join(ROOT, "results/thesis_part1/figs")
F2DIR = os.path.join(ROOT, "results/thesis_part2/figs")
BBS = [("gdn", "GDN"), ("topogdn", "TopoGDN"), ("cstgl", "CST-GL")]
SEEDS = [0, 1, 2, 3, 4, 42]
OMEGA = "#6a3d9a"          # pilot convention for the Omega channel
DPI = 200

VCOL = {"R1_high_confidence": "#1b9e77", "R3_borderline": "#a6761d",
        "R4_ood_suspect": OMEGA, "R4b_ood_rescue": "#b884e0",
        "R2_noisy_sensor": "#e6ab02", "R5_benign_noise": "#d9c79a",
        "R6_data_gap": "#7aa6c2", "missed_quiet": "#cccccc",
        "normal_quiet": "#e8e8e8"}
VORD = ["R1_high_confidence", "R3_borderline", "R4_ood_suspect",
        "R4b_ood_rescue", "R2_noisy_sensor", "R5_benign_noise",
        "R6_data_gap", "missed_quiet"]


def fig_ladder(tidy):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    stages = [("Plain_paper", "plain\n(paper)"), ("M0_residual", "anchored\nM0"),
              ("S2_GBM", "fusion\n(S2)")]
    for ax, (bb, name) in zip(axes, BBS):
        for i, (m, lab) in enumerate(stages):
            v = tidy[(tidy.backbone == bb) & (tidy.method == m) &
                     (tidy.region == "full")].sort_values("seed").F1.values
            x = np.full(len(v), i) + np.linspace(-0.13, 0.13, len(v))
            ax.scatter(x, v, s=22, color="#444444", zorder=3, alpha=0.85)
            ax.hlines(v.mean(), i - 0.28, i + 0.28, color="#d95f02", lw=2.5,
                      zorder=4)
            ax.annotate(f"{v.mean():.3f}", (i, v.mean()),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color="#d95f02")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([s[1] for s in stages], fontsize=9)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(-0.6, len(stages) - 0.4)
    axes[0].set_ylabel("best F1 (full arrays, point-wise)", fontsize=10)
    fig.suptitle("Tier ladder per backbone: every seed (dots) and mean (bar); V2, 6 seeds\n"
                 "(full arrays incl. the S2 training slice -- secondary convention; "
                 "confirmatory comparison = held-out, Fig. F2)", fontsize=10, y=1.06)
    fig.tight_layout()
    fig.savefig(os.path.join(F1DIR, "F1_ladder.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def fig_heldout_deltas(tidy):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (bb, name) in zip(axes, BBS):
        for j, s in enumerate(SEEDS):
            d = {}
            for rg in ["full", "heldout"]:
                g = tidy[(tidy.backbone == bb) & (tidy.seed == s) &
                         (tidy.region == rg)]
                d[rg] = (float(g[g.method == "S2_GBM"].F1.iloc[0]) -
                         float(g[g.method == "M0_residual"].F1.iloc[0]))
            ax.plot([0, 1], [d["full"], d["heldout"]], "-o", ms=4,
                    color="#555555", alpha=0.7, lw=1)
        means = []
        for k, rg in enumerate(["full", "heldout"]):
            g = tidy[(tidy.backbone == bb) & (tidy.region == rg)]
            dm = (g[g.method == "S2_GBM"].sort_values("seed").F1.values -
                  g[g.method == "M0_residual"].sort_values("seed").F1.values)
            means.append(dm.mean())
            ax.hlines(dm.mean(), k - 0.18, k + 0.18, color="#d95f02", lw=3,
                      zorder=4)
        ax.axhline(0, color="#999999", lw=0.8, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["full arrays\n(incl. stacker slice)",
                            "held-out\n(leak-free)"], fontsize=8.5)
        ax.set_title(f"{name}  (mean {means[0]:+.3f} -> {means[1]:+.3f})",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(-0.4, 1.4)
    axes[0].set_ylabel("F1(S2) - F1(M0) per seed", fontsize=10)
    fig.suptitle("Does the stacker gain survive leak-free evaluation? "
                 "Paired per-seed deltas, V2", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(F1DIR, "F2_heldout_deltas.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def fig_verdict_grid(t):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    rows, labels = [], []
    for bb, name in BBS:
        for s in SEEDS:
            g = t[(t.backbone == bb) & (t.seed == s)]
            rows.append({v: int((g.verdict == v).sum()) for v in VORD})
            labels.append(f"{name} s{s}")
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for v in VORD:
        w = np.array([r[v] for r in rows], float)
        ax.barh(y, w, left=left, color=VCOL[v], edgecolor="white", height=0.78,
                label=v.replace("_", " "))
        left += w
    for k in [6, 12]:
        ax.axhline(len(rows) - k - 0.5, color="k", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("events (35 GDN/TopoGDN, 36 CST-GL)", fontsize=9)
    ax.legend(ncol=2, fontsize=7.5, loc="lower right", framealpha=0.95)
    ax.set_title("Event verdict distribution per combo (C-slice thresholds, "
                 "all events)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(F2DIR, "G1_verdict_grid.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def fig_triage_pareto(asum):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pcol = {"P0 all alarms": "#888888", "P1 drop R2": "#1b9e77",
            "P2 drop blips <3 unless R4": "#7570b3",
            "P3 = P1 + P2": "#d95f02", "P4 keep R1/R4 only": "#e7298a"}
    agg = {}
    for k, s in asum.items():
        if not k.endswith("_gbm") or k.startswith("dualstage"):
            continue
        for p in s["policies"]:
            x = p["fa_suppressed_pct"]
            yv = p["events_lost"]
            ax.scatter(x, yv, s=26, color=pcol[p["policy"]], alpha=0.4)
            agg.setdefault(p["policy"], []).append((x, yv))
    for j, (pol, pts) in enumerate(agg.items()):
        xs, ys = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
        ax.scatter(xs + (j - 2) * 1.6, ys, s=200, facecolors="none",
                   edgecolors=pcol[pol], linewidths=2.4, zorder=5,
                   label=f"{pol} (mean)")
    ax.set_xlabel("false-alarm episodes suppressed (%)", fontsize=10)
    ax.set_ylabel("attack events lost", fontsize=10)
    ax.set_title("Verdict/duration suppression policies, gbm source "
                 "(18 combo points per policy + mean ring, means dodged)\n"
                 "descriptive -- not an operating-point recommendation; "
                 "the claim-bearing result is the H3 ordering test",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(F2DIR, "G2_triage_pareto.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def fig_deployment_table(t, canon):
    th = t[t.in_heldout == 1]
    cols = ["attack", "official point", "det m0 k/6\n(G/T/C)",
            "modal verdict (CST-GL)", "modal peak sensor", "on-target"]
    cell_rows = []
    shade = []
    for aid in sorted(th.attack_id.unique()):
        g = th[th.attack_id == aid]
        det = "/".join(str(int(g[g.backbone == b].detected.sum()))
                       for b, _ in BBS)
        gc = g[g.backbone == "cstgl"]
        vs, cnt = np.unique(list(gc.verdict), return_counts=True)
        mx = cnt.max()
        tied = sorted(str(v) for v, c in zip(vs, cnt) if c == mx)
        mv = tied[0].replace("_", " ") + (" (tie)" if len(tied) > 1 else "")
        ps, pc = np.unique(list(gc.peak_sensor), return_counts=True)
        sens = str(ps[np.argmax(pc)])
        pts = {x.replace("-", "").upper()
               for x in (canon.get(aid, {}).get("points") or [])}
        hit = "yes" if sens.replace("-", "").upper() in pts else "no"
        cell_rows.append([f"A{aid}", canon.get(aid, {}).get("attack_point_raw", "-"),
                          det, mv, sens, hit])
        shade.append(aid == 29)
    fig, ax = plt.subplots(figsize=(9.8, 0.42 * len(cell_rows) + 1.2))
    ax.axis("off")
    tab = ax.table(cellText=cell_rows, colLabels=cols, loc="center",
                   cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.35)
    for j in range(len(cols)):
        tab[0, j].set_facecolor("#dddddd")
        tab[0, j].set_text_props(weight="bold")
    for i, sh in enumerate(shade):
        if sh:
            for j in range(len(cols)):
                tab[i + 1, j].set_facecolor("#f3e8ff")
    ax.set_title("The 13 held-out attacks: detection/triage exhibit. "
                 "A29 shaded: pre-declared correct-quiet (official mechanical-"
                 "interlock record; counts toward no recall figure).\n"
                 "'on-target' tests whether the peak sensor is the MANIPULATED "
                 "point (localization), independent of detection success.",
                 fontsize=9.5, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(F2DIR, "G3_deployment_table.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def fig_fingerprints(t):
    ev = t.groupby("attack_id").agg(
        bucket=("bucket", "first") if "bucket" in t else ("category", "first"),
        R=("zpeak_R", "median"), A=("zpeak_A", "median"),
        E=("zpeak_E", "median"), O=("zpeak_O", "median"),
        held=("in_heldout", "max")).reset_index()
    order = {"physical": 0, "deception_only": 1, "deception_with_effect": 2}
    ev["ord"] = ev.bucket.map(order).fillna(3)
    ev = ev.sort_values(["ord", "attack_id"]).reset_index(drop=True)
    M = np.log10(1 + np.maximum(ev[["R", "A", "E", "O"]].values, 0))
    fig, ax = plt.subplots(figsize=(5.4, 8.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["residual", "aleatoric", "epistemic", "Omega"],
                       fontsize=8.5)
    ax.set_yticks(range(len(ev)))
    ax.set_yticklabels([f"A{a}{'*' if h else ''}" for a, h in
                        zip(ev.attack_id, ev.held)], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if M[i, j] < 0.7 * M.max() else "black")
    prev = None
    for i, b in enumerate(ev.bucket):
        if b != prev:
            ax.axhline(i - 0.5, color="white", lw=1.5)
            ax.text(3.62, i, b.replace("_", " "), fontsize=8.5, rotation=90,
                    va="top", color="#333333")
            prev = b
    fig.colorbar(im, ax=ax, shrink=0.6, label="log10(1 + median peak z)")
    ax.set_title("Per-event channel fingerprints (median over 18 combos);\n"
                 "* = held-out attack. Descriptive: the mechanism-bucket\n"
                 "axis showed no significant channel separation (H2)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(F2DIR, "G4_fingerprints.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(F1DIR, exist_ok=True)
    os.makedirs(F2DIR, exist_ok=True)
    tidy = pd.read_csv(f"{ROOT}/results/thesis_part1/fusion_V2_6seed_tidy.csv")
    t = pd.read_csv(f"{ROOT}/results/typing_v1v2/typing_events.csv")
    t = t[t.backbone.isin([b for b, _ in BBS])].copy()
    t["deception"] = t.spoof_gt.astype(int)
    THIRD = {3, 6, 7, 8, 10, 11, 32, 33, 40}
    t["bucket"] = np.where(t.deception == 0, "physical",
                   np.where(t.attack_id.isin(THIRD),
                            "deception_with_effect", "deception_only"))
    asum = json.load(open(f"{ROOT}/results/typing_v1v2/alarm_triage_summary.json"))
    canon = {a["attack_id"]: a for a in json.load(
        open(f"{ROOT}/data/swat/attack_gt_canonical.json"))["attacks"]}

    fig_ladder(tidy)
    fig_heldout_deltas(tidy)
    fig_verdict_grid(t)
    fig_triage_pareto(asum)
    fig_deployment_table(t, canon)
    fig_fingerprints(t)

    cap1 = """# Part 1 figure captions (claims-ladder compliant)

F1_ladder.png -- Tier ladder per backbone (V2, 6 seeds, full arrays): plain
paper-style baseline -> anchored M0 -> S2 stacker fusion. Dots = individual
seeds; orange bar = mean. Reading: instrumentation is approximately
accuracy-neutral on the stable backbone (GDN) and stabilizing on the
unstable ones; fusion lifts the weak backbones. Full-arrays numbers include
the stacker's training slice for S2 (secondary convention; the confirmatory
comparison is held-out, Figure F2).

F2_heldout_deltas.png -- Per-seed S2-minus-M0 deltas, full arrays vs the
held-out region (rows >= 24530, 13 unseen attacks; tau swept within
region). Reading: the stacker gain survives leak-free evaluation on
TopoGDN/CST-GL and is parity on GDN; pooled held-out S2 vs M0 = +0.068,
15/18 seeds, exact sign p = 0.0038 (PART1_STATS.md Section 3).

F3 (PA%K curves) -- generated by p3_pak_figure.py once the curve fleet
completes; mean F1-vs-K with seed band, M0 vs S2 per backbone.
"""
    open(os.path.join(F1DIR, "captions.md"), "w").write(cap1)

    cap2 = """# Part 2 figure captions (claims-ladder compliant)

G1_verdict_grid.png -- Event-verdict distribution for every combo (3
backbones x 6 seeds, all events, deployable C-slice thresholds). Stability
context: modal-verdict share 0.64-0.71 (H4, descriptive).

G2_triage_pareto.png -- Suppression policies on the gbm alarm stream:
percent of false episodes suppressed vs attack events lost; one point per
combo, large marker = policy mean. Descriptive utility view; FA/day
figures are indicative only. The claim-bearing H3 result is the
verdict-ordering test (corroborated 0.271 vs quiet 0.920 held-out false
rate, violation p ~ 0.000), not any single policy point.

G3_deployment_table.png -- The 13 held-out attacks: per-backbone residual
detections (k of 6 seeds), modal CST-GL verdict and peak sensor, and
whether the peak sensor is an official attack point (hit@1 = 7/13;
intent-equipment hit = 0/13 -- localization tracks the manipulated point,
not the attacker's goal). A29 (shaded) is the pre-declared correct-quiet:
the official record states the dosing pumps never started (mechanical
interlock), so no process-data detector can see it; it counts toward no
recall figure. This exhibit is never mechanism-scored.

G4_fingerprints.png -- Per-event channel fingerprints (median peak robust-z
over 18 combos, log scale), events grouped by the canonical mechanism
buckets. Descriptive only: H2 found no significant channel separation
across these buckets (KW p >= 0.38); the figure documents the raw material
the verdicts consume, not a classification claim.
"""
    open(os.path.join(F2DIR, "captions.md"), "w").write(cap2)
    print("wrote figures:",
          sorted(os.listdir(F1DIR)), sorted(os.listdir(F2DIR)))


if __name__ == "__main__":
    main()
