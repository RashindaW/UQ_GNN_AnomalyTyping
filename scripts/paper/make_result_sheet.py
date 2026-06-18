#!/usr/bin/env python3
"""Full per-seed result sheet: every backbone x seed x stage.

Stages:
  M0_residual    -- baseline (residual only, no uncertainty)
  L1/L2/L3       -- uncertainty fused with residual (likelihood scores)
  V1             -- variance-weighted residual
  S1/S2          -- learned fusion (logistic / GBM)
Metrics: PA%K-AUC (threshold-robust headline) and best-F1.
All 5 seeds shown explicitly, plus mean +- std.
Also pulls the extended ranking-metric panel (AUROC/AUPRC/VUS/affiliation).
Writes results/paper/RESULT_SHEET.md (pure ASCII).
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FS = os.path.join(ROOT, "results/paper/fusion_study/fusion_study.csv")
PANEL = os.path.join(ROOT, "results/paper/metrics/extended_panel.csv")
OUT = os.path.join(ROOT, "results/paper/RESULT_SHEET.md")

SEEDS_BASE = [1, 2, 3, 42, 100]
SEEDS_NEW = [7, 17, 23, 88, 256]
SEEDS = SEEDS_BASE + SEEDS_NEW   # up to 10; missing cells render as "-"
BB = ["gdn", "cstgl", "gta", "topogdn"]
METHODS = [("M0_residual", "M0 baseline (residual only)"),
           ("L1_stdres", "Residual+UQ: standardized residual"),
           ("L2_NLPD", "Residual+UQ: Gaussian NLPD"),
           ("L3_Maha", "Residual+UQ: predictive Mahalanobis"),
           ("V1_varweighted", "Residual+UQ: variance-weighted"),
           ("S1_logistic", "Fusion: logistic stacker (interpretable)"),
           ("S2_GBM", "Fusion: gradient-boosting stacker")]


def mean_std(v):
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
    return m, sd


def load_fs():
    d = defaultdict(dict)  # (bb,method) -> {seed: (pak,f1)}
    if not os.path.exists(FS):
        return d
    for r in csv.DictReader(open(FS)):
        d[(r["backbone"], r["method"])][int(r["seed"])] = (float(r["PA_K_AUC"]), float(r["F1"]))
    return d


def fmt(x):
    return f"{x:.4f}" if x == x else "  -   "


def main():
    fs = load_fs()
    L = []
    L.append("# Full Per-Seed Result Sheet (SWaT, 5 seeds)\n")
    L.append("Backbones: GDN, CST-GL, GTA, TopoGDN. Metric blocks: PA%K-AUC "
             "(threshold-robust headline) and best-F1. Each stage shown for every "
             "seed {1,2,3,42,100} plus mean +- std.\n")
    L.append("Stages: M0 = residual-only baseline; L1/L2/L3 and V1 = residual fused "
             "with uncertainty channels; S1/S2 = learned fusion (S1 logistic is the "
             "adopted interpretable method, S2 GBM is the black-box reference).\n")

    for metric_idx, metric_name in [(0, "PA%K-AUC (threshold-robust)"), (1, "best-F1")]:
        L.append(f"\n## METRIC: {metric_name}\n")
        for bb in BB:
            # only show seeds this backbone actually has (5 or 10)
            present = [s for s in SEEDS if any(s in fs.get((bb, m), {}) for m, _ in METHODS)]
            L.append(f"\n### {bb.upper()}  (n={len(present)} seeds: {present})\n")
            L.append("| stage | " + " | ".join(f"s{s}" for s in present) + " | mean | std |")
            L.append("|" + "---|" * (len(present) + 3))
            for m, label in METHODS:
                cells = fs.get((bb, m), {})
                vals = [cells[s][metric_idx] for s in present if s in cells]
                row = [fmt(cells[s][metric_idx]) if s in cells else "  -   " for s in present]
                mmv, ssv = mean_std(vals)
                L.append(f"| {label} | " + " | ".join(row) + f" | {fmt(mmv)} | {fmt(ssv)} |")

    # gain summary (PA%K): M0 baseline vs BOTH learned fusions (S1 logistic, S2 GBM)
    def mm(bb, m, idx):
        c = fs.get((bb, m), {})
        return mean_std([c[s][idx] for s in SEEDS if s in c])

    def wilcox(bb, m):
        try:
            from scipy.stats import wilcoxon
            m0c = fs.get((bb, "M0_residual"), {}); mc = fs.get((bb, m), {})
            ss = [s for s in SEEDS if s in m0c and s in mc]
            if len(ss) < 2:
                return float("nan")
            a = [mc[s][0] for s in ss]; b = [m0c[s][0] for s in ss]
            if all(abs(x - y) < 1e-12 for x, y in zip(a, b)):
                return float("nan")
            return float(wilcoxon(a, b).pvalue)
        except Exception:
            return float("nan")

    L.append("\n## HEADLINE GAIN: M0 baseline vs BOTH learned fusions "
             "(S1 logistic, S2 GBM)\n")
    L.append("PA%K-AUC (mean over the seeds each backbone has; GDN has 10, others 5). "
             "'worst' = worst single-seed value (exposes GBM instability). 'p' = "
             "Wilcoxon signed-rank vs M0 (reachable below 0.05 only at n=10).\n")
    L.append("| backbone | n | M0 | S1 logistic | gain | S1 worst | p | "
             "S2 GBM | gain | S2 worst | p |")
    L.append("|" + "---|" * 11)
    for bb in BB:
        n = max(len(fs.get((bb, "M0_residual"), {})), 0)
        m0p = mm(bb, "M0_residual", 0)[0]
        s1m = mm(bb, "S1_logistic", 0)[0]; s2m = mm(bb, "S2_GBM", 0)[0]
        s1c = fs.get((bb, "S1_logistic"), {}); s2c = fs.get((bb, "S2_GBM"), {})
        s1w = min([s1c[s][0] for s in SEEDS if s in s1c], default=float("nan"))
        s2w = min([s2c[s][0] for s in SEEDS if s in s2c], default=float("nan"))
        L.append(f"| {bb.upper()} | {n} | {fmt(m0p)} | {fmt(s1m)} | {s1m-m0p:+.4f} | "
                 f"{fmt(s1w)} | {wilcox(bb,'S1_logistic'):.3f} | {fmt(s2m)} | "
                 f"{s2m-m0p:+.4f} | {fmt(s2w)} | {wilcox(bb,'S2_GBM'):.3f} |")

    L.append("\nbest-F1 (mean +- std over 5 seeds).\n")
    L.append("| backbone | M0 | S1 logistic | gain | S2 GBM | gain |")
    L.append("|" + "---|" * 6)
    for bb in BB:
        m0f = mm(bb, "M0_residual", 1)[0]
        s1f = mm(bb, "S1_logistic", 1)[0]; s2f = mm(bb, "S2_GBM", 1)[0]
        L.append(f"| {bb.upper()} | {fmt(m0f)} | {fmt(s1f)} | {s1f-m0f:+.4f} | "
                 f"{fmt(s2f)} | {s2f-m0f:+.4f} |")

    # extended ranking panel (per seed) for the M0 baseline
    if os.path.exists(PANEL):
        L.append("\n## EXTENDED RANKING METRICS (M0 baseline score, per seed)\n")
        rows = list(csv.DictReader(open(PANEL)))
        cols = ["AUROC", "AUPRC", "VUS_ROC", "VUS_PR", "aff_f1"]
        L.append("| backbone | seed | " + " | ".join(cols) + " |")
        L.append("|" + "---|" * (len(cols) + 2))
        for bb in BB:
            for r in rows:
                if r["backbone"] == bb:
                    L.append(f"| {bb.upper()} | {r['seed']} | " +
                             " | ".join(f"{float(r[c]):.4f}" if r.get(c) else "-" for c in cols) + " |")

    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n\nwrote {OUT}")


if __name__ == "__main__":
    main()
