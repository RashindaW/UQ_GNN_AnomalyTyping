#!/usr/bin/env python3
"""Thesis Part 1 (detection) FINAL analysis.

Protocol (user-locked 2026-06-06): SWaT V2 split only; ALL six trained seeds
{0,1,2,3,4,42}; four backbones GDN / TopoGDN / CST-GL / DualSTGF (dualstage);
one canonical UQ method; 7 scoring methods; two evaluation regions:
  full    = whole test arrays (comparable to campaign sheets; secondary)
  heldout = arrays rows >= 24530, tau-sweep restricted there (leak-free;
            PRIMARY endpoint for the supervised stackers S1/S2)

Assembles the tidy table from:
  results/baseline_v1v2/fusion_v1v2_seedwise.csv          (3bb, V2, full)
  results/baseline_v1v2/fusion_heldout/seed{S}_V2_heldout.csv   (3bb heldout)
  results/baseline_v1v2/fusion_dualstage/seed{S}_V2_{region}.csv (dualstage)
  results/baseline_v1v2/{bb}/V2/seed{S}/eval.json          (plain baselines;
      gdn plain lives under gdn_plain/)
Asserts ALL 360 expected rows exist (336 fusion + 24 plain), then writes
  results/thesis_part1/fusion_V2_6seed_tidy.csv
  results/thesis_part1/PART1_STATS.md
Std convention: sample std (ddof=1) everywhere. Bootstrap: 10k percentile,
fixed rng seed 20260606. Tests: exact one-sided sign test; exact one-sided
Wilcoxon signed-rank; pre-specified family: S2 primary, S1 secondary,
L1-L3/V1 ablations (descriptive).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, "results/thesis_part1")
SEEDS = [0, 1, 2, 3, 4, 42]
METHODS = ["M0_residual", "L1_stdres", "L2_NLPD", "L3_Maha",
           "V1_varweighted", "S1_logistic", "S2_GBM"]
FUSIONS = METHODS[1:]
BBS = [("gdn", "GDN"), ("topogdn", "TopoGDN"), ("cstgl", "CST-GL"),
       ("dualstage", "DualSTGF")]
# Amendment A1 (2026-06-07, docs/PART2_PREREGISTRATION.md): the thesis scope
# is the three published backbones. Filter via env, default = post-amendment
# scope; set UQ_PART1_BACKBONES=gdn,topogdn,cstgl,dualstage to reproduce the
# archived 4-backbone view.
_bb_env = os.environ.get("UQ_PART1_BACKBONES", "gdn,topogdn,cstgl").split(",")
BBS = [(k, n) for k, n in BBS if k in _bb_env]
PLAIN_DIR = {"gdn": "gdn_plain", "topogdn": "topogdn", "cstgl": "cstgl",
             "dualstage": "dualstage"}
RNG = np.random.default_rng(20260606)
N_BOOT = 10000


def f3(x):
    return f"{x:.3f}"


def ms(v):
    v = np.asarray(v, dtype=float)
    return f"{f3(v.mean())} +/- {f3(v.std(ddof=1))}"


def load_tidy():
    rows = []
    camp = pd.read_csv(f"{ROOT}/results/baseline_v1v2/fusion_v1v2_seedwise.csv")
    camp = camp[(camp.variant == "V2") &
                (camp.backbone.isin(["gdn", "topogdn", "cstgl"]))]
    for _, r in camp.iterrows():
        rows.append((r.backbone, int(r.seed), r.method, "full",
                     float(r.F1), float(r.PA_K_AUC)))
    for s in SEEDS:
        d = pd.read_csv(
            f"{ROOT}/results/baseline_v1v2/fusion_heldout/seed{s}_V2_heldout.csv")
        for _, r in d.iterrows():
            rows.append((r.backbone, int(r.seed), r.method, "heldout",
                         float(r.F1), float(r.PA_K_AUC)))
    if any(k == "dualstage" for k, _ in BBS):
        for region in ["full", "heldout"]:
            for s in SEEDS:
                d = pd.read_csv(f"{ROOT}/results/baseline_v1v2/fusion_dualstage/"
                                f"seed{s}_V2_{region}.csv")
                for _, r in d.iterrows():
                    rows.append((r.backbone, int(r.seed), r.method, region,
                                 float(r.F1), float(r.PA_K_AUC)))
    # plain paper-style baselines (full-arrays harness eval)
    for bb, _ in BBS:
        for s in SEEDS:
            p = (f"{ROOT}/results/baseline_v1v2/{PLAIN_DIR[bb]}/V2/seed{s}/"
                 f"eval.json")
            m0 = json.load(open(p))["baseline_M0"]
            rows.append((bb, s, "Plain_paper", "full",
                         float(m0["F1"]), float(m0["PA_K_AUC"])))
    df = pd.DataFrame(rows, columns=["backbone", "seed", "method", "region",
                                     "F1", "PA_K_AUC"])
    df = df.drop_duplicates(["backbone", "seed", "method", "region"])
    # completeness: 4 bb x 6 seeds x (7 methods x 2 regions + 1 plain) = 360
    missing = []
    for bb, _ in BBS:
        for s in SEEDS:
            for m in METHODS:
                for rg in ["full", "heldout"]:
                    if df[(df.backbone == bb) & (df.seed == s) &
                          (df.method == m) & (df.region == rg)].empty:
                        missing.append((bb, s, m, rg))
            if df[(df.backbone == bb) & (df.seed == s) &
                  (df.method == "Plain_paper")].empty:
                missing.append((bb, s, "Plain_paper", "full"))
    if missing:
        raise SystemExit(f"INCOMPLETE: {len(missing)} cells missing, e.g. "
                         f"{missing[:6]}")
    return df


def vec(df, bb, m, rg, col="F1"):
    d = df[(df.backbone == bb) & (df.method == m) & (df.region == rg)]
    d = d.sort_values("seed")
    assert list(d.seed) == SEEDS, (bb, m, rg, list(d.seed))
    return d[col].values.astype(float)


def battery(x, y):
    """Paired tests for x > y over seeds. Returns dict.

    Ties (zero deltas, possible at the CSVs' 4-dp precision) are dropped
    before both tests (Wilcoxon's own zero_method='wilcox' convention and
    the standard sign-test treatment); n_eff reports the post-drop count.
    CI / d_z are computed on the full delta vector including ties.
    """
    delta = x - y
    n = len(delta)
    nz = delta[delta != 0]
    n_eff = len(nz)
    wins = int((nz > 0).sum())
    if n_eff:
        p_sign = stats.binomtest(wins, n_eff, 0.5,
                                 alternative="greater").pvalue
        p_wil = stats.wilcoxon(nz, alternative="greater",
                               method="exact").pvalue
    else:
        p_sign = p_wil = float("nan")
    bs = RNG.choice(delta, (N_BOOT, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sd = delta.std(ddof=1)
    dz = delta.mean() / sd if sd > 0 else float("inf")
    return dict(mean_delta=delta.mean(), ci_lo=lo, ci_hi=hi, wins=wins,
                n=n_eff, ties=n - n_eff, p_sign=p_sign, p_wilcoxon=p_wil,
                d_z=dz)


def fmt_batt(b):
    tie = f" ({b['ties']} tie)" if b["ties"] else ""
    return (f"{b['mean_delta']:+.3f} [{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}] | "
            f"{b['wins']}/{b['n']}{tie} | {b['p_sign']:.4f} | "
            f"{b['p_wilcoxon']:.4f} | {b['d_z']:+.2f}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = load_tidy()
    df.to_csv(f"{OUTDIR}/fusion_V2_6seed_tidy.csv", index=False)
    L = []
    A = L.append
    A("# Thesis Part 1 (detection): final numbers and statistics")
    A("")
    A("Protocol: SWaT V2 (85/15), seeds {0,1,2,3,4,42} = every trained seed,")
    A(f"{len(BBS)} backbones ({', '.join(n for _, n in BBS)}), one canonical "
      f"UQ method, 7 scoring methods, regions:")
    A("full arrays (secondary) and held-out >= 24530 (primary for S1/S2).")
    A("Sample std (ddof=1). Exact one-sided tests; bootstrap 10k, rng 20260606.")
    A(f"Source: results/thesis_part1/fusion_V2_6seed_tidy.csv "
      f"({len(BBS)*len(SEEDS)*15} rows).")
    A("")

    # ---- 1. Ladder (full arrays) ----
    A("## 1. Ladder: plain paper baseline -> +UQ (anchored M0) -> +fusion")
    A("")
    A("Full-arrays region. 'best fusion' = highest 6-seed mean F1 among the")
    A("six fusion methods (chosen per backbone, disclosed as post hoc; the")
    A("pre-specified confirmatory methods are S2 then S1).")
    A("")
    A("| backbone | plain | anchored M0 | S2_GBM | best fusion | best vs plain | best vs M0 |")
    A("|---|---|---|---|---|---|---|")
    for bb, BB in BBS:
        plain = vec(df, bb, "Plain_paper", "full")
        m0 = vec(df, bb, "M0_residual", "full")
        s2 = vec(df, bb, "S2_GBM", "full")
        means = {m: vec(df, bb, m, "full").mean() for m in FUSIONS}
        best = max(means, key=means.get)
        bv = vec(df, bb, best, "full")
        A(f"| {BB} | {ms(plain)} | {ms(m0)} | {ms(s2)} | {best} {ms(bv)} | "
          f"{bv.mean()-plain.mean():+.3f} | {bv.mean()-m0.mean():+.3f} |")
    A("")

    # ---- 2. All methods, both regions ----
    for col, name in [("F1", "best-F1"), ("PA_K_AUC", "PA%K AUC")]:
        for rg in ["full", "heldout"]:
            A(f"## 2.{'F' if col=='F1' else 'P'}.{rg}: {name}, {rg} region "
              f"(mean +/- std over 6 seeds)")
            A("")
            A("| backbone | " + " | ".join(METHODS) + " |")
            A("|---" * (len(METHODS) + 1) + "|")
            for bb, BB in BBS:
                cells = [ms(vec(df, bb, m, rg, col)) for m in METHODS]
                A(f"| {BB} | " + " | ".join(cells) + " |")
            A("")

    # ---- 3. Statistical battery ----
    A("## 3. Statistical battery (paired over seeds, one-sided 'fusion > M0')")
    A("")
    A("Columns: mean dF1 [95% bootstrap CI] | wins | exact sign p | exact")
    A("Wilcoxon p | Cohen's d_z. n=6 floors: sign 6/6 -> p=0.0156; Wilcoxon")
    A("min p=0.0156 (one-sided). S2 = primary hypothesis, S1 = secondary.")
    A("")
    for rg, tag in [("heldout", "HELD-OUT (primary)"),
                    ("full", "FULL ARRAYS (secondary)")]:
        A(f"### {tag}")
        A("")
        A("| backbone | comparison | dF1 [CI] | wins | sign p | Wilcoxon p | d_z |")
        A("|---|---|---|---|---|---|---|")
        pooled = {"S2_GBM": ([], []), "S1_logistic": ([], [])}
        for bb, BB in BBS:
            m0 = vec(df, bb, "M0_residual", rg)
            for m in ["S2_GBM", "S1_logistic"]:
                x = vec(df, bb, m, rg)
                b = battery(x, m0)
                A(f"| {BB} | {m} vs M0 | " + fmt_batt(b).replace(" | ", " | ")
                  + " |")
                pooled[m][0].extend(x)
                pooled[m][1].extend(m0)
        for m in ["S2_GBM", "S1_logistic"]:
            b = battery(np.array(pooled[m][0]), np.array(pooled[m][1]))
            A(f"| POOLED ({len(pooled[m][0])} pairs) | {m} vs M0 | "
              + fmt_batt(b) + " |")
        A("")
    A("Pooled caveat: seeds are nested within backbones; the pooled Wilcoxon")
    A(f"treats the {len(BBS)*len(SEEDS)} (backbone, seed) configurations as "
      f"exchangeable pairs.")
    A("Per-backbone rows carry no such caveat.")
    A("")

    # ---- 4. Ablations (label-free fusions vs M0, descriptive) ----
    A("## 4. Ablations (label-free fusions vs M0, descriptive, both regions)")
    A("")
    A("| backbone | region | L1 dF1 | L2 dF1 | L3 dF1 | V1 dF1 |")
    A("|---|---|---|---|---|---|")
    for bb, BB in BBS:
        for rg in ["full", "heldout"]:
            m0 = vec(df, bb, "M0_residual", rg).mean()
            cells = [f"{vec(df, bb, m, rg).mean()-m0:+.3f}"
                     for m in ["L1_stdres", "L2_NLPD", "L3_Maha",
                               "V1_varweighted"]]
            A(f"| {BB} | {rg} | " + " | ".join(cells) + " |")
    A("")

    # ---- 5. Stability ----
    A("## 5. Stability (sample std of F1 over 6 seeds, full arrays)")
    A("")
    A("| backbone | plain | M0 | L1 | S1 | S2 | min plain seed | min M0 seed |")
    A("|---|---|---|---|---|---|---|---|")
    for bb, BB in BBS:
        plain = vec(df, bb, "Plain_paper", "full")
        m0 = vec(df, bb, "M0_residual", "full")
        cells = [f3(vec(df, bb, m, "full").std(ddof=1))
                 for m in ["M0_residual", "L1_stdres", "S1_logistic",
                           "S2_GBM"]]
        A(f"| {BB} | {f3(plain.std(ddof=1))} | " + " | ".join(cells) +
          f" | {f3(plain.min())} | {f3(m0.min())} |")
    A("")

    # ---- 6. Per-seed appendix ----
    A("## 6. Per-seed appendix (F1; columns = methods)")
    A("")
    for rg in ["full", "heldout"]:
        for bb, BB in BBS:
            A(f"### {BB}, {rg}")
            A("")
            hdr = ["seed"] + (["Plain_paper"] if rg == "full" else []) + METHODS
            A("| " + " | ".join(hdr) + " |")
            A("|---" * len(hdr) + "|")
            for s in SEEDS:
                cells = [str(s)]
                if rg == "full":
                    cells.append(f3(float(df[(df.backbone == bb) &
                                             (df.seed == s) &
                                             (df.method == "Plain_paper")]
                                          .F1.iloc[0])))
                for m in METHODS:
                    cells.append(f3(float(df[(df.backbone == bb) &
                                             (df.seed == s) &
                                             (df.method == m) &
                                             (df.region == rg)].F1.iloc[0])))
                A("| " + " | ".join(cells) + " |")
            A("")

    open(f"{OUTDIR}/PART1_STATS.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {OUTDIR}/PART1_STATS.md and fusion_V2_6seed_tidy.csv "
          f"({len(df)} rows)")


if __name__ == "__main__":
    main()
