#!/usr/bin/env python3
"""Impact figures for the explainability chapter (docs/IMPACT_FIGURES_PLAN.md).

Verified plan (3-agent review, 2026-06-04). Body figures F1, F3, F4, F5, F6;
appendix F2. Channels are drawn as z OVER THRESHOLD RATIO with one detection
line at 1. Omega recolored #6a3d9a (red-purple confusion fix); channel palette
used only for channels. All GBM panels for events inside the stacker training
slice (arrays 15593 to 24530) carry an in-sample label. No em dashes anywhere.

Renders to results/typing_v1v2/figs_impact/. Runs in rashindaNew-torch-env.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
from typing_rules_v1v2 import load_combo, load_attack_table  # noqa: E402
from analyze_multistage_attacks import estimate_offset  # noqa: E402

OUT = os.path.join(ROOT, "results/typing_v1v2/figs_impact")
GBM = os.path.join(ROOT, "results/typing_v1v2/gbm")
EVCSV = os.path.join(ROOT, "results/baseline_v1v2/attack_event_analysis.csv")
EPCSV = os.path.join(ROOT, "results/typing_v1v2/alarm_triage_episodes.csv")
VAL = (15593, 24530)                       # stacker training slice (array coords)
STEP_S = 10                                # seconds per array step

C = {"R": "#d62728", "A": "#ff7f0e", "E": "#1f77b4", "O": "#6a3d9a"}
LBL = {"R": "Residual", "A": "Aleatoric", "E": "Epistemic", "O": "Omega"}
CAT = {"residual": "#7fbf7b", "uq_only": "#fdb863", "missed": "#cccccc"}
TRUTH = {"true": "#1b9e77", "false": "#e41a1c", "transient": "#8c8c8c"}

_CTX = {}
def ctx(bb, V, seed):
    k = (bb, V, seed)
    if k not in _CTX:
        _CTX[k] = load_combo(bb, V, seed)
    return _CTX[k]

ATTS = load_attack_table()
ATT = {a["aid"]: a for a in ATTS}

_DOFF = {}
def win(bb, V, seed, aid):
    c = ctx(bb, V, seed)
    k = (bb, V, seed)
    if k not in _DOFF:
        _DOFF[k] = estimate_offset(c["lab"], ATTS)
    a = ATT[aid]
    return max(0, a["s"] + _DOFF[k]), min(c["T"], a["e"] + _DOFF[k])

def ratio(c, ch, sl):
    return c[ch][sl] / c["thr"][ch]

def inval_frac(s, e):
    ov = max(0, min(e, VAL[1]) - max(s, VAL[0]))
    return ov / max(e - s, 1)

TRAIN_STATS = None
def train_stats():
    global TRAIN_STATS
    if TRAIN_STATS is None:
        df = pd.read_csv(os.path.join(ROOT, "data/swat/train.csv"), index_col=0)
        TRAIN_STATS = df.mean(), df.std() + 1e-8
    return TRAIN_STATS

TEST_DF = None
def test_df():
    global TEST_DF
    if TEST_DF is None:
        TEST_DF = pd.read_csv(os.path.join(ROOT, "data/swat/test.csv"), index_col=0)
    return TEST_DF

def csv_off(bb):
    return 60 if bb in ("gdn", "topogdn") else 0

def gbm_events(tag):
    p = os.path.join(GBM, f"explain_{tag}.json")
    return json.load(open(p))["events"] if os.path.exists(p) else {}


# ---------------------------------------------------------------- F1
def f1_anatomy():
    cols = [("A22 (easy attack)", "cstgl", 42, 22),
            ("A27 on CST-GL (full miss)", "cstgl", 42, 27),
            ("A27 on GDN (latency rescue)", "gdn", 42, 27)]
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 8.2), sharex="col")
    for j, (title, bb, seed, aid) in enumerate(cols):
        c = ctx(bb, "V2", seed)
        s, e = win(bb, "V2", seed, aid)
        m = max(60, (e - s) // 3)
        sl = slice(max(0, s - m), min(c["T"], e + m))
        t = (np.arange(sl.start, sl.stop) - s) * STEP_S / 60.0   # minutes from onset
        wspan = (0, (e - s) * STEP_S / 60.0)

        # neighbouring attack windows inside the view (light), focal (dark);
        # spans clipped to the view and x-limits pinned so distant attacks
        # cannot stretch the axis
        view = (float(t[0]), float(t[-1]))
        spans = []
        for a2 in ATT.values():
            s2_, e2_ = win(bb, "V2", seed, a2["aid"])
            lo = (s2_ - s) * STEP_S / 60.0
            hi = (e2_ - s) * STEP_S / 60.0
            if a2["aid"] != aid and hi > view[0] and lo < view[1]:
                spans.append((max(lo, view[0]), min(hi, view[1])))

        def shade(ax):
            for sp in spans:
                ax.axvspan(*sp, color="#fdae61", alpha=0.15)
            ax.axvspan(*wspan, color="k", alpha=0.10)
            ax.set_xlim(*view)

        # (a) raw target sensors, z-scored vs train stats (scale floored so
        # near-constant actuators do not explode the axis)
        ax = axes[0, j]
        mu, sd = train_stats()
        df = test_df()
        co = csv_off(bb)
        for tg in ATT[aid]["targets"][:3]:
            if tg in df.columns:
                col = df[tg].to_numpy()
                seg = col[sl.start + co: sl.stop + co]
                scale = max(float(sd[tg]), 0.05 * (col.max() - col.min()), 1e-3)
                ax.plot(t[:len(seg)], (seg - mu[tg]) / scale, lw=1.0, label=tg)
        shade(ax)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_title(title, fontsize=9)
        if j == 0:
            ax.set_ylabel("sensor z\n(train stats, floored scale)", fontsize=7)

        # (b) residual ratio / (c) Omega + epistemic ratios
        for row, chans in ((1, ["R"]), (2, ["O", "E"])):
            ax = axes[row, j]
            for ch in chans:
                style = dict(lw=1.3) if ch != "E" else dict(lw=1.0, ls="--")
                ax.plot(t, ratio(c, ch, sl), color=C[ch], label=LBL[ch], **style)
            ax.axhline(1.0, color="k", lw=1.0)
            shade(ax)
            ax.set_yscale("symlog", linthresh=0.5, linscale=0.6)
            ax.legend(fontsize=6, loc="upper right")
            if j == 0:
                ax.set_ylabel("z / threshold", fontsize=8)
            # first-crossing latency annotation
            for ch in chans:
                rr = ratio(c, ch, slice(s, e))
                hits = np.nonzero(rr > 1.0)[0]
                if hits.size:
                    lat_s = hits[0] * STEP_S
                    lat = f"+{lat_s:.0f}s" if lat_s < 120 else f"+{lat_s/60:.0f}min"
                    ax.annotate(f"{LBL[ch]} {lat}",
                                xy=(hits[0] * STEP_S / 60.0, 1.0),
                                xytext=(hits[0] * STEP_S / 60.0, 3.0),
                                fontsize=6, color=C[ch],
                                arrowprops=dict(arrowstyle="-", color=C[ch], lw=0.6))

        # (d) GBM log-odds + alarm ticks
        ax = axes[3, j]
        tag = f"{bb}_V2_s{seed}"
        ev = gbm_events(tag).get(f"A{aid:02d}") or gbm_events(tag).get(str(aid))
        if ev:
            lo = np.asarray(ev["logodds_series"], float)
            tw = np.arange(len(lo)) * STEP_S / 60.0
            ax.plot(tw, lo, color="#2ca02c", lw=1.2, label="S2 GBM log-odds")
            ax.axhline(0.0, color="k", ls=":", lw=0.8)
            gt_ticks = tw[lo > 0]
            ax.plot(gt_ticks, np.full_like(gt_ticks, lo.min() - 0.5), "|",
                    color="#2ca02c", ms=4, label="GBM alarm")
        rr = ratio(c, "R", slice(s, e))
        m0_ticks = (np.nonzero(rr > 1.0)[0]) * STEP_S / 60.0
        base = (lo.min() - 1.5) if ev else 0.0
        ax.plot(m0_ticks, np.full_like(m0_ticks, base), "|", color=C["R"],
                ms=4, label="M0 alarm")
        iv = inval_frac(s, e)
        if ev and iv > 0:
            ax.text(0.02, 0.04, f"in-sample fold ({iv:.0%} of window in stacker train slice)",
                    transform=ax.transAxes, fontsize=6, style="italic", color="#444444")
        shade(ax)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("minutes from attack onset", fontsize=8)
        if j == 0:
            ax.set_ylabel("GBM log-odds", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "F1_anatomy_A27.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- F3
def f3_coverage():
    rows = [r for r in csv.DictReader(open(EVCSV)) if r["variant"] == "V2"]
    aids = sorted({int(r["attack_id"].lstrip("A")) for r in rows}, reverse=True)
    bbs = ["gdn", "topogdn", "cstgl"]
    det = lambda r, k: (r[k] not in ("", "nan")) and int(float(r[k])) == 1
    cat = np.full((len(aids), 3), "", dtype=object)
    marg_m0 = defaultdict(int); marg_uq = defaultdict(int)
    allruns = list(csv.DictReader(open(EVCSV)))           # both variants, never-fires check
    never = {a for a in aids
             if not any(det(r, "m0_detected") or any(det(r, f"det_{c}") for c in
                        ("upar", "sig2", "omega", "ustr"))
                        for r in allruns if int(r["attack_id"].lstrip("A")) == a)}
    for i, a in enumerate(aids):
        for j, bb in enumerate(bbs):
            sub = [r for r in rows if r["backbone"] == bb
                   and int(r["attack_id"].lstrip("A")) == a]
            if not sub:
                cat[i, j] = "absent"; continue
            m0 = any(det(r, "m0_detected") for r in sub)
            uq = any(any(det(r, f"det_{c}") for c in ("upar", "sig2", "omega", "ustr"))
                     for r in sub)
            cat[i, j] = "residual" if m0 else ("uq_only" if uq else "missed")
            marg_m0[a] += sum(det(r, "m0_detected") for r in sub)
            marg_uq[a] += sum(any(det(r, f"det_{c}") for c in
                                  ("upar", "sig2", "omega", "ustr")) for r in sub)

    fig = plt.figure(figsize=(7.2, 10.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.4], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    cmapv = {"residual": 0, "uq_only": 1, "missed": 2, "absent": 3}
    colors = [CAT["residual"], CAT["uq_only"], CAT["missed"], "#ffffff"]
    img = np.array([[cmapv[cat[i, j]] for j in range(3)] for i in range(len(aids))])
    ax.imshow(img, cmap=matplotlib.colors.ListedColormap(colors), aspect="auto",
              vmin=0, vmax=3)
    for i, a in enumerate(aids):
        if a in never:
            for j in range(3):
                ax.text(j, i, "x", ha="center", va="center", fontsize=7, color="k")
    ax.set_xticks(range(3), ["GDN", "TopoGDN", "CST-GL"], fontsize=8)
    ax.set_yticks(range(len(aids)), [f"A{a:02d}" for a in aids], fontsize=6)
    ax.set_title("Coverage map (V2, any-seed rule)", fontsize=10)
    i10 = aids.index(10)
    ax.annotate("A10: residual dominates UQ\n(26/36 vs 4/36 runs)",
                xy=(-0.45, i10), xytext=(0.0, i10 - 4.5), fontsize=6.5,
                ha="left", arrowprops=dict(arrowstyle="->", lw=0.7))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT[k]) for k in
               ("residual", "uq_only", "missed")]
    ax.legend(handles, ["residual-detected", "UQ-channel-only", "missed by all"],
              fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=3)
    axm = fig.add_subplot(gs[0, 1], sharey=ax)
    y = np.arange(len(aids))
    axm.barh(y - 0.18, [marg_m0[a] for a in aids], height=0.36,
             color=CAT["residual"], label="residual (of 18)")
    axm.barh(y + 0.18, [marg_uq[a] for a in aids], height=0.36,
             color=CAT["uq_only"], label="any UQ (of 18)")
    axm.set_xlim(0, 18); axm.invert_yaxis()
    axm.tick_params(labelleft=False)
    axm.legend(fontsize=6, loc="lower right")
    axm.set_xlabel("seed-backbone cells fired", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "F3_coverage_map.png"), dpi=200)
    plt.close(fig)
    return never


# ---------------------------------------------------------------- F4
def f4_clean_rescue():
    fig = plt.figure(figsize=(8.6, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.3], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for seed, style in ((3, dict(lw=1.4, label="seed 3 (fires)")),
                        (42, dict(lw=1.2, ls="--", label="seed 42 (quiet)"))):
        c = ctx("cstgl", "V2", seed)
        s, e = win("cstgl", "V2", seed, 23)
        m = 60
        sl = slice(max(0, s - m), min(c["T"], e + m))
        t = (np.arange(sl.start, sl.stop) - s) * STEP_S / 60.0
        ax.plot(t, ratio(c, "O", sl), color=C["O"], **style)
        ax.axvspan(0, (e - s) * STEP_S / 60.0, color="k", alpha=0.08)
    ax.axhline(1.0, color="k", lw=1.0)
    ax.set_yscale("symlog", linthresh=0.5, linscale=0.6)
    ax.set_xlabel("minutes from attack onset", fontsize=8)
    ax.set_ylabel("Omega z / threshold", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("A23 on CST-GL V2: the held-out Omega rescue", fontsize=9)

    axr = fig.add_subplot(gs[0, 1])
    seeds = [0, 1, 2, 3, 4, 42]
    ratios = []
    for sd in seeds:
        c = ctx("cstgl", "V2", sd)
        s, e = win("cstgl", "V2", sd, 23)
        ratios.append(float(c["O"][s:e].max() / c["thr"]["O"]))
    x = np.arange(len(seeds))
    for xi, r in zip(x, ratios):
        ax_c = C["O"] if r > 1 else "#aaaaaa"
        axr.plot([xi, xi], [0, r], color=ax_c, lw=1.6)
        axr.plot(xi, r, "o", color=ax_c, ms=5)
    axr.axhline(1.0, color="k", lw=1.0)
    axr.set_xticks(x, [f"s{sd}" for sd in seeds], fontsize=7)
    axr.set_ylabel("peak Omega ratio", fontsize=8)
    fired = sum(r > 1 for r in ratios)
    axr.set_title(f"fires in {fired} of 6 seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "F4_clean_rescue_A23.png"), dpi=200)
    plt.close(fig)
    return ratios


# ---------------------------------------------------------------- F5
def f5_fusion_impact():
    from sklearn.metrics import precision_recall_curve
    rows = [r for r in csv.DictReader(open(EVCSV)) if r["variant"] == "V2"]
    det = lambda r, k: (r[k] not in ("", "nan")) and int(float(r[k])) == 1
    bbs = ["gdn", "topogdn", "cstgl"]
    m0_counts, r4b_counts = {}, {}
    for bb in bbs:
        per_seed_m0, per_seed_r4b = [], []
        for sd in (0, 1, 2, 3, 4, 42):
            sub = [r for r in rows if r["backbone"] == bb and r["seed"] == str(sd)]
            m0 = sum(det(r, "m0_detected") for r in sub)
            r4b = sum((not det(r, "m0_detected")) and det(r, "det_omega") for r in sub)
            per_seed_m0.append(m0); per_seed_r4b.append(m0 + r4b)
        m0_counts[bb] = per_seed_m0; r4b_counts[bb] = per_seed_r4b

    def events_at_best_f1(y, s, lo=None, hi=None):
        sl = slice(lo, hi)
        yy, ss = y[sl], s[sl]
        p, r, t = precision_recall_curve(yy, ss)
        f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
        tau = t[int(np.nanargmax(f1[:-1]))]
        d = np.diff(np.concatenate([[0], yy.astype(int), [0]]))
        ev = list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
        return sum(bool((ss[a:b] >= tau).any()) for a, b in ev), len(ev)

    s2 = defaultdict(list); s2_h = defaultdict(list); m0_h = defaultdict(list)
    for tag, bb in (("gdn_V2_s42", "gdn"), ("topogdn_V2_s42", "topogdn"),
                    ("cstgl_V2_s42", "cstgl"), ("cstgl_V2_s3", "cstgl")):
        d = np.load(os.path.join(GBM, f"{tag}_cache.npz"))
        y, sg, sm = d["label"].astype(int), d["score"], d["feat"][:, 0]
        s2[bb].append(events_at_best_f1(y, sg)[0])
        s2_h[bb].append(events_at_best_f1(y, sg, VAL[1], None)[0])
        m0_h[bb].append(events_at_best_f1(y, sm, VAL[1], None)[0])

    # ALSO compute M0 events at best-F1 from the caches so the right panel
    # compares M0 vs S2 under ONE rule (mixing operating points between bars
    # would mislead; the Q0.995 deployable convention gets its own panel).
    m0_f = defaultdict(list)
    for tag, bb in (("gdn_V2_s42", "gdn"), ("topogdn_V2_s42", "topogdn"),
                    ("cstgl_V2_s42", "cstgl"), ("cstgl_V2_s3", "cstgl")):
        d = np.load(os.path.join(GBM, f"{tag}_cache.npz"))
        y, sm = d["label"].astype(int), d["feat"][:, 0]
        m0_f[bb].append(events_at_best_f1(y, sm)[0])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), width_ratios=[1.25, 1.75])
    shades = ["#bdbdbd", "#74a9cf", "#0570b0"]

    # left: deployable Q0.995 convention, 6 seeds, residual vs +Omega escalation
    ax = axes[0]
    x = np.arange(3)
    for k, (vals, lab) in enumerate(((m0_counts, "M0 residual alarm"),
                                     (r4b_counts, "+ Omega escalation (R4b)"))):
        means = [np.mean(vals[b]) for b in bbs]
        errs = [np.std(vals[b]) / np.sqrt(6) for b in bbs]
        ax.bar(x + (k - 0.5) * 0.36, means, 0.36, yerr=errs, capsize=2,
               color=shades[k], label=lab)
    ax.set_xticks(x, ["GDN", "TopoGDN", "CST-GL"], fontsize=8)
    ax.set_ylabel("events flagged (of 35-36)", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Deployable convention (Q0.995)\n6 seeds, mean +/- se", fontsize=8)

    # right: best-F1 convention from the cached combos, full vs held-out,
    # M0 and S2 under the SAME rule
    axh = axes[1]
    tags = ["gdn s42", "topogdn s42", "cstgl s42", "cstgl s3"]
    vals = {
        "M0 full": [m0_f["gdn"][0], m0_f["topogdn"][0], m0_f["cstgl"][0], m0_f["cstgl"][1]],
        "S2 full": [s2["gdn"][0], s2["topogdn"][0], s2["cstgl"][0], s2["cstgl"][1]],
        "M0 held-out": [m0_h["gdn"][0], m0_h["topogdn"][0], m0_h["cstgl"][0], m0_h["cstgl"][1]],
        "S2 held-out": [s2_h["gdn"][0], s2_h["topogdn"][0], s2_h["cstgl"][0], s2_h["cstgl"][1]],
    }
    xh = np.arange(4)
    for k, (lab, v) in enumerate(vals.items()):
        hatch = "//" if "held" in lab else None
        col = shades[0] if lab.startswith("M0") else shades[2]
        axh.bar(xh + (k - 1.5) * 0.2, v, 0.2, color=col, hatch=hatch,
                edgecolor="white", label=lab)
    axh.set_xticks(xh, tags, fontsize=7)
    axh.set_ylabel("events caught", fontsize=8)
    axh.legend(fontsize=6.5, ncol=2)
    axh.set_title("Best-F1 convention, one rule for both scores\n"
                  "(full 34-35 events; held-out 14 events; n=1 seed per bar)",
                  fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "F5_fusion_impact.png"), dpi=200)
    plt.close(fig)
    return {b: (np.mean(m0_counts[b]), np.mean(r4b_counts[b]),
                m0_f[b], s2[b], m0_h[b], s2_h[b]) for b in bbs}


# ---------------------------------------------------------------- F6
def f6_trustworthy():
    rows = list(csv.DictReader(open(EPCSV)))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fam = lambda v: ("corr" if v.startswith(("R1", "R4")) else
                     ("quiet" if v in ("normal_quiet", "R5_benign_noise", "R6_data_gap")
                      else "other"))
    mk = {"corr": "o", "quiet": "s", "other": "^"}
    fill = {"corr": True, "quiet": False, "other": False}
    for r in rows:
        x = max(int(r["dur"]), 1) * STEP_S
        p = min(float(r["peak_pctl"]), 99.9999)
        y = -np.log10(100.0 - p)
        f = fam(r["verdict"])
        col = TRUTH[r["truth"]]
        ax.scatter(x, y, s=22 if f == "corr" else 16, marker=mk[f],
                   facecolors=col if fill[f] else "none", edgecolors=col,
                   linewidths=0.9, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("episode duration (s, log)", fontsize=9)
    ax.set_ylabel("alarm strength: nines of peak percentile\n"
                  r"$-\log_{10}(100 - \mathrm{pctl})$", fontsize=8)
    n_r4_gbm = sum(1 for r in rows if r["source"] == "gbm"
                   and r["verdict"].startswith("R4"))
    n_r4_gbm_false = sum(1 for r in rows if r["source"] == "gbm"
                         and r["verdict"].startswith("R4") and r["truth"] == "false")
    ax.text(0.02, 0.97, f"R4-typed GBM-stage alarms: {n_r4_gbm_false} false of {n_r4_gbm}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", ec="#888888", lw=0.5))
    from matplotlib.lines import Line2D
    leg1 = [Line2D([], [], marker="o", ls="", mfc=TRUTH[t], mec=TRUTH[t], label=t)
            for t in ("true", "transient", "false")]
    leg2 = [Line2D([], [], marker=mk[f], ls="", mfc=("k" if fill[f] else "none"),
                   mec="k", label=l) for f, l in
            (("corr", "corroborated (R1/R4/R4b)"), ("quiet", "channel-quiet"),
             ("other", "R2/R3"))]
    ax.legend(handles=leg1 + leg2, fontsize=7, loc="lower right", ncol=2)
    ax.set_title("All 267 alarm episodes: corroborated verdicts are trustworthy",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "F6_trustworthy_alarms.png"), dpi=200)
    plt.close(fig)
    return n_r4_gbm, n_r4_gbm_false


# ---------------------------------------------------------------- F2 (appendix)
def f2_gallery():
    rows = [r for r in csv.DictReader(open(EVCSV)) if r["variant"] == "V2"]
    det = lambda r, k: (r[k] not in ("", "nan")) and int(float(r[k])) == 1
    events = [3, 14, 16, 17, 19, 35]
    fired = {a: sum(any(det(r, f"det_{c}") for c in ("upar", "sig2", "omega", "ustr"))
                    for r in rows if int(r["attack_id"].lstrip("A")) == a)
             for a in events}
    # pick best (combo, channel) by peak ratio across V2 combos
    best = {}
    for a in events:
        cand = []
        for bb in ("gdn", "topogdn", "cstgl"):
            for sd in (0, 1, 2, 3, 4, 42):
                c = ctx(bb, "V2", sd)
                s, e = win(bb, "V2", sd, a)
                if e <= s:
                    continue
                for ch in ("A", "E", "O"):
                    cand.append((float(c[ch][s:e].max() / c["thr"][ch]), bb, sd, ch))
        best[a] = max(cand)
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 7.6))
    for k, a in enumerate(events):
        rmax, bb, sd, ch = best[a]
        ax = axes[k // 2, k % 2]
        c = ctx(bb, "V2", sd)
        s, e = win(bb, "V2", sd, a)
        m = 30
        sl = slice(max(0, s - m), min(c["T"], e + m))
        t = (np.arange(sl.start, sl.stop) - s) * STEP_S / 60.0
        ax.plot(t, ratio(c, "R", sl), color="#888888", lw=1.0, label="Residual")
        ax.plot(t, ratio(c, ch, sl), color=C[ch], lw=1.3, label=LBL[ch])
        ax.axhline(1.0, color="k", lw=0.9)
        ax.axvspan(0, (e - s) * STEP_S / 60.0, color="k", alpha=0.08)
        ax.set_yscale("symlog", linthresh=0.5, linscale=0.6)
        ax.set_title(f"A{a:02d}  ({bb} s{sd}, {LBL[ch]})   fired {fired[a]}/18 cells",
                     fontsize=8)
        ax.legend(fontsize=6)
        if k // 2 == 2:
            ax.set_xlabel("minutes from onset", fontsize=8)
        if k % 2 == 0:
            ax.set_ylabel("z / threshold", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "F2_rescue_gallery.png"), dpi=200)
    plt.close(fig)
    return {a: best[a][:1] + best[a][1:] for a in events}, fired


def main():
    os.makedirs(OUT, exist_ok=True)
    f1_anatomy();                print("F1 done", flush=True)
    never = f3_coverage();       print(f"F3 done (never-fires: {sorted(never)})", flush=True)
    r4 = f4_clean_rescue();      print(f"F4 done (A23 ratios: {[round(r,2) for r in r4]})", flush=True)
    f5 = f5_fusion_impact();     print(f"F5 done {f5}", flush=True)
    n, nf = f6_trustworthy();    print(f"F6 done (R4 gbm: {nf} false of {n})", flush=True)
    b, fired = f2_gallery();     print(f"F2 done (selections: {b}; fired: {fired})", flush=True)
    print(f"figures -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
