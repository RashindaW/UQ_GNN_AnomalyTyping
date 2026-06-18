#!/usr/bin/env python3
"""Publication-grade per-event typing cards (redesign of figs/event_*.png).

Fixes over the diagnostic originals: x-axis in MINUTES (the originals labeled
steps as seconds), channels drawn as z over threshold with ONE detection line
at 1 (replacing four dashed threshold lines), focal + neighbouring attack
shading, colorblind-safe palette with linestyle redundancy (Omega #6a3d9a),
plain-language verdict banner with confidence and peak signature, and
single-column sizing. Channels only: the stacker's behaviour has its own
figures in the chapter.

Outputs to results/typing_v1v2/figs_typing_cards/:
  card_{combo}_A{id}.png   for every typed event of the four pilot combos
  exemplars_2x3.png        one exemplar per verdict class for the thesis body
Runs in rashindaNew-torch-env, CPU, ~2 min.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
from typing_rules_v1v2 import load_combo, load_attack_table  # noqa: E402
from analyze_multistage_attacks import estimate_offset  # noqa: E402

from typing_rules_v1v2 import OUTDIR as TYPING_OUTDIR, DATASET  # noqa: E402
OUT = os.path.join(TYPING_OUTDIR, "figs_typing_cards")   # follows UQ_DATASET
EV = os.path.join(TYPING_OUTDIR, "typing_events.csv")
STEP_MIN = 10.0 / 60.0

C = {"R": "#d62728", "A": "#ff7f0e", "E": "#1f77b4", "O": "#6a3d9a"}
_LW = 1.3   # uniform channel line thickness
STYLE = {"R": dict(lw=_LW),
         "A": dict(lw=_LW, ls="-."),
         "E": dict(lw=_LW, ls="--"),
         "O": dict(lw=_LW, ls=":")}   # dotted, uniform width (markers made Omega look thick on long events)
LBL = {"R": "Residual", "A": "Aleatoric", "E": "Epistemic", "O": "Omega"}
VTEXT = {"R1_high_confidence": "R1: high-confidence anomaly",
         "R2_noisy_sensor": "R2: sensor-health, investigate",
         "R3_borderline": "R3: borderline, escalate",
         "R4_ood_suspect": "R4: corroborated OOD, escalate",
         "R4b_ood_rescue": "R4b: OOD rescue (no residual alarm, Omega fires)",
         "R5_benign_noise": "R5: benign noise",
         "R6_data_gap": "R6: data gap",
         "missed_quiet": "missed: all channels quiet",
         "normal_quiet": "normal: all channels quiet"}
BBNAME = {"gdn": "GDN", "topogdn": "TopoGDN", "cstgl": "CST-GL"}

ATTS = load_attack_table()
ATT = {a["aid"]: a for a in ATTS}
_CTX, _DOFF = {}, {}


def ctx(bb, seed):
    k = (bb, seed)
    if k not in _CTX:
        _CTX[k] = load_combo(bb, "V2", seed)
        _DOFF[k] = estimate_offset(_CTX[k]["lab"], ATTS)
    return _CTX[k], _DOFF[k]


def rows():
    out = {}
    for r in csv.DictReader(open(EV)):
        out[(r["backbone"], int(r["seed"]), int(str(r["attack_id"]).lstrip("Aa")))] = r
    return out


def draw_card(ax, bb, seed, aid, meta, show_legend=True):
    c, doff = ctx(bb, seed)
    a = ATT[aid]
    s, e = max(0, a["s"] + doff), min(c["T"], a["e"] + doff)
    m = max(30, (e - s) // 3)
    lo, hi = max(0, s - m), min(c["T"], e + m)
    t = (np.arange(lo, hi) - s) * STEP_MIN
    view = (float(t[0]), float(t[-1]))
    # neighbour attacks (clipped), focal window
    for a2 in ATTS:
        if a2["aid"] == aid:
            continue
        s2, e2 = max(0, a2["s"] + doff), min(c["T"], a2["e"] + doff)
        l2, h2 = (s2 - s) * STEP_MIN, (e2 - s) * STEP_MIN
        if h2 > view[0] and l2 < view[1]:
            ax.axvspan(max(l2, view[0]), min(h2, view[1]), color="#fdae61", alpha=0.20)
    ax.axvspan(0, (e - s) * STEP_MIN, color="k", alpha=0.09)
    for ch in ("A", "E", "O", "R"):
        ax.plot(t, c[ch][lo:hi] / c["thr"][ch], color=C[ch], label=LBL[ch], **STYLE[ch])
    ax.axhline(1.0, color="0.65", lw=1.0, ls=(0, (4, 3)), zorder=0.5,
               label="alarm threshold")  # z/thr=1
    ax.set_yscale("symlog", linthresh=0.5, linscale=0.6)
    ax.set_ylim(bottom=-0.3)
    ax.set_xlim(*view)
    # uniform integer-style x ticks; suppress the "0" y label that collides with 10^-1
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    _yf = ax.yaxis.get_major_formatter()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: "" if v == 0 else _yf(v, p)))
    ax.tick_params(labelsize=11)
    conf = float(meta["confidence"])
    ax.set_title(f"A{aid:02d} on {BBNAME[bb]} (seed {seed})\n"
                 f"{VTEXT.get(meta['verdict'], meta['verdict'])}, "
                 f"agreement {conf:.2f}", fontsize=11)
    if show_legend:
        ax.legend(fontsize=10, loc="upper right", ncol=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = rows()

    # full redesigned gallery for the four pilot combos
    n = 0
    for (bb, seed, aid), meta in sorted(R.items()):
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        draw_card(ax, bb, seed, aid, meta)
        ax.set_xlabel("minutes from attack onset", fontsize=8)
        ax.set_ylabel("z / threshold", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"card_{bb}_V2_s{seed}_A{aid:02d}.png"), dpi=200)
        plt.close(fig)
        n += 1
    print(f"{n} cards rendered", flush=True)

    # verdict exemplars, one per class, for the thesis body (SWaT picks; the
    # WADI exemplars are chosen after the WADI verdicts exist)
    if DATASET != "swat":
        print("exemplars_2x3 skipped (SWaT-specific picks); cards rendered only", flush=True)
        return
    EX = [("gdn", 42, 22),     # R1
          ("cstgl", 42, 26),   # R2 (documented defect)
          ("cstgl", 3, 41),    # R3
          ("cstgl", 3, 28),    # R4
          ("cstgl", 3, 23),    # R4b
          ("gdn", 42, 13)]     # missed quiet (process-invisible)
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 8.8))
    for k, (bb, seed, aid) in enumerate(EX):
        ax = axes[k // 2, k % 2]
        draw_card(ax, bb, seed, aid, R[(bb, seed, aid)], show_legend=(k == 0))
        if k // 2 == 2:
            ax.set_xlabel("minutes from attack onset", fontsize=8)
        if k % 2 == 0:
            ax.set_ylabel("z / threshold", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exemplars_2x3.png"), dpi=200)
    plt.close(fig)
    print("exemplars_2x3.png rendered", flush=True)


if __name__ == "__main__":
    main()
