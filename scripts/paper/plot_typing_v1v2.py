#!/usr/bin/env python3
"""Figures for the rule-based typing pilot (matplotlib, static thesis figures).

Reads results/typing_v1v2/{traces,gbm,typing_events.csv} and renders to
results/typing_v1v2/figs/:
  1. per-event timelines: channel z-series (R, A, E, O) with the High thresholds,
     plus the GBM log-odds panel when available; verdict in the title;
  2. the 2x2 typing-table heatmap (Residual High/Low x Omega High/Low) over all
     typed events, pooled across combos (the Residual-Low x Omega-High cell is
     the OOD-rescue cell);
  3. permutation-importance bar charts per combo (with the inlined caveat).
Runs in rashindaNew-torch-env.
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TDIR = os.path.join(ROOT, "results/typing_v1v2")
FDIR = os.path.join(TDIR, "figs")
CAVEAT = "GBM attribution measures label-predictiveness, not anomaly nature."

CH_COLORS = {"R": "#d62728", "A": "#ff7f0e", "E": "#1f77b4", "O": "#9467bd"}
CH_LABEL = {"R": "Residual z", "A": "Aleatoric z", "E": "Epistemic z", "O": "Omega z"}


def plot_event(trace_path):
    tr = json.load(open(trace_path))
    tag = os.path.basename(trace_path).replace(".json", "")
    combo = "_".join(tag.split("_")[:3])
    aid = tag.split("_")[-1]
    gbm_path = os.path.join(TDIR, "gbm", f"explain_{combo}.json")
    gbm_ev = None
    if os.path.exists(gbm_path):
        g = json.load(open(gbm_path))
        gbm_ev = g.get("events", {}).get(aid)

    n_pan = 2 if gbm_ev else 1
    fig, axes = plt.subplots(n_pan, 1, figsize=(10, 3.2 * n_pan), sharex=True,
                             squeeze=False)
    ax = axes[0, 0]
    t = np.arange(len(tr["series"]["R"]))
    for k in ["R", "A", "E", "O"]:
        ax.plot(t, tr["series"][k], color=CH_COLORS[k], lw=1.2, label=CH_LABEL[k])
        ax.axhline(tr["thresholds"][k], color=CH_COLORS[k], ls="--", lw=0.7, alpha=0.6)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("robust z (symlog)")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.set_title(f"{combo}  A{tr['attack_id']:02d} ({tr['category']}, targets {tr['targets']})\n"
                 f"verdict: {tr['verdict']} (conf {tr['confidence']:.2f}; bits at peak {tr['bits_peak']}; "
                 f"peak sensor {tr['peak_sensor']})", fontsize=9)
    if gbm_ev:
        ax2 = axes[1, 0]
        ax2.plot(np.arange(len(gbm_ev["logodds_series"])), gbm_ev["logodds_series"],
                 color="#2ca02c", lw=1.2, label="S2 GBM log-odds")
        ax2.axhline(0.0, color="k", ls=":", lw=0.7)
        ax2.set_ylabel("GBM log-odds")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.annotate(f"top features at peak: {', '.join(gbm_ev['top_features'])}",
                     xy=(0.01, 0.04), xycoords="axes fraction", fontsize=7)
    axes[-1, 0].set_xlabel("seconds from attack start")
    fig.tight_layout()
    out = os.path.join(FDIR, f"event_{tag}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_heatmap(rows):
    grid = np.zeros((2, 2), int)            # rows: R High/Low; cols: O High/Low
    ms = np.zeros((2, 2), int)
    for r in rows:
        rh = 0 if int(r["detected"]) else 1
        oh = 0 if float(r["zpeak_O"]) > float(r["thr_O"]) else 1
        grid[rh, oh] += 1
        if int(r["n_stages"]) >= 2:
            ms[rh, oh] += 1
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(grid, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]} events\n({ms[i, j]} multi-stage)",
                    ha="center", va="center", fontsize=10,
                    color="white" if grid[i, j] > grid.max() * 0.6 else "black")
    ax.set_xticks([0, 1], ["Omega High", "Omega Low"])
    ax.set_yticks([0, 1], ["Residual High\n(detected)", "Residual Low\n(missed)"])
    ax.text(0, 1.35, "OOD-rescue cell", ha="center", fontsize=8, color="#9467bd")
    ax.set_title("Typing table occupancy, all events, pooled combos", fontsize=10)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    out = os.path.join(FDIR, "typing_heatmap_residual_x_omega.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_importance():
    outs = []
    for p in sorted(glob.glob(os.path.join(TDIR, "gbm", "explain_*.json"))):
        g = json.load(open(p))
        names = g["feature_names"]
        mi = g["permutation_importance_full"]
        means = np.array([mi[n][0] for n in names])
        stds = np.array([mi[n][1] for n in names])
        order = np.argsort(means)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.barh(np.arange(len(names)), means[order], xerr=stds[order],
                color="#1f77b4", alpha=0.85)
        ax.set_yticks(np.arange(len(names)), [names[i] for i in order], fontsize=8)
        ax.set_xlabel("permutation importance (average precision drop)")
        ax.set_title(f"S2 GBM feature importance: {g['combo']} "
                     f"(depth {g['best_hp']['max_depth']}, iters {g['best_hp']['max_iter']})",
                     fontsize=9)
        ax.annotate(CAVEAT, xy=(0.01, -0.32), xycoords="axes fraction", fontsize=7,
                    style="italic")
        fig.tight_layout()
        out = os.path.join(FDIR, f"gbm_importance_{g['combo']}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        outs.append(out)
    return outs


def main():
    os.makedirs(FDIR, exist_ok=True)
    pilot = {3, 8, 22, 23, 26, 27, 28, 30, 33, 36, 38, 39}
    n = 0
    for p in sorted(glob.glob(os.path.join(TDIR, "traces", "*.json"))):
        aid = int(os.path.basename(p).split("_A")[-1].split(".")[0])
        if aid in pilot:
            plot_event(p)
            n += 1
    rows = list(csv.DictReader(open(os.path.join(TDIR, "typing_events.csv"))))
    plot_heatmap(rows)
    imps = plot_importance()
    print(f"figs: {n} event timelines + heatmap + {len(imps)} importance charts -> {FDIR}",
          flush=True)


if __name__ == "__main__":
    main()
