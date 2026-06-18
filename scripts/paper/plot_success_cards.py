#!/usr/bin/env python3
"""Curated 1xN 'confidence-check success' panel for the thesis.

Reuses draw_card from plot_typing_cards.py (so it follows UQ_DATASET) to render
a handful of HAND-PICKED exemplar events side by side: the attacks where the
uncertainty channels most clearly corroborate or rescue the residual flag.

  UQ_SUCCESS="gdn:42:20,gdn:42:27,cstgl:3:23" \
  UQ_TITLE="SWaT" \
  python scripts/paper/plot_success_cards.py
  (prefix UQ_DATASET=wadi for the WADI panel)

Output: <typing OUTDIR>/figs_success/success_<title>.png
"""
import io
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
import plot_typing_cards as PC  # noqa: E402  (follows UQ_DATASET via typing_rules)
from typing_rules_v1v2 import OUTDIR  # noqa: E402


def _set_latex_style(usetex):
    """LaTeX (Computer Modern) fonts. True usetex when the toolchain is present;
    otherwise the bundled cm mathtext + serif, which matches the thesis look."""
    rcParams["axes.unicode_minus"] = False
    if usetex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["Computer Modern Roman"]
    else:
        rcParams["text.usetex"] = False
        rcParams["font.family"] = "serif"
        rcParams["mathtext.fontset"] = "cm"
        rcParams["font.serif"] = ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"]


def _latex_available():
    """Probe a real render so a missing dvipng/latex falls back cleanly."""
    if not (shutil.which("latex") and shutil.which("dvipng")):
        return False
    try:
        _set_latex_style(True)
        f = plt.figure()
        f.text(0.5, 0.5, "A22 on GDN (seed 42), confidence 1.00")
        f.savefig(io.BytesIO(), format="png")
        plt.close(f)
        return True
    except Exception:
        plt.close("all")
        return False


def main():
    usetex = _latex_available()
    _set_latex_style(usetex)
    print(f"[style] LaTeX fonts via {'usetex' if usetex else 'cm-serif mathtext'}", flush=True)
    combos = [(p.split(":")[0], int(p.split(":")[1]), int(p.split(":")[2]))
              for p in os.environ["UQ_SUCCESS"].split(",")]
    title = os.environ.get("UQ_TITLE", "success")
    R = PC.rows()
    n = len(combos)

    # UQ_INDIVIDUAL=1 -> one standalone PNG per card (each with its own
    # legend + axes), named by verdict, under figs_success/individual/
    if os.environ.get("UQ_INDIVIDUAL"):
        out = os.path.join(OUTDIR, "figs_success", "individual")
        os.makedirs(out, exist_ok=True)
        for bb, seed, aid in combos:
            meta = R[(bb, seed, aid)]
            fig, ax = plt.subplots(figsize=(4.7, 3.0))
            PC.draw_card(ax, bb, seed, aid, meta, show_legend=True)
            ax.set_xlabel("minutes from attack onset", fontsize=8)
            ax.set_ylabel("z / threshold", fontsize=8)
            fp = os.path.join(out, f"{meta['verdict']}_{bb}_s{seed}_A{aid:02d}.png")
            fig.tight_layout()
            fig.savefig(fp, dpi=200)
            plt.close(fig)
            print(f"wrote {fp}", flush=True)
        return
    ncol = int(os.environ.get("UQ_NCOL", min(n, 4)))
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.1 * nrow))
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for k, (bb, seed, aid) in enumerate(combos):
        meta = R[(bb, seed, aid)]
        # per-panel legends off; one shared legend is placed at the top
        PC.draw_card(axes[k], bb, seed, aid, meta, show_legend=False)
        if k // ncol == nrow - 1 or k + ncol >= n:
            axes[k].set_xlabel("minutes from attack onset", fontsize=13)
        if k % ncol == 0:
            axes[k].set_ylabel("z / threshold", fontsize=13)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # single shared legend across the top, Residual first, larger text
    handles, labels = axes[0].get_legend_handles_labels()
    order = {"Residual": 0, "Aleatoric": 1, "Epistemic": 2, "Omega": 3,
             "alarm threshold": 4}
    hl = sorted(zip(handles, labels), key=lambda x: order.get(x[1], 9))
    handles, labels = [h for h, _ in hl], [l for _, l in hl]
    leg = fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                     fontsize=14, frameon=False, bbox_to_anchor=(0.5, 1.0),
                     handlelength=2.3, columnspacing=1.6, borderaxespad=0.2)

    out = os.path.join(OUTDIR, "figs_success")
    os.makedirs(out, exist_ok=True)
    # leave headroom for the shared legend, then save as vector PDF (+ PNG preview).
    # bbox_inches="tight" + the legend artist guarantee the wide top legend is not clipped.
    top = 1.0 - 0.55 / (3.1 * nrow)
    fig.tight_layout(rect=[0, 0, 1, top])
    fp_pdf = os.path.join(out, f"success_{title}.pdf")
    fp_png = os.path.join(out, f"success_{title}.png")
    save_kw = dict(bbox_inches="tight", bbox_extra_artists=[leg])
    fig.savefig(fp_pdf, **save_kw)
    fig.savefig(fp_png, dpi=200, **save_kw)
    plt.close(fig)
    print(f"wrote {fp_pdf}", flush=True)


if __name__ == "__main__":
    main()
