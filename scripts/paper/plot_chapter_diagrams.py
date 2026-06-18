#!/usr/bin/env python3
"""Methodology diagrams for the thesis chapter (matplotlib, no graphviz dep).

D1 pipeline overview: backbones -> anchored inference -> uncertainty channels
   -> fusion + typing/triage outputs.
D2 typing decision flow: the ordered 6-rule table as a flowchart.
Renders to results/typing_v1v2/figs_impact/. No em dashes in any text.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/mnt/datassd3/rashinda/UQ_GNN_AnomalyTyping/results/typing_v1v2/figs_impact"
C = {"R": "#d62728", "A": "#ff7f0e", "E": "#1f77b4", "O": "#6a3d9a", "S": "#7f7f7f"}


def box(ax, x, y, w, h, text, fc="#f0f0f0", ec="#555555", fs=8, bold=False, tc="k"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="#444444", lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=11, color=color, lw=lw,
                                 shrinkA=2, shrinkB=2))


def d1_pipeline():
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.8); ax.axis("off")

    # input
    box(ax, 0.15, 1.9, 1.25, 1.0, "SWaT test\nwindows\nW = 60", fc="#e8eef7", fs=8)

    # backbones container
    box(ax, 1.75, 0.55, 1.95, 3.7, "", fc="#fafafa", ec="#999999")
    ax.text(2.725, 4.05, "GNN backbone", ha="center", fontsize=9, fontweight="bold")
    for i, (name, note) in enumerate([("GDN", "attention"),
                                      ("TopoGDN", "attention + topology"),
                                      ("CST-GL", "no attention"),
                                      ("DualSTGF", "this work, pending")]):
        fc = "#fff3cd" if name == "DualSTGF" else "#ffffff"
        box(ax, 1.95, 3.15 - i * 0.78, 1.55, 0.62, f"{name}\n{note}", fc=fc, fs=7)
    arrow(ax, 1.40, 2.4, 1.75, 2.4)

    # anchored inference
    box(ax, 4.05, 1.55, 2.0, 1.7,
        "G-DeltaUQ anchoring\n(one method, all backbones)\n\n"
        "split: h = backbone(x)\nanchored head on\n[h - c ; c],  K = 100\nanchors from val",
        fc="#e8f0e8", fs=7.5)
    arrow(ax, 3.70, 2.4, 4.05, 2.4)

    # channels
    chan = [("Residual  |x - mu|", C["R"]),
            ("Epistemic  U_par = var over anchors", C["E"]),
            ("Aleatoric  sigma2 (NLL head, val)", C["A"]),
            ("Distributional  Omega (Mahalanobis)", C["O"]),
            ("Structural  U_str (attention var)*", C["S"])]
    for i, (txt, col) in enumerate(chan):
        box(ax, 6.45, 3.62 - i * 0.76, 2.45, 0.6, txt, fc="#ffffff", ec=col, fs=7, tc=col)
        arrow(ax, 6.05, 2.4, 6.45, 3.92 - i * 0.76, color=col, lw=1.0)
    ax.text(6.5, 0.32, "* attention backbones only", fontsize=6.5, color=C["S"])

    # outputs
    box(ax, 9.35, 2.95, 1.5, 1.05, "Fusion\nstacker S2\n(GBM)", fc="#dcebf5", fs=8, bold=True)
    box(ax, 9.35, 1.0, 1.5, 1.35, "Rule table\n(6 rules)\ntyping +\nalarm triage",
        fc="#f5e9dc", fs=8, bold=True)
    for i in range(5):
        ymid = 3.92 - i * 0.76
        arrow(ax, 8.90, ymid, 9.35, 3.5, color="#777777", lw=0.8)
        arrow(ax, 8.90, ymid, 9.35, 1.7, color="#777777", lw=0.8)
    ax.text(10.1, 4.25, "detection", ha="center", fontsize=8, style="italic")
    ax.text(10.1, 0.7, "explanation", ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "D1_pipeline_overview.png"), dpi=220)
    plt.close(fig)


def d2_typing_flow():
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.set_xlim(0, 8.2); ax.set_ylim(0, 5.6); ax.axis("off")

    box(ax, 3.0, 4.9, 2.2, 0.55, "channel bits at step t\n[R, A, E, O] vs Q0.995", fc="#e8eef7", fs=8)
    box(ax, 3.25, 3.95, 1.7, 0.55, "Residual high?", fc="#fdecea", ec=C["R"], fs=8, bold=True)
    arrow(ax, 4.1, 4.9, 4.1, 4.5)

    # yes branch (left): ordered precedence
    ax.text(2.65, 4.22, "yes", fontsize=8, color="#333333")
    arrow(ax, 3.25, 4.22, 1.95, 3.6)
    yes = [("Omega high?", "R4  OOD suspect,\ncannot confirm", C["O"]),
           ("Epistemic high?", "R3  borderline,\nescalate", C["E"]),
           ("Aleatoric high?", "R2  noisy sensor*", C["A"]),
           ("else", "R1  high-confidence\nanomaly", "#2ca02c")]
    y = 3.3
    for q, verdict, col in yes:
        if q != "else":
            box(ax, 0.7, y, 1.5, 0.5, q, fc="#ffffff", ec=col, fs=7.5)
            box(ax, 2.5, y, 1.55, 0.5, verdict, fc="#ffffff", ec=col, fs=7, tc=col)
            arrow(ax, 2.2, y + 0.25, 2.5, y + 0.25, color=col, lw=1.0)
            arrow(ax, 1.45, y, 1.45, y - 0.35, color="#888888", lw=0.9)
            ax.text(1.55, y - 0.22, "no", fontsize=7, color="#888888")
        else:
            box(ax, 0.7, y, 1.5, 0.5, "all quiet", fc="#ffffff", ec=col, fs=7.5)
            box(ax, 2.5, y, 1.55, 0.5, verdict, fc="#eaf6ea", ec=col, fs=7, tc=col)
            arrow(ax, 2.2, y + 0.25, 2.5, y + 0.25, color=col, lw=1.0)
        y -= 0.85
    ax.text(0.7, 0.32, "* revision recommended: annotated flag, not dismissal\n"
                       "  (4 of 5 R2 dismissals were real attacks on CST-GL)",
            fontsize=6.5, color=C["A"])

    # no branch (right): dismissals + rescue
    ax.text(5.45, 4.22, "no", fontsize=8, color="#333333")
    arrow(ax, 4.95, 4.22, 6.2, 3.6)
    no = [("Aleatoric high?", "R5  benign noise", C["A"]),
          ("Epistemic high?", "R6  data gap,\ncollect more", C["E"]),
          ("Omega high?", "R4b  OOD rescue,\nescalate", C["O"]),
          ("else", "normal / missed\n(all quiet)", "#888888")]
    y = 3.3
    for q, verdict, col in no:
        if q != "else":
            box(ax, 4.95, y, 1.5, 0.5, q, fc="#ffffff", ec=col, fs=7.5)
            box(ax, 6.75, y, 1.35, 0.5, verdict, fc="#ffffff", ec=col, fs=7, tc=col)
            arrow(ax, 6.45, y + 0.25, 6.75, y + 0.25, color=col, lw=1.0)
            arrow(ax, 5.7, y, 5.7, y - 0.35, color="#888888", lw=0.9)
            ax.text(5.8, y - 0.22, "no", fontsize=7, color="#888888")
        else:
            box(ax, 4.95, y, 1.5, 0.5, "all quiet", fc="#ffffff", ec=col, fs=7.5)
            box(ax, 6.75, y, 1.35, 0.5, verdict, fc="#f2f2f2", ec=col, fs=7, tc=col)
            arrow(ax, 6.45, y + 0.25, 6.75, y + 0.25, color=col, lw=1.0)
        y -= 0.85
    ax.text(4.95, 0.32, "events: detected -> majority over alarm steps;\n"
                        "missed -> window hierarchy Omega > E > A > quiet",
            fontsize=6.5, color="#555555")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "D2_typing_flow.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    d1_pipeline()
    d2_typing_flow()
    print("diagrams ->", OUT)
