#!/usr/bin/env python3
"""Three-stage confusion matrices for the strong/weak contrast pair (GDN, CST-GL).

Cells: pooled windows over six seeds, SWaT held-out, oracle detector thresholds,
Fix-A postproc; stage 3 = fusion alarms minus the dismiss tier (R5/R6/quiet with
the Omega dual-quantile band). Numbers verified against
results/typing_v1v2/triage_threestage_swat.csv and /tmp/bandrule.log.

Outputs thesis/figures/threestage_cm.{pdf,png}.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (TP, FP, FN, TN) pooled
DATA = {
    "GDN": [
        ("Residual baseline", (9480, 361, 4098, 107177)),
        ("Best fusion", (9795, 486, 3783, 107052)),
        ("Fusion + triage", (9738, 425, 3840, 107113)),
    ],
    "CST-GL": [
        ("Residual baseline", (10340, 10552, 3598, 96986)),
        ("Best fusion", (9851, 241, 4087, 107297)),
        ("Fusion + triage", (9843, 186, 4095, 107352)),
    ],
}


def f1pr(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


rcParams["font.family"] = "serif"; rcParams["mathtext.fontset"] = "cm"
rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.6))
for i, (bb, stages) in enumerate(DATA.items()):
    for j, (title, (tp, fp, fn, tn)) in enumerate(stages):
        ax = axes[i][j]
        f, p, r = f1pr(tp, fp, fn)
        # shading: diagonal (TP,TN) light green; FP cell warm tint scaled by share
        shade = np.zeros((2, 2, 3))
        shade[0, 0] = shade[1, 1] = (0.88, 0.95, 0.88)           # TP, TN
        fp_share = min(1.0, fp / 12000)
        shade[0, 1] = (1.0, 0.93 - 0.45 * fp_share, 0.80 - 0.55 * fp_share)  # FP
        shade[1, 0] = (0.96, 0.96, 0.96)                          # FN
        ax.imshow(shade)
        cells = [[("TP", tp), ("FP", fp)], [("FN", fn), ("TN", tn)]]
        for r_ in range(2):
            for c_ in range(2):
                nm, v = cells[r_][c_]
                ax.text(c_, r_ - 0.16, nm, ha="center", va="center",
                        fontsize=11, fontweight="bold", color="0.25")
                ax.text(c_, r_ + 0.12, f"{v:,}", ha="center", va="center", fontsize=12.5)
        ax.set_title(f"{title}\nP={p:.3f}  R={r:.3f}  F1={f:.3f}", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["attack", "nominal"], fontsize=10)
        ax.set_yticklabels(["alarm", "no\nalarm"], fontsize=10)
        if i == 1:
            ax.set_xlabel("ground truth", fontsize=11)
        if j == 0:
            ax.set_ylabel(bb, fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "thesis", "figures", f"threestage_cm.{ext}"),
                dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote thesis/figures/threestage_cm.pdf (+png)")
