#!/usr/bin/env python3
"""Concordance 2x2: on CST-GL's residual (M0) false alarms, do the learned stacker
and the parameter-free typing rule make the same call? Pooled, 6 seeds, held-out,
oracle threshold. Numbers from triage_funnel_cstgl.py (verified).

Outputs thesis/figures/concordance_cstgl.{pdf,png}.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pooled cells (verified, results/typing_v1v2/triage_funnel_cstgl.csv)
both_kill, fus_only, rule_only, both_keep = 10140, 204, 56, 152
M = np.array([[both_kill, fus_only], [rule_only, both_keep]], float)
total = M.sum()
agree = (both_kill + both_keep) / total

rcParams["font.family"] = "serif"; rcParams["mathtext.fontset"] = "cm"
rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(6.4, 4.6))
# shade: agreement diagonal green, disagreement off-diagonal grey
shade = np.array([[0.16, 0.0], [0.0, 0.16]])
ax.imshow(shade, cmap="Greens", vmin=0, vmax=1)
labels = [[f"both drop\n{both_kill:,}", f"stacker drops,\nrule escalates\n{fus_only}"],
          [f"stacker keeps,\nrule dismisses\n{rule_only}", f"both keep\n{both_keep}"]]
for i in range(2):
    for j in range(2):
        kind = "agree" if i == j else "disagree"
        ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=12,
                fontweight="bold" if i == j else "normal",
                color="#1a5d1a" if i == j else "#555555")
ax.set_xticks([0, 1]); ax.set_xticklabels(["rule: dismiss\n(R5/R6/quiet)", "rule: escalate\n(R1--R4)"], fontsize=11)
ax.set_yticks([0, 1]); ax.set_yticklabels(["stacker\nsuppresses", "stacker\nretains"], fontsize=11)
ax.set_xlabel("parameter-free typing rule", fontsize=12)
ax.set_ylabel("learned fusion stacker", fontsize=12)
ax.set_title(f"Agreement on the residual baseline's {int(total):,} false alarms: "
             f"{100*agree:.1f}%\n(CST-GL, SWaT, 6 seeds, held-out, oracle threshold)", fontsize=11.5)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks(np.arange(-.5, 2, 1), minor=True); ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
fig.tight_layout()
for ext in ("pdf", "png"):
    fp = os.path.join(ROOT, "thesis", "figures", f"concordance_cstgl.{ext}")
    fig.savefig(fp, dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote thesis/figures/concordance_cstgl.pdf (+png);  agreement = %.1f%%" % (100 * agree))
