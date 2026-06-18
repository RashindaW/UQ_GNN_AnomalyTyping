#!/usr/bin/env python3
"""F3: PA%K curves (F1 vs K), M0 vs S2 per backbone; mean line + seed band."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CD = os.path.join(ROOT, "results/thesis_part1/pak_curves")
BBS = [("gdn", "GDN"), ("topogdn", "TopoGDN"), ("cstgl", "CST-GL")]
SEEDS = [0, 1, 2, 3, 4, 42]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
for ax, (bb, name) in zip(axes, BBS):
    for meth, col in [("M0", "#555555"), ("S2", "#d95f02")]:
        cur = np.stack([np.load(f"{CD}/{bb}_s{s}.npz")[f"F1_{meth}"]
                        for s in SEEDS])
        K = np.load(f"{CD}/{bb}_s{SEEDS[0]}.npz")["K"]
        ax.fill_between(K, cur.min(0), cur.max(0), color=col, alpha=0.18)
        ax.plot(K, cur.mean(0), color=col, lw=2,
                label=f"{'anchored M0' if meth=='M0' else 'fusion S2'}")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("K (%): segment promoted iff > K% of its steps alarmed",
                  fontsize=8)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("best F1 under PA%K", fontsize=10)
axes[0].legend(fontsize=8.5, loc="lower left")
fig.suptitle("PA%K profile (K=0: lenient point-adjust; K=100: strict "
             "point-wise). Mean over 6 seeds, band = seed min-max; full arrays",
             fontsize=10.5, y=1.03)
fig.tight_layout()
out = os.path.join(ROOT, "results/thesis_part1/figs/F3_pak_curves.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
