#!/usr/bin/env python3
"""GBM-advantage evidence panels: where the S2 GBM fusion detects before the
residual, or detects what the residual misses entirely.

Conventions (all stated on the panels): both scores thresholded at their own
0.995 nominal quantile (equal nominal false-alarm budget); ratio axes with one
detection line at 1 (valid: all thresholds verified positive; the cached GBM
score is log-odds). Two stackers are used and labeled: the standard stacker
(fit on the full validation slice 15593-24530) for events lying OUTSIDE that
slice, and a CLEAN-FIT stacker refit only on rows 15593-20000 (disjoint from
A23-A28) to demonstrate the GDN A27 lead without the in-sample objection.
Selection provenance: /tmp scans reproduced here; honest exclusions: the GDN
A28 lead does NOT survive the clean fit and is not shown.

Renders to results/typing_v1v2/figs_gbm_advantage/. rashindaNew-torch-env.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
from typing_rules_v1v2 import load_attack_table  # noqa: E402
from analyze_multistage_attacks import estimate_offset  # noqa: E402

OUT = os.path.join(ROOT, "results/typing_v1v2/figs_gbm_advantage")
GBM = os.path.join(ROOT, "results/typing_v1v2/gbm")
VAL = (15593, 24530)
FIT = (15593, 20000)
STEP = 10.0
HP = {"gdn_V2_s42": (5, 200), "topogdn_V2_s42": (3, 200),
      "cstgl_V2_s42": (5, 200), "cstgl_V2_s3": (5, 100)}

ATTS = load_attack_table()
_CACHE, _WIN, _CLEAN = {}, {}, {}


def cache(tag):
    if tag not in _CACHE:
        d = np.load(os.path.join(GBM, f"{tag}_cache.npz"))
        _CACHE[tag] = (d["label"].astype(int), d["score"], d["feat"])
        doff = estimate_offset(_CACHE[tag][0], ATTS)
        T = len(_CACHE[tag][0])
        for a in ATTS:
            s, e = max(0, a["s"] + doff), min(T, a["e"] + doff)
            if e > s:
                _WIN[(tag, a["aid"])] = (s, e)
    return _CACHE[tag]


def cleanfit_score(tag):
    if tag not in _CLEAN:
        y, _, X = cache(tag)
        depth, iters = HP[tag]
        clf = HistGradientBoostingClassifier(
            max_depth=depth, max_iter=iters, learning_rate=0.05,
            l2_regularization=1.0, class_weight="balanced", random_state=0)
        clf.fit(X[slice(*FIT)], y[slice(*FIT)])
        _CLEAN[tag] = clf.predict_proba(X)[:, 1]
    return _CLEAN[tag]


def panel(fname, tag, aid, cleanfit, margin, title, note, unit="min"):
    y, sg_std, X = cache(tag)
    sm = X[:, 0]
    sg = cleanfit_score(tag) if cleanfit else sg_std
    nom = y == 0
    tm = np.quantile(sm[nom], 0.995)
    tg = np.quantile(sg[nom], 0.995)
    assert tm > 0 and tg > 0, "ratio display requires positive thresholds"
    rm, rg = sm / tm, sg / tg
    s, e = _WIN[(tag, aid)]
    a, b = max(0, s - margin), min(len(y), e + margin)
    scale = 60.0 if unit == "min" else 3600.0
    t = (np.arange(a, b) - s) * STEP / scale

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    for (tg2, aid2), (s2, e2) in _WIN.items():
        if tg2 == tag and aid2 != aid and e2 > a and s2 < b:
            ax.axvspan(max(s2 - s, a - s) * STEP / scale,
                       min(e2 - s, b - s) * STEP / scale, color="#fdae61", alpha=0.22)
    ax.axvspan(0, (e - s) * STEP / scale, color="k", alpha=0.09)
    ax.plot(t, rm[a:b], color="#d62728", lw=1.1, label="M0 residual / threshold")
    ax.plot(t, rg[a:b], color="#2166ac", lw=1.2, label="S2 GBM / threshold")
    ax.axhline(1.0, color="k", lw=1.0)
    x0, lab = (FIT[1], "clean-fit end") if cleanfit else (VAL[1], "train-slice end")
    if a < x0 < b:
        ax.axvline((x0 - s) * STEP / scale, color="#555555", ls="--", lw=1.0)
        ax.text((x0 - s) * STEP / scale, ax.get_ylim()[1], f" {lab}",
                fontsize=6.5, color="#555555", va="top")
    for r, col in ((rm, "#d62728"), (rg, "#2166ac")):
        h = np.nonzero(r[s:e] >= 1.0)[0]
        if h.size:
            ax.axvline(h[0] * STEP / scale, color=col, ls=":", lw=1.0)
    ax.set_yscale("symlog", linthresh=0.5, linscale=0.6)
    ax.set_ylim(bottom=-0.75)
    ax.set_xlabel(f"{'hours' if unit == 'h' else 'minutes'} from attack onset", fontsize=8)
    ax.set_ylabel("score / threshold", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(title, fontsize=9)
    ax.text(0.01, 0.03, note + "; gray = focal attack, orange = neighbouring attacks",
            transform=ax.transAxes, fontsize=6.5,
            style="italic", color="#444444")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)
    print(fname, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    panel("G1_gdn_A27_cleanfit_lead.png", "gdn_V2_s42", 27, True, 90,
          "A27 on GDN: clean-fit GBM fires 13.5 minutes before the residual",
          "stacker fitted only on rows 15593-20000 (disjoint from A23-A28); "
          "thresholds matched at each score's 0.995 nominal quantile")
    panel("G2_topogdn_A30_miss_catch.png", "topogdn_V2_s42", 30, False, 90,
          "A30 on TopoGDN: residual never crosses, GBM sustained for 19.5 minutes",
          "standard stacker; A30 lies outside the stacker training slice; "
          "matched 0.995 nominal thresholds")
    panel("G3_topogdn_A40_miss_catch.png", "topogdn_V2_s42", 40, False, 60,
          "A40 on TopoGDN: residual never crosses, GBM covers 93 percent of the window",
          "standard stacker; A40 lies outside the stacker training slice; "
          "matched 0.995 nominal thresholds")
    panel("G4_cstgl_s3_A28_cleanhalf.png", "cstgl_V2_s3", 28, False, 360,
          "A28 on CST-GL: GBM holds the 4.5 held-out hours where the residual stays sub-threshold",
          "standard stacker; dashed line = training-slice end; held-out residual peaks at 0.93x "
          "threshold, never crossing; its only two in-window crossings lie inside the slice",
          unit="h")
    panel("G5_cstgl_s3_A39_miss_catch.png", "cstgl_V2_s3", 39, False, 60,
          "A39 on CST-GL: residual never crosses, GBM sustained at 65 percent coverage",
          "standard stacker; A39 lies outside the stacker training slice; "
          "matched 0.995 nominal thresholds")
    panel("G6_cstgl_s42_A30_miss_catch.png", "cstgl_V2_s42", 30, False, 90,
          "A30 on CST-GL (seed 42): residual never crosses, GBM detects within 40 seconds (28 percent coverage)",
          "standard stacker; A30 lies outside the stacker training slice; "
          "matched 0.995 nominal thresholds")


if __name__ == "__main__":
    main()
