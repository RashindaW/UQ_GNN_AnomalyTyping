#!/usr/bin/env python3
"""P3: compute F1-vs-K PA%K curves for the Part-1 figure (M0 vs S2).

Per combo (3 backbones x 6 seeds, V2): M0 = the fusion-convention residual
aggregate (setup_context agg, val-err normalized); S2 = the cached GBM score
(results/typing_v1v2/gbm/{tag}_cache.npz, same fit as the S2 column).
Curve: best-F1 at each K in {0,5,...,100}, n_thresholds=200 (the fusion-CSV
convention's threshold count; finer K grid for a smooth figure).
Writes results/thesis_part1/pak_curves/{bb}_s{seed}.npz.

Usage: p3_pak_curves.py --combo gdn:42
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
from pa_k_metric import best_f1_pa_k  # noqa: E402
import fusion_study as FS  # noqa: E402

DIRMAP = {"gdn": "baseline_v1v2/gdn", "topogdn": "uq_v1v2/topogdn",
          "cstgl": "uq_v1v2/cstgl"}
SPLIT = os.path.join(ROOT, "pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json")
BUNDLE = os.path.join(ROOT, "pretrained/swat_ensemble/calibration_bundle")
K_GRID = np.arange(0, 101, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True)
    args = ap.parse_args()
    bb, seed = args.combo.split(":")
    seed = int(seed)
    out_dir = os.path.join(ROOT, "results/thesis_part1/pak_curves")
    os.makedirs(out_dir, exist_ok=True)
    arr = os.path.join(ROOT, "results", DIRMAP[bb], "V2", f"seed{seed}", "arrays_full.npz")
    ctx = FS.setup_context(argparse.Namespace(arrays=arr, split=SPLIT,
                                              bundle=BUNDLE, slide_win=60,
                                              seed=seed))
    label = np.asarray(ctx["label"])
    cache = np.load(os.path.join(ROOT, "results/typing_v1v2/gbm",
                                 f"{bb}_V2_s{seed}_cache.npz"))
    scores = {"M0": np.asarray(ctx["agg"], float),
              "S2": cache["score"].astype(float)}
    assert len(scores["S2"]) == len(label), (len(scores["S2"]), len(label))
    out = {"K": K_GRID}
    for name, sc in scores.items():
        f1s = [best_f1_pa_k(sc, label, K_pct=float(k), n_thresholds=200)["F1"]
               for k in K_GRID]
        out[f"F1_{name}"] = np.array(f1s)
        print(f"[{bb} s{seed}] {name}: K0={f1s[0]:.3f} K100={f1s[-1]:.3f}",
              flush=True)
    np.savez(os.path.join(out_dir, f"{bb}_s{seed}.npz"), **out)
    print(f"[{bb} s{seed}] wrote curves", flush=True)


if __name__ == "__main__":
    main()
