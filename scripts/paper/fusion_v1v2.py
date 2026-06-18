#!/usr/bin/env python3
"""Fusion methods on the V1/V2 arrays, SEED-WISE. Reuses fusion_study machinery.

For each (backbone, variant, seed) whose arrays.npz carries the UQ channels, runs
the fusion methods (M0 residual baseline + L1 std-resid + L2 NLPD + L3 pred-Maha +
V1 var-weighted + S1 logistic + S2 GBM) and reports per-seed PA%K-AUC and F1.
Writes results/baseline_v1v2/fusion_v1v2_seedwise.csv (one row per
backbone,variant,seed,method).
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
import fusion_study as FS
from fusion_sweep_K100_full import setup_context
from compute_M10_PAK import fit_M10_score

SEEDS = [0, 1, 2, 3, 4, 42]
# UQ_DATASET=wadi retargets the module-level constants that downstream scripts
# (explain_gbm_v1v2) from-import; the CLI --dataset flag mirrors this for main().
_ENV_DS = os.environ.get("UQ_DATASET", "swat")
if _ENV_DS == "wadi":
    SPLIT = os.path.join(ROOT, "pretrained/wadi_ensemble/calibration_bundle/calibration_set_indices.json")
    BUNDLE = os.path.join(ROOT, "pretrained/wadi_ensemble/calibration_bundle")
else:
    SPLIT = os.path.join(ROOT, "pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json")
    BUNDLE = os.path.join(ROOT, "pretrained/swat_ensemble/calibration_bundle")

# Per-backbone arrays_full location: GDN's anchored model wrote them under the
# baseline dir (GDN full-UQ chain); the anchored TopoGDN/CST-GL UQ models write to
# uq_v1v2/. CST-GL omits test_U_str (no attention -> 3 channels); setup_context /
# build_stacker_features already handle the missing-U_str case (7/8 features).
DIRMAP_SWAT = {"gdn": "baseline_v1v2/gdn", "topogdn": "uq_v1v2/topogdn", "cstgl": "uq_v1v2/cstgl", "dualstage": "uq_v1v2/dualstage"}
# WADI campaign tree (V2 only): select with --dataset wadi or UQ_DATASET=wadi
DIRMAP_WADI = {"gdn": "baseline_wadi_v2/gdn", "topogdn": "uq_wadi_v2/topogdn", "cstgl": "uq_wadi_v2/cstgl", "dualstage": "uq_wadi_v2/dualstage"}
DIRMAP = DIRMAP_WADI if _ENV_DS == "wadi" else DIRMAP_SWAT


def has_uq(arr):
    d = np.load(arr)
    return "test_U_par" in d.files and "test_sigma2_ale" in d.files


def promote_omega(arr):
    """Make the REAL Mahalanobis Omega the distributional channel.

    build_omega.py writes the real Omega to test_U_dist_maha_mean but leaves
    test_U_dist as the placeholder (== U_par.mean, redundant with epistemic).
    Both stackers read the distributional channel from test_U_dist, so we
    overwrite it with maha_mean (preserving the placeholder under
    test_U_dist_placeholder). Idempotent: skips if already promoted.
    """
    d = np.load(arr)
    if "test_U_dist_maha_mean" not in d.files:
        return False                              # no real Omega extracted
    if "test_U_dist_placeholder" in d.files:
        return True                               # already promoted
    out = {k: d[k] for k in d.files}
    out["test_U_dist_placeholder"] = d["test_U_dist"]
    out["test_U_dist"] = d["test_U_dist_maha_mean"]
    np.savez_compressed(arr, **out)
    print(f"  promoted real Omega -> test_U_dist in {arr}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,42",
                    help="comma-separated seeds; used to shard work across parallel instances")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/baseline_v1v2/fusion_v1v2_seedwise.csv"))
    ap.add_argument("--backbones", default="gdn,topogdn,cstgl")
    ap.add_argument("--variants", default="V1,V2")
    ap.add_argument("--region", default="full", choices=["full", "heldout"],
                    help="heldout = evaluate only rows past the stacker train slice (>= --held0)")
    ap.add_argument("--dataset", default=_ENV_DS, choices=["swat", "wadi"],
                    help="wadi switches DIRMAP + split/bundle + held0 defaults to the WADI campaign")
    ap.add_argument("--split", default=None, help="override calibration_set_indices.json path")
    ap.add_argument("--bundle", default=None, help="override calibration bundle dir")
    ap.add_argument("--held0", type=int, default=None,
                    help="held-out region start, WINDOWED coords (swat 24530, wadi 9445)")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.dataset == "wadi":
        dirmap = DIRMAP_WADI
        split = a.split or os.path.join(ROOT, "pretrained/wadi_ensemble/calibration_bundle/calibration_set_indices.json")
        bundle = a.bundle or os.path.join(ROOT, "pretrained/wadi_ensemble/calibration_bundle")
        HELD0 = a.held0 if a.held0 is not None else 9445
    else:
        dirmap = DIRMAP_SWAT
        split = a.split or SPLIT
        bundle = a.bundle or BUNDLE
        HELD0 = a.held0 if a.held0 is not None else 24530
    rows = []
    for bb in a.backbones.split(","):
        for V in a.variants.split(","):
            for s in seeds:
                full = os.path.join(ROOT, f"results/{dirmap[bb]}/{V}/seed{s}/arrays_full.npz")
                base = os.path.join(ROOT, f"results/{dirmap[bb]}/{V}/seed{s}/arrays.npz")
                arr = full if os.path.exists(full) else base   # prefer full-UQ arrays
                if not os.path.exists(arr):
                    continue
                if arr == full:
                    promote_omega(arr)            # real Omega -> distributional channel
                if not has_uq(arr):
                    rows.append(dict(backbone=bb, variant=V, seed=s, method="(no UQ channels)", PA_K_AUC="", F1=""))
                    print(f"[{bb} {V} s{s}] NO UQ channels -> needs extraction", flush=True)
                    continue
                ctx = setup_context(argparse.Namespace(arrays=arr, split=split, bundle=bundle, slide_win=60, seed=s))
                lab = np.asarray(ctx["label"])
                scores = FS.build_scores(ctx)
                try:
                    scores["S2_GBM"], _ = fit_M10_score(ctx)
                except Exception as e:
                    print(f"  gbm fail {bb} {V} s{s}: {e}", flush=True)
                try:
                    scores["S1_logistic"], _ = FS.logistic_stacker(ctx)
                except Exception as e:
                    print(f"  log fail {bb} {V} s{s}: {e}", flush=True)
                for m, sc in scores.items():
                    if a.region == "heldout":
                        pak, f1 = FS.eval_method(np.asarray(sc)[HELD0:], lab[HELD0:])
                    else:
                        pak, f1 = FS.eval_method(sc, lab)
                    rows.append(dict(backbone=bb, variant=V, seed=s, method=m,
                                     PA_K_AUC=round(pak, 4), F1=round(f1, 4)))
                    print(f"[{bb} {V} s{s} {m}] PA%K={pak:.4f} F1={f1:.4f}", flush=True)

    out = a.out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "variant", "seed", "method", "PA_K_AUC", "F1"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
