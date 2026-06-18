#!/usr/bin/env python3
"""Explain the S2_GBM fusion stacker: global importance + per-event profiles.

Mirrors compute_M10_PAK.fit_M10_score exactly (same grid, same selection by
eval_score_full F1 on the full arrays) but KEEPS the best estimator, then:
  1. persists the fitted model (joblib) + the cached feature matrix/labels;
  2. global explanation: sklearn permutation_importance (n_repeats=20,
     scoring=average_precision) on the full arrays and on the test-minus-val
     rows (to avoid crediting training fit), with the exact ordered feature
     names; 1-D partial dependence for the top features and the 2-D
     (agg_z, U_dist) pair;
  3. per-event explanation for the pilot attacks: in-window GBM log-odds
     series, the peak log-odds step, and the feature values + feature z
     (vs the nominal feature distribution) at that step.

Caveat (stated wherever these numbers are reported): GBM attribution measures
label-predictiveness of the features, not the physical nature of the anomaly;
the rule table is the typing instrument.

Runs in rashindaNew-torch-env, CPU, ~2-3 min per combo.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))

from fusion_sweep_K100_full import setup_context, build_stacker_features, eval_score_full  # noqa: E402
from fusion_v1v2 import promote_omega, DIRMAP, SPLIT, BUNDLE  # noqa: E402
from analyze_multistage_attacks import estimate_offset  # noqa: E402
from typing_rules_v1v2 import load_attack_table, PILOT_EVENTS, COMBOS  # noqa: E402
from typing_rules_v1v2 import OUTDIR as TYPING_OUTDIR  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance, partial_dependence  # noqa: E402

OUTDIR = os.path.join(TYPING_OUTDIR, "gbm")    # follows UQ_DATASET via typing_rules

FEAT8 = ["agg_z", "U_par_max_v", "U_par_mean_v", "sigma_ale_max_v",
         "U_str_mean_e", "U_dist", "U_par_max_x_agg", "log_sigma_tot_max"]
FEAT7 = [f for f in FEAT8 if f != "U_str_mean_e"]


def fit_best_gbm(ctx):
    """compute_M10_PAK.fit_M10_score, but returns the fitted best estimator."""
    label = ctx["label"]
    feat = build_stacker_features(ctx)
    val_idx = ctx["val_idx"]
    best = (-1.0, None, None, None)
    for depth in (2, 3, 5):
        for n_iter in (50, 100, 200):
            gb = HistGradientBoostingClassifier(
                max_depth=depth, max_iter=n_iter, learning_rate=0.05,
                l2_regularization=1.0, random_state=ctx["seed"],
                class_weight="balanced")
            gb.fit(feat[val_idx], label[val_idx])
            proba = gb.predict_proba(feat)[:, 1]
            s = np.log(np.clip(proba, 1e-8, 1 - 1e-8) /
                       np.clip(1 - proba, 1e-8, 1 - 1e-8))
            res = eval_score_full(s, label)
            if res["F1"] > best[0]:
                best = (res["F1"], gb, s, dict(max_depth=depth, max_iter=n_iter, **res))
    return best[1], best[2], best[3], feat, np.asarray(label), np.asarray(val_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=",".join(str(x) for x in PILOT_EVENTS))
    ap.add_argument("--combos", default="", help="override: bb:V:seed,bb:V:seed,...")
    args = ap.parse_args()
    pilot = {int(x) for x in args.events.split(",")}
    combos = COMBOS
    if args.combos:
        combos = [(c.split(":")[0], c.split(":")[1], int(c.split(":")[2]))
                  for c in args.combos.split(",")]
    os.makedirs(OUTDIR, exist_ok=True)
    atts = load_attack_table()

    for bb, V, seed in combos:
        arr = os.path.join(ROOT, "results", DIRMAP[bb], V, f"seed{seed}", "arrays_full.npz")
        promote_omega(arr)
        ctx = setup_context(argparse.Namespace(arrays=arr, split=SPLIT, bundle=BUNDLE,
                                               slide_win=60, seed=seed))
        gb, score, hp, feat, label, val_idx = fit_best_gbm(ctx)
        names = FEAT8 if feat.shape[1] == 8 else FEAT7
        tag = f"{bb}_{V}_s{seed}"
        joblib.dump(gb, os.path.join(OUTDIR, f"{tag}.joblib"))
        np.savez_compressed(os.path.join(OUTDIR, f"{tag}_cache.npz"),
                            feat=feat.astype(np.float32), label=label,
                            val_idx=val_idx, score=score.astype(np.float32))
        print(f"[{tag}] best HP depth={hp['max_depth']} iter={hp['max_iter']} "
              f"F1={hp['F1']:.4f} ({feat.shape[1]} features)", flush=True)

        # ---- global: permutation importance (full + test-minus-val) ----
        rng = np.random.RandomState(0)
        sub = rng.choice(len(label), size=min(20000, len(label)), replace=False)  # speed
        pi_full = permutation_importance(gb, feat[sub], label[sub], n_repeats=20,
                                         scoring="average_precision", random_state=0)
        nonval = np.ones(len(label), bool); nonval[val_idx] = False
        nv_idx = np.nonzero(nonval)[0]
        nv_sub = rng.choice(nv_idx, size=min(20000, len(nv_idx)), replace=False)
        pi_test = permutation_importance(gb, feat[nv_sub], label[nv_sub], n_repeats=20,
                                         scoring="average_precision", random_state=0)
        order = np.argsort(-pi_full.importances_mean)
        print("  permutation importance (full):", flush=True)
        for i in order:
            print(f"    {names[i]:18s} {pi_full.importances_mean[i]:+.4f} "
                  f"± {pi_full.importances_std[i]:.4f}", flush=True)

        # ---- partial dependence: top-3 1-D + (agg_z, U_dist) 2-D ----
        top3 = [int(i) for i in order[:3]]
        pdp = {}
        for i in top3:
            r = partial_dependence(gb, feat[sub], features=[i], grid_resolution=30)
            pdp[names[i]] = dict(grid=[float(x) for x in r["grid_values"][0]],
                                 avg=[float(x) for x in np.ravel(r["average"])])
        iu = names.index("U_dist")
        r2 = partial_dependence(gb, feat[sub], features=[(0, iu)], grid_resolution=15)
        pdp["agg_z__x__U_dist"] = dict(
            grid0=[float(x) for x in r2["grid_values"][0]],
            grid1=[float(x) for x in r2["grid_values"][1]],
            avg=[[float(v) for v in row] for row in r2["average"][0]])

        # ---- per-event profiles ----
        lab_int = label.astype(int)
        doff = estimate_offset(lab_int, atts)
        nominal = lab_int == 0
        fmean = feat[nominal].mean(0); fstd = feat[nominal].std(0) + 1e-9
        events = {}
        for a in atts:
            if a["aid"] not in pilot:
                continue
            s0, e0 = max(0, a["s"] + doff), min(len(lab_int), a["e"] + doff)
            if e0 <= s0:
                continue
            W = slice(s0, e0)
            pk = int(np.argmax(score[W]))
            fz = (feat[s0 + pk] - fmean) / fstd
            events[f"A{a['aid']:02d}"] = dict(
                start=s0, end=e0,
                logodds_peak=round(float(score[W].max()), 3),
                logodds_mean=round(float(score[W].mean()), 3),
                logodds_frac_pos=round(float((score[W] > 0).mean()), 3),
                peak_offset_s=pk,
                feat_at_peak={n: round(float(v), 3) for n, v in zip(names, feat[s0 + pk])},
                feat_z_at_peak={n: round(float(v), 2) for n, v in zip(names, fz)},
                top_features=[names[i] for i in np.argsort(-np.abs(fz))[:3]],
                logodds_series=[round(float(x), 3) for x in score[W]],
            )
        out = dict(combo=tag, best_hp={k: hp[k] for k in ("max_depth", "max_iter", "F1")},
                   feature_names=names,
                   permutation_importance_full={n: [float(pi_full.importances_mean[i]),
                                                    float(pi_full.importances_std[i])]
                                                for i, n in enumerate(names)},
                   permutation_importance_testminusval={n: [float(pi_test.importances_mean[i]),
                                                            float(pi_test.importances_std[i])]
                                                        for i, n in enumerate(names)},
                   partial_dependence=pdp, events=events,
                   caveat="GBM attribution measures label-predictiveness, not anomaly nature.")
        with open(os.path.join(OUTDIR, f"explain_{tag}.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(f"  wrote {OUTDIR}/explain_{tag}.json ({len(events)} pilot events)", flush=True)


if __name__ == "__main__":
    main()
