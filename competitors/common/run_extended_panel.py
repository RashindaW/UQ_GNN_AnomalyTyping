#!/usr/bin/env python3
"""Compute the Track-B extended metric panel for GDN + competitor backbones.

For every (backbone, seed) it builds the M0 residual top-1 anomaly score
DIRECTLY from the cached npz (the same pipeline as
competitors/common/eval_from_arrays.evaluate_baseline_only:
build_full_err_scores(..., before_num=5) -> topk_aggregate(..., 1)), then runs
competitors/common/extra_metrics.full_panel on it.

We deliberately DO NOT call fusion_sweep_K100_full.setup_context here, because
(a) setup_context requires the UQ channels (test_U_par, ...) that the competitor
M0 *baseline* npz files do not carry, and (b) it reads val via a key that some
caches do not expose. The M0 residual score only needs mu/gt, so building it
directly keeps the panel runnable on every backbone while remaining the IDENTICAL
canonical M0 channel -> apples-to-apples with the existing F1 / PA%K table.

Writes (NEW files only):
    results/paper/metrics/extended_panel.csv
    results/paper/metrics/extended_panel.json
    results/paper/metrics/extended_report.md

Pure CPU, no training, reads only cached arrays.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
COMMON = Path(__file__).resolve().parent
for p in (REPO_ROOT, SCRIPTS, COMMON):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sweep_eval_gdeltauq import build_full_err_scores, topk_aggregate  # noqa: E402
import extra_metrics as em  # noqa: E402

VUS_WINDOW = 100   # buffer half-widths 0..100 (Paparrizos range-AUC volume)
M0_SMOOTH = 5      # before_num used by eval_from_arrays.evaluate_baseline_only
M0_TOPK = 1


def m0_scores_from_npz(arrays_path: str):
    """Return (test_score, test_label, val_score) for the canonical M0 channel.

    test_score mirrors eval_from_arrays.evaluate_baseline_only exactly. val_score
    applies the identical pipeline with the val slice as its own (anomaly-free)
    calibration+eval slice, yielding the nominal band for a val-fit threshold.
    """
    d = np.load(arrays_path, allow_pickle=True)
    test_mu = d["test_mu_bar"].astype(np.float64)
    test_gt = d["test_ground_truth"].astype(np.float64)
    val_mu = d["val_mu_bar"].astype(np.float64)
    val_gt = d["val_ground_truth"].astype(np.float64)
    label = d["test_attack_label"].astype(np.int8).ravel()

    full_scores = build_full_err_scores(test_mu, test_gt, val_mu, val_gt, M0_SMOOTH)
    test_score = topk_aggregate(full_scores, M0_TOPK).astype(np.float64)

    val_full = build_full_err_scores(val_mu, val_gt, val_mu, val_gt, M0_SMOOTH)
    val_score = topk_aggregate(val_full, M0_TOPK).astype(np.float64)
    return test_score, label, val_score


def panel_for(arrays_path: str, vus_window: int = VUS_WINDOW):
    score, label, val_score = m0_scores_from_npz(arrays_path)
    return em.full_panel(score, label, val_score=val_score, vus_window=vus_window,
                         pred_for_event_metrics="oracle")


def _discover():
    items = []
    gdn_map = {
        "42": "results/gdn/ref_seed42/arrays.npz",
        "1": "results/gdn/seed1/arrays.npz",
        "2": "results/gdn/seed2/arrays.npz",
        "3": "results/gdn/seed3/arrays.npz",
        "100": "results/gdn/seed100/arrays.npz",
    }
    for s, p in gdn_map.items():
        fp = REPO_ROOT / p
        if fp.exists():
            items.append(("gdn", s, str(fp)))

    comp_dirs = {
        "cstgl": "results/competitors/cstgl",
        "gta": "results/competitors/gta",
        "topogdn": "results/competitors/topogdn",
    }
    for name, d in comp_dirs.items():
        for s in ("42", "1", "2", "3", "100"):
            for fp in (REPO_ROOT / d / f"seed{s}_baseline_arrays.npz",
                       REPO_ROOT / d / f"{name}_seed{s}_baseline_arrays.npz"):
                if fp.exists():
                    items.append((name, s, str(fp)))
                    break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-seed42-competitors", action="store_true")
    ap.add_argument("--vus-window", type=int, default=VUS_WINDOW)
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results/paper/metrics"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    work = _discover()
    if args.only_seed42_competitors:
        work = [w for w in work if (w[0] == "gdn" or w[1] == "42")]
    print("discovered %d (backbone,seed) targets" % len(work), flush=True)

    rows = []
    for (backbone, seed, path) in work:
        print("[panel] %s seed%s :: %s" % (backbone, seed, path), flush=True)
        try:
            panel = panel_for(path, vus_window=args.vus_window)
        except Exception as exc:
            import traceback
            print("   ERROR: %r\n%s" % (exc, traceback.format_exc()), flush=True)
            continue
        row = {"backbone": backbone, "seed": seed}
        row.update(panel)
        rows.append(row)
        print("   AUROC=%.4f AUPRC=%.4f VUS_ROC=%.4f VUS_PR=%.4f rawF1_or=%.4f "
              "rawF1_vf=%s affP=%.4f affR=%.4f"
              % (panel["AUROC"], panel["AUPRC"], panel["VUS_ROC"], panel["VUS_PR"],
                 panel["raw_F1_oracle"],
                 ("%.4f" % panel.get("raw_F1_valfit", float("nan"))),
                 panel["aff_precision"], panel["aff_recall"]), flush=True)

    csv_path = outdir / "extended_panel.csv"
    fields = ["backbone", "seed", "base_rate",
              "AUROC", "AUPRC", "VUS_ROC", "VUS_PR", "vus_window",
              "raw_F1_oracle", "raw_P_oracle", "raw_R_oracle", "raw_tau_oracle",
              "raw_F1_valfit", "raw_P_valfit", "raw_R_valfit", "raw_tau_valfit",
              "aff_precision", "aff_recall", "aff_f1",
              "n_true_events", "n_detected_events", "n_missed_events",
              "n_false_alarm_runs", "n_pred_runs"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d data rows)" % (csv_path, len(rows)), flush=True)

    (outdir / "extended_panel.json").write_text(json.dumps(rows, indent=2))
    _write_report(rows, outdir / "extended_report.md", args.vus_window)
    print("wrote %s" % (outdir / "extended_report.md"), flush=True)
    print("PANEL_COMPLETE rows=%d" % len(rows), flush=True)
    return len(rows)


def _fmt(x, nd=4):
    try:
        if x is None:
            return "n/a"
        if isinstance(x, float) and (x != x):
            return "nan"
        return ("%." + str(nd) + "f") % float(x)
    except Exception:
        return str(x)


def _write_report(rows, path, vus_window):
    lines = []
    A = lines.append
    A("# Track-B Extended Detection Metrics -- Panel Report")
    A("")
    A("Module: competitors/common/extra_metrics.py  (unit tests: test_extra_metrics.py)")
    A("Score evaluated: M0 residual top-1 anomaly score (before_num=5, topk=1),")
    A("built directly from each npz with the same pipeline as")
    A("eval_from_arrays.evaluate_baseline_only -> apples-to-apples with the")
    A("existing F1 / PA%K table.")
    A("VUS buffer half-width range: 0..%d (range-AUC volume, Paparrizos VLDB 2022)." % vus_window)
    A("Affiliation P/R: probabilistic distance form of Huet KDD 2022 (see module docstring).")
    A("raw-F1 = NON point-adjusted F1; oracle = test-swept tau (ceiling, labelled oracle),")
    A("val-fit = nominal-max tau from the anomaly-free validation residual (primary).")
    A("Event-level + affiliation metrics use the oracle raw-F1 hard prediction.")
    A("")

    A("## Table 1 -- Threshold-robust + ranking metrics (M0 score)")
    A("")
    hdr = ["backbone", "seed", "AUROC", "AUPRC", "VUS_ROC", "VUS_PR",
           "rawF1_valfit", "rawF1_oracle", "aff_P", "aff_R", "aff_F1"]
    A("| " + " | ".join(hdr) + " |")
    A("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        cells = [r["backbone"], r["seed"],
                 _fmt(r.get("AUROC")), _fmt(r.get("AUPRC")),
                 _fmt(r.get("VUS_ROC")), _fmt(r.get("VUS_PR")),
                 _fmt(r.get("raw_F1_valfit")), _fmt(r.get("raw_F1_oracle")),
                 _fmt(r.get("aff_precision")), _fmt(r.get("aff_recall")),
                 _fmt(r.get("aff_f1"))]
        A("| " + " | ".join(str(c) for c in cells) + " |")
    A("")

    A("## Table 2 -- Event-level counts (oracle raw-F1 threshold)")
    A("")
    hdr2 = ["backbone", "seed", "n_true", "n_detected", "n_missed",
            "n_FA_runs", "n_pred_runs"]
    A("| " + " | ".join(hdr2) + " |")
    A("|" + "|".join(["---"] * len(hdr2)) + "|")
    for r in rows:
        cells = [r["backbone"], r["seed"],
                 r.get("n_true_events"), r.get("n_detected_events"),
                 r.get("n_missed_events"), r.get("n_false_alarm_runs"),
                 r.get("n_pred_runs")]
        A("| " + " | ".join(str(c) for c in cells) + " |")
    A("")

    gdn_rows = [r for r in rows if r["backbone"] == "gdn"]
    if gdn_rows:
        A("## GDN across %d seeds (mean +/- std)" % len(gdn_rows))
        A("")
        keys = [("AUROC", "AUROC"), ("AUPRC", "AUPRC"),
                ("VUS_ROC", "VUS_ROC"), ("VUS_PR", "VUS_PR"),
                ("raw_F1_valfit", "rawF1_valfit"), ("raw_F1_oracle", "rawF1_oracle"),
                ("aff_precision", "aff_P"), ("aff_recall", "aff_R")]
        A("| metric | mean | std |")
        A("|---|---|---|")
        for k, lab in keys:
            vals = np.array([float(r[k]) for r in gdn_rows if r.get(k) is not None
                             and not (isinstance(r[k], float) and r[k] != r[k])], dtype=float)
            if vals.size:
                A("| %s | %s | %s |" % (lab, _fmt(vals.mean()), _fmt(vals.std())))
        A("")

    A("## Ranking note -- do these metrics agree with PA%K? (seed42)")
    A("")
    s42 = {r["backbone"]: r for r in rows if r["seed"] == "42"}
    metrics_for_rank = [("AUPRC", "AUPRC"), ("VUS_ROC", "VUS_ROC"),
                        ("VUS_PR", "VUS_PR"), ("raw_F1_oracle", "raw-F1(oracle)"),
                        ("aff_f1", "affiliation-F1")]
    if len(s42) >= 2:
        A("Per-metric ranking of seed42 backbones (best -> worst):")
        A("")
        for k, lab in metrics_for_rank:
            ranked = sorted(
                [(b, s42[b].get(k)) for b in s42 if s42[b].get(k) is not None
                 and not (isinstance(s42[b].get(k), float) and s42[b].get(k) != s42[b].get(k))],
                key=lambda t: t[1], reverse=True,
            )
            order = " > ".join("%s(%s)" % (b, _fmt(v)) for b, v in ranked)
            A("- %s: %s" % (lab, order))
        A("")
    A("Interpretation: PA%K-AUC (already in the repo) point-adjusts a whole event")
    A("to positive once a fraction K of it is hit, inflating recall on the long")
    A("SWaT attack segments. The metrics above are NOT point-adjusted, so they")
    A("reorder methods whenever a backbone wins on PA%K only by catching a sliver")
    A("of each long event:")
    A("  - AUPRC / VUS-PR reward dense, well-ranked anomaly scores under heavy")
    A("    class imbalance (base rate ~0.12) and penalise false-alarm mass that")
    A("    point-adjustment hides.")
    A("  - raw-F1 (no PA) is the strict per-timestep score; a large gap between")
    A("    raw-F1 and PA%K F1 flags methods that lean on point-adjustment.")
    A("  - affiliation P/R is threshold/length aware and credits near-misses")
    A("    without the all-or-nothing event flip of PA, ranking methods by")
    A("    temporal localisation quality rather than event coverage alone.")
    A("Recommendation (prospectus 5.1): report VUS-ROC/PR + AUPRC + affiliation")
    A("alongside PA%K; do not rank on PA%K alone.")
    A("")

    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
