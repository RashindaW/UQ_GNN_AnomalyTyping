#!/usr/bin/env bash
# Phase-1 K-anchor sweep extension: K in {60, 70, 80, 90, 100} on the
# sw=60, 70:10:20 G-DeltaUQ seed=42 model.
#
# K=60/70/80/90 run in parallel on cuda:0/1/2/3, then K=100 runs on
# cuda:0 once the first batch is done. Same anchor_seed=0 / random
# strategy as the K=20/30/50 sweep so all bundles are nested
# (apples-to-apples).
#
# Rollup merges the new results with the prior K=10/20/30/50 sweep in
# results/postproc_threshold_fixA_kcal_sw60/0516-025453/ to produce a
# single K=10..100 curve.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
CKPT=pretrained/swat_gdeltauq_sw60/best_0513-211014.pt
HP=pretrained/swat_gdeltauq_sw60/hyperparameters_0513-211014.json
SPLIT=data/swat/gdeltauq_split.json
PRIOR_ROLLUP_DIR=results/postproc_threshold_fixA_kcal_sw60/0516-025453
mkdir -p logs

DATESTR=$(date +%m%d-%H%M%S)
ROLLUP_DIR="results/postproc_threshold_fixA_kcal_sw60/${DATESTR}_K60_100"
mkdir -p "$ROLLUP_DIR"

run_one_K() {
  local K=$1
  local DEV=$2
  local TAG="K${K}"
  local LOG="logs/kcal_seed42_sw60_${TAG}_${DATESTR}.log"
  local BUNDLE="pretrained/swat_gdeltauq_sw60/calibration_bundle_K${K}"
  local EVAL_DIR="results/swat_gdeltauq_sw60_paper_protocol_K${K}"

  {
    echo "=========================================================="
    echo "=== K=${K} dev=${DEV}  $(date)"
    echo "=========================================================="

    echo "--- CALIBRATE  $(date)"
    "$PY" scripts/calibrate_gdeltauq.py \
      -checkpoint "$CKPT" \
      -hyperparameters "$HP" \
      -split_path "$SPLIT" \
      -K_anchors "$K" -anchor_seed 0 -anchor_strategy random \
      -save_dir "$BUNDLE" \
      -device "$DEV"

    echo ""
    echo "--- EVAL  $(date)"
    "$PY" scripts/eval_paper_protocol_gdeltauq.py \
      -checkpoint "$CKPT" \
      -hyperparameters "$HP" \
      -bundle_dir "$BUNDLE" \
      -split_path "$SPLIT" \
      -topk 1 -device "$DEV" \
      -results_dir "$EVAL_DIR"

    NEW_ARRAYS=$(ls -t "${EVAL_DIR}"/*/arrays.npz | head -1)
    echo ""
    echo "--- Fix-A postproc on ${NEW_ARRAYS}  $(date)"
    "$PY" scripts/sweep_postproc_threshold.py \
      -arrays "$NEW_ARRAYS" \
      -out_root "${ROLLUP_DIR}/${TAG}" \
      -topk_grid 1 -smoothing_grid 5 -W_grid 5 -G_grid 5 \
      -n_taus 400

    echo ""
    echo "=== K=${K}: DONE  $(date)"
  } >> "$LOG" 2>&1
}

echo "=========================================================="
echo "=== Phase-1 K-sweep extension on sw=60 seed=42  $(date)"
echo "=== Batch 1: K in {60, 70, 80, 90} on cuda:0/1/2/3 (parallel)"
echo "=========================================================="

run_one_K 60 cuda:0 & PID60=$!
run_one_K 70 cuda:1 & PID70=$!
run_one_K 80 cuda:2 & PID80=$!
run_one_K 90 cuda:3 & PID90=$!
echo "K=60 pid=$PID60  K=70 pid=$PID70  K=80 pid=$PID80  K=90 pid=$PID90"

wait "$PID60" || echo "K=60 stream exited non-zero"
wait "$PID70" || echo "K=70 stream exited non-zero"
wait "$PID80" || echo "K=80 stream exited non-zero"
wait "$PID90" || echo "K=90 stream exited non-zero"

echo ""
echo "=========================================================="
echo "=== Batch 2: K=100 on cuda:0  $(date)"
echo "=========================================================="

run_one_K 100 cuda:0 & PID100=$!
echo "K=100 pid=$PID100"
wait "$PID100" || echo "K=100 stream exited non-zero"

echo ""
echo "=========================================================="
echo "=== ROLLUP (merging K=10..100)  $(date)"
echo "=========================================================="

"$PY" - <<PYEOF
import csv
import json
import numpy as np
from glob import glob
from pathlib import Path

new_dir = Path("$ROLLUP_DIR")
prior_dir = Path("$PRIOR_ROLLUP_DIR")
rows = []

# K=10 baseline (from prior sweep) and K=20/30/50 from prior rollup CSV.
prior_csv = prior_dir / "kcal_fixA_rollup.csv"
if prior_csv.exists():
    import csv as _csv
    with open(prior_csv) as f:
        for r in _csv.DictReader(f):
            rows.append({k: r[k] for k in r if k})

# K=60/70/80/90/100 (new).
for K in (60, 70, 80, 90, 100):
    matches = sorted(glob(str(new_dir / f"K{K}/*/best_fixA.json")),
                     reverse=True)
    if not matches:
        rows.append({"K": K, "config": None,
                     "F1_fixA": None, "P_fixA": None, "R_fixA": None,
                     "tau_fixA": None, "q_fixA": None,
                     "F1_legacy": None, "P_legacy": None, "R_legacy": None,
                     "best_path": None})
        continue
    with open(matches[0]) as f:
        d = json.load(f)
    rows.append({
        "K": K,
        "config": d.get("config"),
        "F1_fixA": d.get("F1_fixA"),
        "P_fixA": d.get("P_fixA"),
        "R_fixA": d.get("R_fixA"),
        "tau_fixA": d.get("tau_fixA"),
        "q_fixA": d.get("q_fixA"),
        "F1_legacy": d.get("F1_legacy"),
        "P_legacy": d.get("P_legacy"),
        "R_legacy": d.get("R_legacy"),
        "best_path": matches[0],
    })

def _f(v):
    try:
        return float(v) if v not in (None, '', 'None') else None
    except (TypeError, ValueError):
        return None

for r in rows:
    r["K"] = int(r["K"]) if r["K"] not in (None, '') else None
    for k in ("F1_fixA", "P_fixA", "R_fixA", "tau_fixA", "q_fixA",
             "F1_legacy", "P_legacy", "R_legacy"):
        r[k] = _f(r.get(k))

rows = [r for r in rows if r["K"] is not None]
rows.sort(key=lambda r: r["K"])

print()
print("Per-K Fix-A on sw=60 seed=42 (config=tk1_sm5_W5_G5):")
print(f"{'K':>4s}  {'F1_fixA':>8s} {'P_fixA':>8s} {'R_fixA':>8s} "
      f"{'tau_fixA':>8s} {'q_fixA':>8s}  {'F1_legacy':>9s}")
for r in rows:
    if r['F1_fixA'] is None:
        print(f"{r['K']:>4d}  {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} "
              f"{'N/A':>8s} {'N/A':>8s}  {'N/A':>9s}")
    else:
        print(f"{r['K']:>4d}  "
              f"{r['F1_fixA']:>8.4f} {r['P_fixA']:>8.4f} {r['R_fixA']:>8.4f} "
              f"{r['tau_fixA']:>8.2f} {r['q_fixA']:>8.4f}  "
              f"{r['F1_legacy']:>9.4f}")

f1s = [r['F1_fixA'] for r in rows if r['F1_fixA'] is not None]
if f1s:
    best = max(rows, key=lambda r: (r['F1_fixA'] or -1))
    print()
    print(f"best K = {best['K']}  F1_fixA = {best['F1_fixA']:.4f}  "
          f"P = {best['P_fixA']:.4f}  R = {best['R_fixA']:.4f}")

out_csv = new_dir / "kcal_fixA_rollup_K10_100.csv"
fields = ["K", "config", "F1_fixA", "P_fixA", "R_fixA", "tau_fixA",
          "q_fixA", "F1_legacy", "P_legacy", "R_legacy", "best_path"]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nrollup -> {out_csv}")
PYEOF

echo ""
echo "=========================================================="
echo "=== ALL DONE  $(date)"
echo "=========================================================="
