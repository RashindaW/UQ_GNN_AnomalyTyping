#!/usr/bin/env bash
# Phase-1 K-anchor sweep on the sw=60, 70:10:20 G-DeltaUQ seed=42 model.
#
# For K in {20, 30, 50}: recalibrate anchor pool (random sampling, fixed
# anchor_seed=0 so the K=50 pool contains the K=30 pool contains K=20 —
# apples-to-apples comparison), re-run paper-protocol eval, then apply
# Fix-A postproc (tk1_sm5_W5_G5) to the resulting arrays.npz.
#
# K=10 is the existing baseline (results/swat_gdeltauq_sw60_paper_protocol/
# 0513-211654/) — included in the rollup but not re-run.
#
# Parallel across cuda:0/1/2 (one K per GPU). cuda:3 stays idle.
# Designed for tmux session `kcal-seed42-sw60`.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
CKPT=pretrained/swat_gdeltauq_sw60/best_0513-211014.pt
HP=pretrained/swat_gdeltauq_sw60/hyperparameters_0513-211014.json
SPLIT=data/swat/gdeltauq_split.json
mkdir -p logs

DATESTR=$(date +%m%d-%H%M%S)
ROLLUP_DIR="results/postproc_threshold_fixA_kcal_sw60/${DATESTR}"
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
echo "=== Phase-1 K-sweep on sw=60 seed=42  $(date)"
echo "=== K in {20, 30, 50} parallel across cuda:0/1/2"
echo "=========================================================="

run_one_K 20 cuda:0 & PID20=$!
run_one_K 30 cuda:1 & PID30=$!
run_one_K 50 cuda:2 & PID50=$!
echo "K=20 pid=$PID20  K=30 pid=$PID30  K=50 pid=$PID50"

wait "$PID20" || echo "K=20 stream exited non-zero"
wait "$PID30" || echo "K=30 stream exited non-zero"
wait "$PID50" || echo "K=50 stream exited non-zero"

echo ""
echo "=========================================================="
echo "=== ROLLUP across K  $(date)"
echo "=========================================================="

"$PY" - <<PYEOF
import csv
import json
import numpy as np
from glob import glob
from pathlib import Path

rollup_dir = Path("$ROLLUP_DIR")
rows = []

# K=10 baseline (existing).
baseline_arrays = "results/swat_gdeltauq_sw60_paper_protocol/0513-211654/arrays.npz"
baseline_fixA = "pretrained/postproc_threshold_fixA"  # not used
# Use the existing best_fixA.json from the original sweep at this dir:
existing = sorted(glob(
    "results/postproc_threshold_fixA/0513-211654_*/best_fixA.json"
), reverse=True)
if existing:
    with open(existing[0]) as f:
        d = json.load(f)
    rows.append({
        "K": 10,
        "config": d.get("config"),
        "F1_fixA": d.get("F1_fixA"),
        "P_fixA": d.get("P_fixA"),
        "R_fixA": d.get("R_fixA"),
        "tau_fixA": d.get("tau_fixA"),
        "q_fixA": d.get("q_fixA"),
        "F1_legacy": d.get("F1_legacy"),
        "P_legacy": d.get("P_legacy"),
        "R_legacy": d.get("R_legacy"),
        "best_path": existing[0],
    })

# K=20, 30, 50 (new).
for K in (20, 30, 50):
    matches = sorted(glob(str(rollup_dir / f"K{K}/*/best_fixA.json")),
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

# Sort by K
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

out_csv = rollup_dir / "kcal_fixA_rollup.csv"
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
