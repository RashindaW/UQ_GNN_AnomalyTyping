#!/usr/bin/env bash
# Phase-1 K-anchor sweep extension #2: K in {120, 140, 160, 180, 200} on
# the sw=60, 70:10:20 G-DeltaUQ seed=42 model.
#
# K=120/140/160/180 run in parallel on cuda:0/1/2/3, then K=200 on
# cuda:0 once batch 1 is done. Same anchor_seed=0 / random strategy as
# the prior sweeps so all bundles are nested (apples-to-apples).
#
# Rollup merges the new results with the K=10..100 rollup at
# results/postproc_threshold_fixA_kcal_sw60/0516-030554_K60_100/ to
# produce a single K=10..200 curve.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
CKPT=pretrained/swat_gdeltauq_sw60/best_0513-211014.pt
HP=pretrained/swat_gdeltauq_sw60/hyperparameters_0513-211014.json
SPLIT=data/swat/gdeltauq_split.json
PRIOR_ROLLUP_CSV=results/postproc_threshold_fixA_kcal_sw60/0516-030554_K60_100/kcal_fixA_rollup_K10_100.csv
mkdir -p logs

DATESTR=$(date +%m%d-%H%M%S)
ROLLUP_DIR="results/postproc_threshold_fixA_kcal_sw60/${DATESTR}_K120_200"
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
echo "=== Phase-1 K-sweep #2 on sw=60 seed=42  $(date)"
echo "=== Batch 1: K in {120, 140, 160, 180} on cuda:0/1/2/3 (parallel)"
echo "=========================================================="

run_one_K 120 cuda:0 & PID120=$!
run_one_K 140 cuda:1 & PID140=$!
run_one_K 160 cuda:2 & PID160=$!
run_one_K 180 cuda:3 & PID180=$!
echo "K=120 pid=$PID120  K=140 pid=$PID140  K=160 pid=$PID160  K=180 pid=$PID180"

wait "$PID120" || echo "K=120 stream exited non-zero"
wait "$PID140" || echo "K=140 stream exited non-zero"
wait "$PID160" || echo "K=160 stream exited non-zero"
wait "$PID180" || echo "K=180 stream exited non-zero"

echo ""
echo "=========================================================="
echo "=== Batch 2: K=200 on cuda:0  $(date)"
echo "=========================================================="

run_one_K 200 cuda:0 & PID200=$!
echo "K=200 pid=$PID200"
wait "$PID200" || echo "K=200 stream exited non-zero"

echo ""
echo "=========================================================="
echo "=== ROLLUP (merging K=10..200)  $(date)"
echo "=========================================================="

"$PY" - <<PYEOF
import csv
import json
import numpy as np
from glob import glob
from pathlib import Path

new_dir = Path("$ROLLUP_DIR")
prior_csv = Path("$PRIOR_ROLLUP_CSV")
rows = []

# Prior K=10..100 rollup.
if prior_csv.exists():
    import csv as _csv
    with open(prior_csv) as f:
        for r in _csv.DictReader(f):
            rows.append({k: r[k] for k in r if k})

# New K=120/140/160/180/200.
for K in (120, 140, 160, 180, 200):
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

out_csv = new_dir / "kcal_fixA_rollup_K10_200.csv"
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
