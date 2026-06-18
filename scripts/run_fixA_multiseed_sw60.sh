#!/usr/bin/env bash
# Multi-seed Fix-A postproc on sw=60, 70:10:20 G-DeltaUQ arrays.
#
# Each seed already has cached arrays.npz from the paper-protocol eval.
# We apply sweep_postproc_threshold.py with the Fix-A best HP combination
# (config tk1_sm5_W5_G5: topk=1, smoothing=5, extend_W=5, merge_G=5) and
# collect per-seed F1/P/R, then print mean / std across seeds.
#
# CPU-only. Designed for tmux session `fixA-multiseed-sw60`.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
mkdir -p logs results/postproc_threshold_fixA_multiseed_sw60

declare -A ARRAYS=(
  [1]="results/swat_gdeltauq_70_sw60_seed1_paper_protocol/0514-202100/arrays.npz"
  [2]="results/swat_gdeltauq_70_sw60_seed2_paper_protocol/0514-202606/arrays.npz"
  [3]="results/swat_gdeltauq_70_sw60_seed3_paper_protocol/0514-202239/arrays.npz"
  [42]="results/swat_gdeltauq_sw60_paper_protocol/0513-211654/arrays.npz"
)

DATESTR=$(date +%m%d-%H%M%S)
ROLLUP_DIR="results/postproc_threshold_fixA_multiseed_sw60/${DATESTR}"
mkdir -p "$ROLLUP_DIR"

echo "=========================================================="
echo "=== FixA multi-seed postproc (sw=60, 70:10:20)  $(date)"
echo "=========================================================="

for SEED in 1 2 3 42; do
  ARR="${ARRAYS[$SEED]}"
  if [[ ! -f "$ARR" ]]; then
    echo "ERROR: missing arrays for seed=${SEED}: $ARR" >&2
    exit 1
  fi
  echo ""
  echo "--- seed=${SEED}: postproc on $ARR  $(date)"
  "$PY" scripts/sweep_postproc_threshold.py \
    -arrays "$ARR" \
    -out_root "$ROLLUP_DIR/seed${SEED}" \
    -topk_grid 1 -smoothing_grid 5 -W_grid 5 -G_grid 5 \
    -n_taus 400 \
    2>&1 | tee "logs/fixA_seed${SEED}_${DATESTR}.log"
done

echo ""
echo "=========================================================="
echo "=== ROLLUP  $(date)"
echo "=========================================================="

"$PY" - <<PYEOF
import csv
import json
import numpy as np
from glob import glob
from pathlib import Path

rollup_dir = Path("$ROLLUP_DIR")
rows = []
for seed in (1, 2, 3, 42):
    matches = sorted(glob(str(rollup_dir / f"seed{seed}/*/best_fixA.json")),
                     reverse=True)
    if not matches:
        rows.append({"seed": seed, "config": None,
                     "F1_fixA": None, "P_fixA": None, "R_fixA": None,
                     "tau_fixA": None, "q_fixA": None,
                     "F1_legacy": None, "lift": None, "best_path": None})
        continue
    with open(matches[0]) as f:
        d = json.load(f)
    rows.append({
        "seed": seed,
        "config": d.get("config"),
        "F1_fixA": d.get("F1_fixA"),
        "P_fixA": d.get("P_fixA"),
        "R_fixA": d.get("R_fixA"),
        "tau_fixA": d.get("tau_fixA"),
        "q_fixA": d.get("q_fixA"),
        "F1_legacy": d.get("F1_legacy"),
        "lift": d.get("lift"),
        "best_path": matches[0],
    })

print()
print("Per-seed Fix-A results (sw=60, 70:10:20, config=tk1_sm5_W5_G5):")
print(f"{'seed':>4s}  {'F1_fixA':>8s} {'P_fixA':>8s} {'R_fixA':>8s} "
      f"{'tau_fixA':>8s} {'q_fixA':>8s}  {'F1_legacy':>9s}")
for r in rows:
    if r['F1_fixA'] is None:
        print(f"{r['seed']:>4d}  {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} "
              f"{'N/A':>8s} {'N/A':>8s}  {'N/A':>9s}")
    else:
        print(f"{r['seed']:>4d}  "
              f"{r['F1_fixA']:>8.4f} {r['P_fixA']:>8.4f} {r['R_fixA']:>8.4f} "
              f"{r['tau_fixA']:>8.2f} {r['q_fixA']:>8.4f}  "
              f"{r['F1_legacy']:>9.4f}")

f1s = [r['F1_fixA'] for r in rows if r['F1_fixA'] is not None]
ps  = [r['P_fixA']  for r in rows if r['P_fixA']  is not None]
rs  = [r['R_fixA']  for r in rows if r['R_fixA']  is not None]
if f1s:
    print()
    print(f"n={len(f1s)} seeds")
    print(f"F1_fixA : mean={np.mean(f1s):.4f}  std={np.std(f1s, ddof=1):.4f}  "
          f"min={min(f1s):.4f}  max={max(f1s):.4f}")
    print(f"P_fixA  : mean={np.mean(ps):.4f}  std={np.std(ps, ddof=1):.4f}")
    print(f"R_fixA  : mean={np.mean(rs):.4f}  std={np.std(rs, ddof=1):.4f}")

out_csv = rollup_dir / "fixA_multiseed_rollup.csv"
fields = ["seed", "config", "F1_fixA", "P_fixA", "R_fixA", "tau_fixA",
          "q_fixA", "F1_legacy", "lift", "best_path"]
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
