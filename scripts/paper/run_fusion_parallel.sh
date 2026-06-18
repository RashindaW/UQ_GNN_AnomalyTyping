#!/usr/bin/env bash
# Run fusion_v1v2.py sharded by seed (6 parallel instances), then merge the per-seed
# CSVs into the single seedwise CSV. CPU-only; ~6x faster than the sequential sweep.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
OUT=results/baseline_v1v2; mkdir -p "$OUT/fusion_logs"
SEEDS=(0 1 2 3 4 42)
echo "[fusion-par] launching ${#SEEDS[@]} per-seed instances $(date)"
pids=()
for S in "${SEEDS[@]}"; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" scripts/paper/fusion_v1v2.py \
    --seeds "$S" --out "$OUT/fusion_v1v2_seed${S}.csv" > "$OUT/fusion_logs/seed${S}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[fusion-par] instances done $(date); merging"
M="$OUT/fusion_v1v2_seedwise.csv"; first=1
for S in "${SEEDS[@]}"; do
  f="$OUT/fusion_v1v2_seed${S}.csv"
  [ -f "$f" ] || { echo "  MISSING $f"; continue; }
  if [ "$first" -eq 1 ]; then cat "$f" > "$M"; first=0; else tail -n +2 "$f" >> "$M"; fi
done
echo "[fusion-par] merged -> $M ($(($(wc -l < "$M")-1)) rows) $(date)"
