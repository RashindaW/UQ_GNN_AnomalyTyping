#!/usr/bin/env bash
# Held-out (leak-free) fusion eval for the thesis protocol: V2, seeds 0-4, 3 backbones.
# All 7 methods evaluated on arrays rows >= 24530 (past the stacker train slice),
# tau-sweep restricted to the same region -> uniform convention across methods.
# Sharded by seed (5 parallel CPU instances), merged at the end.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
D=results/baseline_v1v2/fusion_heldout
mkdir -p "$D"
SEEDS=(0 1 2 3 4)
echo "[heldout-v2] launching ${#SEEDS[@]} per-seed instances $(date)"
pids=()
for S in "${SEEDS[@]}"; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" scripts/paper/fusion_v1v2.py \
    --seeds "$S" --variants V2 --region heldout \
    --out "$D/seed${S}_V2_heldout.csv" > "$D/seed${S}_V2.log" 2>&1 &
  pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[heldout-v2] instances done rc=$rc $(date); merging"
M="$D/fusion_v1v2_heldout_V2_seeds0-4.csv"; first=1
for S in "${SEEDS[@]}"; do
  f="$D/seed${S}_V2_heldout.csv"
  [ -f "$f" ] || { echo "  MISSING $f"; rc=1; continue; }
  if [ "$first" -eq 1 ]; then cat "$f" > "$M"; first=0; else tail -n +2 "$f" >> "$M"; fi
done
echo "[heldout-v2] merged -> $M ($(($(wc -l < "$M")-1)) rows) rc=$rc $(date)"
exit "$rc"
