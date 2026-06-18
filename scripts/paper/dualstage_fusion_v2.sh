#!/usr/bin/env bash
# DualSTGF (dualstage) V2 seeds 0-4 fusion eval for thesis Part 1 table parity:
# full-arrays region first, then held-out region (rows >= 24530), each sharded
# by seed (5 parallel CPU instances) and merged. Mirrors heldout_v2_seeds04.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
D=results/baseline_v1v2/fusion_dualstage
mkdir -p "$D"
SEEDS=(0 1 2 3 4)
rc=0
for REGION in full heldout; do
  echo "[dualstage-fusion] region=$REGION launching ${#SEEDS[@]} instances $(date)"
  pids=()
  for S in "${SEEDS[@]}"; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" scripts/paper/fusion_v1v2.py \
      --backbones dualstage --variants V2 --seeds "$S" --region "$REGION" \
      --out "$D/seed${S}_V2_${REGION}.csv" > "$D/seed${S}_${REGION}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  M="$D/fusion_dualstage_V2_${REGION}_seeds0-4.csv"; first=1
  for S in "${SEEDS[@]}"; do
    f="$D/seed${S}_V2_${REGION}.csv"
    [ -f "$f" ] || { echo "  MISSING $f"; rc=1; continue; }
    if [ "$first" -eq 1 ]; then cat "$f" > "$M"; first=0; else tail -n +2 "$f" >> "$M"; fi
  done
  echo "[dualstage-fusion] region=$REGION merged -> $M ($(($(wc -l < "$M")-1)) rows) rc=$rc $(date)"
done
exit "$rc"
