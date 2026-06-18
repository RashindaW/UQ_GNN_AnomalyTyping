#!/usr/bin/env bash
# Re-score the 12 GDN V1/V2 baseline arrays SEQUENTIALLY with capped threads.
# Arrays already exist (GPU work done); this just runs the cheap M0 eval without
# the 13-way CPU thrash that drove load to 741. ~1-2 min each, ~20 min total.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
SEEDS=(0 1 2 3 4 42)
OUT=results/baseline_v1v2/gdn
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
echo "[rescore-gdn] $(date)"
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  arr="$OUT/$V/seed$S/arrays.npz"
  [ -f "$arr" ] || { echo "$V-s$S: NO ARRAYS, skip"; continue; }
  echo "[rescore] $V-s$S $(date '+%H:%M:%S')"
  "$PY" competitors/common/eval_from_arrays.py --arrays "$arr" \
    --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
    --baseline-only --label "GDN-$V-s$S" --out "$OUT/$V/seed$S/eval.json" \
    > "$OUT/logs/rescore_${V}_seed${S}.log" 2>&1 \
    && echo 0 > "$OUT/logs/${V}_seed${S}.done" \
    || echo "$V-s$S FAILED rc=$?"
done; done
echo "[rescore-gdn] DONE $(date)"
echo "done sentinels: $(ls $OUT/logs/*.done 2>/dev/null | wc -l)/12"
echo 0 > "$OUT/main.done"
