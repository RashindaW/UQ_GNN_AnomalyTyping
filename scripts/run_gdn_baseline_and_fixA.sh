#!/usr/bin/env bash
# Stage 1: Train original GDN (Deng & Hooi) on SWaT at slide_win ∈ {5, 15, 60}.
# Stage 2: Apply Fix A (post-proc-aware threshold) to every available arrays.npz
#          across both the GDN baselines and the GDeltaUQ pipeline outputs.
#
# Each Stage-1 training fans out to a separate CUDA so all three run in
# parallel. Stage 2 runs CPU-only after all GDN trainings finish.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
mkdir -p logs

train_gdn() {
  local SW=$1
  local DEV=$2
  local TAG="swat_gdn_sw${SW}"
  local LOG="logs/gdn_${TAG}.log"
  {
    echo "=== ${TAG}: TRAIN sw=${SW} dev=${DEV}  $(date)"
    "$PY" main.py \
      -dataset swat \
      -model gdn \
      -save_path_pattern "$TAG" \
      -slide_win "$SW" \
      -slide_stride 1 \
      -batch 128 \
      -epoch 100 \
      -dim 64 \
      -out_layer_num 1 \
      -out_layer_inter_dim 128 \
      -val_ratio 0.1 \
      -topk 15 \
      -decay 0.0 \
      -random_seed 42 \
      -report best \
      -save_arrays \
      -device "$DEV" \
      -comment "GDN baseline sw=${SW}"
    echo "=== ${TAG}: DONE  $(date)"
  } &> "$LOG"
}

echo "=========================================================="
echo "=== STAGE 1: GDN BASELINES  $(date)"
echo "=========================================================="

train_gdn  5 cuda:0 & PID5=$!
train_gdn 15 cuda:1 & PID15=$!
train_gdn 60 cuda:2 & PID60=$!
echo "GDN sw=5  pid=$PID5  log=logs/gdn_swat_gdn_sw5.log"
echo "GDN sw=15 pid=$PID15 log=logs/gdn_swat_gdn_sw15.log"
echo "GDN sw=60 pid=$PID60 log=logs/gdn_swat_gdn_sw60.log"

wait "$PID5"  || echo "GDN sw=5 exited non-zero"
wait "$PID15" || echo "GDN sw=15 exited non-zero"
wait "$PID60" || echo "GDN sw=60 exited non-zero"

echo "=========================================================="
echo "=== STAGE 2: FIX A ACROSS ALL ARRAYS.NPZ  $(date)"
echo "=========================================================="

# Collect every arrays.npz we want to evaluate.
FIXA_OUT_ROOT=results/postproc_threshold_fixA
mkdir -p "$FIXA_OUT_ROOT"

declare -A ARRAYS_MAP
ARRAYS_MAP[gdeltauq_sw5]=results/swat_gdeltauq_paper_protocol/0511-222735/arrays.npz
ARRAYS_MAP[gdeltauq_sw60]=results/swat_gdeltauq_sw60_paper_protocol/0513-211654/arrays.npz
ARRAYS_MAP[gdeltauq_sw120]=results/swat_gdeltauq_sw120_paper_protocol/0514-174137/arrays.npz
# Newly produced GDN baselines (resolve at runtime to the latest matching file).
for SW in 5 15 60; do
  P=$(ls -t "results/swat_gdn_sw${SW}"/*_arrays.npz 2>/dev/null | head -1 || true)
  if [[ -n "$P" ]]; then
    ARRAYS_MAP[gdn_sw${SW}]="$P"
  fi
done

for KEY in "${!ARRAYS_MAP[@]}"; do
  ARR="${ARRAYS_MAP[$KEY]}"
  if [[ -z "$ARR" || ! -f "$ARR" ]]; then
    echo "skip ${KEY}: missing $ARR"
    continue
  fi
  echo "--- Fix A on ${KEY}: ${ARR}"
  "$PY" scripts/sweep_postproc_threshold.py \
    -arrays "$ARR" \
    -out_root "$FIXA_OUT_ROOT" \
    -configs paper cheap paperW3 cheapW3 \
    &> "logs/fixA_${KEY}.log"
  echo "  log -> logs/fixA_${KEY}.log"
done

echo "=========================================================="
echo "=== STAGE 3: ROLLUP  $(date)"
echo "=========================================================="

"$PY" - <<'PYEOF'
import csv
import json
from glob import glob
from pathlib import Path
import pandas as pd

# Collect all fixA_sweep.csv outputs and stitch into one rollup.
csvs = sorted(glob('results/postproc_threshold_fixA/*/fixA_sweep.csv'))
frames = []
for c in csvs:
    df = pd.read_csv(c)
    df['source'] = Path(c).parent.name
    frames.append(df)
if not frames:
    print('No fixA_sweep.csv files found.')
else:
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(['source', 'config']).reset_index(drop=True)
    out = 'results/fixA_rollup.csv'
    full.to_csv(out, index=False)
    print(f'rollup -> {out}')
    print(full[['source', 'config', 'F1_legacy', 'F1_fixA', 'lift',
                'P_fixA', 'R_fixA']].to_string(
        index=False, float_format=lambda v: f'{v:.4f}'))
PYEOF

echo "=========================================================="
echo "=== ALL DONE  $(date)"
echo "=========================================================="
