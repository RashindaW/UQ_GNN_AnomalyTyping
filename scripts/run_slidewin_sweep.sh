#!/usr/bin/env bash
# Q1-(1): Slide_win sweep for GDN_GDeltaUQ on SWaT.
#
# For each slide_win ∈ {10, 15, 30, 60}:
#   1. Train GDN_GDeltaUQ at that slide_win (writes per-slide_win
#      pretrained dir via -save_path_pattern).
#   2. Calibrate with K=10 anchors.
#   3. Run paper-protocol eval (writes arrays.npz + report.json).
#
# After the sweep, emit results/swat_gdeltauq_sw_rollup.csv with one row
# per slide_win giving F1/P/R/AUC.
#
# Designed to run inside a tmux session (no terminal interaction).

set -euo pipefail

cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
SPLIT=data/swat/gdeltauq_split.json
SLIDEWINS=(10 15 30 60)

mkdir -p results

for SW in "${SLIDEWINS[@]}"; do
  TAG="swat_gdeltauq_sw${SW}"
  PRETRAIN_DIR="pretrained/${TAG}"
  RESULTS_DIR="results/${TAG}_paper_protocol"
  CAL_DIR="${PRETRAIN_DIR}/calibration_bundle"

  echo "============================================================"
  echo "=== slide_win=${SW}  TRAIN  ($(date))"
  echo "============================================================"

  "$PY" train_gdeltauq_main.py \
    -dataset swat \
    -slide_win "$SW" \
    -slide_stride 1 \
    -epoch 100 \
    -batch 128 \
    -dim 64 \
    -out_layer_num 1 \
    -out_layer_inter_dim 128 \
    -topk 15 \
    -n_gnn_layers 2 \
    -K_anchors 10 \
    -decay 0.0 \
    -random_seed 42 \
    -split_path "$SPLIT" \
    -save_path_pattern "${TAG}" \
    -device cuda:0 \
    -comment "Q1-(1) slide_win=${SW} sweep"

  # Newest checkpoint + hyperparameters in the per-slide_win dir.
  CKPT=$(ls -t "${PRETRAIN_DIR}"/best_*.pt 2>/dev/null | head -1)
  HP=$(ls -t "${PRETRAIN_DIR}"/hyperparameters_*.json 2>/dev/null | head -1)
  if [[ -z "$CKPT" || -z "$HP" ]]; then
    echo "ERROR: missing checkpoint or hyperparameters in ${PRETRAIN_DIR}" >&2
    exit 1
  fi
  echo "checkpoint: $CKPT"
  echo "hyperparams: $HP"

  echo "============================================================"
  echo "=== slide_win=${SW}  CALIBRATE  ($(date))"
  echo "============================================================"

  "$PY" scripts/calibrate_gdeltauq.py \
    -checkpoint "$CKPT" \
    -hyperparameters "$HP" \
    -split_path "$SPLIT" \
    -K_anchors 10 -anchor_seed 0 \
    -save_dir "$CAL_DIR" \
    -device cuda:0

  echo "============================================================"
  echo "=== slide_win=${SW}  EVAL  ($(date))"
  echo "============================================================"

  "$PY" scripts/eval_paper_protocol_gdeltauq.py \
    -checkpoint "$CKPT" \
    -hyperparameters "$HP" \
    -bundle_dir "$CAL_DIR" \
    -split_path "$SPLIT" \
    -topk 1 \
    -device cuda:0 \
    -results_dir "$RESULTS_DIR"
done

echo "============================================================"
echo "=== WRITING ROLLUP  ($(date))"
echo "============================================================"

ROLLUP=results/swat_gdeltauq_sw_rollup.csv
"$PY" - <<'PYEOF'
import json
from glob import glob
from pathlib import Path
import csv

rows = []
for sw in (10, 15, 30, 60):
    reports = sorted(glob(f'results/swat_gdeltauq_sw{sw}_paper_protocol/*/report.json'),
                     reverse=True)
    if not reports:
        rows.append({'slide_win': sw, 'F1': None, 'P': None, 'R': None,
                     'AUC': None, 'threshold': None, 'report_path': None})
        continue
    with open(reports[0]) as f:
        r = json.load(f)
    pp = r['paper_protocol']
    rows.append({
        'slide_win': sw,
        'F1': pp['F1'],
        'P': pp['precision'],
        'R': pp['recall'],
        'AUC': pp['AUC'],
        'threshold': pp['threshold'],
        'report_path': reports[0],
    })

out = 'results/swat_gdeltauq_sw_rollup.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'rollup -> {out}')
for r in rows:
    print(r)
PYEOF

echo "=== SLIDE_WIN SWEEP DONE ==="
