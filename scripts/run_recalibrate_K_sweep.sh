#!/usr/bin/env bash
# Thread A driver: recalibrate the G-DeltaUQ anchor pool at K in {20, 30, 50}
# and re-run the GDN paper-protocol eval for each bundle. Designed to be
# launched inside tmux session `gdeltauq-kcal` (see README of the plan file).
#
# Deterministic anchor sampling with -anchor_seed 0 means K=50 contains K=30
# contains K=20, so the comparison across K is apples-to-apples.

set -euo pipefail

cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
CKPT=pretrained/swat_gdeltauq/best_0511-213455.pt
HP=pretrained/swat_gdeltauq/hyperparameters_0511-213455.json
SPLIT=data/swat/gdeltauq_split.json
DEV=cuda:0

for K in 20 30 50; do
  echo "=== K=$K CALIBRATE ==="
  "$PY" scripts/calibrate_gdeltauq.py \
    -checkpoint "$CKPT" \
    -hyperparameters "$HP" \
    -split_path "$SPLIT" \
    -K_anchors "$K" -anchor_seed 0 \
    -save_dir "pretrained/swat_gdeltauq/calibration_bundle_K${K}" \
    -device "$DEV"

  echo "=== K=$K EVAL ==="
  "$PY" scripts/eval_paper_protocol_gdeltauq.py \
    -checkpoint "$CKPT" \
    -hyperparameters "$HP" \
    -bundle_dir "pretrained/swat_gdeltauq/calibration_bundle_K${K}" \
    -split_path "$SPLIT" \
    -topk 1 -device "$DEV" \
    -results_dir "results/swat_gdeltauq_paper_protocol_K${K}"
done

echo "=== KCAL DONE ==="
