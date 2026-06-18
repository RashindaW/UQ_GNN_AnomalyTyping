#!/usr/bin/env bash
# Re-calibrate + eval an EXISTING GDN-GDeltaUQ checkpoint at K=100, then
# eval_from_arrays -> baseline M0 + M10 JSON. No retraining (K_anchors does not
# affect training, so the seed1/2/3 checkpoints are config-identical to the ref).
# Usage: run_gdn_calibrate_seed.sh <seed> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
S=$1; G=$2
export CUDA_VISIBLE_DEVICES=$G
TAG=swat_gdeltauq_70_sw60_seed${S}
K=100
CKPT=$(ls -t pretrained/$TAG/best_*.pt | head -1)
HP=$(ls -t pretrained/$TAG/hyperparameters_*.json | head -1)
echo "[s$S g$G] CKPT=$CKPT  $(date)"

echo "[s$S g$G] CALIBRATE K=$K $(date)"
$PY scripts/calibrate_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
  -K_anchors $K -anchor_seed 0 -anchor_strategy random \
  -save_dir pretrained/$TAG/calibration_bundle_K$K -device cuda:0 \
  || { echo "[s$S] CALIB FAIL"; exit 1; }

echo "[s$S g$G] EVAL $(date)"
$PY scripts/eval_paper_protocol_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
  -bundle_dir pretrained/$TAG/calibration_bundle_K$K \
  -results_dir results/${TAG}_K$K -topk 1 -device cuda:0 \
  || { echo "[s$S] EVAL FAIL"; exit 1; }

ARR=$(ls -t results/${TAG}_K$K/*/arrays.npz | head -1)
echo "[s$S g$G] ARR=$ARR"

echo "[s$S g$G] EVAL_FROM_ARRAYS $(date)"
$PY competitors/common/eval_from_arrays.py --arrays "$ARR" \
  --split pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json \
  --bundle pretrained/swat_ensemble/calibration_bundle \
  --slide_win 60 --label GDN-s$S --out results/competitors/gdn/seed${S}.json

echo "[s$S g$G] DONE $(date)"
