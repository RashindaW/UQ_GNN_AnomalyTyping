#!/usr/bin/env bash
# Full GDN-GDeltaUQ chain for ONE fresh seed at 70/10/20 sw60 K=100.
# Chain: train -> calibrate(K=100) -> eval_paper_protocol -> eval_from_arrays.
# Usage: run_gdn_seed_70sw60.sh <seed> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
S=$1; G=$2
export CUDA_VISIBLE_DEVICES=$G
TAG=swat_gdeltauq_70_sw60_seed${S}
SPLIT=data/swat/gdeltauq_split.json
K=100

echo "[s$S g$G] TRAIN $(date)"
$PY train_gdeltauq_main.py -dataset swat -slide_win 60 -slide_stride 1 \
  -epoch 100 -batch 128 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 \
  -topk 15 -n_gnn_layers 2 -K_anchors $K -decay 0.0 -random_seed $S \
  -split_path $SPLIT -save_path_pattern $TAG -device cuda:0 \
  || { echo "[s$S] TRAIN FAIL"; exit 1; }

CKPT=$(ls -t pretrained/$TAG/best_*.pt | head -1)
HP=$(ls -t pretrained/$TAG/hyperparameters_*.json | head -1)
echo "[s$S g$G] CKPT=$CKPT"

echo "[s$S g$G] CALIBRATE K=$K $(date)"
$PY scripts/calibrate_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
  -split_path $SPLIT -K_anchors $K -anchor_seed 0 -anchor_strategy random \
  -save_dir pretrained/$TAG/calibration_bundle_K$K -device cuda:0 \
  || { echo "[s$S] CALIB FAIL"; exit 1; }

echo "[s$S g$G] EVAL $(date)"
$PY scripts/eval_paper_protocol_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
  -bundle_dir pretrained/$TAG/calibration_bundle_K$K -split_path $SPLIT \
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
