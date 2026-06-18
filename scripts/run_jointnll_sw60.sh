#!/usr/bin/env bash
# Q1-3: Joint NLL training of GDN_GDeltaUQ + AleatoricHead at sw=60.
# Single seed=42, 70:10:20 split. Sequential train -> calibrate -> eval.
# Runs on cuda:0; cuda:1/2/3 stay free for other concurrent work.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
SPLIT=data/swat/gdeltauq_split.json
TAG=swat_gdeltauq_jointnll_sw60
DEV=cuda:0

mkdir -p logs

echo "=== ${TAG}: TRAIN  $(date)"
"$PY" train_gdeltauq_jointnll_main.py \
  -dataset swat -slide_win 60 -slide_stride 1 \
  -epoch 100 -batch 128 -dim 64 -out_layer_num 1 \
  -out_layer_inter_dim 128 -topk 15 -n_gnn_layers 2 \
  -K_anchors 10 -decay 0.0 -random_seed 42 \
  -split_path "$SPLIT" -save_path_pattern "$TAG" \
  -device "$DEV" -comment "Q1-3 joint NLL sw=60"

CKPT=$(ls -t "pretrained/${TAG}"/best_*.pt | head -1)
HP=$(ls -t "pretrained/${TAG}"/hyperparameters_*.json | head -1)
HEAD=$(ls -t "pretrained/${TAG}"/aleatoric_head_*.pt | head -1)
echo "checkpoint: $CKPT"
echo "head:       $HEAD"
echo "hp:         $HP"

echo "=== ${TAG}: CALIBRATE (joint-trained head reused)  $(date)"
"$PY" scripts/calibrate_gdeltauq.py \
  -checkpoint "$CKPT" -hyperparameters "$HP" \
  -split_path "$SPLIT" \
  -K_anchors 10 -anchor_seed 0 -anchor_strategy random \
  -pretrained_head_path "$HEAD" \
  -save_dir "pretrained/${TAG}/calibration_bundle" \
  -device "$DEV"

echo "=== ${TAG}: EVAL  $(date)"
"$PY" scripts/eval_paper_protocol_gdeltauq.py \
  -checkpoint "$CKPT" -hyperparameters "$HP" \
  -bundle_dir "pretrained/${TAG}/calibration_bundle" \
  -split_path "$SPLIT" \
  -topk 1 -device "$DEV" \
  -results_dir "results/${TAG}_paper_protocol"

echo "=== ${TAG}: DONE  $(date)"
