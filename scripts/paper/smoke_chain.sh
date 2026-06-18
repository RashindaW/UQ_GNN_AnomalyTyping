#!/usr/bin/env bash
# 1-epoch end-to-end smoke of the GDN+G-DeltaUQ retrain chain on real SWaT data.
# Throwaway tag, epoch=1, K=5 -> finishes in minutes. Validates every entry
# point (train -> calibrate -> eval_paper_protocol[arrays.npz] -> eval_from_arrays)
# before committing the real 5-seed GPU run. Cleans its own outputs on success.
#
# Usage: bash scripts/paper/smoke_chain.sh [gpu_index]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
G="${1:-$("$PY" scripts/paper/gpu_pick.py 1 2>/dev/null || echo 0)}"
export CUDA_VISIBLE_DEVICES="$G"
TAG=swat_gdeltauq_SMOKE
SPLIT=data/swat/gdeltauq_split.json
K=5
echo "[smoke] gpu=$G tag=$TAG K=$K  $(date)"

echo "[smoke 1/4] TRAIN (epoch=1)"
"$PY" train_gdeltauq_main.py -dataset swat -slide_win 60 -slide_stride 1 \
  -epoch 1 -batch 128 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 \
  -topk 15 -n_gnn_layers 2 -K_anchors "$K" -decay 0.0 -random_seed 42 \
  -split_path "$SPLIT" -save_path_pattern "$TAG" -device cuda:0 \
  || { echo "[smoke] TRAIN FAIL rc=$?"; exit 11; }

CKPT="$(ls -t pretrained/$TAG/best_*.pt 2>/dev/null | head -1)"
HP="$(ls -t pretrained/$TAG/hyperparameters_*.json 2>/dev/null | head -1)"
[ -n "$CKPT" ] && [ -n "$HP" ] || { echo "[smoke] NO CKPT/HP"; exit 11; }
echo "[smoke] ckpt=$CKPT"

echo "[smoke 2/4] CALIBRATE (K=$K)"
"$PY" scripts/calibrate_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
  -split_path "$SPLIT" -K_anchors "$K" -anchor_seed 0 -anchor_strategy random \
  -aleatoric_epochs 1 -save_dir "pretrained/$TAG/calibration_bundle_K$K" -device cuda:0 \
  || { echo "[smoke] CALIB FAIL rc=$?"; exit 12; }

echo "[smoke 3/4] EVAL_PAPER_PROTOCOL -> arrays.npz"
"$PY" scripts/eval_paper_protocol_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
  -bundle_dir "pretrained/$TAG/calibration_bundle_K$K" -split_path "$SPLIT" \
  -results_dir "results/${TAG}_K$K" -topk 1 -device cuda:0 \
  || { echo "[smoke] EVALPP FAIL rc=$?"; exit 13; }

ARR="$(ls -t results/${TAG}_K$K/*/arrays.npz 2>/dev/null | head -1)"
[ -n "$ARR" ] || { echo "[smoke] NO ARRAYS"; exit 13; }
echo "[smoke] arr=$ARR"

echo "[smoke 4/4] EVAL_FROM_ARRAYS"
"$PY" competitors/common/eval_from_arrays.py --arrays "$ARR" \
  --split pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json \
  --bundle pretrained/swat_ensemble/calibration_bundle \
  --slide_win 60 --seed 42 --label GDN-SMOKE --out /tmp/smoke_eval.json \
  || { echo "[smoke] EVALARR FAIL rc=$?"; exit 14; }

echo "[smoke] eval.json:"; "$PY" -c "import json;d=json.load(open('/tmp/smoke_eval.json'));b=d['baseline_M0'];print('M0 F1=%.4f PA%%K=%.4f (1-epoch, expect LOW)'%(b['F1'],b['PA_K_AUC']))"
echo "[smoke] CLEANUP throwaway outputs"
rm -rf "pretrained/$TAG" "results/${TAG}_K$K"
echo "[smoke] ALL STEPS PASSED  $(date)"
