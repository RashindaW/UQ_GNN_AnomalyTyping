#!/usr/bin/env bash
# Train the GDN_GDeltaUQ variant on SWaT.  Uses the chronological 70/10/20
# split written by scripts/build_gdeltauq_split.py.
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
SPLIT_PATH="${SPLIT_PATH:-data/swat/gdeltauq_split.json}"

python train_gdeltauq_main.py \
    -dataset swat \
    -slide_win 5 -slide_stride 1 \
    -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 -topk 15 \
    -n_gnn_layers 2 \
    -K_anchors 10 \
    -batch 128 -epoch "$EPOCHS" \
    -random_seed "$SEED" \
    -split_path "$SPLIT_PATH" \
    -save_path_pattern swat_gdeltauq \
    -device "$DEVICE"
