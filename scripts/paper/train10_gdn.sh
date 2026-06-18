#!/usr/bin/env bash
# Train 5 NEW GDN+G-DeltaUQ seeds {7,17,23,88,256} to reach 10 seeds total.
# Full chain per seed: train -> calibrate(K=100) -> eval_paper_protocol(arrays.npz)
# -> eval_from_arrays. Mirrors the proven scripts/run_gdn_seed_70sw60.sh.
# Seed-parallel across 4 GPUs, good-neighbour mem-cap. Isolated _s<seed> tags.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
PY="${PY:-/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
SPLIT=data/swat/gdeltauq_split.json
K=100
SEEDS=(7 17 23 88 256)
GPUS=(0 1 2 3)
OUT="$REPO/results/train10/gdn"; mkdir -p "$OUT/logs"
echo "[gdn10] seeds ${SEEDS[*]} on GPUs ${GPUS[*]} $(date)"

run_seed() {
  local S="$1" G="$2"
  local TAG="swat_gdeltauq_70_sw60_seed${S}"
  local log="$OUT/logs/seed${S}.log"; local sent="$OUT/logs/seed${S}.done"
  export CUDA_VISIBLE_DEVICES="$G"
  {
    echo "== seed $S gpu $G $(date) =="
    if ls pretrained/$TAG/best_*.pt >/dev/null 2>&1; then echo "train SKIP (ckpt exists)"; else
      "$PY" train_gdeltauq_main.py -dataset swat -slide_win 60 -slide_stride 1 \
        -epoch 100 -batch 128 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 \
        -topk 15 -n_gnn_layers 2 -K_anchors $K -decay 0.0 -random_seed $S \
        -split_path $SPLIT -save_path_pattern $TAG -device cuda:0 \
        || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    fi
    local CKPT HP; CKPT=$(ls -t pretrained/$TAG/best_*.pt|head -1); HP=$(ls -t pretrained/$TAG/hyperparameters_*.json|head -1)
    "$PY" scripts/calibrate_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
      -split_path $SPLIT -K_anchors $K -anchor_seed 0 -anchor_strategy random \
      -save_dir pretrained/$TAG/calibration_bundle_K$K -device cuda:0 \
      || { echo "CALIB FAIL $?"; echo 12 > "$sent"; return; }
    "$PY" scripts/eval_paper_protocol_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
      -bundle_dir pretrained/$TAG/calibration_bundle_K$K -split_path $SPLIT \
      -results_dir results/${TAG}_K$K -topk 1 -device cuda:0 \
      || { echo "EVALPP FAIL $?"; echo 13 > "$sent"; return; }
    local ARR; ARR=$(ls -t results/${TAG}_K$K/*/arrays.npz|head -1)
    mkdir -p results/gdn/seed${S}; cp "$ARR" results/gdn/seed${S}/arrays.npz
    "$PY" competitors/common/eval_from_arrays.py --arrays "$ARR" \
      --split pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json \
      --bundle pretrained/swat_ensemble/calibration_bundle \
      --slide_win 60 --seed $S --label GDN-s$S --out results/competitors/gdn/seed${S}.json \
      || { echo "EVALARR FAIL $?"; echo 14 > "$sent"; return; }
    echo "== seed $S DONE $(date) =="; echo 0 > "$sent"
  } > "$log" 2>&1
}

# launch all 5 across the 4 GPUs (seed 256 waits for a free GPU after first 4)
i=0
for S in "${SEEDS[@]}"; do
  G=${GPUS[$((i % 4))]}
  run_seed "$S" "$G" &
  i=$((i+1))
  # stagger so 5th seed lands after one finishes; simple: launch first 4, wait none, 5th shares gpu0
done
wait
echo "[gdn10] ALL DONE $(date)"
for S in "${SEEDS[@]}"; do echo "seed$S rc=$(cat $OUT/logs/seed${S}.done 2>/dev/null)"; done
