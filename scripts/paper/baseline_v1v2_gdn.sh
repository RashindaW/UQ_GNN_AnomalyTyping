#!/usr/bin/env bash
# Baseline (no-uncertainty) GDN forecaster under V1 and V2 splits, seeds {0..9,42}.
# V1: train[0,70%) val[85,100%]; V2: train[0,85%) val[85,100%]. Stored SEPARATELY.
# Round-robin all 11x2=22 runs across 4 GPUs (balanced). M0 residual score only.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
SEEDS=(0 1 2 3 4 42)
GPUS=(0 1 2 3)
OUT=results/baseline_v1v2/gdn; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
echo "[gdn-v1v2] seeds ${SEEDS[*]} x {V1,V2} $(date)"

run_one() {
  local V="$1" S="$2" G="$3"
  local SPLIT="data/swat/split_${V}_baseline.json"
  local TAG="swat_gdeltauq_${V}_seed${S}"
  local lg="$OUT/logs/${V}_seed${S}.log"; local sent="$OUT/logs/${V}_seed${S}.done"
  export CUDA_VISIBLE_DEVICES="$G"
  {
    echo "== GDN $V seed $S gpu $G $(date) =="
    if ls pretrained/$TAG/best_*.pt >/dev/null 2>&1; then echo "train SKIP"; else
      "$PY" train_gdeltauq_main.py -dataset swat -slide_win 60 -slide_stride 1 \
        -epoch 100 -batch 128 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 \
        -topk 15 -n_gnn_layers 2 -K_anchors 100 -decay 0.0 -random_seed $S \
        -split_path "$SPLIT" -save_path_pattern "$TAG" -device cuda:0 \
        || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    fi
    local CKPT HP; CKPT=$(ls -t pretrained/$TAG/best_*.pt|head -1); HP=$(ls -t pretrained/$TAG/hyperparameters_*.json|head -1)
    # calibrate (needed to produce arrays via eval_paper_protocol; M0 ignores UQ)
    "$PY" scripts/calibrate_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
      -split_path "$SPLIT" -K_anchors 100 -anchor_seed 0 -anchor_strategy random \
      -save_dir pretrained/$TAG/calibration_bundle_K100 -device cuda:0 \
      || { echo "CALIB FAIL $?"; echo 12 > "$sent"; return; }
    "$PY" scripts/eval_paper_protocol_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
      -bundle_dir pretrained/$TAG/calibration_bundle_K100 -split_path "$SPLIT" \
      -results_dir results/${TAG}_K100 -topk 1 -device cuda:0 \
      || { echo "EVALPP FAIL $?"; echo 13 > "$sent"; return; }
    local ARR; ARR=$(ls -t results/${TAG}_K100/*/arrays.npz|head -1)
    mkdir -p "$OUT/$V/seed$S"; cp "$ARR" "$OUT/$V/seed$S/arrays.npz"
    "$PY" competitors/common/eval_from_arrays.py --arrays "$ARR" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed $S \
      --label "GDN-$V-s$S" --out "$OUT/$V/seed$S/eval.json" \
      || { echo "EVALARR FAIL $?"; echo 14 > "$sent"; return; }
    echo "== GDN $V seed $S DONE $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

# round-robin all (V,seed) jobs across 4 GPUs, max 8 concurrent (2 per GPU;
# GDN is tiny ~0.9GB so 2/card fits comfortably in the shared ~15GB free).
CONC=12
i=0; pids=()
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  G=${GPUS[$((i % 4))]}      # round-robin GPUs -> 2 jobs land on each of the 4 cards
  run_one "$V" "$S" "$G" & pids+=("$!")
  i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done; done
for p in "${pids[@]}"; do wait "$p"; done
echo "[gdn-v1v2] ALL DONE $(date)"
for V in V1 V2; do for S in "${SEEDS[@]}"; do echo -n "$V-s$S:rc$(cat $OUT/logs/${V}_seed${S}.done 2>/dev/null) "; done; echo; done
