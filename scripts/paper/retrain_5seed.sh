#!/usr/bin/env bash
# Phase A2: clean 5-seed GDN+G-DeltaUQ retrain on SWaT (sw=60, 70:10:20).
# MIRRORS the proven chain in scripts/run_gdn_seed_70sw60.sh VERBATIM
# (train -> calibrate K=100 -> eval_paper_protocol[writes arrays.npz] ->
#  eval_from_arrays), but writes to ISOLATED *_rt tags so the verified
# checkpoints/results are never clobbered.
#
# Seed-parallel, 1 seed/GPU, waves of 2, good-neighbour 2-GPU mem-capped.
# Usage: bash scripts/paper/retrain_5seed.sh
# (intended to be wrapped by scripts/paper/tmux_launch.sh)
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
PY="${PY:-/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# seed42 first: it is the reference (verified M0=0.8109) -> validates the
# whole chain before the other seeds finish.
SEEDS=(42 1 2 3 100)
SPLIT="data/swat/gdeltauq_split.json"
K=100
OUTROOT="$REPO/results/retrain_v2"; mkdir -p "$OUTROOT/logs"

# Pick 2 good-neighbour GPUs (most free memory).
GPUSTR="$("$PY" "$REPO/scripts/paper/gpu_pick.py" 2 2>>"$OUTROOT/logs/gpu_pick.err" || echo "0,1")"
IFS=',' read -r -a GPUS <<< "$GPUSTR"
[ "${#GPUS[@]}" -ge 2 ] || GPUS=(0 1)
echo "[retrain] using GPUs: ${GPUS[*]}  (seeds: ${SEEDS[*]})  $(date)"
echo "[retrain] alloc_conf=$PYTORCH_CUDA_ALLOC_CONF  K=$K  split=$SPLIT"

run_seed() {
  local S="$1" G="$2"
  local TAG="swat_gdeltauq_70_sw60_seed${S}_rt"
  local log="$OUTROOT/logs/seed${S}.log"
  local sent="$OUTROOT/logs/seed${S}.done"
  export CUDA_VISIBLE_DEVICES="$G"   # within this subshell only
  {
    echo "==== seed=$S gpu=$G TAG=$TAG  $(date) ===="

    # Idempotent: skip train if a checkpoint already exists.
    if ls "pretrained/$TAG"/best_*.pt >/dev/null 2>&1; then
      echo "[1/4] train SKIP (checkpoint exists)"
    else
      echo "[1/4] train $(date)"
      "$PY" train_gdeltauq_main.py -dataset swat -slide_win 60 -slide_stride 1 \
        -epoch 100 -batch 128 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 \
        -topk 15 -n_gnn_layers 2 -K_anchors "$K" -decay 0.0 -random_seed "$S" \
        -split_path "$SPLIT" -save_path_pattern "$TAG" -device cuda:0 \
        || { echo "TRAIN FAIL rc=$?"; echo 11 > "$sent"; return; }
    fi

    local CKPT HP
    CKPT="$(ls -t pretrained/$TAG/best_*.pt 2>/dev/null | head -1)"
    HP="$(ls -t pretrained/$TAG/hyperparameters_*.json 2>/dev/null | head -1)"
    [ -n "$CKPT" ] && [ -n "$HP" ] || { echo "NO CKPT/HP"; echo 11 > "$sent"; return; }
    echo "[ckpt] $CKPT"

    echo "[2/4] calibrate K=$K $(date)"
    "$PY" scripts/calibrate_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
      -split_path "$SPLIT" -K_anchors "$K" -anchor_seed 0 -anchor_strategy random \
      -save_dir "pretrained/$TAG/calibration_bundle_K$K" -device cuda:0 \
      || { echo "CALIB FAIL rc=$?"; echo 12 > "$sent"; return; }

    echo "[3/4] eval_paper_protocol -> arrays.npz $(date)"
    "$PY" scripts/eval_paper_protocol_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
      -bundle_dir "pretrained/$TAG/calibration_bundle_K$K" -split_path "$SPLIT" \
      -results_dir "results/${TAG}_K$K" -topk 1 -device cuda:0 \
      || { echo "EVALPP FAIL rc=$?"; echo 13 > "$sent"; return; }

    local ARR
    ARR="$(ls -t results/${TAG}_K$K/*/arrays.npz 2>/dev/null | head -1)"
    [ -n "$ARR" ] || { echo "NO ARRAYS"; echo 13 > "$sent"; return; }
    echo "[arr] $ARR"

    echo "[4/4] eval_from_arrays $(date)"
    "$PY" competitors/common/eval_from_arrays.py --arrays "$ARR" \
      --split pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json \
      --bundle pretrained/swat_ensemble/calibration_bundle \
      --slide_win 60 --seed "$S" --label "GDN-seed${S}-rt" \
      --out "results/competitors/gdn_rt/seed${S}.json" \
      || { echo "EVALARR FAIL rc=$?"; echo 14 > "$sent"; return; }

    echo "==== seed=$S DONE  $(date) ===="; echo 0 > "$sent"
  } > "$log" 2>&1
}

# Waves of 2 across the 2 GPUs.
i=0
while [ "$i" -lt "${#SEEDS[@]}" ]; do
  pids=()
  for j in 0 1; do
    idx=$((i + j)); [ "$idx" -lt "${#SEEDS[@]}" ] || break
    S="${SEEDS[$idx]}"; G="${GPUS[$j]}"
    echo "[wave] launch seed=$S on gpu=$G"
    ( run_seed "$S" "$G" ) & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  i=$((i + 2))
done

# Rollup from the eval_from_arrays JSONs.
ROLLUP="$OUTROOT/rollup.csv"; echo "seed,rc,M0_F1,M10_F1,M0_PAK_AUC,M10_PAK_AUC" > "$ROLLUP"
for S in "${SEEDS[@]}"; do
  sent="$OUTROOT/logs/seed${S}.done"; rc="$(cat "$sent" 2>/dev/null || echo NA)"
  ev="$REPO/results/competitors/gdn_rt/seed${S}.json"
  "$PY" - "$S" "$rc" "$ev" >> "$ROLLUP" 2>/dev/null <<'PYEOF'
import json, sys
S, rc, ev = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(ev)); b, m = d["baseline_M0"], d["M10"]
    print(f'{S},{rc},{b["F1"]:.4f},{m["F1"]:.4f},{b["PA_K_AUC"]:.4f},{m["PA_K_AUC"]:.4f}')
except Exception:
    print(f'{S},{rc},NA,NA,NA,NA')
PYEOF
done
echo "[retrain] rollup -> $ROLLUP"; cat "$ROLLUP"
echo "[retrain] ALL DONE $(date)"
