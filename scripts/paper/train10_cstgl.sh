#!/usr/bin/env bash
# Train 5 NEW CST-GL seeds {7,17,23,88,256} at the native config (matches the
# existing expswat5_<seed>.pth checkpoints). Then build the full UQ arrays.
# Runs in the cstgl conda env. Seeds sharded across GPUs 0-3, good-neighbour.
set -uo pipefail
cd "$(dirname "$0")/../../competitors/CST-GL"
PYC=/home/rashinda/.conda/envs/cstgl/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
SEEDS=(7 17 23 88 256)
GPUS=(0 1 2 3)
LOG=../../results/train10/cstgl; mkdir -p "$LOG/logs"
echo "[cstgl10] seeds ${SEEDS[*]} $(date)"

run_seed() {
  local S="$1" G="$2"
  local lg="$LOG/logs/seed${S}.log"; local sent="$LOG/logs/seed${S}.done"
  export CUDA_VISIBLE_DEVICES="$G"
  {
    echo "== cstgl seed $S gpu $G $(date) =="
    if [ -f save/expswat5_${S}.pth ]; then echo "train SKIP (ckpt exists)"; else
      "$PYC" run.py --data data/swat_canon --expid swat5 --seed $S --epochs 20 \
        --num_nodes 51 --subgraph_size 15 --delays "[0,6,30,60,120,180,360]" \
        --seq_in_len 60 --device cuda:0 --skip_scorer \
        || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    fi
    [ -f save/expswat5_${S}.pth ] || { echo "NO CKPT"; echo 11 > "$sent"; return; }
    echo "== cstgl seed $S trained $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

i=0
for S in "${SEEDS[@]}"; do run_seed "$S" "${GPUS[$((i % 4))]}" & i=$((i+1)); done
wait
echo "[cstgl10] training DONE $(date); seeds: $(for S in ${SEEDS[@]}; do echo -n "$S:rc$(cat $LOG/logs/seed${S}.done 2>/dev/null) "; done)"
