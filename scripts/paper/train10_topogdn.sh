#!/usr/bin/env bash
# Train 5 NEW TopoGDN seeds {7,17,23,88,256}. Mirrors the existing topo_s<seed>
# checkpoints (slide_win 60, dim 128, topk 15). Runs in the topogdn conda env.
# Seeds sharded across GPUs, good-neighbour. Saves to pretrained/topo_s<seed>/.
set -uo pipefail
cd "$(dirname "$0")/../../competitors/TopoGDN"
PYT=/home/rashinda/.conda/envs/topogdn/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
SEEDS=(7 17 23 88 256)
GPUS=(0 1 2 3)
LOG=../../results/train10/topogdn; mkdir -p "$LOG/logs"
echo "[topo10] seeds ${SEEDS[*]} $(date)"

run_seed() {
  local S="$1" G="$2"
  local lg="$LOG/logs/seed${S}.log"; local sent="$LOG/logs/seed${S}.done"
  export CUDA_VISIBLE_DEVICES="$G"
  {
    echo "== topo seed $S gpu $G $(date) =="
    if ls pretrained/topo_s${S}/best_*.pt >/dev/null 2>&1; then echo "train SKIP"; else
      "$PYT" main.py -dataset swat -slide_win 60 -dim 128 -topk 15 -epoch 30 \
        -batch 128 -random_seed $S -save_path_pattern topo_s${S} -report best -device cuda:0 \
        || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    fi
    ls pretrained/topo_s${S}/best_*.pt >/dev/null 2>&1 || { echo "NO CKPT"; echo 11 > "$sent"; return; }
    echo "== topo seed $S trained $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}
i=0
for S in "${SEEDS[@]}"; do run_seed "$S" "${GPUS[$((i % 4))]}" & i=$((i+1)); done
wait
echo "[topo10] DONE $(date); $(for S in ${SEEDS[@]}; do echo -n "$S:rc$(cat $LOG/logs/seed${S}.done 2>/dev/null) "; done)"
