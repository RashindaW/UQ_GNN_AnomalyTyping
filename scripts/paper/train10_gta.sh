#!/usr/bin/env bash
# Train 5 NEW GTA seeds {7,17,23,88,256}. Matches the existing checkpoint setting
# gta_SWaT_ftM_sl60_ll30_pl1_nl3_dm128_nh8_el3_dl2_df128_atprob_ebfixed_seed<S>_0.
# Runs in the gta conda env. Seeds sharded across GPUs.
set -uo pipefail
cd "$(dirname "$0")/../../competitors/GTA"
PYG=/home/rashinda/.conda/envs/gta/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
SEEDS=(7 17 23 88 256)
GPUS=(0 1 2 3)
LOG=../../results/train10/gta; mkdir -p "$LOG/logs"
echo "[gta10] seeds ${SEEDS[*]} $(date)"

run_seed() {
  local S="$1" G="$2"
  local lg="$LOG/logs/seed${S}.log"; local sent="$LOG/logs/seed${S}.done"
  local dir="checkpoints/gta_SWaT_ftM_sl60_ll30_pl1_nl3_dm128_nh8_el3_dl2_df128_atprob_ebfixed_seed${S}_0"
  export CUDA_VISIBLE_DEVICES="$G"
  {
    echo "== gta seed $S gpu $G $(date) =="
    if [ -f "$dir/checkpoint.pth" ]; then echo "train SKIP"; else
      "$PYG" main_gta_dad.py --model gta --data SWaT --root_path ./data --data_path SWaT \
        --features M --seq_len 60 --label_len 30 --pred_len 1 --num_nodes 51 \
        --num_levels 3 --d_model 128 --n_heads 8 \
        --e_layers 3 --d_layers 2 --d_ff 128 --attn prob --embed fixed \
        --train_epochs 6 --batch_size 32 --itr 1 --des seed${S} --seed $S --gpu 0 \
        || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    fi
    [ -f "$dir/checkpoint.pth" ] || { echo "NO CKPT ($dir)"; echo 11 > "$sent"; return; }
    echo "== gta seed $S trained $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}
i=0
for S in "${SEEDS[@]}"; do run_seed "$S" "${GPUS[$((i % 4))]}" & i=$((i+1)); done
wait
echo "[gta10] DONE $(date); $(for S in ${SEEDS[@]}; do echo -n "$S:rc$(cat $LOG/logs/seed${S}.done 2>/dev/null) "; done)"
