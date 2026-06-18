#!/usr/bin/env bash
# Emit the real Omega channel (Mahalanobis + kNN on anchored penultimate) for
# seeds 1,2,3,100 (seed42 already done). 2 GPUs, good-neighbour, waves of 2.
# Each writes results/gdn/seed<S>/arrays_omega.npz (originals untouched) and
# prints AUROC of each Omega variant vs attack_label.
# Usage: bash scripts/paper/emit_omega_seeds.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
SPLIT=data/swat/gdeltauq_split.json
OUT=results/paper/omega; mkdir -p "$OUT/logs"

GPUSTR="$("$PY" scripts/paper/gpu_pick.py 2 2>/dev/null || echo 0,1)"
IFS=',' read -r -a GPUS <<< "$GPUSTR"; [ "${#GPUS[@]}" -ge 2 ] || GPUS=(0 1)
echo "[omega] GPUs ${GPUS[*]}  $(date)"

SEEDS=(1 2 3 100)
emit() {
  local S="$1" G="$2"
  local d="pretrained/swat_gdeltauq_70_sw60_seed${S}"
  local ck hp; ck=$(ls -t $d/best_*.pt|head -1); hp=$(ls -t $d/hyperparameters_*.json|head -1)
  local log="$OUT/logs/seed${S}.log"; local sent="$OUT/logs/seed${S}.done"
  CUDA_VISIBLE_DEVICES="$G" "$PY" scripts/paper/build_omega.py \
    --checkpoint "$ck" --hyperparameters "$hp" \
    --bundle_dir "$d/calibration_bundle_K100" --split "$SPLIT" \
    --in_arrays "results/gdn/seed${S}/arrays.npz" \
    --out_arrays "results/gdn/seed${S}/arrays_omega.npz" \
    --device cuda:0 > "$log" 2>&1
  echo "$?" > "$sent"
}

i=0
while [ "$i" -lt "${#SEEDS[@]}" ]; do
  pids=()
  for j in 0 1; do
    idx=$((i+j)); [ "$idx" -lt "${#SEEDS[@]}" ] || break
    S="${SEEDS[$idx]}"; G="${GPUS[$j]}"; echo "[wave] seed$S -> gpu$G"
    emit "$S" "$G" & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  i=$((i+2))
done

echo "[omega] AUROC rollup:"
for S in "${SEEDS[@]}"; do
  echo "--- seed$S (rc=$(cat $OUT/logs/seed${S}.done 2>/dev/null)) ---"
  grep -iE 'placeholder|maha|knn|AUROC' "$OUT/logs/seed${S}.log" 2>/dev/null | tail -5
done
echo "[omega] ALL DONE $(date)"
