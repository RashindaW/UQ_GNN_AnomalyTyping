#!/usr/bin/env bash
# Train M heteroscedastic GDN_UQ members on SWaT, all running in parallel.
# M is determined by the number of seeds passed via the SEEDS env var.
# Members are mapped to GPUs round-robin (member m → GPU m % NUM_GPUS).
#
# Override defaults via env vars:
#   SEEDS="5 17 42 100 314"          override seed list (M = #seeds)
#   PYTHON=/path/to/python           Python interpreter
#   SKIP_EXISTING=1                  skip members that already have a checkpoint
#                                    + hyperparameters.json (default: 1)
#   LOGVAR_CLAMP_LOW=-3.0            log_var clamp lower bound
#   LOGVAR_CLAMP_HIGH=3.0            log_var clamp upper bound
#   LOGVAR_L2=0.01                   L2 regulariser weight on log_var
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DATASET="swat"
SEEDS_STR="${SEEDS:-5 17 42 100 314}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
read -ra SEEDS <<< "$SEEDS_STR"
M="${#SEEDS[@]}"

if [[ "$M" -lt 1 ]]; then
  echo "ERROR: no seeds provided (SEEDS env var)" >&2; exit 1
fi

# Detect available GPUs (default 4 if nvidia-smi unavailable).
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
  NUM_GPUS=4
fi
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1

ENSEMBLE_ROOT="pretrained/swat_ensemble"
mkdir -p "$ENSEMBLE_ROOT"

# Hyperparameters from GDN paper §4.4 for SWaT, plus Stage-2 σ-saturation fixes.
COMMON_ARGS=(
  -model gdn_uq
  -dataset "$DATASET"
  -batch 32
  -epoch 30
  -slide_win 5
  -slide_stride 1
  -dim 64
  -out_layer_num 1
  -out_layer_inter_dim 128
  -val_ratio 0.2
  -decay 0
  -topk 15
  -report best
  -device cuda
  -logvar_clamp_low "${LOGVAR_CLAMP_LOW:--3.0}"
  -logvar_clamp_high "${LOGVAR_CLAMP_HIGH:-3.0}"
  -logvar_l2 "${LOGVAR_L2:-0.01}"
)

declare -a PIDS=()
declare -a LOG_FILES=()
declare -a MEMBER_NAMES=()
declare -a SKIPPED=()

start_ts=$(date +%s)
echo "[ensemble] M=$M  seeds=${SEEDS[*]}  num_gpus=$NUM_GPUS  skip_existing=$SKIP_EXISTING"
echo "[ensemble] started at $(date -Iseconds)"

for ((m=0; m<M; m++)); do
  seed="${SEEDS[$m]}"
  gpu=$((m % NUM_GPUS))
  member_dir="$ENSEMBLE_ROOT/member_$(printf '%02d' "$m")_seed_${seed}"
  mkdir -p "$member_dir"
  log_file="$member_dir/train.log"

  # Skip-existing check: needs both a checkpoint and a hyperparameters.json
  # (the latter ensures the trained config is recorded for build_manifest.py).
  has_ckpt=0
  if compgen -G "$member_dir/best_*.pt" > /dev/null; then has_ckpt=1; fi
  has_hp=0
  if [[ -f "$member_dir/hyperparameters.json" ]]; then has_hp=1; fi
  if [[ "$SKIP_EXISTING" == "1" ]] && [[ "$has_ckpt" -eq 1 ]] && [[ "$has_hp" -eq 1 ]]; then
    echo "[ensemble] member $m (seed $seed): SKIP (existing checkpoint + hyperparameters.json)"
    SKIPPED+=("$m")
    continue
  fi

  echo "[ensemble] member $m: seed=$seed gpu=$gpu log=$log_file"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" main.py \
    "${COMMON_ARGS[@]}" \
    -random_seed "$seed" \
    -save_path_pattern "swat_ensemble/member_$(printf '%02d' "$m")_seed_${seed}" \
    > "$log_file" 2>&1 &
  PIDS+=($!)
  LOG_FILES+=("$log_file")
  MEMBER_NAMES+=("member $m (seed $seed)")
done

failed=0
if [[ ${#PIDS[@]} -gt 0 ]]; then
  echo "[ensemble] launched ${#PIDS[@]} new training jobs (PIDs: ${PIDS[*]}); waiting..."
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if wait "$pid"; then
      echo "[ensemble] ${MEMBER_NAMES[$i]} OK"
    else
      rc=$?
      echo "[ensemble] ${MEMBER_NAMES[$i]} FAILED (exit $rc); see ${LOG_FILES[$i]}" >&2
      failed=1
    fi
  done
else
  echo "[ensemble] all $M members already trained (no new training launched)"
fi

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
echo "[ensemble] elapsed: ${elapsed}s ($(printf '%dh%dm%ds' $((elapsed/3600)) $(((elapsed/60)%60)) $((elapsed%60))))"
echo "[ensemble] new=${#PIDS[@]}  skipped=${#SKIPPED[@]}  total=$M"

if [[ "$failed" -ne 0 ]]; then
  echo "[ensemble] one or more members failed; not building manifest" >&2
  exit 1
fi

echo "[ensemble] building manifest..."
"$PYTHON" scripts/build_manifest.py
echo "[ensemble] done."
