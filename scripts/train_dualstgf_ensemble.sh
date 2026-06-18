#!/usr/bin/env bash
# Train M=5 DualSTGF_UQ members on SWaT, all running concurrently.
# GPU mapping: members 0,4 share GPU 0; members 1,2,3 each on GPU 1/2/3.
# Mirrors scripts/train_ensemble.sh exactly (only the python entrypoint differs).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DATASET="swat"
SEEDS_STR="${SEEDS:-5 17 42 100 314}"
read -ra SEEDS <<< "$SEEDS_STR"

if [[ "${#SEEDS[@]}" -ne 5 ]]; then
  echo "ERROR: expected 5 seeds, got ${#SEEDS[@]}: ${SEEDS[*]}" >&2
  exit 1
fi

GPUS=(0 1 2 3 0)
ENSEMBLE_ROOT="pretrained/dualstgf_ensemble"
mkdir -p "$ENSEMBLE_ROOT"

# DualSTGF on SWaT — paper-aligned defaults (see plan):
COMMON_ARGS=(
  -dataset "$DATASET"
  -window_size 60
  -train_stride 1
  -val_stride 5
  -batch 32
  -epoch 50
  -lr 1e-3
  -weight_decay 1e-3
  -early_stop_patience 15
  -gnn_embed_dim 64
  -temp_node_embed_dim 16
  -recon_hidden_dim 10
  -topk 15
  -num_gnn_layers 1
  -with_variance_head 1
  -device cuda
)

declare -a PIDS=()
declare -a LOG_FILES=()

start_ts=$(date +%s)
echo "[ensemble-dualstgf] launching 5 members, seeds=${SEEDS[*]}, started at $(date -Iseconds)"

for m in 0 1 2 3 4; do
  seed="${SEEDS[$m]}"
  gpu="${GPUS[$m]}"
  member_dir="$ENSEMBLE_ROOT/member_$(printf '%02d' "$m")_seed_${seed}"
  mkdir -p "$member_dir"
  log_file="$member_dir/train.log"
  LOG_FILES+=("$log_file")
  echo "[ensemble-dualstgf] member $m: seed=$seed gpu=$gpu log=$log_file"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_dualstgf.py \
    "${COMMON_ARGS[@]}" \
    -random_seed "$seed" \
    -save_path_pattern "member_$(printf '%02d' "$m")_seed_${seed}" \
    > "$log_file" 2>&1 &
  PIDS+=($!)
done

echo "[ensemble-dualstgf] all 5 launched (PIDs: ${PIDS[*]}); waiting..."

failed=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  if wait "$pid"; then
    echo "[ensemble-dualstgf] member $i (pid $pid) finished OK"
  else
    rc=$?
    echo "[ensemble-dualstgf] member $i (pid $pid) FAILED (exit $rc); see ${LOG_FILES[$i]}" >&2
    failed=1
  fi
done

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
echo "[ensemble-dualstgf] elapsed: ${elapsed}s ($(printf '%dh%dm%ds' $((elapsed/3600)) $(((elapsed/60)%60)) $((elapsed%60))))"

if [[ "$failed" -ne 0 ]]; then
  echo "[ensemble-dualstgf] one or more members failed; not building manifest" >&2
  exit 1
fi

echo "[ensemble-dualstgf] all members completed; building manifest..."
"$PYTHON" scripts/build_manifest_dualstgf.py
echo "[ensemble-dualstgf] done."
