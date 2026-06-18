#!/bin/bash
# Run cf_graph_static.py across 4 GPUs in parallel inside this tmux session,
# then merge the shard outputs and run the unsupervised validator.
#
# Designed to be invoked inside a tmux session so the user can disconnect:
#   tmux new-session -d -s cf-parallel "bash scripts/run_cf_parallel.sh"
#   tmux attach -t cf-parallel        # optional
#
# Reattach later: tmux attach -t cf-parallel
# Follow a single shard's log without attaching:
#   tail -f results/cf_static_graph/<datestr>/logs/shard_0.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
NUM_SHARDS=${NUM_SHARDS:-4}
N_DIVERSE=${N_DIVERSE:-5}
K_EDGE_MAX=${K_EDGE_MAX:-15}
K_NODE_MAX=${K_NODE_MAX:-5}

DIVERSITY_MODE=${DIVERSITY_MODE:-first_edge}
FAITHFUL_SMOOTHING=${FAITHFUL_SMOOTHING:-0}   # set to 1 to enable
RUN_TAG=${RUN_TAG:-}                          # optional suffix on the out dir

DATESTR=$(date +%m%d-%H%M%S)${RUN_TAG:+_$RUN_TAG}
OUT_DIR="results/cf_static_graph/$DATESTR"
mkdir -p "$OUT_DIR/per_segment" "$OUT_DIR/logs"
echo "[parallel] OUT_DIR=$OUT_DIR  NUM_SHARDS=$NUM_SHARDS  DIVERSITY_MODE=$DIVERSITY_MODE  FAITHFUL_SMOOTHING=$FAITHFUL_SMOOTHING"

FAITHFUL_FLAG=""
if [ "$FAITHFUL_SMOOTHING" = "1" ]; then
  FAITHFUL_FLAG="--faithful-smoothing"
fi

# Launch one process per GPU in the background
for I in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_DEV="cuda:$I"
  LOG="$OUT_DIR/logs/shard_$I.log"
  echo "[parallel] launching shard $I on $CUDA_DEV  log=$LOG"
  $PY scripts/cf_graph_static.py \
    --device "$CUDA_DEV" \
    --shard-idx "$I" \
    --num-shards "$NUM_SHARDS" \
    --out-dir-fixed "$OUT_DIR" \
    --N "$N_DIVERSE" \
    --K-edge-max "$K_EDGE_MAX" \
    --K-node-max "$K_NODE_MAX" \
    --diversity-mode "$DIVERSITY_MODE" \
    $FAITHFUL_FLAG \
    > "$LOG" 2>&1 &
done

echo "[parallel] all $NUM_SHARDS shards launched; waiting..."
wait
echo "[parallel] all shards completed."

echo "[parallel] merging shard CSVs..."
$PY scripts/cf_merge_shards.py --run-dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/logs/merge.log"

echo "[parallel] running validator..."
$PY scripts/cf_unsupervised_validate.py --run-dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/logs/validate.log"

echo "[parallel] all done."
echo "  SUMMARY:           $OUT_DIR/SUMMARY.md"
echo "  per-segment CSV:   $OUT_DIR/cost_per_segment.csv"
echo "  per-anchor CSV:    $OUT_DIR/cf_per_anchor.csv"
