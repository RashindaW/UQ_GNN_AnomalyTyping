#!/usr/bin/env bash
# Retry runner for DualSTAGE V1/V2 baselines: only runs missing arrays.npz.
# Memory-aware: picks the GPU with most free memory at job start, staggers
# launches, batch 64, allocator anti-fragmentation. One retry per run.
set -u
cd /mnt/datassd3/rashinda/UQ_GNN_AnomalyTyping

PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SEEDS=(0 1 2 3 4 42); CONC=10; EPOCH=50; BATCH=64
OUT=results/baseline_v1v2/dualstage; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle

pick_gpu() {  # GPU index with most free MiB
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' '
}

run_one() {
  local V="$1" S="$2"
  local lg="$OUT/logs/${V}_seed${S}.log"; local sent="$OUT/logs/${V}_seed${S}.done"
  local rdir="$OUT/$V/seed$S"; mkdir -p "$rdir"
  for attempt in 1 2; do
    local G; G="$(pick_gpu)"
    {
      echo "== dualstage $V s$S gpu $G attempt $attempt epoch=$EPOCH batch=$BATCH $(date) =="
      CUDA_VISIBLE_DEVICES="$G" "$PYT" competitors/common/baseline_v1v2_dualstage.py \
          --variant "$V" --seed "$S" --epochs "$EPOCH" --batch "$BATCH" --device cuda:0 \
        && break || echo "TRAIN FAIL rc=$? (attempt $attempt)"
    } >> "$lg" 2>&1
    sleep 120
  done
  if [ ! -f "$rdir/arrays.npz" ]; then echo 11 > "$sent"; return; fi
  OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$rdir/arrays.npz" \
    --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
    --baseline-only --label "DualSTAGE-$V-s$S" --out "$rdir/eval.json" >> "$lg" 2>&1 \
    || { echo 14 > "$sent"; return; }
  echo 0 > "$sent"
}

i=0
for V in V1 V2; do
  for S in "${SEEDS[@]}"; do
    rdir="$OUT/$V/seed$S"
    if [ -f "$rdir/arrays.npz" ]; then echo "skip $V s$S (arrays exist)"; continue; fi
    if [ ! -f "$OUT/logs/${V}_seed${S}.done" ] && \
       [ -n "$(find "$OUT/logs/${V}_seed${S}.log" -mmin -10 2>/dev/null)" ]; then
      echo "skip $V s$S (live run from first launch)"; continue
    fi
    rm -f "$OUT/logs/${V}_seed${S}.done"
    sleep $((10 + (i % CONC) * 15))   # stagger allocator startups
    run_one "$V" "$S" &
    i=$((i + 1))
    if (( i % CONC == 0 )); then wait; fi
  done
done
wait
ok=0
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  [ -f "$OUT/$V/seed$S/arrays.npz" ] && ok=$((ok + 1))
done; done
echo "[dualstage-retry] arrays present: $ok/12 $(date)"
