#!/usr/bin/env bash
# Distribute the DualSTGF WADI chain (baseline -> anchored train -> fulluq K=100
# -> eval) over the 4 GPUs, co-resident with the running TopoGDN fleet. Spreads
# 6 seeds round-robin (2+2+1+1) at CONC=2/GPU. A free-memory pre-flight gate
# before each seed is a safety valve (DualSTGF and TopoGDN together fit in 46 GB).
# Detached (setsid). Idempotent: skips a seed whose arrays_full.npz + eval exist.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYT=/home/rashinda/.conda/envs/topogdn/bin/python      # dualstage runs in topogdn env
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export UQ_DATASET=wadi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
BASE=results/baseline_wadi_v2/dualstage
UQ=results/uq_wadi_v2/dualstage
SPLIT=pretrained/wadi_ensemble/calibration_bundle/calibration_set_indices.json
BUNDLE=pretrained/wadi_ensemble/calibration_bundle
# WADI DualSTGF at batch 64 needs ~40 GB (NxN attention, N=123 ~6x SWaT). batch 32
# (~20 GB) fits alongside the TopoGDN fleet; one dual per GPU (CONC=1).
GPUS=(0 1 2 3); SEEDS=(0 1 2 3 4 42); EPOCH=50; BATCH=32; MIN_FREE=24000
mkdir -p "$BASE/logs" "$UQ/logs" "$UQ/queue"
printf '%s\n' "${SEEDS[@]}" > "$UQ/queue/list"

wait_mem() {  # G : block until GPU G has >= MIN_FREE MiB free
  while :; do
    local f; f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null)
    [ -n "$f" ] && [ "$f" -ge "$MIN_FREE" ] && return 0
    sleep 60
  done
}

run_one() {  # G S
  local G="$1" S="$2"
  local bd="$BASE/V2/seed$2" ud="$UQ/V2/seed$2" lg="$UQ/logs/V2_seed$2.log"
  mkdir -p "$bd" "$ud"; echo "$G" > "$ud/gpu"
  if [ -f "$ud/arrays_full.npz" ] && [ -f "$ud/eval_m0.json" ]; then
    echo "[dual] s$S already complete, skip" >> "$lg"; return 0; fi
  {
    echo "== dual-wadi s$S gpu$G BASELINE $(date) =="
    if [ ! -f "$bd/arrays.npz" ]; then
      CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 "$PYT" \
        competitors/common/baseline_v1v2_dualstage.py --variant V2 --seed "$S" \
        --epochs "$EPOCH" --batch "$BATCH" --device cuda:0 --out-root "$BASE" \
        || { echo "BASELINE rc=$? s$S"; return 1; }
    fi
    echo "== dual-wadi s$S ANCHORED $(date) =="
    if [ ! -f "$ud/best.pt" ]; then
      CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 "$PYT" \
        competitors/common/v1v2_dualstage_gdeltauq_train.py --variant V2 --seed "$S" \
        --epochs "$EPOCH" --batch "$BATCH" --device cuda:0 --out-root "$UQ" \
        || { echo "ANCHORED rc=$? s$S"; return 1; }
    fi
    echo "== dual-wadi s$S FULLUQ $(date) =="
    CUDA_VISIBLE_DEVICES=$G "$PYT" competitors/common/v1v2_dualstage_gdeltauq_fulluq.py \
      --variant V2 --seed "$S" --K_anchors 100 --anchor_seed 0 --batch 32 --device cuda:0 \
      --uq-root "$UQ" --base-arrays "$bd/arrays.npz" \
      || { echo "FULLUQ rc=$? s$S"; return 1; }
    [ -f "$ud/arrays_full.npz" ] || { echo "NO ARRAYS_FULL s$S"; return 1; }
    echo "== dual-wadi s$S EVAL $(date) =="
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$ud/arrays_full.npz" \
      --split "$SPLIT" --bundle "$BUNDLE" --slide_win 60 --seed "$S" --baseline-only \
      --label "DualSTGF-UQ-wadi-V2-s$S" --out "$ud/eval_m0.json" \
      || { echo "EVAL rc=$? s$S"; return 1; }
    echo "== dual-wadi s$S DONE rc=0 $(date) =="
  } >> "$lg" 2>&1
}

worker() {  # G
  local G="$1" S
  while :; do
    S=$( ( flock 9; head -n1 "$UQ/queue/list"; sed -i '1d' "$UQ/queue/list" ) 9>"$UQ/queue/lock" )
    [ -z "$S" ] && break
    echo "[dual] gpu$G -> seed$S (mem gate) $(date)"
    wait_mem "$G"
    run_one "$G" "$S" || echo "[dual] gpu$G seed$S FAILED"
  done
  echo "[dual] gpu$G drained $(date)"
}

CONC="${CONC_OVERRIDE:-1}"   # one dual per GPU (each ~20 GB) co-resident with TopoGDN
echo "[dual-wadi-dist] START $(date) gpus=${GPUS[*]} conc/gpu=$CONC batch=$BATCH min_free=${MIN_FREE}MiB"
for c in $(seq 1 "$CONC"); do for G in "${GPUS[@]}"; do worker "$G" & sleep 1; done; done
wait
echo "[dual-wadi-dist] ALL DONE $(date)"
echo "arrays_full: $(ls $UQ/V2/seed*/arrays_full.npz 2>/dev/null | wc -l)/${#SEEDS[@]}"
