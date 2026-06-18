#!/usr/bin/env bash
# Distribute the TopoGDN WADI anchored chain (train -> fulluq K=100 -> eval) over
# all 4 GPUs, one seed per GPU at full speed, auto-balancing via an atomic queue.
# Fresh start: deletes any partial anchored checkpoints/arrays first.
# Detached (setsid) so it survives the launching session. Idempotent per seed.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOPO_DUMP_GATED=0 TOPO_GRAD_CLIP=1000 TOPO_LR_WARMUP_ITERS=500
UOUT=results/uq_wadi_v2/topogdn
SPLIT=pretrained/wadi_ensemble/calibration_bundle/calibration_set_indices.json
BUNDLE=pretrained/wadi_ensemble/calibration_bundle
GPUS=(0 1 2 3)
SEEDS=(0 1 2 3 4 42)
mkdir -p "$UOUT/logs" "$UOUT/queue"

# fresh: clear partial anchored artifacts + per-seed logs (keep plain baseline arrays)
for S in "${SEEDS[@]}"; do
  rm -f "$UOUT/V2/seed$S/best.pt" "$UOUT/V2/seed$S/arrays_full.npz" "$UOUT/V2/seed$S/eval_m0.json"
  : > "$UOUT/logs/V2_seed$S.log"
done
printf '%s\n' "${SEEDS[@]}" > "$UOUT/queue/list"

run_one() {  # G S
  local G="$1" S="$2" d="$UOUT/V2/seed$2" lg="$UOUT/logs/V2_seed$2.log"
  mkdir -p "$d"; echo "$G" > "$d/gpu"
  {
    echo "== topo-wadi UQ V2 s$S gpu$G TRAIN $(date) =="
    CUDA_VISIBLE_DEVICES=$G "$PYT" competitors/common/v1v2_topogdn_gdeltauq_train.py \
      --variant V2 --seed "$S" --epoch 50 --dataset wadi --topk 30 --device cuda:0 \
      || { echo "TRAIN rc=$? s$S"; return 1; }
    echo "== topo-wadi UQ V2 s$S FULLUQ $(date) =="
    CUDA_VISIBLE_DEVICES=$G "$PYT" competitors/common/v1v2_topogdn_gdeltauq_fulluq.py \
      --variant V2 --seed "$S" --K_anchors 100 --anchor_seed 0 --dataset wadi --device cuda:0 \
      || { echo "FULLUQ rc=$? s$S"; return 1; }
    [ -f "$d/arrays_full.npz" ] || { echo "NO ARRAYS_FULL s$S"; return 1; }
    echo "== topo-wadi UQ V2 s$S EVAL $(date) =="
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$d/arrays_full.npz" \
      --split "$SPLIT" --bundle "$BUNDLE" --slide_win 60 --seed "$S" --baseline-only \
      --label "TopoGDN-UQ-wadi-V2-s$S" --out "$d/eval_m0.json" \
      || { echo "EVAL rc=$? s$S"; return 1; }
    echo "== topo-wadi UQ V2 s$S DONE rc=0 $(date) =="
  } >> "$lg" 2>&1
}

worker() {  # G
  local G="$1" S
  while :; do
    S=$( ( flock 9; head -n1 "$UOUT/queue/list"; sed -i '1d' "$UOUT/queue/list" ) 9>"$UOUT/queue/lock" )
    [ -z "$S" ] && break
    echo "[dist] gpu$G -> seed$S $(date)"
    run_one "$G" "$S" || echo "[dist] gpu$G seed$S FAILED"
  done
  echo "[dist] gpu$G drained $(date)"
}

CONC="${CONC_OVERRIDE:-2}"   # workers per GPU (mem ~3GB/job, 46GB avail; CPU 128 cores)
echo "[topo-wadi-dist] START $(date)  gpus=${GPUS[*]} conc/gpu=$CONC seeds=${SEEDS[*]}"
for c in $(seq 1 "$CONC"); do for G in "${GPUS[@]}"; do worker "$G" & sleep 1; done; done
wait
echo "[topo-wadi-dist] ALL DONE $(date)"
echo "arrays_full: $(ls $UOUT/V2/seed*/arrays_full.npz 2>/dev/null | wc -l)/${#SEEDS[@]}"
