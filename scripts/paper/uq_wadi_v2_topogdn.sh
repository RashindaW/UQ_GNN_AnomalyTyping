#!/usr/bin/env bash
# WADI V2 TopoGDN: real baseline (topo+MSConv) + anchored TopoGDN_GDeltaUQ chain,
# seeds {0,1,2,3,4,42}. CUDA 0 ONLY, one training at a time (123-node homology is
# the heaviest fleet), free-memory gate, idempotent. Run the 1-epoch PROFILE GATE
# before launching this fleet (see plan: if an epoch exceeds ~30 min, stop).
# Per seed: baseline train+eval -> anchored train -> fulluq (K=100) -> eval_m0.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOPO_DUMP_GATED=0 TOPO_GRAD_CLIP=1000 TOPO_LR_WARMUP_ITERS=500
SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 4 42}); EPOCH=50; MIN_FREE=12000; CONC="${CONC_OVERRIDE:-6}"
BOUT=results/baseline_wadi_v2/topogdn; UOUT=results/uq_wadi_v2/topogdn
mkdir -p "$BOUT/logs" "$UOUT/logs"
EVAL_SPLIT=pretrained/wadi_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/wadi_ensemble/calibration_bundle
echo "[topo-wadi] V2 seeds ${SEEDS[*]} epoch=$EPOCH cuda0-only $(date)"

wait_for_free() {
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "$free" -ge "$MIN_FREE" ] && return 0
    echo "[gate] cuda0 free=${free}MiB < ${MIN_FREE}, waiting $(date)"; sleep 120
  done
}

run_seed() {
  local S="$1"
  local blg="$BOUT/logs/V2_seed${S}.log" bsent="$BOUT/logs/V2_seed${S}.done"
  local ulg="$UOUT/logs/V2_seed${S}.log" usent="$UOUT/logs/V2_seed${S}.done"
  local brdir="$BOUT/V2/seed$S" urdir="$UOUT/V2/seed$S"
  mkdir -p "$brdir" "$urdir"

  # ---- 1. real-TopoGDN baseline ----
  if [ ! -f "$brdir/eval.json" ]; then
    {
      echo "== topo-wadi baseline V2 s$S epoch=$EPOCH $(date) =="
      if [ ! -f "$brdir/arrays.npz" ]; then
        CUDA_VISIBLE_DEVICES=0 "$PYT" competitors/common/baseline_v1v2_topogdn.py \
            --variant V2 --seed "$S" --epoch "$EPOCH" --dataset wadi --topk 30 --device cuda:0 \
          || { echo "TRAIN FAIL rc=$?"; echo 11 > "$bsent"; return 1; }
      fi
      [ -f "$brdir/arrays.npz" ] || { echo "NO ARRAYS"; echo 12 > "$bsent"; return 1; }
      OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$brdir/arrays.npz" \
        --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
        --baseline-only --label "TopoGDN-wadi-V2-s$S" --out "$brdir/eval.json" \
        || { echo "EVAL FAIL rc=$?"; echo 14 > "$bsent"; return 1; }
      echo "== topo-wadi baseline V2 s$S DONE $(date) =="; echo 0 > "$bsent"
    } > "$blg" 2>&1 || return 1
  fi

  # ---- 2. anchored UQ chain ----
  wait_for_free
  {
    echo "== topo-wadi UQ V2 s$S epoch=$EPOCH $(date) =="
    if [ ! -f "$urdir/best.pt" ]; then
      CUDA_VISIBLE_DEVICES=0 "$PYT" competitors/common/v1v2_topogdn_gdeltauq_train.py \
          --variant V2 --seed "$S" --epoch "$EPOCH" --dataset wadi --topk 30 --device cuda:0 \
        || { echo "TRAIN FAIL rc=$?"; echo 11 > "$usent"; return 1; }
    fi
    CUDA_VISIBLE_DEVICES=0 "$PYT" competitors/common/v1v2_topogdn_gdeltauq_fulluq.py \
        --variant V2 --seed "$S" --K_anchors 100 --anchor_seed 0 --dataset wadi --device cuda:0 \
      || { echo "EXTRACT FAIL rc=$?"; echo 12 > "$usent"; return 1; }
    [ -f "$urdir/arrays_full.npz" ] || { echo "NO ARRAYS_FULL"; echo 13 > "$usent"; return 1; }
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$urdir/arrays_full.npz" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "TopoGDN-UQ-wadi-V2-s$S" --out "$urdir/eval_m0.json" \
      || { echo "EVAL FAIL rc=$?"; echo 14 > "$usent"; return 1; }
    echo "== topo-wadi UQ V2 s$S DONE $(date) =="; echo 0 > "$usent"
  } > "$ulg" 2>&1
}

# wave-parallel seeds (user: 6 at a time on cuda 0); 90 s stagger so each
# job's memory footprint materializes before the gate admits the next
i=0; pids=()
for S in "${SEEDS[@]}"; do
  wait_for_free
  ( run_seed "$S" || echo "[topo-wadi] seed $S FAILED (see logs)" ) & pids+=("$!")
  i=$((i+1)); sleep 90
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done
for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
echo "[topo-wadi] ALL DONE $(date)"
echo "baseline: $(for S in ${SEEDS[@]}; do echo -n V2-s$S:rc$(cat $BOUT/logs/V2_seed${S}.done 2>/dev/null)\ ; done)"
echo "uq:       $(for S in ${SEEDS[@]}; do echo -n V2-s$S:rc$(cat $UOUT/logs/V2_seed${S}.done 2>/dev/null)\ ; done)"
echo "arrays_full: $(ls $UOUT/V2/seed*/arrays_full.npz 2>/dev/null|wc -l)/${#SEEDS[@]}"
echo "M0 F1: $(for S in ${SEEDS[@]}; do f=$(grep -oE '"F1"[: ]+[0-9.]+' $BOUT/V2/seed$S/eval.json 2>/dev/null | grep -oE '[0-9.]+$'); echo -n "V2-s$S:${f:-?} "; done)"
