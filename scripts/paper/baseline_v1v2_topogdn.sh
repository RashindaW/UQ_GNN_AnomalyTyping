#!/usr/bin/env bash
# Baseline TopoGDN forecaster under V1/V2 contiguous splits, seeds {0,1,2,3,4,42}.
# 12 runs, round-robin across 4 GPUs (CONC concurrent). Then M0-score each arrays.npz.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # avoid CPU thrash in scoring
SEEDS=(0 1 2 3 4 42); GPUS=(0 1 2 3); CONC=12
OUT=results/baseline_v1v2/topogdn; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
echo "[topo-v1v2] seeds ${SEEDS[*]} x {V1,V2} $(date)"

run_one() {
  local V="$1" S="$2" G="$3"
  local lg="$OUT/logs/${V}_seed${S}.log"; local sent="$OUT/logs/${V}_seed${S}.done"
  export CUDA_VISIBLE_DEVICES="$G"
  {
    echo "== topo $V s$S gpu $G $(date) =="
    arr="$OUT/$V/seed$S/arrays.npz"
    if [ ! -f "$arr" ]; then
      "$PYT" competitors/common/baseline_v1v2_topogdn.py --variant "$V" --seed "$S" --epoch 30 --device cuda:0 \
        || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    fi
    [ -f "$arr" ] || { echo "NO ARRAYS"; echo 11 > "$sent"; return; }
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$arr" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "TOPO-$V-s$S" --out "$OUT/$V/seed$S/eval.json" \
      || { echo "EVAL FAIL $?"; echo 14 > "$sent"; return; }
    echo "== topo $V s$S DONE $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

i=0; pids=()
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  run_one "$V" "$S" "${GPUS[$((i % 4))]}" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done; done
for p in "${pids[@]}"; do wait "$p"; done
echo "[topo-v1v2] ALL DONE $(date); $(for V in V1 V2; do for S in ${SEEDS[@]}; do echo -n $V-s$S:rc$(cat $OUT/logs/${V}_seed${S}.done 2>/dev/null)\ ; done; done)"
