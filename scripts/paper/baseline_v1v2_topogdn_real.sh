#!/usr/bin/env bash
# REAL TopoGDN baseline (no uncertainty) under V1/V2 contiguous splits.
#
# Fixes the prior baseline, which silently ran plain GDN: this version enables the
# persistent-homology TopologyLayer (use_topo=True) AND the multi-scale temporal
# convolution (MSConv=TCN1d) in competitors/common/baseline_v1v2_topogdn.py, and
# uses the PyG-2.x-compatible topoPooling (single-graph slice fix).
#
#   V1: train windows [0,70%), val [85,100%)   (70% of training data)
#   V2: train windows [0,85%), val [85,100%)   (85% of training data)
# seeds {0,1,2,3,4,42} x {V1,V2} = 12 runs, round-robin balanced across 4 GPUs,
# 12-wide. Window W=60 (matches GDN). Early stopping (window 15) inside train().
#
# Topology needs the compiled persistent-homology extension -> train in the topogdn
# env; M0 detection scoring uses the shared harness in the rashindaNew env.
# Baseline ONLY (train + M0). Uncertainty (anchoring/aleatoric/structural) is a
# separate later phase.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOPO_DUMP_GATED=0                       # do not write shared gated_edge_index.txt
export TOPO_GRAD_CLIP=1000                     # clip grad-norm (raw-scale MSE -> ~900-2900 norms; tames inflation)
export TOPO_LR_WARMUP_ITERS=500               # linear lr warmup ~1 epoch (stabilizes Adam's early steps)
SEEDS=(0 1 2 3 4 42); GPUS=(0 1 2 3); CONC=12; EPOCH=50
OUT=results/baseline_v1v2/topogdn; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
echo "[topo-real-v1v2] REAL topo+MSConv, W=60, epoch=$EPOCH (early-stop 15), seeds ${SEEDS[*]} x {V1,V2}, ${CONC}-wide $(date)"

run_one() {
  local V="$1" S="$2" G="$3"
  local lg="$OUT/logs/${V}_seed${S}.log"; local sent="$OUT/logs/${V}_seed${S}.done"
  local rdir="$OUT/$V/seed$S"; mkdir -p "$rdir"
  {
    echo "== topo-real $V s$S gpu $G epoch=$EPOCH $(date) =="
    CUDA_VISIBLE_DEVICES="$G" "$PYT" competitors/common/baseline_v1v2_topogdn.py \
        --variant "$V" --seed "$S" --epoch "$EPOCH" --device cuda:0 \
      || { echo "TRAIN FAIL rc=$?"; echo 11 > "$sent"; return; }
    [ -f "$rdir/arrays.npz" ] || { echo "NO ARRAYS"; echo 12 > "$sent"; return; }
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$rdir/arrays.npz" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "TopoGDN-$V-s$S" --out "$rdir/eval.json" \
      || { echo "EVAL FAIL rc=$?"; echo 14 > "$sent"; return; }
    echo "== topo-real $V s$S DONE $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

i=0; pids=()
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  run_one "$V" "$S" "${GPUS[$((i % 4))]}" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done; done
for p in "${pids[@]}"; do wait "$p"; done

echo "[topo-real-v1v2] ALL DONE $(date): $(for V in V1 V2; do for S in ${SEEDS[@]}; do echo -n $V-s$S:rc$(cat $OUT/logs/${V}_seed${S}.done 2>/dev/null)\ ; done; done)"
echo "TopoGDN real arrays: $(ls $OUT/V*/seed*/arrays.npz 2>/dev/null | wc -l)/12"
echo "M0 F1: $(for V in V1 V2; do for S in ${SEEDS[@]}; do f=$(grep -oE '"F1"[: ]+[0-9.]+' $OUT/$V/seed$S/eval.json 2>/dev/null | grep -oE '[0-9.]+$'); echo -n "$V-s$S:${f:-?} "; done; done)"
