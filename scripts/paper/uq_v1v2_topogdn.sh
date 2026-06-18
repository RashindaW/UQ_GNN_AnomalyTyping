#!/usr/bin/env bash
# Anchored TopoGDN_GDeltaUQ -- the ONE UQ method on the real TopoGDN backbone.
# Per (V,seed): train the 2-layer anchored model (topo in layer 0 + MSConv, G-DeltaUQ
# anchoring at the final graph layer) -> calibrate K=100 anchor pool + aleatoric head
# on the held-out 15% -> K-anchor extract (mu_bar/U_par/U_str/sigma2_ale/Omega) ->
# arrays_full.npz -> anchored-M0 eval. V1/V2 x seeds{0,1,2,3,4,42} = 12 runs,
# round-robin balanced across 4 GPUs. Topology runs in the topogdn env; M0 scoring in
# the rashindaNew env. Fusion is a separate later step (fusion_v1v2.py).
set -uo pipefail
cd "$(dirname "$0")/../.."
PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOPO_DUMP_GATED=0 TOPO_GRAD_CLIP=1000 TOPO_LR_WARMUP_ITERS=500
SEEDS=(0 1 2 3 4 42); GPUS=(0 2 3); CONC=12; EPOCH=50    # cuda:0/2/3, 12-wide (cuda:0 freed + CPU headroom)
OUT=results/uq_v1v2/topogdn; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
echo "[topo-uq-v1v2] anchored TopoGDN_GDeltaUQ K=100, epoch=$EPOCH, seeds ${SEEDS[*]} x {V1,V2}, ${CONC}-wide $(date)"

run_one() {
  local V="$1" S="$2" G="$3"
  local lg="$OUT/logs/${V}_seed${S}.log"; local sent="$OUT/logs/${V}_seed${S}.done"
  local rdir="$OUT/$V/seed$S"; mkdir -p "$rdir"
  {
    echo "== topo-uq $V s$S gpu $G epoch=$EPOCH $(date) =="
    if [ ! -f "$rdir/best.pt" ]; then
      CUDA_VISIBLE_DEVICES="$G" "$PYT" competitors/common/v1v2_topogdn_gdeltauq_train.py \
          --variant "$V" --seed "$S" --epoch "$EPOCH" --device cuda:0 \
        || { echo "TRAIN FAIL rc=$?"; echo 11 > "$sent"; return; }
    fi
    CUDA_VISIBLE_DEVICES="$G" "$PYT" competitors/common/v1v2_topogdn_gdeltauq_fulluq.py \
        --variant "$V" --seed "$S" --K_anchors 100 --anchor_seed 0 --device cuda:0 \
      || { echo "EXTRACT FAIL rc=$?"; echo 12 > "$sent"; return; }
    [ -f "$rdir/arrays_full.npz" ] || { echo "NO ARRAYS_FULL"; echo 13 > "$sent"; return; }
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$rdir/arrays_full.npz" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "TopoGDN-UQ-$V-s$S" --out "$rdir/eval_m0.json" \
      || { echo "EVAL FAIL rc=$?"; echo 14 > "$sent"; return; }
    echo "== topo-uq $V s$S DONE $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

i=0; pids=()
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  run_one "$V" "$S" "${GPUS[$((i % ${#GPUS[@]}))]}" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done; done
for p in "${pids[@]}"; do wait "$p"; done

echo "[topo-uq-v1v2] ALL DONE $(date): $(for V in V1 V2; do for S in ${SEEDS[@]}; do echo -n $V-s$S:rc$(cat $OUT/logs/${V}_seed${S}.done 2>/dev/null)\ ; done; done)"
echo "arrays_full: $(ls $OUT/V*/seed*/arrays_full.npz 2>/dev/null|wc -l)/12"
