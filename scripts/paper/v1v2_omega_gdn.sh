#!/usr/bin/env bash
# Splice the real Mahalanobis Omega into the 12 GDN V1/V2 arrays.
# GDN V1/V2 already have real aleatoric+epistemic+U_str; only U_dist is a
# placeholder. build_omega.py re-runs K-anchor inference on the V1/V2 ckpt,
# fits per-node Gaussian on the split's TRAIN rows, scores Omega on test,
# and writes arrays_full.npz next to the baseline arrays.
# Sequential, capped threads, good-neighbour single GPU per run (2 at a time).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
SEEDS=(0 1 2 3 4 42); GPUS=(0 1 2 3); CONC=12   # all 12 in one wave (3/GPU)
OUT=results/baseline_v1v2/gdn; mkdir -p "$OUT/omega_logs"
echo "[gdn-omega-v1v2] $(date)"

run_one() {
  local V="$1" S="$2" G="$3"
  local d="pretrained/swat_gdeltauq_${V}_seed${S}"
  local ck hp; ck=$(ls -t $d/best_*.pt 2>/dev/null|head -1); hp=$(ls -t $d/hyperparameters_*.json 2>/dev/null|head -1)
  local inarr="$OUT/$V/seed$S/arrays.npz"
  local outarr="$OUT/$V/seed$S/arrays_full.npz"
  local split="data/swat/split_${V}_baseline.json"
  local lg="$OUT/omega_logs/${V}_seed${S}.log"; local sent="$OUT/omega_logs/${V}_seed${S}.done"
  {
    echo "== gdn-omega $V s$S gpu $G $(date) =="
    [ -n "$ck" ] && [ -n "$hp" ] && [ -f "$inarr" ] || { echo "MISSING inputs ck=$ck inarr=$inarr"; echo 11 > "$sent"; return; }
    CUDA_VISIBLE_DEVICES="$G" "$PY" scripts/paper/build_omega.py \
      --checkpoint "$ck" --hyperparameters "$hp" --bundle_dir "$d/calibration_bundle_K100" \
      --split "$split" --in_arrays "$inarr" --out_arrays "$outarr" --device cuda:0 \
      || { echo "OMEGA FAIL $?"; echo 12 > "$sent"; return; }
    [ -f "$outarr" ] && { echo "DONE $outarr"; echo 0 > "$sent"; } || { echo "NO OUT"; echo 13 > "$sent"; }
  } > "$lg" 2>&1
}

i=0; pids=()
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  run_one "$V" "$S" "${GPUS[$((i % 4))]}" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done; done
for p in "${pids[@]}"; do wait "$p"; done
echo "[gdn-omega-v1v2] DONE $(date); $(for V in V1 V2; do for S in ${SEEDS[@]}; do echo -n $V-s$S:rc$(cat $OUT/omega_logs/${V}_seed${S}.done 2>/dev/null)\ ; done; done)"
