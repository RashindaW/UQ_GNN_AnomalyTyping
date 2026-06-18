#!/usr/bin/env bash
# Splice the real Mahalanobis Omega into the 6 WADI V2 GDN arrays.
# Clone of v1v2_omega_gdn.sh: dataset wadi, V2 only, CUDA 0 ONLY (2 at a time).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 4 42}); CONC="${CONC_OVERRIDE:-6}"; MIN_FREE=4000
OUT=results/baseline_wadi_v2/gdn; mkdir -p "$OUT/omega_logs"
echo "[gdn-omega-wadi] $(date)"

wait_for_free() {
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "$free" -ge "$MIN_FREE" ] && return 0
    echo "[gate] cuda0 free=${free}MiB < ${MIN_FREE}, waiting $(date)"; sleep 120
  done
}

run_one() {
  local S="$1"
  local d="pretrained/wadi_gdeltauq_V2_seed${S}"
  local ck hp; ck=$(ls -t $d/best_*.pt 2>/dev/null|head -1); hp=$(ls -t $d/hyperparameters_*.json 2>/dev/null|head -1)
  local inarr="$OUT/V2/seed$S/arrays.npz"
  local outarr="$OUT/V2/seed$S/arrays_full.npz"
  local split="data/wadi/split_V2_baseline.json"
  local lg="$OUT/omega_logs/V2_seed${S}.log"; local sent="$OUT/omega_logs/V2_seed${S}.done"
  {
    echo "== gdn-omega-wadi V2 s$S $(date) =="
    [ -n "$ck" ] && [ -n "$hp" ] && [ -f "$inarr" ] || { echo "MISSING inputs ck=$ck inarr=$inarr"; echo 11 > "$sent"; return; }
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/paper/build_omega.py \
      --checkpoint "$ck" --hyperparameters "$hp" --bundle_dir "$d/calibration_bundle_K100" \
      --split "$split" --in_arrays "$inarr" --out_arrays "$outarr" --device cuda:0 \
      || { echo "OMEGA FAIL $?"; echo 12 > "$sent"; return; }
    [ -f "$outarr" ] && { echo "DONE $outarr"; echo 0 > "$sent"; } || { echo "NO OUT"; echo 13 > "$sent"; }
  } > "$lg" 2>&1
}

i=0; pids=()
for S in "${SEEDS[@]}"; do
  wait_for_free
  run_one "$S" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[gdn-omega-wadi] DONE $(date); $(for S in ${SEEDS[@]}; do echo -n V2-s$S:rc$(cat $OUT/omega_logs/V2_seed${S}.done 2>/dev/null)\ ; done)"
