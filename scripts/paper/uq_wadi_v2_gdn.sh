#!/usr/bin/env bash
# WADI V2 anchored G-DeltaUQ GDN chain, seeds {0,1,2,3,4,42}.
# Clone of baseline_v1v2_gdn.sh restricted to: dataset wadi, V2 only,
# CUDA 0 ONLY (free-memory gate, 2 concurrent), topk 30.
# Chain per seed: train_gdeltauq_main -> calibrate (K=100 anchors) ->
# eval_paper_protocol (writes arrays.npz with U_par/ale/U_str) -> copy +
# baseline-only eval (fusion runs later via fusion_v1v2.py).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 4 42}); CONC="${CONC_OVERRIDE:-6}"; MIN_FREE=6000
OUT=results/baseline_wadi_v2/gdn; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/wadi_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/wadi_ensemble/calibration_bundle
echo "[gdn-uq-wadi] V2 seeds ${SEEDS[*]} cuda0-only $(date)"

wait_for_free() {
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "$free" -ge "$MIN_FREE" ] && return 0
    echo "[gate] cuda0 free=${free}MiB < ${MIN_FREE}, waiting $(date)"; sleep 120
  done
}

run_one() {
  local S="$1"
  local SPLIT="data/wadi/split_V2_baseline.json"
  local TAG="wadi_gdeltauq_V2_seed${S}"
  local lg="$OUT/logs/V2_seed${S}.log"; local sent="$OUT/logs/V2_seed${S}.done"
  export CUDA_VISIBLE_DEVICES=0
  {
    echo "== GDN-UQ wadi V2 seed $S $(date) =="
    if ls pretrained/$TAG/best_*.pt >/dev/null 2>&1; then echo "train SKIP"; else
      "$PY" train_gdeltauq_main.py -dataset wadi -slide_win 60 -slide_stride 1 \
        -epoch 100 -batch 128 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 \
        -topk 30 -n_gnn_layers 2 -K_anchors 100 -decay 0.0 -random_seed $S \
        -split_path "$SPLIT" -save_path_pattern "$TAG" -device cuda:0 \
        || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    fi
    local CKPT HP; CKPT=$(ls -t pretrained/$TAG/best_*.pt|head -1); HP=$(ls -t pretrained/$TAG/hyperparameters_*.json|head -1)
    "$PY" scripts/calibrate_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
      -split_path "$SPLIT" -K_anchors 100 -anchor_seed 0 -anchor_strategy random \
      -save_dir pretrained/$TAG/calibration_bundle_K100 -device cuda:0 \
      || { echo "CALIB FAIL $?"; echo 12 > "$sent"; return; }
    "$PY" scripts/eval_paper_protocol_gdeltauq.py -checkpoint "$CKPT" -hyperparameters "$HP" \
      -bundle_dir pretrained/$TAG/calibration_bundle_K100 -split_path "$SPLIT" \
      -results_dir results/${TAG}_K100 -topk 1 -device cuda:0 \
      || { echo "EVALPP FAIL $?"; echo 13 > "$sent"; return; }
    local ARR; ARR=$(ls -t results/${TAG}_K100/*/arrays.npz|head -1)
    mkdir -p "$OUT/V2/seed$S"; cp "$ARR" "$OUT/V2/seed$S/arrays.npz"
    "$PY" competitors/common/eval_from_arrays.py --arrays "$ARR" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed $S \
      --baseline-only --label "GDN-wadi-V2-s$S" --out "$OUT/V2/seed$S/eval.json" \
      || { echo "EVALARR FAIL $?"; echo 14 > "$sent"; return; }
    echo "== GDN-UQ wadi V2 seed $S DONE $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

i=0; pids=()
for S in "${SEEDS[@]}"; do
  wait_for_free
  run_one "$S" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[gdn-uq-wadi] ALL DONE $(date)"
for S in "${SEEDS[@]}"; do echo -n "V2-s$S:rc$(cat $OUT/logs/V2_seed${S}.done 2>/dev/null) "; done; echo
