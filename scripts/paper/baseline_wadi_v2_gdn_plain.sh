#!/usr/bin/env bash
# WADI V2 plain GDN baseline (paper format, no anchoring), seeds {0,1,2,3,4,42}.
# Clone of baseline_v1v2_gdn_plain.sh restricted to: dataset wadi, V2 only,
# CUDA 0 ONLY (shared card: free-memory gate, 2 concurrent max), topk 30
# (the GDN paper's own WADI setting; the single declared HP deviation).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 4 42}); CONC="${CONC_OVERRIDE:-6}"; MIN_FREE=6000
OUT=results/baseline_wadi_v2/gdn_plain; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/wadi_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/wadi_ensemble/calibration_bundle
echo "[gdn-plain-wadi] V2 seeds ${SEEDS[*]} cuda0-only $(date)"

wait_for_free() {  # good-neighbour gate: never launch into a busy card
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "$free" -ge "$MIN_FREE" ] && return 0
    echo "[gate] cuda0 free=${free}MiB < ${MIN_FREE}, waiting $(date)"; sleep 120
  done
}

run_one() {
  local S="$1"
  local SPLIT="data/wadi/split_V2_baseline.json"
  local lg="$OUT/logs/V2_seed${S}.log"; local sent="$OUT/logs/V2_seed${S}.done"
  local rdir="$OUT/V2/seed$S"; mkdir -p "$rdir"
  local tag="gdn_plain_wadi_V2_seed${S}"
  {
    echo "== gdn-plain-wadi V2 s$S $(date) =="
    rm -rf "results/$tag" "pretrained/$tag"
    CUDA_VISIBLE_DEVICES=0 "$PY" main.py -dataset wadi -model gdn \
        -slide_win 60 -slide_stride 1 -epoch 100 -batch 128 -dim 64 \
        -out_layer_num 1 -out_layer_inter_dim 128 -topk 30 -decay 0.0 \
        -random_seed "$S" -report best -split_path "$SPLIT" -save_arrays \
        -save_path_pattern "$tag" -device cuda:0 \
      || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    local SRC; SRC=$(ls -t results/$tag/*_arrays.npz 2>/dev/null | head -1)
    [ -n "$SRC" ] || { echo "NO ARRAYS"; echo 12 > "$sent"; return; }
    "$PY" - "$SRC" "$rdir/arrays.npz" <<'PYEOF' || { echo "REMAP FAIL $?"; echo 13 > "$sent"; return; }
import sys, numpy as np
z = np.load(sys.argv[1]); d = {k: z[k] for k in z.files}
d['test_mu_bar'] = d.pop('test_predict'); d['val_mu_bar'] = d.pop('val_predict')
np.savez_compressed(sys.argv[2], **d); print('remapped ->', sys.argv[2])
PYEOF
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$rdir/arrays.npz" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "GDN-plain-wadi-V2-s$S" --out "$rdir/eval.json" \
      || { echo "EVAL FAIL $?"; echo 14 > "$sent"; return; }
    echo "== gdn-plain-wadi V2 s$S DONE $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

i=0; pids=()
for S in "${SEEDS[@]}"; do
  wait_for_free
  run_one "$S" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[gdn-plain-wadi] ALL DONE $(date): $(for S in ${SEEDS[@]}; do echo -n V2-s$S:rc$(cat $OUT/logs/V2_seed${S}.done 2>/dev/null)\ ; done)"
echo "M0 F1: $(for S in ${SEEDS[@]}; do f=$(grep -oE '"F1"[: ]*[0-9.]+' $OUT/V2/seed$S/eval.json 2>/dev/null|grep -oE '[0-9.]+$'|head -1); echo -n "V2-s$S:${f:-?} "; done)"
