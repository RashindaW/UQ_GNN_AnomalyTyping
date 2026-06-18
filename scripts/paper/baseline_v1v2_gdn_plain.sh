#!/usr/bin/env bash
# Plain paper-format GDN baseline (original d-ailin GDN: single GNN layer, attention,
# top-k graph, MSE/residual -- NO anchoring, NO NLL head) under V1/V2 splits,
# seeds {0,1,2,3,4,42} = 12 runs, round-robin across 4 GPUs. Gives every backbone a
# clean plain-M0 reference (vs the anchored-M0). Same data / W=60 / V1/V2 split files /
# eval harness as the other backbones, so it's directly comparable.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SEEDS=(0 1 2 3 4 42); GPUS=(0 1 2 3); CONC=12
OUT=results/baseline_v1v2/gdn_plain; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
echo "[gdn-plain-v1v2] plain GDN (paper format, no anchoring), seeds ${SEEDS[*]} x {V1,V2} $(date)"

run_one() {
  local V="$1" S="$2" G="$3"
  local SPLIT="data/swat/split_${V}_baseline.json"
  local lg="$OUT/logs/${V}_seed${S}.log"; local sent="$OUT/logs/${V}_seed${S}.done"
  local rdir="$OUT/$V/seed$S"; mkdir -p "$rdir"
  local tag="gdn_plain_${V}_seed${S}"
  {
    echo "== gdn-plain $V s$S gpu $G $(date) =="
    rm -rf "results/$tag" "pretrained/$tag"
    CUDA_VISIBLE_DEVICES="$G" "$PY" main.py -dataset swat -model gdn \
        -slide_win 60 -slide_stride 1 -epoch 100 -batch 128 -dim 64 \
        -out_layer_num 1 -out_layer_inter_dim 128 -topk 15 -decay 0.0 \
        -random_seed "$S" -report best -split_path "$SPLIT" -save_arrays \
        -save_path_pattern "$tag" -device cuda:0 \
      || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    local SRC; SRC=$(ls -t results/$tag/*_arrays.npz 2>/dev/null | head -1)
    [ -n "$SRC" ] || { echo "NO ARRAYS"; echo 12 > "$sent"; return; }
    # remap main.py arrays -> eval_from_arrays schema (predict -> mu_bar)
    "$PY" - "$SRC" "$rdir/arrays.npz" <<'PYEOF' || { echo "REMAP FAIL $?"; echo 13 > "$sent"; return; }
import sys, numpy as np
z = np.load(sys.argv[1]); d = {k: z[k] for k in z.files}
d['test_mu_bar'] = d.pop('test_predict'); d['val_mu_bar'] = d.pop('val_predict')
np.savez_compressed(sys.argv[2], **d); print('remapped ->', sys.argv[2])
PYEOF
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$rdir/arrays.npz" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "GDN-plain-$V-s$S" --out "$rdir/eval.json" \
      || { echo "EVAL FAIL $?"; echo 14 > "$sent"; return; }
    echo "== gdn-plain $V s$S DONE $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

i=0; pids=()
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  run_one "$V" "$S" "${GPUS[$((i % 4))]}" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done; done
for p in "${pids[@]}"; do wait "$p"; done
echo "[gdn-plain-v1v2] ALL DONE $(date): $(for V in V1 V2; do for S in ${SEEDS[@]}; do echo -n $V-s$S:rc$(cat $OUT/logs/${V}_seed${S}.done 2>/dev/null)\ ; done; done)"
echo "M0 F1: $(for V in V1 V2; do for S in ${SEEDS[@]}; do f=$(grep -oE '"F1"[: ]*[0-9.]+' $OUT/$V/seed$S/eval.json 2>/dev/null|grep -oE '[0-9.]+$'|head -1); echo -n "$V-s$S:${f:-?} "; done; done)"
