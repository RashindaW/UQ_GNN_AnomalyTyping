#!/usr/bin/env bash
# WADI V2 CST-GL: plain baseline + anchored CSTGL_GDeltaUQ chain, seeds {0,1,2,3,4,42}.
# Per seed (sequential, CUDA 0 ONLY, free-memory gate, 1 training at a time):
#   1. run.py plain baseline on wadi_canon_V2 (--num_nodes 123 --subgraph_size 30)
#      -> assemble baseline arrays.npz (labels from wadi_canon/test_label.pkl)
#      -> baseline-only eval
#   2. anchored train (v1v2_cstgl_gdeltauq_train --dataset wadi)
#      -> fulluq extract (K=100, ale head, Omega) -> anchored-M0 eval
# Trains in cstgl env; evals in rashindaNew env. Idempotent (skips finished steps).
set -uo pipefail
cd "$(dirname "$0")/../.."
PYC=/home/rashinda/.conda/envs/cstgl/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 4 42}); EPOCHS=20; MIN_FREE=10000; CONC="${CONC_OVERRIDE:-6}"
BOUT=results/baseline_wadi_v2/cstgl; UOUT=results/uq_wadi_v2/cstgl
mkdir -p "$BOUT/logs" "$UOUT/logs"
EVAL_SPLIT=pretrained/wadi_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/wadi_ensemble/calibration_bundle
CST=competitors/CST-GL
echo "[cstgl-wadi] V2 seeds ${SEEDS[*]} cuda0-only $(date)"

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

  # ---- 1. plain baseline ----
  {
    echo "== cstgl-wadi baseline V2 s$S $(date) =="
    if [ ! -f "$brdir/test_pred_${S}.npy" ]; then
      ( cd "$CST" && CUDA_VISIBLE_DEVICES=0 "$PYC" run.py --data data/wadi_canon_V2 \
          --expid V2wadi --seed $S --epochs $EPOCHS --num_nodes 123 --subgraph_size 30 \
          --delays "[0,6,30,60,120,180,360]" --seq_in_len 60 --skip_scorer \
          --batch_size 32 \
          --save_result "$(cd ../.. && pwd)/$brdir/" --device cuda:0 ) \
        || { echo "TRAIN FAIL $?"; echo 11 > "$bsent"; return 1; }
    fi
    [ -f "$brdir/test_pred_${S}.npy" ] || { echo "NO FORECAST"; echo 11 > "$bsent"; return 1; }
    "$PY" - "$brdir" "$S" "$CST/data/wadi_canon/test_label.pkl" <<'PYEOF' || { echo "ASSEMBLE FAIL $?"; echo 12 > "$bsent"; return 1; }
import sys, numpy as np, pickle
rdir, S = sys.argv[1], sys.argv[2]
mu = np.load(f"{rdir}/test_pred_{S}.npy"); gt = np.load(f"{rdir}/test_label_{S}.npy")
vmu = np.load(f"{rdir}/val_pred_{S}.npy"); vgt = np.load(f"{rdir}/val_label_{S}.npy")
lab = np.asarray(pickle.load(open(sys.argv[3],'rb'))).astype(np.int8).reshape(-1)
T = mu.shape[0]
assert lab.shape[0] >= T, f"label too short ({lab.shape[0]} < {T})"
lab = lab[-T:]
np.savez_compressed(f"{rdir}/arrays.npz", test_mu_bar=mu.astype(np.float32),
    test_ground_truth=gt.astype(np.float32), test_attack_label=lab,
    val_mu_bar=vmu.astype(np.float32), val_ground_truth=vgt.astype(np.float32))
print(f"assembled arrays.npz T={T} attack_rate={lab.mean():.3f}")
PYEOF
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$brdir/arrays.npz" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "CSTGL-wadi-V2-s$S" --out "$brdir/eval.json" \
      || { echo "EVAL FAIL $?"; echo 14 > "$bsent"; return 1; }
    echo "== cstgl-wadi baseline V2 s$S DONE $(date) =="; echo 0 > "$bsent"
  } > "$blg" 2>&1 || return 1

  # ---- 2. anchored UQ chain ----
  wait_for_free
  {
    echo "== cstgl-wadi UQ V2 s$S epochs=$EPOCHS $(date) =="
    if [ ! -f "$urdir/best.pt" ]; then
      CUDA_VISIBLE_DEVICES=0 "$PYC" competitors/common/v1v2_cstgl_gdeltauq_train.py \
          --variant V2 --seed "$S" --epochs "$EPOCHS" --dataset wadi --device cuda:0 \
        || { echo "TRAIN FAIL rc=$?"; echo 11 > "$usent"; return 1; }
    fi
    CUDA_VISIBLE_DEVICES=0 "$PYC" competitors/common/v1v2_cstgl_gdeltauq_fulluq.py \
        --variant V2 --seed "$S" --K_anchors 100 --anchor_seed 0 --dataset wadi --device cuda:0 \
      || { echo "EXTRACT FAIL rc=$?"; echo 12 > "$usent"; return 1; }
    [ -f "$urdir/arrays_full.npz" ] || { echo "NO ARRAYS_FULL"; echo 13 > "$usent"; return 1; }
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$urdir/arrays_full.npz" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "CSTGL-UQ-wadi-V2-s$S" --out "$urdir/eval_m0.json" \
      || { echo "EVAL FAIL rc=$?"; echo 14 > "$usent"; return 1; }
    echo "== cstgl-wadi UQ V2 s$S DONE $(date) =="; echo 0 > "$usent"
  } > "$ulg" 2>&1
}

# wave-parallel seeds (user: 6 at a time on cuda 0); 90 s stagger so each
# job's memory footprint materializes before the gate admits the next
i=0; pids=()
for S in "${SEEDS[@]}"; do
  wait_for_free
  ( run_seed "$S" || echo "[cstgl-wadi] seed $S FAILED (see logs)" ) & pids+=("$!")
  i=$((i+1)); sleep 90
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done
for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
echo "[cstgl-wadi] ALL DONE $(date)"
echo "baseline: $(for S in ${SEEDS[@]}; do echo -n V2-s$S:rc$(cat $BOUT/logs/V2_seed${S}.done 2>/dev/null)\ ; done)"
echo "uq:       $(for S in ${SEEDS[@]}; do echo -n V2-s$S:rc$(cat $UOUT/logs/V2_seed${S}.done 2>/dev/null)\ ; done)"
echo "arrays_full: $(ls $UOUT/V2/seed*/arrays_full.npz 2>/dev/null|wc -l)/${#SEEDS[@]}"
