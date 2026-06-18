#!/usr/bin/env bash
# Baseline CST-GL forecaster under V1/V2 contiguous splits, seeds {0,1,2,3,4,42}.
# Uses re-sliced swat_canon_{V1,V2} datasets (made by reslice_cstgl_v1v2.py).
# Trains (skip_scorer), saves forecasts, assembles a baseline arrays.npz, M0-scores it.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYC=/home/rashinda/.conda/envs/cstgl/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SEEDS=(0 1 2 3 4 42); GPUS=(0 1 2 3); CONC=12
OUT=results/baseline_v1v2/cstgl; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
CST=competitors/CST-GL
echo "[cstgl-v1v2] seeds ${SEEDS[*]} x {V1,V2} $(date)"

run_one() {
  local V="$1" S="$2" G="$3"
  local lg="$OUT/logs/${V}_seed${S}.log"; local sent="$OUT/logs/${V}_seed${S}.done"
  local rdir="$OUT/$V/seed$S"; mkdir -p "$rdir"
  export CUDA_VISIBLE_DEVICES="$G"
  {
    echo "== cstgl $V s$S gpu $G $(date) =="
    if [ ! -f "$rdir/test_pred_${S}.npy" ]; then
      ( cd "$CST" && CUDA_VISIBLE_DEVICES="$G" "$PYC" run.py --data data/swat_canon_${V} \
          --expid ${V}base --seed $S --epochs 20 --num_nodes 51 --subgraph_size 15 \
          --delays "[0,6,30,60,120,180,360]" --seq_in_len 60 --skip_scorer \
          --batch_size 32 \
          --save_result "$(cd ../.. && pwd)/$rdir/" --device cuda:0 ) \
        || { echo "TRAIN FAIL $?"; echo 11 > "$sent"; return; }
    fi
    [ -f "$rdir/test_pred_${S}.npy" ] || { echo "NO FORECAST"; echo 11 > "$sent"; return; }
    # assemble baseline arrays.npz: mu=test_pred, gt=test_label(=realy observed y),
    # attack labels from the canonical swat_canon test.npz labels (binary).
    "$PY" - "$rdir" "$S" "$CST/data/swat_canon/test_label.pkl" <<'PYEOF' || { echo "ASSEMBLE FAIL $?"; echo 12 > "$sent"; return; }
import sys, os, numpy as np, pickle
rdir, S = sys.argv[1], sys.argv[2]
mu = np.load(f"{rdir}/test_pred_{S}.npy")          # (T,V) forecast
gt = np.load(f"{rdir}/test_label_{S}.npy")          # (T,V) observed y (realy)
vmu = np.load(f"{rdir}/val_pred_{S}.npy"); vgt = np.load(f"{rdir}/val_label_{S}.npy")
# binary attack labels: load canonical test labels (pkl) and align to T
try:
    lab = np.asarray(pickle.load(open(sys.argv[3],'rb'))).astype(np.int8).reshape(-1)
except Exception:
    lab = np.load(os.path.join(os.path.dirname(sys.argv[3]),'test.npz'))['labels'].astype(np.int8).reshape(-1) if os.path.exists(os.path.join(os.path.dirname(sys.argv[3]),'test.npz')) else None
T = mu.shape[0]
if lab is None or lab.shape[0] < T:
    raise SystemExit(f"label load failed (T={T}, lab={None if lab is None else lab.shape})")
lab = lab[-T:]
np.savez_compressed(f"{rdir}/arrays.npz", test_mu_bar=mu.astype(np.float32),
    test_ground_truth=gt.astype(np.float32), test_attack_label=lab,
    val_mu_bar=vmu.astype(np.float32), val_ground_truth=vgt.astype(np.float32))
print(f"assembled arrays.npz T={T} attack_rate={lab.mean():.3f}")
PYEOF
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$rdir/arrays.npz" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "CSTGL-$V-s$S" --out "$rdir/eval.json" \
      || { echo "EVAL FAIL $?"; echo 14 > "$sent"; return; }
    echo "== cstgl $V s$S DONE $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

i=0; pids=()
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  run_one "$V" "$S" "${GPUS[$((i % 4))]}" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done; done
for p in "${pids[@]}"; do wait "$p"; done
echo "[cstgl-v1v2] ALL DONE $(date); $(for V in V1 V2; do for S in ${SEEDS[@]}; do echo -n $V-s$S:rc$(cat $OUT/logs/${V}_seed${S}.done 2>/dev/null)\ ; done; done)"
