#!/usr/bin/env bash
# Ready-to-run GPU command to emit the REAL Omega OOD channel for seed42 and
# splice it into a NEW arrays file (the verified arrays.npz is NOT overwritten).
#
# Launch inside tmux. GPUs are SHARED -- this picks ONE least-busy device via
# scripts/paper/gpu_pick.py (good-neighbour) and caps the CUDA allocator split.
#
# Usage:
#   bash scripts/paper/run_omega_seed42.sh
# Optional: pass a device explicitly to override auto-pick, e.g.
#   bash scripts/paper/run_omega_seed42.sh cuda:0
set -u

ROOT=/mnt/datassd3/rashinda/UQ_GNN_AnomalyTyping
cd "$ROOT" || exit 1
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python

BUNDLE_ROOT="$ROOT/pretrained/swat_gdeltauq_sw60"
CKPT=$(ls "$BUNDLE_ROOT"/best_*.pt 2>/dev/null | head -1)
HP=$(ls "$BUNDLE_ROOT"/hyperparameters_*.json 2>/dev/null | head -1)
BUNDLE_DIR="$BUNDLE_ROOT/calibration_bundle_K100"
SPLIT="$ROOT/data/swat/gdeltauq_split.json"
IN_ARRAYS="$ROOT/results/gdn/ref_seed42/arrays.npz"
OUT_ARRAYS="$ROOT/results/gdn/ref_seed42/arrays_omega.npz"

# Device: argument overrides auto-pick. When CUDA_VISIBLE_DEVICES pins one GPU it
# is remapped to index 0, so cuda:0 is the right device string.
if [ "$#" -ge 1 ]; then
  DEV="$1"
else
  GPU=$("$PY" "$ROOT/scripts/paper/gpu_pick.py" 1 2>/dev/null | tr -d '[:space:]')
  [ -z "$GPU" ] && GPU=0
  DEV="cuda:0"
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "checkpoint  = $CKPT"
echo "hyperparams = $HP"
echo "bundle_dir  = $BUNDLE_DIR"
echo "in_arrays   = $IN_ARRAYS"
echo "out_arrays  = $OUT_ARRAYS"
echo "device      = $DEV  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"

"$PY" "$ROOT/scripts/paper/build_omega.py" \
  --checkpoint "$CKPT" \
  --hyperparameters "$HP" \
  --bundle_dir "$BUNDLE_DIR" \
  --split "$SPLIT" \
  --in_arrays "$IN_ARRAYS" \
  --out_arrays "$OUT_ARRAYS" \
  --device "$DEV"
