#!/usr/bin/env bash
# End-to-end M10 evaluation for a causal-RESTRICT-trained GDN_GDeltaUQ.
# (Pre-top-K restriction on the acyclic actuator-exogenous DAG scaffold,
#  cf. run_causal_mask_M10.sh which AND-s the domain mask after top-K.)
#
# Usage:
#   MODE=pure    SEED=42 GPU=0 bash scripts/run_causal_restrict_M10.sh
#   MODE=augment SEED=42 GPU=1 bash scripts/run_causal_restrict_M10.sh
#
# Stages (each resumable if its output exists):
#   1. Train base GDN_GDeltaUQ with -causal_restrict (mode pure|augment)
#   2. Calibrate at K=100 (anchor pool + aleatoric head + q_v)
#   3. Eval at K=100 -> arrays.npz
#   4. Run M10 fusion + record PA%K
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEED="${SEED:-42}"
GPU="${GPU:-0}"
SW="${SW:-60}"
EP="${EP:-100}"
K="${K:-100}"
TOPK="${TOPK:-15}"
NLAYERS="${NLAYERS:-2}"
BATCH="${BATCH:-128}"
DATASET="${DATASET:-swat}"
MODE="${MODE:-pure}"            # pure | augment

PY="${PY:-/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python}"
RUN_TAG="${RUN_TAG:-causal${MODE}_sw${SW}_seed${SEED}}"
PRE_DIR="pretrained/swat_gdeltauq_${RUN_TAG}"
RES_DIR="results/causal_restrict_M10/${RUN_TAG}"
BUNDLE_DIR="${PRE_DIR}/calibration_bundle_K${K}"
ARRAYS_DIR="${RES_DIR}/eval_K${K}"
FUSION_DIR="${RES_DIR}/fusion"

mkdir -p "$PRE_DIR" "$RES_DIR" "$ARRAYS_DIR" "$FUSION_DIR"
LOG="${RES_DIR}/pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "================================================================="
echo "[$(date -Iseconds)] START restrict mode=$MODE seed=$SEED gpu=$GPU sw=$SW ep=$EP K=$K"
echo "  RUN_TAG=$RUN_TAG"
echo "================================================================="

CRESTRICT="data/swat/causal_scaffold_dag.npy"

# ---------------------------------------------------------------------------
# STAGE 1: train base
# ---------------------------------------------------------------------------
CKPT=$(ls -t "${PRE_DIR}"/best_*.pt 2>/dev/null | head -1 || true)
HP=$(ls -t "${PRE_DIR}"/hyperparameters_*.json 2>/dev/null | head -1 || true)
if [[ -n "$CKPT" && -n "$HP" ]]; then
  echo "[stage 1] SKIP -- checkpoint exists: $CKPT"
else
  echo "[stage 1] train base GDN_GDeltaUQ with causal restrict (mode=$MODE)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_gdeltauq_main.py \
    -dataset "$DATASET" \
    -slide_win "$SW" -slide_stride 1 -batch "$BATCH" \
    -epoch "$EP" -topk "$TOPK" -n_gnn_layers "$NLAYERS" \
    -random_seed "$SEED" \
    -split_path data/swat/gdeltauq_split.json \
    -save_path_pattern "swat_gdeltauq_${RUN_TAG}" \
    -causal_restrict "$CRESTRICT" -causal_restrict_mode "$MODE" \
    -causal_restrict_keep_self 1 \
    -device cuda:0
  CKPT=$(ls -t "${PRE_DIR}"/best_*.pt | head -1)
  HP=$(ls -t "${PRE_DIR}"/hyperparameters_*.json | head -1)
fi
echo "[stage 1] CKPT=$CKPT"
echo "[stage 1] HP=$HP"

# ---------------------------------------------------------------------------
# STAGE 2: calibrate at K=100
# ---------------------------------------------------------------------------
if [[ -f "${BUNDLE_DIR}/anchor_pool.pt" && -f "${BUNDLE_DIR}/aleatoric_head.pt" ]]; then
  echo "[stage 2] SKIP -- bundle exists: $BUNDLE_DIR"
else
  echo "[stage 2] calibrate K=$K"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/calibrate_gdeltauq.py \
    -checkpoint "$CKPT" \
    -hyperparameters "$HP" \
    -split_path data/swat/gdeltauq_split.json \
    -K_anchors "$K" -anchor_seed 0 -anchor_strategy random \
    -aleatoric_epochs 5 -aleatoric_batch 32 \
    -bonferroni \
    -save_dir "$BUNDLE_DIR" \
    -device cuda:0
fi

# ---------------------------------------------------------------------------
# STAGE 3: eval (K-anchor inference -> arrays.npz)
# ---------------------------------------------------------------------------
ARRAYS_NPZ=$(ls -t "${ARRAYS_DIR}"/*/arrays.npz 2>/dev/null | head -1 || true)
if [[ -n "$ARRAYS_NPZ" ]]; then
  echo "[stage 3] SKIP -- arrays exist: $ARRAYS_NPZ"
else
  echo "[stage 3] paper-protocol eval @ K=$K"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_paper_protocol_gdeltauq.py \
    -checkpoint "$CKPT" \
    -hyperparameters "$HP" \
    -bundle_dir "$BUNDLE_DIR" \
    -split_path data/swat/gdeltauq_split.json \
    -topk 1 -device cuda:0 \
    -results_dir "$ARRAYS_DIR"
  ARRAYS_NPZ=$(ls -t "${ARRAYS_DIR}"/*/arrays.npz | head -1)
fi
echo "[stage 3] ARRAYS_NPZ=$ARRAYS_NPZ"

# ---------------------------------------------------------------------------
# STAGE 4: M10 fusion + PA%K
# ---------------------------------------------------------------------------
FUSION_SPLIT="${FUSION_SPLIT:-pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json}"

echo "[stage 4] M10 fusion (using $FUSION_SPLIT)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/fusion_sweep_K100_full.py \
  -arrays "$ARRAYS_NPZ" \
  -split "$FUSION_SPLIT" \
  -bundle "$BUNDLE_DIR" \
  -slide_win "$SW" \
  -methods M0 M10 \
  -seed "$SEED" \
  -out_root "$FUSION_DIR"

FUSION_RUN=$(ls -t "${FUSION_DIR}"/*/SUMMARY.md 2>/dev/null | head -1 || true)
echo "[stage 4] FUSION SUMMARY=$FUSION_RUN"

echo "[stage 4] PA%K computation"
"$PY" scripts/compute_M10_PAK.py \
  --arrays "$ARRAYS_NPZ" \
  --split "$FUSION_SPLIT" \
  --slide_win "$SW" \
  --bundle "$BUNDLE_DIR" \
  --seed "$SEED" \
  --out "${RES_DIR}/M10_PAK.json"

echo "================================================================="
echo "[$(date -Iseconds)] DONE restrict mode=$MODE seed=$SEED  results in $RES_DIR"
echo "================================================================="
