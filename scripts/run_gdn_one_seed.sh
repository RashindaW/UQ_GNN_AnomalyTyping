#!/usr/bin/env bash
# Full GDN G-DeltaUQ chain for ONE seed on ONE GPU:
#   train (sw60, K=100, aleatoric ON) -> calibrate K=100 -> eval (arrays.npz)
#   -> eval_from_arrays (baseline M0 + M10 JSON)
# Usage: run_gdn_one_seed.sh <seed> <gpu>
# Config reconstructed to match the canonical reference
# (results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655 -> M0=0.811 M10=0.839):
#   slide_win 60, K_anchors 100, aleatoric ON; dim64/n_gnn2/topk15/anchor=data are
#   train_gdeltauq_main.py defaults. seed42 doubles as the config-reproduction check.
set +e
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
S=$1; G=$2
export CUDA_VISIBLE_DEVICES=$G
SP=swat_gd_sw60_s$S

echo "[s$S g$G] TRAIN $(date)"
$PY train_gdeltauq_main.py -dataset swat -slide_win 60 -slide_stride 1 \
  -K_anchors 100 -epoch 100 -batch 128 -aleatoric -random_seed $S \
  -save_path_pattern $SP -device cuda:0 || { echo "[s$S] TRAIN FAILED"; exit 1; }

RUN=$(ls -td pretrained/${SP}_* 2>/dev/null | head -1)
if [ -z "$RUN" ]; then echo "[s$S] no run dir found"; exit 1; fi
echo "[s$S g$G] run dir = $RUN"

echo "[s$S g$G] CALIBRATE $(date)"
$PY scripts/calibrate_gdeltauq.py --run "$RUN" --K 100 --device cuda:0 \
  || { echo "[s$S] CALIBRATE FAILED"; exit 1; }

echo "[s$S g$G] EVAL $(date)"
$PY scripts/eval_paper_protocol_gdeltauq.py --run "$RUN" --K 100 --out $SP \
  || { echo "[s$S] EVAL FAILED"; exit 1; }

ARR=$(ls -td results/${SP}/*/arrays.npz 2>/dev/null | head -1)
echo "[s$S g$G] arrays = $ARR"

echo "[s$S g$G] EVAL_FROM_ARRAYS $(date)"
$PY competitors/common/eval_from_arrays.py --arrays "$ARR" \
  --split pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json \
  --bundle pretrained/swat_ensemble/calibration_bundle \
  --slide_win 60 --label GDN-s$S --out results/competitors/gdn/seed${S}.json

echo "[s$S g$G] CHAIN DONE $(date)"
