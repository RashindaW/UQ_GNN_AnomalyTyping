#!/usr/bin/env bash
# Apples-to-apples multi-seed comparison at sw=60 with the original 70:10:20
# split (data/swat/gdeltauq_split.json — same file the original G-DeltaUQ
# pipeline uses).
#
# - GDN: train at seeds {1, 2, 3, 42}.
# - G-DeltaUQ: train at seeds {1, 2, 3} (seed=42 already exists from prior
#   slide_win sweep at results/swat_gdeltauq_sw60_paper_protocol/0513-211654/).
#
# Distribution across 4 GPUs (each stream paired GDN→G-DeltaUQ where
# possible to balance load):
#   cuda:0  GDN seed=1   then G-DeltaUQ seed=1
#   cuda:1  GDN seed=2   then G-DeltaUQ seed=2
#   cuda:2  GDN seed=3   then G-DeltaUQ seed=3
#   cuda:3  GDN seed=42  (G-DeltaUQ seed=42 already present)

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
SPLIT=data/swat/gdeltauq_split.json
mkdir -p logs

train_gdn_sw60_seed() {
  local SEED=$1
  local DEV=$2
  local TAG="swat_gdn_70_sw60_seed${SEED}"
  local LOG="logs/${TAG}.log"
  echo "=== ${TAG}: TRAIN seed=${SEED} dev=${DEV}  $(date)"
  "$PY" main.py \
    -dataset swat -model gdn \
    -save_path_pattern "$TAG" \
    -slide_win 60 -slide_stride 1 \
    -batch 128 -epoch 100 -dim 64 \
    -out_layer_num 1 -out_layer_inter_dim 128 \
    -topk 15 -decay 0.0 -random_seed "$SEED" \
    -val_ratio 0.1 -report best \
    -split_path "$SPLIT" \
    -save_arrays \
    -device "$DEV" \
    -comment "70:10:20 GDN sw=60 seed=${SEED}" >> "$LOG" 2>&1
  echo "=== ${TAG}: DONE  $(date)"
}

train_cal_eval_gdeltauq_sw60_seed() {
  local SEED=$1
  local DEV=$2
  local TAG="swat_gdeltauq_70_sw60_seed${SEED}"
  local LOG="logs/${TAG}.log"
  echo "=== ${TAG}: TRAIN seed=${SEED} dev=${DEV}  $(date)"
  "$PY" train_gdeltauq_main.py \
    -dataset swat -slide_win 60 -slide_stride 1 \
    -epoch 100 -batch 128 -dim 64 -out_layer_num 1 \
    -out_layer_inter_dim 128 -topk 15 -n_gnn_layers 2 \
    -K_anchors 10 -decay 0.0 -random_seed "$SEED" \
    -split_path "$SPLIT" -save_path_pattern "$TAG" \
    -device "$DEV" -comment "70:10:20 GDeltaUQ sw=60 seed=${SEED}" >> "$LOG" 2>&1

  CKPT=$(ls -t "pretrained/${TAG}"/best_*.pt | head -1)
  HP=$(ls -t "pretrained/${TAG}"/hyperparameters_*.json | head -1)

  echo "=== ${TAG}: CALIBRATE  $(date)"
  "$PY" scripts/calibrate_gdeltauq.py \
    -checkpoint "$CKPT" -hyperparameters "$HP" \
    -split_path "$SPLIT" \
    -K_anchors 10 -anchor_seed 0 -anchor_strategy random \
    -save_dir "pretrained/${TAG}/calibration_bundle" \
    -device "$DEV" >> "$LOG" 2>&1

  echo "=== ${TAG}: EVAL  $(date)"
  "$PY" scripts/eval_paper_protocol_gdeltauq.py \
    -checkpoint "$CKPT" -hyperparameters "$HP" \
    -bundle_dir "pretrained/${TAG}/calibration_bundle" \
    -split_path "$SPLIT" \
    -topk 1 -device "$DEV" \
    -results_dir "results/${TAG}_paper_protocol" >> "$LOG" 2>&1
  echo "=== ${TAG}: DONE  $(date)"
}

paired_stream() {
  local SEED=$1
  local DEV=$2
  {
    train_gdn_sw60_seed "$SEED" "$DEV"
    train_cal_eval_gdeltauq_sw60_seed "$SEED" "$DEV"
  } &> "logs/sw60_70_10_20_seed${SEED}_${DEV//:/_}.log"
}

solo_gdn_stream() {
  local SEED=$1
  local DEV=$2
  {
    train_gdn_sw60_seed "$SEED" "$DEV"
  } &> "logs/sw60_70_10_20_seed${SEED}_${DEV//:/_}.log"
}

echo "=========================================================="
echo "=== STAGE 1: PARALLEL TRAIN (sw=60, 70:10:20)  $(date)"
echo "=========================================================="

paired_stream  1 cuda:0 & PID1=$!
paired_stream  2 cuda:1 & PID2=$!
paired_stream  3 cuda:2 & PID3=$!
solo_gdn_stream 42 cuda:3 & PID42=$!
echo "cuda:0 (GDN+GDeltaUQ seed=1) pid=$PID1"
echo "cuda:1 (GDN+GDeltaUQ seed=2) pid=$PID2"
echo "cuda:2 (GDN+GDeltaUQ seed=3) pid=$PID3"
echo "cuda:3 (GDN seed=42 only)    pid=$PID42"

wait "$PID1"  || echo "stream 1 exited non-zero"
wait "$PID2"  || echo "stream 2 exited non-zero"
wait "$PID3"  || echo "stream 3 exited non-zero"
wait "$PID42" || echo "stream 42 exited non-zero"

echo "=========================================================="
echo "=== STAGE 2: ROLLUP  $(date)"
echo "=========================================================="

"$PY" - <<'PYEOF'
import csv
import json
import numpy as np
from glob import glob
from pathlib import Path

def latest(p):
    rs = sorted(glob(p), reverse=True)
    return rs[0] if rs else None

rows = []

# GDN: arrays.npz contains F1/precision/recall scalars (from main.py)
for seed in (1, 2, 3, 42):
    arr = latest(f'results/swat_gdn_70_sw60_seed{seed}/*_arrays.npz')
    if arr is None:
        rows.append({'method': 'GDN_70_sw60', 'seed': seed,
                     'F1': None, 'P': None, 'R': None, 'AUC': None,
                     'arrays': None})
        continue
    d = np.load(arr)
    rows.append({
        'method': 'GDN_70_sw60', 'seed': seed,
        'F1': float(d['F1']), 'P': float(d['precision']),
        'R': float(d['recall']), 'AUC': None,
        'arrays': arr,
    })

# G-DeltaUQ: paper-protocol report.json. Seeds 1/2/3 are new; seed=42
# reuses the existing slide_win sweep result.
for seed in (1, 2, 3):
    rep = latest(f'results/swat_gdeltauq_70_sw60_seed{seed}_paper_protocol/*/report.json')
    if rep is None:
        rows.append({'method': 'GDeltaUQ_70_sw60', 'seed': seed,
                     'F1': None, 'P': None, 'R': None, 'AUC': None,
                     'arrays': None})
        continue
    with open(rep) as f:
        r = json.load(f)
    pp = r['paper_protocol']
    rows.append({
        'method': 'GDeltaUQ_70_sw60', 'seed': seed,
        'F1': pp['F1'], 'P': pp['precision'], 'R': pp['recall'],
        'AUC': pp['AUC'], 'arrays': rep,
    })

# Reuse existing seed=42 from slide_win sweep
existing = latest('results/swat_gdeltauq_sw60_paper_protocol/*/report.json')
if existing:
    with open(existing) as f:
        r = json.load(f)
    pp = r['paper_protocol']
    rows.append({
        'method': 'GDeltaUQ_70_sw60', 'seed': 42,
        'F1': pp['F1'], 'P': pp['precision'], 'R': pp['recall'],
        'AUC': pp['AUC'], 'arrays': existing,
    })

print('Per-seed results (sw=60, 70:10:20):\n')
print(f'{"method":<22s} {"seed":>4s} {"F1":>8s} {"P":>8s} {"R":>8s}')
for r in rows:
    f1s = f"{r['F1']:.4f}" if r['F1'] is not None else 'N/A'
    ps  = f"{r['P']:.4f}"  if r['P']  is not None else 'N/A'
    rs  = f"{r['R']:.4f}"  if r['R']  is not None else 'N/A'
    print(f'{r["method"]:<22s} {r["seed"]:>4d} {f1s:>8s} {ps:>8s} {rs:>8s}')

print('\nPer-method stats:')
for m in ['GDN_70_sw60', 'GDeltaUQ_70_sw60']:
    f1s = [r['F1'] for r in rows if r['method'] == m and r['F1'] is not None]
    if f1s:
        print(f'  {m}: n={len(f1s)} mean={np.mean(f1s):.4f} '
              f'std={np.std(f1s):.4f} min={min(f1s):.4f} max={max(f1s):.4f}')

# Headline comparison
gdn_f1 = [r['F1'] for r in rows if r['method'] == 'GDN_70_sw60'      and r['F1'] is not None]
gdq_f1 = [r['F1'] for r in rows if r['method'] == 'GDeltaUQ_70_sw60' and r['F1'] is not None]
if gdn_f1 and gdq_f1:
    print(f'\nMean F1 delta (G-DeltaUQ - GDN): '
          f'{np.mean(gdq_f1) - np.mean(gdn_f1):+.4f}')
    print(f'Best F1 delta (G-DeltaUQ best - GDN best): '
          f'{max(gdq_f1) - max(gdn_f1):+.4f}')

out = 'results/sw60_70_10_20_multiseed_rollup.csv'
fields = ['method', 'seed', 'F1', 'P', 'R', 'AUC', 'arrays']
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'\nrollup -> {out}')
PYEOF

echo "=========================================================="
echo "=== ALL DONE  $(date)"
echo "=========================================================="
