#!/usr/bin/env bash
# Apples-to-apples comparison at the 80/10/10 split.
#
# For each slide_win in {5, 15, 60} (in parallel on cuda:0/1/2):
#   1. Train G-DeltaUQ on the 80% train slice + 10% val + 10% aleatoric.
#   2. Calibrate G-DeltaUQ + run paper-protocol eval (saves arrays.npz).
#   3. Train GDN on the SAME 80% + 10% (ignoring the 10% aleatoric slice).
#   4. Eval GDN with -save_arrays so Fix A can run on it.
#
# After all per-sw streams finish (Stage 1):
#   Stage 2: apply Fix A to every new arrays.npz (CPU-only).
#   Stage 3: write a rollup CSV comparing GDN vs G-DeltaUQ at fair split.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
SPLIT=data/swat/gdeltauq_split_80_10_10.json
mkdir -p logs

# Single-stream chain: G-DeltaUQ train+cal+eval, then GDN train+eval.
chain_for_sw() {
  local SW=$1
  local DEV=$2
  local LOG=logs/fair80_sw${SW}.log
  {
    GTAG="swat_gdeltauq_80_sw${SW}"
    DTAG="swat_gdn_80_sw${SW}"

    echo "=== sw=${SW} dev=${DEV}: GDeltaUQ TRAIN  $(date)"
    "$PY" train_gdeltauq_main.py \
      -dataset swat -slide_win "$SW" -slide_stride 1 \
      -epoch 100 -batch 128 -dim 64 -out_layer_num 1 \
      -out_layer_inter_dim 128 -topk 15 -n_gnn_layers 2 \
      -K_anchors 10 -decay 0.0 -random_seed 42 \
      -split_path "$SPLIT" -save_path_pattern "$GTAG" \
      -device "$DEV" -comment "fair80 sw=${SW}"

    GCKPT=$(ls -t "pretrained/${GTAG}"/best_*.pt | head -1)
    GHP=$(ls -t "pretrained/${GTAG}"/hyperparameters_*.json | head -1)

    echo "=== sw=${SW}: GDeltaUQ CALIBRATE  $(date)"
    "$PY" scripts/calibrate_gdeltauq.py \
      -checkpoint "$GCKPT" -hyperparameters "$GHP" \
      -split_path "$SPLIT" \
      -K_anchors 10 -anchor_seed 0 -anchor_strategy random \
      -save_dir "pretrained/${GTAG}/calibration_bundle" \
      -device "$DEV"

    echo "=== sw=${SW}: GDeltaUQ EVAL  $(date)"
    "$PY" scripts/eval_paper_protocol_gdeltauq.py \
      -checkpoint "$GCKPT" -hyperparameters "$GHP" \
      -bundle_dir "pretrained/${GTAG}/calibration_bundle" \
      -split_path "$SPLIT" \
      -topk 1 -device "$DEV" \
      -results_dir "results/${GTAG}_paper_protocol"

    echo "=== sw=${SW}: GDN TRAIN  $(date)"
    "$PY" main.py \
      -dataset swat -model gdn \
      -save_path_pattern "$DTAG" \
      -slide_win "$SW" -slide_stride 1 \
      -batch 128 -epoch 100 -dim 64 \
      -out_layer_num 1 -out_layer_inter_dim 128 \
      -topk 15 -decay 0.0 -random_seed 42 \
      -val_ratio 0.1 -report best \
      -split_path "$SPLIT" \
      -save_arrays \
      -device "$DEV" \
      -comment "fair80 GDN sw=${SW}"

    echo "=== sw=${SW}: DONE  $(date)"
  } &> "$LOG"
}

echo "=========================================================="
echo "=== STAGE 1: PARALLEL TRAIN+CAL+EVAL (sw 5/15/60)  $(date)"
echo "=========================================================="

chain_for_sw  5 cuda:0 & PID5=$!
chain_for_sw 15 cuda:1 & PID15=$!
chain_for_sw 60 cuda:2 & PID60=$!
echo "sw=5  pid=$PID5  log=logs/fair80_sw5.log"
echo "sw=15 pid=$PID15 log=logs/fair80_sw15.log"
echo "sw=60 pid=$PID60 log=logs/fair80_sw60.log"

wait "$PID5"  || echo "sw=5  exited non-zero"
wait "$PID15" || echo "sw=15 exited non-zero"
wait "$PID60" || echo "sw=60 exited non-zero"

echo "=========================================================="
echo "=== STAGE 2: FIX A ON ALL NEW ARRAYS  $(date)"
echo "=========================================================="

FIXA_OUT=results/postproc_threshold_fixA
mkdir -p "$FIXA_OUT"

for SW in 5 15 60; do
  for KIND in gdeltauq_80 gdn_80; do
    if [[ "$KIND" == gdeltauq_80 ]]; then
      ARR=$(ls -t "results/swat_${KIND}_sw${SW}_paper_protocol"/*/arrays.npz 2>/dev/null | head -1 || true)
    else
      ARR=$(ls -t "results/swat_${KIND}_sw${SW}"/*_arrays.npz 2>/dev/null | head -1 || true)
    fi
    if [[ -z "$ARR" || ! -f "$ARR" ]]; then
      echo "skip ${KIND}_sw${SW}: missing arrays"
      continue
    fi
    echo "--- Fix A: ${KIND}_sw${SW}  ${ARR}"
    "$PY" scripts/sweep_postproc_threshold.py \
      -arrays "$ARR" \
      -out_root "$FIXA_OUT" \
      -configs paper cheap paperW3 cheapW3 \
      &> "logs/fixA_${KIND}_sw${SW}.log"
  done
done

echo "=========================================================="
echo "=== STAGE 3: FAIR-SPLIT ROLLUP  $(date)"
echo "=========================================================="

"$PY" - <<'PYEOF'
import csv
import json
from glob import glob
from pathlib import Path
import pandas as pd

rows = []

def latest(p):
    rs = sorted(glob(p), reverse=True)
    return rs[0] if rs else None

# G-DeltaUQ paper-protocol reports
for sw in (5, 15, 60):
    p = latest(f'results/swat_gdeltauq_80_sw{sw}_paper_protocol/*/report.json')
    if p:
        with open(p) as f:
            r = json.load(f)
        pp = r['paper_protocol']
        rows.append({
            'method': 'GDeltaUQ_80', 'slide_win': sw,
            'F1_paper': pp['F1'], 'P_paper': pp['precision'],
            'R_paper': pp['recall'], 'AUC_paper': pp['AUC'],
            'report': p,
        })

# GDN: re-derive paper-protocol F1 from arrays.npz (saved by main.py;
# already includes F1 etc. as scalars in the npz).
import numpy as np
for sw in (5, 15, 60):
    arr = latest(f'results/swat_gdn_80_sw{sw}/*_arrays.npz')
    if arr:
        d = np.load(arr)
        rows.append({
            'method': 'GDN_80', 'slide_win': sw,
            'F1_paper': float(d['F1']),
            'P_paper': float(d['precision']),
            'R_paper': float(d['recall']),
            'AUC_paper': None,
            'report': arr,
        })

# Pull Fix-A best-config F1 per (method, sw).
for fixa_dir in sorted(glob('results/postproc_threshold_fixA/*')):
    csvp = Path(fixa_dir) / 'fixA_sweep.csv'
    if not csvp.exists():
        continue
    df = pd.read_csv(csvp)
    src = Path(fixa_dir).name
    # Filter to fair-80 sources only.
    if 'gdn_80' not in src and 'gdeltauq_80' not in src:
        continue
    best = df.loc[df['F1_fixA'].idxmax()]
    method = 'GDN_80' if 'gdn_80' in src else 'GDeltaUQ_80'
    # Extract sw from source name
    sw = None
    for s in (5, 15, 60):
        if f'sw{s}' in src:
            sw = s
            break
    if sw is None:
        continue
    # Find matching paper row and add fix-A best
    for r in rows:
        if r['method'] == method and r['slide_win'] == sw:
            r['F1_fixA_best'] = float(best['F1_fixA'])
            r['fixA_config']  = best['config']
            break

out = 'results/fair80_rollup.csv'
fields = ['method', 'slide_win', 'F1_paper', 'F1_fixA_best',
          'fixA_config', 'P_paper', 'R_paper', 'AUC_paper', 'report']
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, '') for k in fields})
print(f'rollup -> {out}')
print(pd.DataFrame(rows).to_string(index=False))
PYEOF

echo "=========================================================="
echo "=== ALL FAIR80 DONE  $(date)"
echo "=========================================================="
