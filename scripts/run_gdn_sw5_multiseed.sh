#!/usr/bin/env bash
# Verify whether GDN sw=5 fair-split F1=0.5937 (seed=42) is real or a
# bad-seed artifact. Train at seeds {1, 2, 3} on cuda:0, cuda:2, cuda:3
# (cuda:1 is occupied by the in-flight fair80 sw=15 stream).
#
# All four seeds (1, 2, 3 plus the pre-existing 42 from the fair80 chain)
# are then summarized in a rollup CSV.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
SPLIT=data/swat/gdeltauq_split_80_10_10.json
mkdir -p logs

train_gdn_sw5_seed() {
  local SEED=$1
  local DEV=$2
  local TAG="swat_gdn_80_sw5_seed${SEED}"
  local LOG="logs/gdn_sw5_seed${SEED}.log"
  {
    echo "=== ${TAG}: TRAIN seed=${SEED} dev=${DEV}  $(date)"
    "$PY" main.py \
      -dataset swat -model gdn \
      -save_path_pattern "$TAG" \
      -slide_win 5 -slide_stride 1 \
      -batch 128 -epoch 100 -dim 64 \
      -out_layer_num 1 -out_layer_inter_dim 128 \
      -topk 15 -decay 0.0 -random_seed "$SEED" \
      -val_ratio 0.1 -report best \
      -split_path "$SPLIT" \
      -save_arrays \
      -device "$DEV" \
      -comment "fair80 GDN sw=5 seed=${SEED}"
    echo "=== ${TAG}: DONE  $(date)"
  } &> "$LOG"
}

echo "=========================================================="
echo "=== GDN sw=5 MULTI-SEED (1, 2, 3) on cuda:0/2/3  $(date)"
echo "=========================================================="

train_gdn_sw5_seed 1 cuda:0 & PID1=$!
train_gdn_sw5_seed 2 cuda:2 & PID2=$!
train_gdn_sw5_seed 3 cuda:3 & PID3=$!
echo "seed=1 pid=$PID1 log=logs/gdn_sw5_seed1.log"
echo "seed=2 pid=$PID2 log=logs/gdn_sw5_seed2.log"
echo "seed=3 pid=$PID3 log=logs/gdn_sw5_seed3.log"

wait "$PID1" || echo "seed=1 exited non-zero"
wait "$PID2" || echo "seed=2 exited non-zero"
wait "$PID3" || echo "seed=3 exited non-zero"

echo "=========================================================="
echo "=== SEED-MULTI ROLLUP  $(date)"
echo "=========================================================="

"$PY" - <<'PYEOF'
import csv
import numpy as np
from glob import glob
from pathlib import Path

def latest(p):
    rs = sorted(glob(p), reverse=True)
    return rs[0] if rs else None

rows = []
for seed_label, glob_pat in [
    ('seed=1',  'results/swat_gdn_80_sw5_seed1/*_arrays.npz'),
    ('seed=2',  'results/swat_gdn_80_sw5_seed2/*_arrays.npz'),
    ('seed=3',  'results/swat_gdn_80_sw5_seed3/*_arrays.npz'),
    ('seed=42', 'results/swat_gdn_80_sw5/*_arrays.npz'),
]:
    arr = latest(glob_pat)
    if not arr:
        rows.append({'seed': seed_label, 'F1': None, 'P': None, 'R': None,
                     'arrays': None})
        continue
    d = np.load(arr)
    rows.append({
        'seed': seed_label,
        'F1': float(d['F1']),
        'P':  float(d['precision']),
        'R':  float(d['recall']),
        'arrays': arr,
    })

f1s = [r['F1'] for r in rows if r['F1'] is not None]
print('Per-seed F1:')
for r in rows:
    print(f"  {r['seed']:8s} F1={r['F1']:.4f}  P={r['P']:.4f}  R={r['R']:.4f}"
          if r['F1'] is not None else f"  {r['seed']:8s} (no result)")
if f1s:
    print(f"\nF1 stats: mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}  "
          f"min={min(f1s):.4f}  max={max(f1s):.4f}")

out = 'results/gdn_sw5_seed_multi_rollup.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['seed', 'F1', 'P', 'R', 'arrays'])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'\nrollup -> {out}')
PYEOF

echo "=========================================================="
echo "=== ALL DONE  $(date)"
echo "=========================================================="
