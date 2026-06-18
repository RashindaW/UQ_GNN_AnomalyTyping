#!/usr/bin/env bash
# Top-level chain: run Q2-(1) temporal sharpening first (fast, CPU-only),
# then Q1-(1) slide_win sweep (slow, GPU). Designed for tmux q2q1-chain.

set -euo pipefail

cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python

echo "============================================================"
echo "=== Q2-(1) TEMPORAL SHARPENING  ($(date))"
echo "============================================================"

"$PY" scripts/temporal_sharpen_uq.py \
  -arrays results/swat_gdeltauq_paper_protocol/0511-222735/arrays.npz \
  -out_root results/uq_attack_assoc/sharpen \
  -W_grid 0 1 3 5 10 30 \
  -L_grid 0 3 10 30 100 200 500 1000 \
  -permutations 1000

echo "============================================================"
echo "=== Q1-(1) SLIDE_WIN SWEEP  ($(date))"
echo "============================================================"

bash scripts/run_slidewin_sweep.sh

echo "============================================================"
echo "=== ALL DONE  ($(date))"
echo "============================================================"
