#!/usr/bin/env bash
# Phase-2 (Thread B) on the sw=60 seed=42 G-DeltaUQ model arrays.
#
# Re-runs the three UQ-vs-attack-window analyses that previously ran on
# the sw=15 model (results/uq_attack_assoc/0513-200355/, .../stacked/),
# now on the optimal HP combination identified by the Fix-A best result
# (sw=60, 70:10:20, seed=42, F1=0.7951 with config tk1_sm5_W5_G5).
#
#   §1-§3  analyze_uq_attack_association.py  -> marginal coincidence,
#                                               PTaRP saturation,
#                                               permutation test
#   Q2-1   temporal_sharpen_uq.py            -> causal Δ_W on each UQ
#                                               signal
#   §5     stack_hybrid_on_cheapsweep.py     -> stacked OR/AND on the
#                                               cheap-sweep best base
#
# CPU-only. Designed for tmux session `phase2-seed42-sw60`.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
ARR=results/swat_gdeltauq_sw60_paper_protocol/0513-211654/arrays.npz

if [[ ! -f "$ARR" ]]; then
  echo "ERROR: missing arrays $ARR" >&2
  exit 1
fi

DATESTR=$(date +%m%d-%H%M%S)
mkdir -p logs

LOG_ASSOC="logs/phase2_assoc_seed42_sw60_${DATESTR}.log"
LOG_SHARP="logs/phase2_sharpen_seed42_sw60_${DATESTR}.log"
LOG_STACK="logs/phase2_stack_seed42_sw60_${DATESTR}.log"

echo "=========================================================="
echo "=== Phase-2 on seed=42 sw=60 arrays  $(date)"
echo "=== arrays: $ARR"
echo "=========================================================="

echo ""
echo "--- §1-§3 marginal + PTaRP + permutation  $(date)"
"$PY" scripts/analyze_uq_attack_association.py \
  -arrays "$ARR" \
  -out_root results/uq_attack_assoc_sw60 \
  -permutations 1000 \
  2>&1 | tee "$LOG_ASSOC"

echo ""
echo "--- Q2-1 temporal sharpening (causal Δ_W)  $(date)"
"$PY" scripts/temporal_sharpen_uq.py \
  -arrays "$ARR" \
  -out_root results/uq_attack_assoc_sw60/sharpen \
  -permutations 1000 \
  2>&1 | tee "$LOG_SHARP"

echo ""
echo "--- §5 stacked OR/AND on cheap-sweep best  $(date)"
"$PY" scripts/stack_hybrid_on_cheapsweep.py \
  -arrays "$ARR" \
  -out_root results/uq_attack_assoc_sw60/stacked \
  2>&1 | tee "$LOG_STACK"

echo ""
echo "=========================================================="
echo "=== Phase-2 ALL DONE  $(date)"
echo ""
echo "Outputs:"
echo "  results/uq_attack_assoc_sw60/<datestr>/"
echo "  results/uq_attack_assoc_sw60/sharpen/<datestr>/"
echo "  results/uq_attack_assoc_sw60/stacked/<datestr>/"
echo "=========================================================="
