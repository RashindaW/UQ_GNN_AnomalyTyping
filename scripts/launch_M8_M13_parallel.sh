#!/usr/bin/env bash
# Launch the remaining fusion methods (M8–M13) in 6 parallel tmux sessions.
# M8 is split into 4 chunks (each ~156 of 625 HP configs).
# M9-M13 each get their own session.
#
# Usage:
#   bash scripts/launch_M8_M13_parallel.sh

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
ARR=results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655/arrays.npz
SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
BUNDLE=pretrained/swat_gdeltauq_sw60/calibration_bundle_K100
SW=60

mkdir -p logs

DATESTR=$(date +%m%d-%H%M%S)
M8_OUT_ROOT="results/fusion_K100_full_M8_chunked/${DATESTR}"
mkdir -p "$M8_OUT_ROOT"

# ---- M8: 4 parallel chunks (5^4=625 configs split into ~156 each) ----
for CHUNK in 0 1 2 3; do
  SESSION="fusion-M8c${CHUNK}"
  LOG="logs/m8_chunk${CHUNK}_${DATESTR}.log"
  echo "Launching $SESSION (chunk $CHUNK of 4)..."
  tmux new-session -d -s "$SESSION" \
    "$PY scripts/m8_chunked.py \
       -arrays '$ARR' -split '$SPLIT' -bundle '$BUNDLE' -slide_win $SW \
       -chunk_idx $CHUNK -n_chunks 4 \
       -out_root '$M8_OUT_ROOT' \
       2>&1 | tee '$LOG'"
done

# ---- M9 + M10: stackers (LogReg + GBM). Run together in one tmux. ----
SESSION="fusion-M9M10"
LOG="logs/m9_m10_${DATESTR}.log"
M9M10_OUT_ROOT="results/fusion_K100_full_M9M10/${DATESTR}"
echo "Launching $SESSION..."
tmux new-session -d -s "$SESSION" \
  "$PY scripts/fusion_sweep_K100_full.py \
     -arrays '$ARR' -split '$SPLIT' -bundle '$BUNDLE' -slide_win $SW \
     -methods M9 M10 -seed 42 \
     -out_root '$M9M10_OUT_ROOT' \
     2>&1 | tee '$LOG'"

# ---- M11 + M12 + M13: combos + triple-OR + adaptive. ----
SESSION="fusion-M11M13"
LOG="logs/m11_m12_m13_${DATESTR}.log"
M11M13_OUT_ROOT="results/fusion_K100_full_M11M13/${DATESTR}"
echo "Launching $SESSION..."
tmux new-session -d -s "$SESSION" \
  "$PY scripts/fusion_sweep_K100_full.py \
     -arrays '$ARR' -split '$SPLIT' -bundle '$BUNDLE' -slide_win $SW \
     -methods M11 M12 M13 -seed 42 \
     -out_root '$M11M13_OUT_ROOT' \
     2>&1 | tee '$LOG'"

echo ""
echo "All 6 parallel sessions launched:"
echo "  fusion-M8c0, fusion-M8c1, fusion-M8c2, fusion-M8c3 (M8 chunks)"
echo "  fusion-M9M10  (stackers)"
echo "  fusion-M11M13 (combos + triple-OR + adaptive)"
echo ""
echo "tmux ls:"
tmux ls 2>&1 | grep -E "fusion-M|metrics"
echo ""
echo "DATESTR = $DATESTR (use this to find the output dirs)"
echo "  M8: $M8_OUT_ROOT/chunk_*/best.json"
echo "  M9-M10: $M9M10_OUT_ROOT"
echo "  M11-M13: $M11M13_OUT_ROOT"
