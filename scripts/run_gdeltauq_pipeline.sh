#!/usr/bin/env bash
# End-to-end GDN_GDeltaUQ pipeline: split -> train -> calibrate -> detect.
#
# Expects:
#   - data/swat/train.csv and data/swat/test.csv populated.
#   - GPU available (set DEVICE=cpu to override).
#
# CHECKPOINT and HYPERPARAMETERS env vars can be passed to skip training and
# point at an existing run.
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
SPLIT_PATH="${SPLIT_PATH:-data/swat/gdeltauq_split.json}"

echo "=== step 1: build 70/10/20 split ==="
python scripts/build_gdeltauq_split.py \
    -dataset swat -out_path "$SPLIT_PATH"

if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "=== step 2: train ==="
    DEVICE="$DEVICE" SEED="$SEED" EPOCHS="$EPOCHS" \
        SPLIT_PATH="$SPLIT_PATH" \
        bash scripts/train_gdeltauq.sh

    echo "Set CHECKPOINT and HYPERPARAMETERS to the values printed above, then"
    echo "re-run with those env vars to continue to calibration + detection."
    exit 0
fi

HYPERPARAMETERS="${HYPERPARAMETERS:?HYPERPARAMETERS must be set when CHECKPOINT is set}"

echo "=== step 3: calibrate ==="
python scripts/calibrate_gdeltauq.py \
    -checkpoint "$CHECKPOINT" \
    -hyperparameters "$HYPERPARAMETERS" \
    -split_path "$SPLIT_PATH" \
    -device "$DEVICE"

echo "=== step 4: detect on final-test ==="
python scripts/detect_gdeltauq.py \
    -checkpoint "$CHECKPOINT" \
    -hyperparameters "$HYPERPARAMETERS" \
    -device "$DEVICE"
