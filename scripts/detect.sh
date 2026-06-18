#!/usr/bin/env bash
# Run inference + detection on the final-test slice using the calibration bundle.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
MANIFEST="${MANIFEST:-pretrained/swat_ensemble/manifest.json}"

"$PYTHON" scripts/detect.py --device "$DEVICE" --manifest "$MANIFEST" "$@"
