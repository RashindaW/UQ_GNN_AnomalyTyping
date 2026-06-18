#!/usr/bin/env bash
# Build the calibration bundle for the trained ensemble.
# Override defaults via env vars:
#   PYTHON=/path/to/python   use a specific Python interpreter
#   DEVICE=cuda:0            override GPU
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
MANIFEST="${MANIFEST:-pretrained/swat_ensemble/manifest.json}"

# The test-set split is shared across both backends (GDN + DualSTGF). Build it
# once under the canonical swat_ensemble bundle; calibrate.py copies it into
# the active bundle dir if missing.
if [[ ! -f "pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json" ]]; then
  "$PYTHON" scripts/build_test_split.py
fi

"$PYTHON" scripts/calibrate.py --device "$DEVICE" --manifest "$MANIFEST" "$@"
