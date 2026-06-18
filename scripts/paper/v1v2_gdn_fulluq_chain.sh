#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
echo "[gdn-chain] 1/2 Omega extraction $(date)"
bash scripts/paper/v1v2_omega_gdn.sh
echo "[gdn-chain] 2/2 seed-wise fusion on full-UQ arrays $(date)"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" scripts/paper/fusion_v1v2.py
echo "[gdn-chain] DONE $(date)"
echo "GDN full arrays: $(ls results/baseline_v1v2/gdn/V*/seed*/arrays_full.npz 2>/dev/null|wc -l)/12"
