#!/usr/bin/env bash
# Seed-42 extension for the thesis Part 1 fusion evals (protocol = all 6 trained
# seeds). Produces the three missing pieces:
#   (A) gdn/topogdn/cstgl V2 s42 held-out        -> fusion_heldout/seed42_V2_heldout.csv
#   (B) dualstage V2 s42 full, then held-out     -> fusion_dualstage/seed42_V2_{full,heldout}.csv
# (B) runs its two regions SEQUENTIALLY: the first touch Omega-promotes the s42
# npz in place; parallel full+heldout on the same file would race that rewrite.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
H=results/baseline_v1v2/fusion_heldout
D=results/baseline_v1v2/fusion_dualstage
echo "[s42-ext] launching $(date)"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" scripts/paper/fusion_v1v2.py \
  --seeds 42 --variants V2 --region heldout \
  --out "$H/seed42_V2_heldout.csv" > "$H/seed42_V2.log" 2>&1 &
A=$!
(
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" scripts/paper/fusion_v1v2.py \
    --backbones dualstage --seeds 42 --variants V2 --region full \
    --out "$D/seed42_V2_full.csv" > "$D/seed42_full.log" 2>&1
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" scripts/paper/fusion_v1v2.py \
    --backbones dualstage --seeds 42 --variants V2 --region heldout \
    --out "$D/seed42_V2_heldout.csv" > "$D/seed42_heldout.log" 2>&1
) &
B=$!
rc=0
wait "$A" || rc=1
wait "$B" || rc=1
for f in "$H/seed42_V2_heldout.csv" "$D/seed42_V2_full.csv" "$D/seed42_V2_heldout.csv"; do
  [ -f "$f" ] && echo "  $(wc -l < "$f") lines $f" || { echo "  MISSING $f"; rc=1; }
done
echo "[s42-ext] done rc=$rc $(date)"
exit "$rc"
