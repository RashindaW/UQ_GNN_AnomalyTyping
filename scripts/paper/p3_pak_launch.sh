#!/usr/bin/env bash
# PA%K curve fleet: 18 combos, 6-wide waves.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p results/thesis_part1/pak_curves
rc=0; n=0; pids=()
for bb in gdn topogdn cstgl; do for s in 0 1 2 3 4 42; do
  $PY scripts/paper/p3_pak_curves.py --combo "$bb:$s" \
    > "results/thesis_part1/pak_curves/${bb}_s${s}.log" 2>&1 &
  pids+=("$!"); n=$((n+1))
  if [ "$n" -ge 6 ]; then for p in "${pids[@]}"; do wait "$p" || rc=1; done; pids=(); n=0; fi
done; done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
NC=$(ls results/thesis_part1/pak_curves/*.npz 2>/dev/null | wc -l)
echo "[pak] done rc=$rc curves=$NC/18"
[ "$NC" -eq 18 ] || exit 1
exit "$rc"
