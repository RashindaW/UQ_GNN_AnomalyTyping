#!/usr/bin/env bash
# TopoGDN V1/V2 full-UQ chain (GDN-only counterpart already done):
#   1. extract full UQ for 12 TopoGDN baselines (epistemic+aleatoric+Omega+U_str)
#   2. re-run seed-wise fusion (now includes topogdn rows; promote_omega is
#      idempotent so the already-promoted GDN arrays + numbers are unchanged)
# Good-neighbour: 12-wide one wave (3/GPU), threads capped to avoid CPU thrash.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
SEEDS=(0 1 2 3 4 42); GPUS=(0 1 2 3); CONC=12
OUT=results/baseline_v1v2/topogdn; mkdir -p "$OUT/fulluq_logs"

echo "[topo-chain] 1/2 full-UQ extraction (12-wide) $(date)"
run_one() {
  local V="$1" S="$2" G="$3"
  local lg="$OUT/fulluq_logs/${V}_seed${S}.log"; local sent="$OUT/fulluq_logs/${V}_seed${S}.done"
  {
    echo "== topo-fulluq $V s$S gpu $G $(date) =="
    CUDA_VISIBLE_DEVICES="$G" "$PYT" competitors/common/v1v2_fulluq_topogdn.py \
      --variant "$V" --seed "$S" --device cuda:0
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$OUT/$V/seed$S/arrays_full.npz" ]; then echo 0 > "$sent"; echo "DONE"; else echo "$rc" > "$sent"; echo "FAIL rc=$rc"; fi
  } > "$lg" 2>&1
}
i=0; pids=()
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  run_one "$V" "$S" "${GPUS[$((i % 4))]}" & pids+=("$!"); i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done; done
for p in "${pids[@]}"; do wait "$p"; done
echo "[topo-chain] extraction done $(date): $(for V in V1 V2; do for S in ${SEEDS[@]}; do echo -n $V-s$S:rc$(cat $OUT/fulluq_logs/${V}_seed${S}.done 2>/dev/null)\ ; done; done)"

echo "[topo-chain] 2/2 seed-wise fusion on full-UQ arrays $(date)"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" scripts/paper/fusion_v1v2.py
echo "[topo-chain] DONE $(date)"
echo "TopoGDN full arrays: $(ls $OUT/V*/seed*/arrays_full.npz 2>/dev/null|wc -l)/12"
