#!/usr/bin/env bash
# Orchestrate the full V1/V2 UQ pipeline in dependency order:
#   0. wait for in-flight GDN fusion + CST-GL V1/V2 training to finish
#   1. GDN real-Omega splice (12)         [rashindaNew env]
#   2. TopoGDN full UQ (12)               [topogdn env]
#   3. CST-GL full UQ (12)                [cstgl env]
#   4. seed-wise fusion on arrays_full    [rashindaNew env] -> fusion_v1v2_seedwise.csv
# Good-neighbour: CONC concurrent, capped threads. Each extraction is one
# (variant,seed); we batch them CONC-wide per backbone.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PYC=/home/rashinda/.conda/envs/cstgl/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
SEEDS=(0 1 2 3 4 42); GPUS=(0 1 2 3); CONC=4
B=results/baseline_v1v2

echo "[orch] START $(date)"

# ---- 0. wait for prerequisites ----
echo "[orch] waiting for in-flight GDN fusion + CST-GL training..."
until grep -q "wrote " "$B/fusion_v1v2.log" 2>/dev/null || ! pgrep -f fusion_v1v2.py >/dev/null; do sleep 30; done
until [ "$(ls $B/cstgl/logs/*_seed*.done 2>/dev/null | wc -l)" -ge 12 ] || ! pgrep -f 'run.py.*base' >/dev/null; do sleep 60; done
echo "[orch] prerequisites clear $(date)"

# generic CONC-wide batch runner: $1=tag $2=cmd-template (uses {V} {S} {G})
run_batch() {
  local tag="$1"; shift; local tmpl="$*"
  local i=0; local pids=()
  for V in V1 V2; do for S in "${SEEDS[@]}"; do
    local G=${GPUS[$((i % 4))]}
    local cmd=${tmpl//\{V\}/$V}; cmd=${cmd//\{S\}/$S}; cmd=${cmd//\{G\}/$G}
    ( eval "$cmd" > "$B/_orch_${tag}_${V}_s${S}.log" 2>&1 ) & pids+=("$!"); i=$((i+1))
    if [ $((i % CONC)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
  done; done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[orch] $tag batch done $(date)"
}

# ---- 1. GDN real Omega (uses the dedicated runner which already loops 12) ----
echo "[orch] 1/4 GDN Omega $(date)"
bash scripts/paper/v1v2_omega_gdn.sh || echo "[orch] gdn omega had failures (continuing)"

# ---- 2. TopoGDN full UQ ----
echo "[orch] 2/4 TopoGDN full UQ $(date)"
run_batch topo "CUDA_VISIBLE_DEVICES={G} $PYT competitors/common/v1v2_fulluq_topogdn.py --variant {V} --seed {S} --device cuda:0"

# ---- 3. CST-GL full UQ ----
echo "[orch] 3/4 CST-GL full UQ $(date)"
run_batch cstgl "CUDA_VISIBLE_DEVICES={G} $PYC competitors/common/v1v2_fulluq_cstgl.py --variant {V} --seed {S} --device cuda:0"

# ---- 4. seed-wise fusion on full-UQ arrays ----
echo "[orch] 4/4 seed-wise fusion $(date)"
OMP_NUM_THREADS=8 "$PY" scripts/paper/fusion_v1v2.py || echo "[orch] fusion had issues"

echo "[orch] ALL DONE $(date)"
echo "[orch] full arrays: gdn=$(ls $B/gdn/V*/seed*/arrays_full.npz 2>/dev/null|wc -l) topo=$(ls $B/topogdn/V*/seed*/arrays_full.npz 2>/dev/null|wc -l) cstgl=$(ls $B/cstgl/V*/seed*/arrays_full.npz 2>/dev/null|wc -l) /12 each"
