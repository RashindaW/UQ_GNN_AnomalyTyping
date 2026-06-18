#!/usr/bin/env bash
# Capacity-aware scheduler for the DualSTAGE anchored-UQ chain (V2-only).
# Per (V2, seed): train (needs >=9 GB, ~6.5 GB held) -> fulluq (needs >=4 GB,
# eval-mode, also requires the plain-baseline arrays for labels) -> eval_m0.
# setsid-detached children: restarting this scheduler never kills runs.
# Exits when all 6 have arrays_full.npz + eval_m0.json.
set -u
cd /mnt/datassd3/rashinda/UQ_GNN_AnomalyTyping

PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SEEDS=(0 1 2 3 4 42); EPOCH=50; BATCH=64
MIN_FREE_TRAIN=9000; MIN_FREE_EXTRACT=4000; MAX_PER=2; CYCLE=180
UQ=results/uq_v1v2/dualstage
BASE=results/baseline_v1v2/dualstage
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
mkdir -p "$UQ/logs"

live() {  # pattern V S -> 0 if alive
  ps -eo args | grep "[v]1v2_dualstage_gdeltauq_$1.py" | grep -q -- "--variant $2 --seed $3 --"
}
any_ours_on_gpu() {  # G -> count our dualstage jobs pinned to G via .gpu files
  local n=0
  for f in "$UQ"/logs/*.gpu "$BASE"/logs/*.gpu; do
    [ -f "$f" ] || continue
    [ "$(cat "$f")" = "$1" ] || continue
    local base; base="$(basename "$f" .gpu)"
    local V="${base%%_seed*}"; local S="${base##*seed}"
    if live train "$V" "$S" || live fulluq "$V" "$S" \
       || ps -eo args | grep "[b]aseline_v1v2_dualstage.py" | grep -q -- "--variant $V --seed $S --"; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}
launch() {  # mode V S G extra-args...
  local mode="$1" V="$2" S="$3" G="$4"; shift 4
  local lg="$UQ/logs/${V}_seed${S}.log"
  echo "$G" > "$UQ/logs/${V}_seed${S}.gpu"
  setsid bash -c "
    echo \"== uq-$mode $V s$S gpu $G \$(date) ==\"
    CUDA_VISIBLE_DEVICES=$G '$PYT' competitors/common/v1v2_dualstage_gdeltauq_${mode}.py \
      --variant $V --seed $S --device cuda:0 $*
    echo \"== uq-$mode $V s$S rc=\$? \$(date) ==\"
  " >> "$lg" 2>&1 < /dev/null &
}

while true; do
  done_full=0; done_eval=0; launched=0
  declare -A FREE=()
  while IFS=, read -r idx free; do
    FREE[$(echo "$idx" | tr -d ' ')]="$(echo "$free" | tr -d ' ')"
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  for V in V2; do
    for S in "${SEEDS[@]}"; do
      udir="$UQ/$V/seed$S"; mkdir -p "$udir"
      if [ -f "$udir/arrays_full.npz" ]; then
        done_full=$((done_full + 1))
        if [ ! -f "$udir/eval_m0.json" ]; then
          OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py \
            --arrays "$udir/arrays_full.npz" --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" \
            --slide_win 60 --seed "$S" --baseline-only --label "DualSTAGE-UQ-$V-s$S" \
            --out "$udir/eval_m0.json" >> "$UQ/logs/${V}_seed${S}.log" 2>&1 \
            && echo "[uq-sched] eval_m0 done $V s$S"
        fi
        [ -f "$udir/eval_m0.json" ] && done_eval=$((done_eval + 1))
        continue
      fi
      live train "$V" "$S" && continue
      live fulluq "$V" "$S" && continue
      if [ -f "$udir/best.pt" ]; then
        # extraction stage: needs baseline arrays for labels
        [ -f "$BASE/$V/seed$S/arrays.npz" ] || continue
        for G in 0 1 2 3; do
          free="${FREE[$G]:-0}"
          [ "$free" -ge "$MIN_FREE_EXTRACT" ] || continue
          [ "$(any_ours_on_gpu "$G")" -lt "$MAX_PER" ] || continue
          launch fulluq "$V" "$S" "$G" "--batch 128"
          FREE[$G]=$((free - 3000)); launched=$((launched + 1)); break
        done
      else
        for G in 0 1 2 3; do
          free="${FREE[$G]:-0}"
          [ "$free" -ge "$MIN_FREE_TRAIN" ] || continue
          [ "$(any_ours_on_gpu "$G")" -lt "$MAX_PER" ] || continue
          launch train "$V" "$S" "$G" "--epochs $EPOCH --batch $BATCH"
          FREE[$G]=$((free - 8000)); launched=$((launched + 1)); break
        done
      fi
    done
  done
  echo "[uq-sched] $(date +%H:%M) arrays_full=$done_full/6 eval_m0=$done_eval/6 launched_now=$launched free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ',')"
  if [ "$done_full" -ge 6 ] && [ "$done_eval" -ge 6 ]; then
    echo "[uq-sched] ALL 6 UQ RUNS COMPLETE $(date)"; break
  fi
  sleep "$CYCLE"
done
