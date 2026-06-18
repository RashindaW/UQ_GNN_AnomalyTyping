#!/usr/bin/env bash
# Capacity-aware scheduler for the 12 DualSTAGE V1/V2 baseline runs.
# Every cycle: launch pending runs (no arrays.npz, no live process) on GPUs
# with >= MIN_FREE MiB free and < MAX_PER our jobs; eval any finished run
# missing eval.json. Exits when all 12 runs have arrays.npz + eval.json.
# Designed to survive other users' load swings on the shared A40 box.
set -u
cd /mnt/datassd3/rashinda/UQ_GNN_AnomalyTyping

PYT=/home/rashinda/.conda/envs/topogdn/bin/python
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SEEDS=(0 1 2 3 4 42); EPOCH=50; BATCH=64
MIN_FREE=9000   # MiB needed to place one run: training holds ~6.5 GB
                # (autograd through the 60-snapshot attention loop) + context
MAX_PER=2       # max of our runs per GPU
CYCLE=180
OUT=results/baseline_v1v2/dualstage
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
mkdir -p "$OUT/logs"

live_run() {  # V S -> 0 if a training process for this combo is alive
  ps -eo args | grep "[d]ualstage.py" | grep -q -- "--variant $1 --seed $2 --"
}

our_live_on_gpu() {  # G -> count of our live runs whose .gpu file says G
  local n=0
  for f in "$OUT"/logs/*.gpu; do
    [ -f "$f" ] || continue
    [ "$(cat "$f")" = "$1" ] || continue
    local base; base="$(basename "$f" .gpu)"
    local V="${base%%_seed*}"; local S="${base##*seed}"
    live_run "$V" "$S" && n=$((n + 1))
  done
  echo "$n"
}

launch() {  # V S G
  local V="$1" S="$2" G="$3"
  local lg="$OUT/logs/${V}_seed${S}.log"
  echo "$G" > "$OUT/logs/${V}_seed${S}.gpu"
  rm -f "$OUT/logs/${V}_seed${S}.done"
  # setsid detaches the run from the scheduler's session: restarting or
  # killing the scheduler must never kill in-flight trainings.
  setsid bash -c "
    echo \"== sched launch $V s$S gpu $G batch=$BATCH \$(date) ==\"
    CUDA_VISIBLE_DEVICES=$G '$PYT' competitors/common/baseline_v1v2_dualstage.py \
      --variant $V --seed $S --epochs $EPOCH --batch $BATCH --device cuda:0
    echo \"== sched run $V s$S rc=\$? \$(date) ==\"
  " >> "$lg" 2>&1 < /dev/null &
}

while true; do
  done_arr=0; done_eval=0; launched=0
  # free memory per GPU, refreshed each cycle
  declare -A FREE=()
  while IFS=, read -r idx free; do
    FREE[$(echo "$idx" | tr -d ' ')]="$(echo "$free" | tr -d ' ')"
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

  for V in V2; do   # V1 stopped by user decision 2026-06-04; revert to "V1 V2" to resume
    for S in "${SEEDS[@]}"; do
      rdir="$OUT/$V/seed$S"
      if [ -f "$rdir/arrays.npz" ]; then
        done_arr=$((done_arr + 1))
        if [ ! -f "$rdir/eval.json" ]; then
          OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py \
            --arrays "$rdir/arrays.npz" --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" \
            --slide_win 60 --seed "$S" --baseline-only --label "DualSTAGE-$V-s$S" \
            --out "$rdir/eval.json" >> "$OUT/logs/${V}_seed${S}.log" 2>&1 \
            && echo "[sched] eval done $V s$S"
        fi
        [ -f "$rdir/eval.json" ] && done_eval=$((done_eval + 1))
        continue
      fi
      live_run "$V" "$S" && continue
      # pending: find a GPU with room
      for G in 0 1 2 3; do
        free="${FREE[$G]:-0}"
        [ "$free" -ge "$MIN_FREE" ] || continue
        [ "$(our_live_on_gpu "$G")" -lt "$MAX_PER" ] || continue
        launch "$V" "$S" "$G"
        FREE[$G]=$((free - 8000))   # reservation: one training job's footprint
        launched=$((launched + 1))
        break
      done
    done
  done
  echo "[sched] $(date +%H:%M) arrays=$done_arr/6 evals=$done_eval/6 launched_now=$launched free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ',')"
  if [ "$done_arr" -ge 6 ] && [ "$done_eval" -ge 6 ]; then
    echo "[sched] ALL 6 V2 RUNS COMPLETE $(date)"; break
  fi
  sleep "$CYCLE"
done
