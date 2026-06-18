#!/usr/bin/env bash
# Plain DualSTAGE (dual-view, paper config) V1/V2 baseline: 12 runs, 12-wide.
# Mirrors baseline_v1v2_topogdn_real.sh. Run inside tmux.
set -u
cd /mnt/datassd3/rashinda/UQ_GNN_AnomalyTyping

PYT=/home/rashinda/.conda/envs/topogdn/bin/python          # train env (dualstgf imports clean)
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python  # eval env
export OMP_NUM_THREADS=4
SEEDS=(0 1 2 3 4 42); GPUS=(0 1 2 3); CONC=12; EPOCH=50; BATCH=128
OUT=results/baseline_v1v2/dualstage; mkdir -p "$OUT/logs"
EVAL_SPLIT=pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json
EVAL_BUNDLE=pretrained/swat_ensemble/calibration_bundle
echo "[dualstage-v1v2] dual-view paper config, W=60, epoch=$EPOCH (early-stop 15), batch=$BATCH, seeds ${SEEDS[*]} x {V1,V2}, ${CONC}-wide $(date)"

run_one() {
  local V="$1" S="$2" G="$3"
  local lg="$OUT/logs/${V}_seed${S}.log"; local sent="$OUT/logs/${V}_seed${S}.done"
  local rdir="$OUT/$V/seed$S"; mkdir -p "$rdir"
  {
    echo "== dualstage $V s$S gpu $G epoch=$EPOCH $(date) =="
    CUDA_VISIBLE_DEVICES="$G" "$PYT" competitors/common/baseline_v1v2_dualstage.py \
        --variant "$V" --seed "$S" --epochs "$EPOCH" --batch "$BATCH" --device cuda:0 \
      || { echo "TRAIN FAIL rc=$?"; echo 11 > "$sent"; return; }
    [ -f "$rdir/arrays.npz" ] || { echo "NO ARRAYS"; echo 12 > "$sent"; return; }
    OMP_NUM_THREADS=4 "$PY" competitors/common/eval_from_arrays.py --arrays "$rdir/arrays.npz" \
      --split "$EVAL_SPLIT" --bundle "$EVAL_BUNDLE" --slide_win 60 --seed "$S" \
      --baseline-only --label "DualSTAGE-$V-s$S" --out "$rdir/eval.json" \
      || { echo "EVAL FAIL rc=$?"; echo 14 > "$sent"; return; }
    echo "== dualstage $V s$S DONE $(date) =="; echo 0 > "$sent"
  } > "$lg" 2>&1
}

i=0
for V in V1 V2; do
  for S in "${SEEDS[@]}"; do
    G="${GPUS[$((i % ${#GPUS[@]}))]}"
    run_one "$V" "$S" "$G" &
    i=$((i + 1))
    if (( i % CONC == 0 )); then wait; fi
  done
done
wait

ok=0
for V in V1 V2; do for S in "${SEEDS[@]}"; do
  [ -f "$OUT/$V/seed$S/arrays.npz" ] && ok=$((ok + 1))
done; done
echo "[dualstage-v1v2] arrays present: $ok/12 $(date)"
for f in "$OUT"/logs/*.done; do echo "$f: $(cat "$f")"; done
