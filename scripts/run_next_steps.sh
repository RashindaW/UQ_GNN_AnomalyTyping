#!/usr/bin/env bash
# Orchestrator for the four next-step experiments on top of the slide_win
# sweep results. Distributes across 4 CUDAs + CPU:
#
#   T1 (CPU, parallel): stack hybrid OR + AND-gate sweep on sw=60 arrays.
#   T2 (cuda:1, cuda:2, cuda:3, parallel): sw=15 retrain at seeds 1, 2, 3.
#   T3 (cuda:0, parallel with T2): sw=120 retrain.
#   T4 (cuda:0, after T3 finishes): diverse-anchor calibration + eval on
#       sw=60 checkpoint (Q2-2).
#
# Final step: write results/next_steps_rollup.csv with one row per
# experiment outcome.

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
SPLIT=data/swat/gdeltauq_split.json
SW60_ARRAYS=results/swat_gdeltauq_sw60_paper_protocol/0513-211654/arrays.npz

mkdir -p logs

# Helper: train + cal + eval at a given (slide_win, seed, save_pattern, dev).
train_cal_eval() {
  local SW=$1
  local SEED=$2
  local TAG=$3
  local DEV=$4
  local LOG=$5

  {
    echo "=== ${TAG}: TRAIN sw=${SW} seed=${SEED} dev=${DEV}  $(date)"
    "$PY" train_gdeltauq_main.py \
      -dataset swat -slide_win "$SW" -slide_stride 1 \
      -epoch 100 -batch 128 -dim 64 -out_layer_num 1 \
      -out_layer_inter_dim 128 -topk 15 -n_gnn_layers 2 \
      -K_anchors 10 -decay 0.0 -random_seed "$SEED" \
      -split_path "$SPLIT" -save_path_pattern "$TAG" \
      -device "$DEV" -comment "next-steps ${TAG}"

    CKPT=$(ls -t "pretrained/${TAG}"/best_*.pt 2>/dev/null | head -1)
    HP=$(ls -t "pretrained/${TAG}"/hyperparameters_*.json 2>/dev/null | head -1)
    if [[ -z "$CKPT" || -z "$HP" ]]; then
      echo "ERROR: missing artifacts in pretrained/${TAG}" >&2
      exit 1
    fi

    echo "=== ${TAG}: CALIBRATE  $(date)"
    "$PY" scripts/calibrate_gdeltauq.py \
      -checkpoint "$CKPT" -hyperparameters "$HP" \
      -split_path "$SPLIT" \
      -K_anchors 10 -anchor_seed 0 -anchor_strategy random \
      -save_dir "pretrained/${TAG}/calibration_bundle" \
      -device "$DEV"

    echo "=== ${TAG}: EVAL  $(date)"
    "$PY" scripts/eval_paper_protocol_gdeltauq.py \
      -checkpoint "$CKPT" -hyperparameters "$HP" \
      -bundle_dir "pretrained/${TAG}/calibration_bundle" \
      -split_path "$SPLIT" \
      -topk 1 -device "$DEV" \
      -results_dir "results/${TAG}_paper_protocol"

    echo "=== ${TAG}: DONE  $(date)"
  } &> "$LOG"
}

echo "=========================================================="
echo "=== LAUNCHING NEXT-STEPS CHAIN  $(date)"
echo "=========================================================="

# T1 (CPU): stack hybrid on sw=60 arrays.
{
  echo "=== T1 STACK HYBRID OR+AND on sw=60  $(date)"
  "$PY" scripts/stack_hybrid_on_cheapsweep.py \
    -arrays "$SW60_ARRAYS" \
    -out_root results/uq_attack_assoc/stacked_sw60 \
    -rules OR_A OR_B AND_A AND_B
  echo "=== T1: DONE  $(date)"
} &> logs/t1_stack_sw60.log &
T1_PID=$!
echo "T1 (stack sw=60) pid=$T1_PID  log=logs/t1_stack_sw60.log"

# T2: sw=15 seeds 1, 2, 3 in parallel on cuda:1/2/3.
train_cal_eval 15 1 swat_gdeltauq_sw15_seed1 cuda:1 logs/t2a_sw15_s1.log &
T2A_PID=$!
train_cal_eval 15 2 swat_gdeltauq_sw15_seed2 cuda:2 logs/t2b_sw15_s2.log &
T2B_PID=$!
train_cal_eval 15 3 swat_gdeltauq_sw15_seed3 cuda:3 logs/t2c_sw15_s3.log &
T2C_PID=$!
echo "T2A (sw=15 seed=1) pid=$T2A_PID  log=logs/t2a_sw15_s1.log"
echo "T2B (sw=15 seed=2) pid=$T2B_PID  log=logs/t2b_sw15_s2.log"
echo "T2C (sw=15 seed=3) pid=$T2C_PID  log=logs/t2c_sw15_s3.log"

# T3: sw=120 retrain on cuda:0.
train_cal_eval 120 42 swat_gdeltauq_sw120 cuda:0 logs/t3_sw120.log &
T3_PID=$!
echo "T3  (sw=120 seed=42) pid=$T3_PID  log=logs/t3_sw120.log"

echo "waiting on T3 (sw=120) so cuda:0 is free for T4 ..."
wait "$T3_PID" && echo "T3 done." || echo "T3 exited non-zero (continuing)."

# T4 (cuda:0, after T3): diverse-anchor cal + eval on sw=60 checkpoint.
{
  echo "=== T4 DIVERSE ANCHOR (sw=60)  $(date)"
  CKPT=$(ls -t pretrained/swat_gdeltauq_sw60/best_*.pt | head -1)
  HP=$(ls -t pretrained/swat_gdeltauq_sw60/hyperparameters_*.json | head -1)
  "$PY" scripts/calibrate_gdeltauq.py \
    -checkpoint "$CKPT" -hyperparameters "$HP" \
    -split_path "$SPLIT" \
    -K_anchors 10 -anchor_seed 0 -anchor_strategy diverse \
    -save_dir pretrained/swat_gdeltauq_sw60/calibration_bundle_diverse \
    -device cuda:0
  "$PY" scripts/eval_paper_protocol_gdeltauq.py \
    -checkpoint "$CKPT" -hyperparameters "$HP" \
    -bundle_dir pretrained/swat_gdeltauq_sw60/calibration_bundle_diverse \
    -split_path "$SPLIT" \
    -topk 1 -device cuda:0 \
    -results_dir results/swat_gdeltauq_sw60_diverse_paper_protocol
  echo "=== T4: DONE  $(date)"
} &> logs/t4_diverse_sw60.log

echo "waiting on T1, T2A, T2B, T2C ..."
wait "$T1_PID"  || echo "T1 exited non-zero"
wait "$T2A_PID" || echo "T2A exited non-zero"
wait "$T2B_PID" || echo "T2B exited non-zero"
wait "$T2C_PID" || echo "T2C exited non-zero"

echo "=========================================================="
echo "=== WRITING NEXT-STEPS ROLLUP  $(date)"
echo "=========================================================="

"$PY" - <<'PYEOF'
import json
from glob import glob
import csv

rows = []

def latest_report(pattern):
    rs = sorted(glob(pattern), reverse=True)
    return rs[0] if rs else None

def add_paper_row(name, pattern, **extra):
    p = latest_report(pattern)
    if not p:
        rows.append({'experiment': name, 'F1': None, 'P': None, 'R': None,
                     'AUC': None, 'threshold': None, 'note': 'NO REPORT',
                     'report_path': None, **extra})
        return
    with open(p) as f:
        r = json.load(f)
    pp = r['paper_protocol']
    rows.append({
        'experiment': name,
        'F1': pp['F1'], 'P': pp['precision'], 'R': pp['recall'],
        'AUC': pp['AUC'], 'threshold': pp['threshold'],
        'note': '', 'report_path': p, **extra,
    })

for seed in (1, 2, 3):
    add_paper_row(
        f'sw15_seed{seed}',
        f'results/swat_gdeltauq_sw15_seed{seed}_paper_protocol/*/report.json',
    )
add_paper_row(
    'sw120_seed42',
    'results/swat_gdeltauq_sw120_paper_protocol/*/report.json',
)
add_paper_row(
    'sw60_diverse_K10',
    'results/swat_gdeltauq_sw60_diverse_paper_protocol/*/report.json',
)

# Stack hybrid sw=60: read best_stacked.json
js = sorted(glob('results/uq_attack_assoc/stacked_sw60/*/best_stacked.json'),
            reverse=True)
if js:
    with open(js[0]) as f:
        r = json.load(f)
    bs = r['best_stacked']
    thr = bs.get('tau_r')
    if thr is None:
        thr = bs.get('tau_r_prime')
    rows.append({
        'experiment': 'sw60_stacked_OR_AND',
        'F1': bs['F1'], 'P': bs['P'], 'R': bs['R'],
        'AUC': None, 'threshold': thr,
        'note': f"rule={bs['rule']} signal={bs['signal']} tau_s={bs['tau_s']}",
        'report_path': js[0],
    })

out = 'results/next_steps_rollup.csv'
fields = ['experiment', 'F1', 'P', 'R', 'AUC', 'threshold', 'note',
          'report_path']
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, '') for k in fields})
print(f'rollup -> {out}')
for row in rows:
    print(row)
PYEOF

echo "=========================================================="
echo "=== ALL NEXT-STEPS DONE  $(date)"
echo "=========================================================="
