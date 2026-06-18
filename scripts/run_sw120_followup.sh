#!/usr/bin/env bash
# Replicate the sw=60 follow-up experiments at sw=120.
#
#   T1 (CPU): stack_hybrid_on_cheapsweep.py with OR_A/OR_B/AND_A/AND_B on
#             sw=120 arrays.
#   T2 (cuda:0): diverse-anchor calibration + paper-protocol eval on the
#                sw=120 checkpoint.
#
# Both run in parallel. cuda:1-3 stay idle (T1 has no GPU work).

set -euo pipefail
cd /mnt/datassd3/rashinda/CF_Uncertainity_for_STGNN

PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
SPLIT=data/swat/gdeltauq_split.json

SW120_ARRAYS=$(ls -t results/swat_gdeltauq_sw120_paper_protocol/*/arrays.npz | head -1)
SW120_CKPT=$(ls -t pretrained/swat_gdeltauq_sw120/best_*.pt | head -1)
SW120_HP=$(ls -t pretrained/swat_gdeltauq_sw120/hyperparameters_*.json | head -1)

echo "sw=120 arrays:        $SW120_ARRAYS"
echo "sw=120 checkpoint:    $SW120_CKPT"
echo "sw=120 hyperparams:   $SW120_HP"
echo "==========================================================="
echo "=== LAUNCH SW120 FOLLOW-UP  $(date)"
echo "==========================================================="

mkdir -p logs

# T1 (CPU): stack hybrid on sw=120 arrays
{
  echo "=== T1 STACK HYBRID OR+AND on sw=120  $(date)"
  "$PY" scripts/stack_hybrid_on_cheapsweep.py \
    -arrays "$SW120_ARRAYS" \
    -out_root results/uq_attack_assoc/stacked_sw120 \
    -rules OR_A OR_B AND_A AND_B
  echo "=== T1: DONE  $(date)"
} &> logs/t1_stack_sw120.log &
T1_PID=$!
echo "T1 (stack sw=120, CPU) pid=$T1_PID  log=logs/t1_stack_sw120.log"

# T2 (cuda:0): diverse-anchor cal + eval on sw=120
{
  echo "=== T2 DIVERSE ANCHOR (sw=120, cuda:0)  $(date)"
  "$PY" scripts/calibrate_gdeltauq.py \
    -checkpoint "$SW120_CKPT" -hyperparameters "$SW120_HP" \
    -split_path "$SPLIT" \
    -K_anchors 10 -anchor_seed 0 -anchor_strategy diverse \
    -save_dir pretrained/swat_gdeltauq_sw120/calibration_bundle_diverse \
    -device cuda:0
  "$PY" scripts/eval_paper_protocol_gdeltauq.py \
    -checkpoint "$SW120_CKPT" -hyperparameters "$SW120_HP" \
    -bundle_dir pretrained/swat_gdeltauq_sw120/calibration_bundle_diverse \
    -split_path "$SPLIT" \
    -topk 1 -device cuda:0 \
    -results_dir results/swat_gdeltauq_sw120_diverse_paper_protocol
  echo "=== T2: DONE  $(date)"
} &> logs/t2_diverse_sw120.log &
T2_PID=$!
echo "T2 (diverse sw=120, cuda:0) pid=$T2_PID  log=logs/t2_diverse_sw120.log"

echo "waiting on T1, T2 ..."
wait "$T1_PID" || echo "T1 exited non-zero"
wait "$T2_PID" || echo "T2 exited non-zero"

echo "==========================================================="
echo "=== WRITING SW120 FOLLOWUP ROLLUP  $(date)"
echo "==========================================================="

"$PY" - <<'PYEOF'
import json
from glob import glob
import csv

rows = []

def latest(p):
    rs = sorted(glob(p), reverse=True)
    return rs[0] if rs else None

# Diverse anchor on sw=120
p = latest('results/swat_gdeltauq_sw120_diverse_paper_protocol/*/report.json')
if p:
    with open(p) as f:
        r = json.load(f)
    pp = r['paper_protocol']
    rows.append({
        'experiment': 'sw120_diverse_K10',
        'F1': pp['F1'], 'P': pp['precision'], 'R': pp['recall'],
        'AUC': pp['AUC'], 'threshold': pp['threshold'],
        'note': '', 'report_path': p,
    })

# Stack hybrid on sw=120
p = latest('results/uq_attack_assoc/stacked_sw120/*/best_stacked.json')
if p:
    with open(p) as f:
        r = json.load(f)
    bs = r['best_stacked']
    thr = bs.get('tau_r')
    if thr is None:
        thr = bs.get('tau_r_prime')
    rows.append({
        'experiment': 'sw120_stacked_OR_AND',
        'F1': bs['F1'], 'P': bs['P'], 'R': bs['R'],
        'AUC': None, 'threshold': thr,
        'note': f"rule={bs['rule']} signal={bs['signal']} tau_s={bs['tau_s']}",
        'report_path': p,
    })

out = 'results/sw120_followup_rollup.csv'
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

echo "==========================================================="
echo "=== ALL SW120 FOLLOWUP DONE  $(date)"
echo "==========================================================="
