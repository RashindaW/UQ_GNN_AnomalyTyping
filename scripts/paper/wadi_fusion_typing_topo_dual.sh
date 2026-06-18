#!/usr/bin/env bash
# WADI CPU analysis for TopoGDN + DualSTGF (the two whose training just finished):
#   1. fusion ladder per backbone, both regions -> fusion_wadi_{bb}_{seedwise,heldout}.csv
#   2. typing rule re-run over ALL FOUR backbones x 6 seeds (summary is overwritten,
#      so it must cover everything) -> results/typing_wadi_v2/{typing_events.csv,
#      typing_summary.json, traces/}.
# CPU only, minutes. Idempotent (overwrites its own outputs).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export UQ_DATASET=wadi
OUT=results/baseline_wadi_v2; mkdir -p "$OUT/fusion_logs"

echo "[wadi-fus] start $(date)"
# fusion_v1v2 REWRITES arrays_full.npz in place (Omega promotion), so the two
# regions of one backbone MUST run sequentially (same files). Different backbones
# touch different files, so they are safe to run in parallel.
run_bb() {
  local bb="$1"
  for reg in heldout full; do
    local suff; suff=$([ "$reg" = "full" ] && echo seedwise || echo heldout)
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" scripts/paper/fusion_v1v2.py --dataset wadi \
      --backbones "$bb" --variants V2 --seeds 0,1,2,3,4,42 --region "$reg" \
      --out "$OUT/fusion_wadi_${bb}_${suff}.csv" \
      > "$OUT/fusion_logs/wadi_${bb}_${reg}.log" 2>&1 || echo "[wadi-fus] $bb $reg FAILED"
  done
}
run_bb topogdn & run_bb dualstage & wait
echo "[wadi-fus] fusion done $(date)"
for bb in topogdn dualstage; do for reg in heldout seedwise; do
  f="$OUT/fusion_wadi_${bb}_${reg}.csv"
  echo "  $f: $([ -f "$f" ] && echo "$(($(wc -l < "$f")-1)) rows" || echo MISSING)"
done; done

echo "[wadi-typ] start $(date)"
COMBOS=$(for bb in gdn topogdn cstgl dualstage; do for s in 0 1 2 3 4 42; do printf '%s:V2:%s,' "$bb" "$s"; done; done | sed 's/,$//')
UQ_DATASET=wadi UQ_COMBOS="$COMBOS" OMP_NUM_THREADS=4 "$PY" scripts/paper/typing_rules_v1v2.py \
  > results/typing_wadi_v2/typing_run.log 2>&1 && echo "[wadi-typ] typing OK $(date)" \
  || { echo "[wadi-typ] typing FAILED $(date)"; tail -20 results/typing_wadi_v2/typing_run.log; }
echo "typing backbones in summary:"; grep -oE '"(gdn|topogdn|cstgl|dualstage)_V2_s[0-9]+"' results/typing_wadi_v2/typing_summary.json 2>/dev/null | sed 's/_V2.*//' | tr -d '"' | sort | uniq -c
echo "[wadi] ALL DONE $(date)"
