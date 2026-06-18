#!/usr/bin/env bash
# P1: Part-2 typing/triage expansion to all 24 combos (4 backbones x 6 seeds
# x V2), per docs/PART2_PREREGISTRATION.md. Phases:
#   A) build the 14 missing GBM score caches (parallel waves, OMP-capped)
#   B) typing engine, all events, all 24 combos
#   C) alarm triage, both sources, all 24 combos
#   D) C7 DualSTGF first-typing screen + integrity counts -> P1_RUN_REPORT
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

SEEDS="0 1 2 3 4 42"
ALL24=$(for bb in gdn topogdn cstgl dualstage; do for s in $SEEDS; do printf "%s:V2:%s," "$bb" "$s"; done; done | sed 's/,$//')
MISSING="gdn:V2:0 gdn:V2:1 gdn:V2:2 gdn:V2:3 gdn:V2:4 topogdn:V2:0 topogdn:V2:1 topogdn:V2:2 topogdn:V2:3 topogdn:V2:4 cstgl:V2:0 cstgl:V2:1 cstgl:V2:2 cstgl:V2:4"

echo "[P1-A] building 14 missing GBM caches, 7-wide waves $(date)"
rc=0; n=0; pids=()
for c in $MISSING; do
  $PY scripts/paper/explain_gbm_v1v2.py --combos "$c" \
     > "results/typing_v1v2/gbm/build_${c//:/_}.log" 2>&1 &
  pids+=("$!"); n=$((n+1))
  if [ "$n" -ge 7 ]; then for p in "${pids[@]}"; do wait "$p" || rc=1; done; pids=(); n=0; fi
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[P1-A] caches done rc=$rc $(date)"
NC=$(ls results/typing_v1v2/gbm/*_cache.npz 2>/dev/null | wc -l)
echo "[P1-A] cache count: $NC (expect 24)"
[ "$NC" -ge 24 ] || { echo "CACHE BUILD INCOMPLETE"; exit 1; }

echo "[P1-B] typing, all events, 24 combos $(date)"
UQ_COMBOS="$ALL24" $PY scripts/paper/typing_rules_v1v2.py --all-events \
  > results/thesis_part2/p1_typing.log 2>&1 || { echo "TYPING FAIL"; exit 1; }

echo "[P1-C] triage, 24 combos $(date)"
UQ_COMBOS="$ALL24" $PY scripts/paper/alarm_triage_v1v2.py \
  > results/thesis_part2/p1_triage.log 2>&1 || { echo "TRIAGE FAIL"; exit 1; }

echo "[P1-D] C7 screen + integrity $(date)"
$PY - <<'EOF' | tee results/thesis_part2/P1_RUN_REPORT.txt
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "scripts/paper")
t = pd.read_csv("results/typing_v1v2/typing_events.csv")
e = pd.read_csv("results/typing_v1v2/alarm_triage_episodes.csv")
print("== P1 integrity ==")
print("typing rows:", len(t), "| combos:", t.groupby(["backbone","seed"]).ngroups,
      "| heldout event rows:", int(t.in_heldout.sum()))
print("episode rows:", len(e), "| combo-sources:", e.groupby(["backbone","seed","source"]).ngroups,
      "| heldout episodes:", int(e.in_heldout.sum()))
assert t.groupby(["backbone","seed"]).ngroups == 24, "expected 24 typing combos"
assert e.groupby(["backbone","seed","source"]).ngroups == 48, "expected 48 combo-sources"

print("\n== C7 DualSTGF first-typing screen (frozen predicate, prereg S5) ==")
os.environ["UQ_COMBOS"] = ""  # not needed; direct load below
from typing_rules_v1v2 import load_combo, c_slice_thresholds, load_attack_table
from analyze_multistage_attacks import estimate_offset
atts = load_attack_table()
fails = []
ds = t[t.backbone=="dualstage"]
r2_share = float((ds[ds.detected==1].verdict=="R2_noisy_sensor").mean())
print(f"(a) R2 share of DETECTED dualstage events (pooled 6 seeds): {r2_share:.3f}  (fail if > 0.10)")
if r2_share > 0.10: fails.append("a")
for seed in [0,1,2,3,4,42]:
    ctx = load_combo("dualstage","V2",seed)
    doff = estimate_offset(ctx["lab"], atts)
    thr = c_slice_thresholds(ctx, doff)
    ah = (ctx["A"] > thr["A"]).astype(float)
    r = float(np.corrcoef(ah, (ctx["lab"]==1).astype(float))[0,1])
    print(f"(b) dualstage s{seed}: point-biserial corr(A-High, label) = {r:.3f}  (fail if > 0.30)")
    if r > 0.30: fails.append(f"b:s{seed}")
print("C7 VERDICT:", "FAIL " + ",".join(fails) if fails else "PASS")

print("\n== per-backbone quick shape (events detected / R2 / quiet, pooled seeds) ==")
for bb,g in t.groupby("backbone"):
    print(f"{bb:10s} events={len(g):4d} detected={int(g.detected.sum()):4d} "
          f"R2={int((g.verdict=='R2_noisy_sensor').sum()):3d} "
          f"missedquiet={int((g.verdict=='missed_quiet').sum()):3d}")
EOF
echo "[P1] done rc=$? $(date)"
