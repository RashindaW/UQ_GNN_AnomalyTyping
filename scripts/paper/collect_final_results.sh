#!/usr/bin/env bash
# Collect everything needed to reproduce the paper (SWaT complete, WADI partial)
# into final_results/. Safe: copies only, never deletes. Idempotent.
set +e
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
FR="$ROOT/final_results"
mkdir -p "$FR/SWaT"/{detection,typing,arrays,models} "$FR/WADI"/{detection,typing,arrays,models}

echo "== SWaT detection =="
cp -f results/thesis_part1/*.csv results/thesis_part1/*.md "$FR/SWaT/detection/" 2>/dev/null

echo "== SWaT typing (csv + gbm caches) =="
cp -f results/typing_v1v2/*.csv "$FR/SWaT/typing/" 2>/dev/null
cp -rf results/typing_v1v2/gbm "$FR/SWaT/typing/gbm" 2>/dev/null

echo "== SWaT arrays_full.npz (4 backbones x 6 seeds) =="
for bb in baseline_v1v2/gdn uq_v1v2/topogdn uq_v1v2/cstgl uq_v1v2/dualstage; do
  name="$(basename "$bb")"
  for s in 0 1 2 3 4 42; do
    src="results/$bb/V2/seed$s/arrays_full.npz"
    [ -f "$src" ] && { mkdir -p "$FR/SWaT/arrays/$name/seed$s"; cp -f "$src" "$FR/SWaT/arrays/$name/seed$s/"; }
  done
done

echo "== SWaT models (checkpoints + calibration bundle) =="
cp -rf pretrained/swat_ensemble/calibration_bundle "$FR/SWaT/models/calibration_bundle" 2>/dev/null
for s in 0 1 2 3 4 42; do cp -rf pretrained/swat_gdeltauq_V2_seed${s}* "$FR/SWaT/models/" 2>/dev/null; done
for bb in topogdn cstgl dualstage; do for s in 0 1 2 3 4 42; do
  src="results/uq_v1v2/$bb/V2/seed$s/best.pt"
  [ -f "$src" ] && { mkdir -p "$FR/SWaT/models/$bb/seed$s"; cp -f "$src" "$FR/SWaT/models/$bb/seed$s/"; }
done; done

echo "== WADI detection (fusion + rankcount csv) =="
cp -f results/baseline_wadi_v2/fusion_wadi_*.csv results/baseline_wadi_v2/m0_rankcount_*.csv "$FR/WADI/detection/" 2>/dev/null

echo "== WADI typing (csv + json + gbm) =="
cp -f results/typing_wadi_v2/*.csv results/typing_wadi_v2/*.json "$FR/WADI/typing/" 2>/dev/null
cp -rf results/typing_wadi_v2/gbm "$FR/WADI/typing/gbm" 2>/dev/null

echo "== WADI arrays_full.npz (done: gdn, cstgl) =="
for s in 0 1 2 3 4 42; do
  for pair in "baseline_wadi_v2/gdn:gdn" "uq_wadi_v2/cstgl:cstgl"; do
    src="results/${pair%%:*}/V2/seed$s/arrays_full.npz"; name="${pair##*:}"
    [ -f "$src" ] && { mkdir -p "$FR/WADI/arrays/$name/seed$s"; cp -f "$src" "$FR/WADI/arrays/$name/seed$s/"; }
  done
done

echo "== WADI models (checkpoints + calibration bundle) =="
cp -rf pretrained/wadi_ensemble/calibration_bundle "$FR/WADI/models/calibration_bundle" 2>/dev/null
for s in 0 1 2 3 4 42; do cp -rf pretrained/wadi_gdeltauq_V2_seed${s} pretrained/gdn_plain_wadi_V2_seed${s} "$FR/WADI/models/" 2>/dev/null; done
for s in 0 1 2 3 4 42; do
  src="results/uq_wadi_v2/cstgl/V2/seed$s/best.pt"
  [ -f "$src" ] && { mkdir -p "$FR/WADI/models/cstgl/seed$s"; cp -f "$src" "$FR/WADI/models/cstgl/seed$s/"; }
done

echo "== sizes =="
du -sh "$FR"/SWaT "$FR"/WADI 2>/dev/null
echo "DONE"
exit 0
