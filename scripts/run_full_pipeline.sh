#!/usr/bin/env bash
# End-to-end GDN_UQ pipeline: train → calibrate → detect → plots → dashboard.
#
# Default M=10 ensemble (extends an existing M=5 run by training 5 more members).
# Skips already-trained members based on checkpoint + hyperparameters.json
# presence, so re-running is cheap.
#
# Override defaults via env vars:
#   PYTHON=/path/to/python                Python interpreter (with torch + pyg + plotly)
#   DEVICE=cuda:0                         GPU for calibrate / detect
#   SEEDS="5 17 42 100 314 7 23 88 256 999"   ensemble seed list (M = #seeds)
#   SKIP_TRAIN=0                          1 to skip the training phase
#   SKIP_PLOTS=0                          1 to skip per-node + per-attack plots
#   BACKUP_BUNDLE=1                       1 to back up calibration_bundle before recalibrate
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-5 17 42 100 314 7 23 88 256 999}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
BACKUP_BUNDLE="${BACKUP_BUNDLE:-1}"

TS=$(date +%m%d-%H%M%S)
RUN_DIR="results/dashboards/run_$TS"
mkdir -p "$RUN_DIR"
RUN_LOG="$RUN_DIR/pipeline.log"

# Tee everything to both stdout and the run log.
exec > >(tee -a "$RUN_LOG") 2>&1

n_seeds=$(echo "$SEEDS" | wc -w)
banner() { echo; printf '#%.0s' {1..78}; echo; echo "# $*"; printf '#%.0s' {1..78}; echo; }

banner "GDN_UQ end-to-end pipeline  (M=$n_seeds, run_dir=$RUN_DIR)"
echo "  Python:      $PYTHON"
echo "  Device:      $DEVICE"
echo "  Seeds:       $SEEDS"
echo "  Skip train:  $SKIP_TRAIN"
echo "  Skip plots:  $SKIP_PLOTS"
echo "  Started:     $(date -Iseconds)"

# Sanity: data must be prepped.
if [[ ! -f data/swat/train.csv ]] || [[ ! -f data/swat/test.csv ]]; then
  echo "ERROR: data/swat/{train,test}.csv missing. Run scripts/prepare_swat.py first." >&2
  exit 1
fi

# Backup the existing calibration bundle (if any) so we can compare M=N vs M=M.
if [[ "$BACKUP_BUNDLE" == "1" ]] && [[ -d pretrained/swat_ensemble/calibration_bundle ]]; then
  bundle_backup="pretrained/swat_ensemble/calibration_bundle.before_$TS"
  if [[ ! -e "$bundle_backup" ]]; then
    cp -a pretrained/swat_ensemble/calibration_bundle "$bundle_backup"
    echo "[pipeline] backed up calibration_bundle → $bundle_backup"
  fi
fi

# ---------------------------------------------------------------------------
banner "STEP 1 — train ensemble (skip-existing)"
# ---------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "[pipeline] SKIP_TRAIN=1 → skipping training"
else
  PYTHON="$PYTHON" SEEDS="$SEEDS" SKIP_EXISTING=1 bash scripts/train_ensemble.sh
fi

# ---------------------------------------------------------------------------
banner "STEP 2 — rebuild calibration_set_indices for current test.csv"
# ---------------------------------------------------------------------------
"$PYTHON" scripts/build_test_split.py

# ---------------------------------------------------------------------------
banner "STEP 3 — calibrate"
# ---------------------------------------------------------------------------
PYTHON="$PYTHON" DEVICE="$DEVICE" bash scripts/calibrate.sh

# ---------------------------------------------------------------------------
banner "STEP 4 — detect (writes results/swat_ensemble/<datestr>/)"
# ---------------------------------------------------------------------------
PYTHON="$PYTHON" DEVICE="$DEVICE" bash scripts/detect.sh

if [[ "$SKIP_PLOTS" == "1" ]]; then
  echo "[pipeline] SKIP_PLOTS=1 → skipping plots"
else
  banner "STEP 5 — per-node HTML plots (51 sensors, on CPU)"
  "$PYTHON" scripts/plot_gdn_test.py --device cpu

  banner "STEP 6 — per-attack HTML plots + 4-aggregator alignment scoring (CPU)"
  "$PYTHON" scripts/plot_gdn_attacks.py --device cpu
fi

# ---------------------------------------------------------------------------
banner "STEP 7 — render single-page dashboard"
# ---------------------------------------------------------------------------
"$PYTHON" scripts/build_dashboard.py --out-dir "$RUN_DIR" --m "$n_seeds" --seeds "$SEEDS"

# ---------------------------------------------------------------------------
banner "DONE"
# ---------------------------------------------------------------------------
echo "[pipeline] dashboard:  $RUN_DIR/dashboard.html"
echo "[pipeline] run log:    $RUN_LOG"
echo "[pipeline] finished:   $(date -Iseconds)"
