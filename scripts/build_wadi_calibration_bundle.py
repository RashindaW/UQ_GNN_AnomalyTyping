#!/usr/bin/env python3
"""Build the WADI V2 split + calibration-bundle indices.

Outputs:
  data/wadi/split_V2_baseline.json
      {train_rows:[0,0.85N], val_rows:[0.85N,N]} on train.csv rows,
      same schema as data/swat/split_V2_baseline.json.
  pretrained/wadi_ensemble/calibration_bundle/calibration_set_indices.json
      C_row_range / labeled_val_range / final_test_range in TEST-ROW
      coordinates (setup_context subtracts slide_win itself), same key
      set as the SWaT bundle, plus documented convenience keys:
      wadi_c_end_windowed  = C_end  - 60   (typing --c-end)
      wadi_held0_windowed  = val_end - 60  (fusion --held0)

Boundary policy: target fractions C=35% / val end=55% (SWaT analogue),
each boundary snapped to the centre of a nominal gap (no attack row
within +-slide_win), never bisecting an attack; the labeled val slice
must contain >= 3 whole attack runs (search widens the val window if
starved). All snaps and per-slice attack counts are printed and stored.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WADI = os.path.join(ROOT, "data/wadi")
BUNDLE = os.path.join(ROOT, "pretrained/wadi_ensemble/calibration_bundle")
SLIDE_WIN = 60


def label_runs(label: np.ndarray):
    runs, i, n = [], 0, len(label)
    while i < n:
        if label[i]:
            j = i
            while j < n and label[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def snap_to_gap(label: np.ndarray, target: int, w: int = SLIDE_WIN) -> int:
    """Nearest b to target with no attack row in [b-w, b+w)."""
    n = len(label)
    for d in range(0, n):
        for b in (target - d, target + d):
            if w <= b <= n - w and label[b - w:b + w].sum() == 0:
                return int(b)
    raise RuntimeError("no nominal gap found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-frac", type=float, default=0.35)
    ap.add_argument("--val-frac", type=float, default=0.55,
                    help="val END as a fraction of test rows")
    ap.add_argument("--min-val-attacks", type=int, default=3)
    a = ap.parse_args()

    # ---- train split ----
    n_train = len(pd.read_csv(os.path.join(WADI, "train.csv"), index_col=0))
    cut = int(round(0.85 * n_train))
    split = dict(
        dataset="wadi", total_rows=n_train,
        train_rows=[0, cut], val_rows=[cut, n_train],
        aleatoric_rows=[cut, n_train], variant="V2_baseline",
        note="BASELINE V2: train first 85%, val last 15% [85,100); "
             "no uncertainty. aleatoric_rows=val (ignored for M0).",
    )
    with open(os.path.join(WADI, "split_V2_baseline.json"), "w") as f:
        json.dump(split, f, indent=2)
    print(f"[wadi-cal] split_V2_baseline.json: train [0,{cut}) val [{cut},{n_train})", flush=True)

    # ---- test calibration indices ----
    test = pd.read_csv(os.path.join(WADI, "test.csv"), index_col=0)
    label = test["attack"].to_numpy().astype(np.int8)
    n = len(label)
    runs = label_runs(label)
    print(f"[wadi-cal] test rows {n}, attack rows {int(label.sum())}, runs {len(runs)}", flush=True)

    candidates = [(a.c_frac, a.val_frac)]
    candidates += [(c, v) for c in (0.35, 0.30, 0.25) for v in (0.55, 0.60, 0.65)
                   if (c, v) != (a.c_frac, a.val_frac)]
    chosen = None
    for cf, vf in candidates:
        c_end = snap_to_gap(label, int(round(cf * n)))
        v_end = snap_to_gap(label, int(round(vf * n)))
        in_val = [(s, e) for (s, e) in runs if s >= c_end and e <= v_end]
        if len(in_val) >= a.min_val_attacks:
            chosen = (c_end, v_end, in_val, cf, vf)
            break
        print(f"[wadi-cal] fractions ({cf},{vf}) -> only {len(in_val)} val runs, retrying", flush=True)
    if chosen is None:
        raise RuntimeError("no boundary placement satisfies the val-attack minimum")
    c_end, v_end, in_val, cf, vf = chosen
    in_c = [(s, e) for (s, e) in runs if e <= c_end]
    in_held = [(s, e) for (s, e) in runs if s >= v_end]
    crossing = len(runs) - len(in_c) - len(in_val) - len(in_held)
    assert crossing == 0, f"{crossing} attack runs bisected by a boundary"

    c_zero = np.flatnonzero(label[:c_end] == 0)
    n_nominal = int((label == 0).sum())
    out = dict(
        test_csv=os.path.join(ROOT, "data/wadi/test.csv"),
        n_rows=int(n), n_attack=int(label.sum()),
        C_row_range=[0, int(c_end)],
        C_attack_zero_indices=[int(i) for i in c_zero],
        labeled_val_range=[int(c_end), int(v_end)],
        final_test_range=[int(v_end), int(n)],
        c_target_rows=int(len(c_zero)),
        c_target_fraction=float(len(c_zero) / n_nominal),
        n_nominal_in_test=n_nominal,
        wadi_c_end_windowed=int(c_end - SLIDE_WIN),
        wadi_held0_windowed=int(v_end - SLIDE_WIN),
        boundary_policy=f"targets C={cf} valEnd={vf}, snapped to nominal gaps >= {SLIDE_WIN}",
        attacks_per_slice=dict(C=len(in_c), val=len(in_val), heldout=len(in_held)),
    )
    os.makedirs(BUNDLE, exist_ok=True)
    with open(os.path.join(BUNDLE, "calibration_set_indices.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wadi-cal] C [0,{c_end}) val [{c_end},{v_end}) held [{v_end},{n})", flush=True)
    print(f"[wadi-cal] attack runs per slice: C={len(in_c)} val={len(in_val)} held={len(in_held)}", flush=True)
    print(f"[wadi-cal] WADI_CEND_WINDOWED={c_end - SLIDE_WIN}  WADI_HELD0_WINDOWED={v_end - SLIDE_WIN}", flush=True)
    print(f"[wadi-cal] wrote {os.path.join(BUNDLE, 'calibration_set_indices.json')}", flush=True)


if __name__ == "__main__":
    main()
