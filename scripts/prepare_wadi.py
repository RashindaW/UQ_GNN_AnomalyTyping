#!/usr/bin/env python3
"""Prepare WADI.A1 (Oct 2017) for the V2 uncertainty campaign.

Mirrors the data/swat contract produced by prepare_swat.py:
  data/wadi/train.csv   index + sensor columns (no attack column)
  data/wadi/test.csv    index + sensor columns + trailing integer `attack`
  data/wadi/list.txt    one sensor name per line (model node order)
  data/wadi/prep_meta.json  anchors, scaler, drops, NaN stats, row counts
  data/wadi/scaler_minmax.csv  per-column train-fit min/max

WADI-specific handling (all decisions documented in prep_meta.json):
  - the two raw CSVs have DIFFERENT header offsets (14days has metadata
    lines, attackdata starts at the header): header row auto-detected by
    scanning for the line that starts with "Row,Date,Time";
  - sensor names carry a Windows-path prefix: stripped to the part after
    the last backslash;
  - columns that are all-NaN in EITHER file are dropped from BOTH
    (intersection rule); "Unnamed" columns dropped;
  - remaining NaNs: ffill -> bfill -> train-column-mean;
  - values are MIN-MAX NORMALIZED with train-fit parameters (the GDN
    paper's own WADI convention; WADI carries totalizer channels whose
    raw magnitudes are unsuitable for unnormalized model input);
  - test labels are built from the corrected attack_description.xlsx
    windows (15 attacks; xlsx typos fixed and cross-checked against
    competitors/CST-GL/generate_data/generate_wadi_data.ipynb and
    competitors/TopoGDN/scripts/wadi_mark_label.py) by per-second
    datetime membership BEFORE downsampling;
  - 10x downsample: median for sensors, MAX for the label (any attack
    second in the block marks the block);
  - stabilization trim (default 2160 rows post-downsample = 6 h) is
    applied to TRAIN ONLY (reference convention; test is untrimmed so
    attack 1 and the time anchor stay intact).

Runs in rashindaNew-torch-env, CPU, a few minutes (the 778 MB read
dominates).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/WADI/raw")
OUT = os.path.join(ROOT, "data/wadi")

# Corrected transcription of attack_description.xlsx (primary source).
# Fixes applied to the xlsx: duplicate S.No "2" for the 4th attack, year
# "1947" on attack 5, "11.30:40" -> 11:30:40, month "07" -> 10 on the
# Oct-11 rows. Attack 7 keeps its two overlapping sub-windows (1_AIT_002
# and 2_MV_003). Cross-checks: CST-GL notebook (merges A4 into A3's
# window, A9 date typo) and TopoGDN wadi_mark_label.py (A9 end-date typo).
ATTACK_WINDOWS = [
    (1,  "2017-10-09 19:25:00", "2017-10-09 19:50:16"),
    (2,  "2017-10-10 10:24:10", "2017-10-10 10:34:00"),
    (3,  "2017-10-10 10:55:00", "2017-10-10 11:24:00"),
    (4,  "2017-10-10 11:07:46", "2017-10-10 11:12:15"),   # nested in A3
    (5,  "2017-10-10 11:30:40", "2017-10-10 11:44:50"),
    (6,  "2017-10-10 13:39:30", "2017-10-10 13:50:40"),
    (7,  "2017-10-10 14:48:17", "2017-10-10 14:59:55"),   # 1_AIT_002 leg
    (7,  "2017-10-10 14:53:44", "2017-10-10 15:00:32"),   # 2_MV_003 leg
    (8,  "2017-10-10 17:40:00", "2017-10-10 17:49:40"),
    (9,  "2017-10-11 10:55:00", "2017-10-11 10:56:27"),
    (10, "2017-10-11 11:17:54", "2017-10-11 11:31:20"),
    (11, "2017-10-11 11:36:31", "2017-10-11 11:47:00"),
    (12, "2017-10-11 11:59:00", "2017-10-11 12:05:00"),
    (13, "2017-10-11 12:07:30", "2017-10-11 12:10:52"),
    (14, "2017-10-11 12:16:00", "2017-10-11 12:25:36"),
    (15, "2017-10-11 15:26:30", "2017-10-11 15:37:00"),
]


def find_header_row(path: str, max_scan: int = 10) -> int:
    with open(path, "r", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if line.startswith("Row,Date,Time"):
                return i
    raise RuntimeError(f"no 'Row,Date,Time' header in first {max_scan} lines of {path}")


def strip_prefix(col: str) -> str:
    return col.rsplit("\\", 1)[-1].strip() if "\\" in col else col.strip()


def parse_datetimes(df: pd.DataFrame, tag: str) -> pd.Series:
    combined = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
    for fmt in ("%m/%d/%Y %I:%M:%S.%f %p", "%m/%d/%Y %I:%M:%S %p",
                "%d/%m/%Y %I:%M:%S.%f %p", "%m/%d/%Y %H:%M:%S"):
        try:
            dt = pd.to_datetime(combined, format=fmt)
            print(f"[prep] {tag}: datetime format {fmt}", flush=True)
            return dt
        except (ValueError, TypeError):
            continue
    print(f"[prep] {tag}: falling back to generic datetime parse (slow)", flush=True)
    return pd.to_datetime(combined, errors="coerce")


def load_raw(path: str, tag: str):
    hdr = find_header_row(path)
    print(f"[prep] {tag}: header at line {hdr}", flush=True)
    df = pd.read_csv(path, header=hdr, low_memory=False, skip_blank_lines=False)
    df.columns = [c.strip() for c in df.columns]
    drop_unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if drop_unnamed:
        print(f"[prep] {tag}: dropping {len(drop_unnamed)} Unnamed columns", flush=True)
        df = df.drop(columns=drop_unnamed)
    dt = parse_datetimes(df, tag)
    bad = dt.isna()
    if bad.any():
        print(f"[prep] {tag}: dropping {int(bad.sum())} rows with unparseable datetime", flush=True)
        df, dt = df[~bad].reset_index(drop=True), dt[~bad].reset_index(drop=True)
    df = df.drop(columns=[c for c in ("Row", "Date", "Time") if c in df.columns])
    df.columns = [strip_prefix(c) for c in df.columns]
    # cadence report
    gaps = dt.diff().dt.total_seconds().dropna()
    n_irreg = int((gaps != 1.0).sum())
    print(f"[prep] {tag}: {len(df)} rows, {df.shape[1]} sensor cols, "
          f"span {dt.iloc[0]} .. {dt.iloc[-1]}, irregular steps: {n_irreg}", flush=True)
    return df, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downsample", type=int, default=10)
    ap.add_argument("--stabilization-trim", type=int, default=2160,
                    help="rows dropped from the START of TRAIN, post-downsample")
    ap.add_argument("--train-csv", default=os.path.join(RAW, "WADI_14days.csv"))
    ap.add_argument("--test-csv", default=os.path.join(RAW, "WADI_attackdata.csv"))
    a = ap.parse_args()

    train, dt_tr = load_raw(a.train_csv, "train(14days)")
    test, dt_te = load_raw(a.test_csv, "test(attackdata)")

    # ---- dead-column intersection rule ----
    dead_tr = set(train.columns[train.isna().all()])
    dead_te = set(test.columns[test.isna().all()])
    dead = sorted(dead_tr | dead_te)
    if dead:
        print(f"[prep] dropping {len(dead)} all-NaN columns (union of both files): {dead}", flush=True)
    common = [c for c in train.columns if c in set(test.columns) and c not in dead]
    only_tr = [c for c in train.columns if c not in set(test.columns)]
    only_te = [c for c in test.columns if c not in set(train.columns)]
    if only_tr or only_te:
        print(f"[prep] non-shared columns dropped: train-only {only_tr}, test-only {only_te}", flush=True)
    train, test = train[common], test[common]
    assert list(train.columns) == list(test.columns)

    train = train.apply(pd.to_numeric, errors="coerce")
    test = test.apply(pd.to_numeric, errors="coerce")

    # ---- NaN fill: ffill -> bfill -> train column mean ----
    nan_tr, nan_te = int(train.isna().sum().sum()), int(test.isna().sum().sum())
    nan_cols = {c: int(n) for c, n in train.isna().sum().items() if n} | \
               {c: int(n) for c, n in test.isna().sum().items() if n}
    train = train.ffill().bfill()
    test = test.ffill().bfill()
    col_mean = train.mean()
    train = train.fillna(col_mean)
    test = test.fillna(col_mean)
    print(f"[prep] NaNs filled: train {nan_tr}, test {nan_te} "
          f"({len(nan_cols)} columns affected)", flush=True)
    assert not train.isna().any().any() and not test.isna().any().any()

    # ---- per-second labels from corrected xlsx windows ----
    lab = np.zeros(len(test), dtype=np.int8)
    te_vals = dt_te.values
    for aid, s, e in ATTACK_WINDOWS:
        m = (te_vals >= np.datetime64(s)) & (te_vals <= np.datetime64(e))
        lab[m] = 1
        if not m.any():
            print(f"[prep] WARNING: attack {aid} window {s}..{e} matched 0 rows", flush=True)
    print(f"[prep] raw test label fraction: {lab.mean():.4f} ({int(lab.sum())} of {len(lab)} s)", flush=True)

    # ---- min-max normalize, train-fit ----
    cmin, cmax = train.min(), train.max()
    rng = (cmax - cmin).replace(0, 1.0)
    train = (train - cmin) / rng
    test = (test - cmin) / rng
    n_const = int((cmax == cmin).sum())
    print(f"[prep] min-max normalized (train-fit); constant train columns: {n_const}", flush=True)

    # ---- 10x downsample: median sensors, max label, first datetime ----
    d = a.downsample
    g_tr = np.arange(len(train)) // d
    g_te = np.arange(len(test)) // d
    train_ds = train.groupby(g_tr).median()
    test_ds = test.groupby(g_te).median()
    lab_ds = pd.Series(lab).groupby(g_te).max().to_numpy()
    anchor_tr = dt_tr.iloc[0]
    anchor_te = dt_te.iloc[0]

    # ---- stabilization trim, train only ----
    if a.stabilization_trim > 0:
        train_ds = train_ds.iloc[a.stabilization_trim:].reset_index(drop=True)
        anchor_tr_eff = anchor_tr + pd.Timedelta(seconds=a.stabilization_trim * d)
        print(f"[prep] trimmed first {a.stabilization_trim} downsampled rows from TRAIN "
              f"({a.stabilization_trim * d / 3600:.1f} h)", flush=True)
    else:
        anchor_tr_eff = anchor_tr

    test_ds = test_ds.reset_index(drop=True)
    test_ds["attack"] = lab_ds.astype(int)

    # ---- write ----
    os.makedirs(OUT, exist_ok=True)
    train_ds.to_csv(os.path.join(OUT, "train.csv"))
    test_ds.to_csv(os.path.join(OUT, "test.csv"))
    with open(os.path.join(OUT, "list.txt"), "w") as f:
        f.write("\n".join(train_ds.columns) + "\n")
    pd.DataFrame({"col": cmin.index, "min": cmin.values, "max": cmax.values}).to_csv(
        os.path.join(OUT, "scaler_minmax.csv"), index=False)

    meta = dict(
        source="WADI.A1_9 Oct 2017",
        downsample=d,
        stabilization_trim_train_rows=a.stabilization_trim,
        train_rows=len(train_ds), test_rows=len(test_ds),
        n_sensors=train_ds.shape[1],
        train_anchor_raw=str(anchor_tr), train_anchor_effective=str(anchor_tr_eff),
        test_anchor=str(anchor_te),
        test_span=[str(dt_te.iloc[0]), str(dt_te.iloc[-1])],
        label_fraction_raw=float(lab.mean()),
        label_fraction_downsampled=float(lab_ds.mean()),
        normalization="minmax_train_fit",
        constant_train_columns=n_const,
        dropped_dead_columns=dead,
        dropped_nonshared={"train_only": only_tr, "test_only": only_te},
        nan_filled={"train": nan_tr, "test": nan_te, "by_column": nan_cols},
        label_convention="per-second membership in corrected xlsx windows, "
                         "max over each 10s block",
        attack_windows=[(aid, s, e) for aid, s, e in ATTACK_WINDOWS],
        built=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(os.path.join(OUT, "prep_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[prep] wrote data/wadi: train {train_ds.shape}, test {test_ds.shape} "
          f"(label frac {lab_ds.mean():.4f}), {train_ds.shape[1]} sensors", flush=True)
    print("[prep] DONE", flush=True)


if __name__ == "__main__":
    main()
