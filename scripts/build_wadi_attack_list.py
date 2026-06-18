#!/usr/bin/env python3
"""Build data/wadi/attack_list.csv + attack_targets.json (SWaT schema).

Adapts scripts/cf_attack_manifest.py to WADI.A1. Source of truth is the
corrected transcription of attack_description.xlsx (typos fixed: the
duplicate S.No 2 for attack 4, year 1947 on attack 5, "11.30:40",
month 07 on the Oct-11 rows; cross-checked against the CST-GL notebook
and TopoGDN's wadi_mark_label.py, both of which carry their own typos).

Schema (identical to data/swat/attack_list.csv so typing_rules_v1v2
consumes it unchanged):
  attack_id, start_time, end_time, raw_attack_point, actual_change,
  no_physical_impact, targets, in_model, impact_sensors, n_points,
  n_stages, category, start_idx, end_idx, label_coverage

Conventions mirrored from cf_attack_manifest.py:
  - start_idx/end_idx in WINDOWED array coords:
    round((dt - TEST_ANCHOR)/10s) - SLIDE_WIN, plus a global offset
    calibrated by maximum overlap with the empirical label runs
    (windowed label = test.csv attack column shifted by SLIDE_WIN);
  - category SSSP/SSMP/MSSP/MSMP from device count x stage count
    (stage = leading digit of the WADI device name);
  - `targets` holds the matched MODEL COLUMN names (e.g. 1_MV_001_STATUS)
    so norm_name matching against list.txt is exact; n_points counts
    DEVICES, not columns (a 2_PIC_003 maps to its CO/PV/SP columns but
    is one attacked point);
  - device -> column matching is prefix-based after normalization with
    the documented WADI alias LIT -> LT (the manifest writes 2LIT002,
    the historian logs 2_LT_002_PV).

Gates: every device maps to >= 1 in-model column (else listed);
mean label_coverage on actual_change attacks >= 0.8.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WADI = os.path.join(ROOT, "data/wadi")
DOWNSAMPLE = 10
SLIDE_WIN = 60

# aid, start, end, devices, actual_change, impact_devices, note
# actual_change True = the targeted variable was physically changed
# (actuator/setpoint manipulation); False = sensor reading spoofed.
ATTACKS = [
    (1,  "2017-10-09 19:25:00", "2017-10-09 19:50:16", ["1_MV_001"], True,
     [], "motorized valve opened; overflow of primary grid tank"),
    (2,  "2017-10-10 10:24:10", "2017-10-10 10:34:00", ["1_FIT_001"], False,
     [], "false readings to 1FIT001; chemical dosing pumps start"),
    (3,  "2017-10-10 10:55:00", "2017-10-10 11:24:00", ["2_LIT_002"], False,
     [], "stealthy fast level ramp 70-80%; elevated reservoir drained"),
    (4,  "2017-10-10 11:07:46", "2017-10-10 11:12:15", ["1_AIT_001"], False,
     [], "reading 176 -> 640; raw water tank drain valves open (impact till 11:27); nested inside attack 3 window"),
    (5,  "2017-10-10 11:30:40", "2017-10-10 11:44:50",
     ["2_MCV_101", "2_MCV_201", "2_MCV_301", "2_MCV_401", "2_MCV_501", "2_MCV_601"], True,
     [], "all inlet valves to 0%; no water to consumers"),
    (6,  "2017-10-10 13:39:30", "2017-10-10 13:50:40", ["2_MCV_101", "2_MCV_201"], True,
     [], "valves opened to 50%; contaminated water to elevated reservoir"),
    (7,  "2017-10-10 14:48:17", "2017-10-10 15:00:32", ["1_AIT_002", "2_MV_003"], True,
     ["1_MV_002"], "1AIT002 value 0.5 -> 6 plus 2MV003 opened (two overlapping legs, union window); drain valve 1MV002 opens"),
    (8,  "2017-10-10 17:40:00", "2017-10-10 17:49:40", ["2_MCV_007"], True,
     [], "valve opened to 30%; water leakage"),
    (9,  "2017-10-11 10:55:00", "2017-10-11 10:56:27", ["1_P_005", "1_P_006"], True,
     [], "pump 5 on, pump 6 off; pipe bursts"),
    (10, "2017-10-11 11:17:54", "2017-10-11 11:31:20", ["1_MV_001"], True,
     ["2_LIT_002", "1_LIT_001"], "randomized open/close after staging 2LIT002 at 70% and 1LIT001 at 40%"),
    (11, "2017-10-11 11:36:31", "2017-10-11 11:47:00", ["2_MCV_007"], True,
     ["2_FIT_002"], "valve at 50%; booster never starts as 2FIT002 reads above required flow"),
    (12, "2017-10-11 11:59:00", "2017-10-11 12:05:00", ["2_MCV_007"], True,
     [], "valve to 100% in 10% steps; booster on, water wasted"),
    (13, "2017-10-11 12:07:30", "2017-10-11 12:10:52", ["2_PIC_003"], True,
     [], "setpoint 1 bar -> 0.25 bar; intermittent supply to consumer tanks"),
    (14, "2017-10-11 12:16:00", "2017-10-11 12:25:36", ["1_P_001", "1_P_003"], True,
     [], "chemical dosing pumps off"),
    (15, "2017-10-11 15:26:30", "2017-10-11 15:37:00", ["2_LIT_002"], False,
     [], "stealthy slow level ramp 70-80%; overflow of elevated reservoir"),
]

ALIASES = {"LIT": "LT"}  # manifest device family -> historian tag family


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def device_key(dev: str) -> str:
    k = norm(dev)
    for a, b in ALIASES.items():
        k = k.replace(a, b)
    return k


def stage_of(dev: str) -> int | None:
    m = re.match(r"\s*(\d)", dev)
    return int(m.group(1)) if m else None


def match_columns(dev: str, sensors: list[str]) -> list[str]:
    key = device_key(dev)
    return [s for s in sensors if norm(s).startswith(key)]


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


def total_overlap(intervals, runs) -> int:
    tot = 0
    for (a, b) in intervals:
        for (c, d) in runs:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                tot += hi - lo
    return tot


def calibrate_offset(naive_intervals, runs, search=range(-600, 601)):
    best_c, best_ov = 0, -1
    for c in search:
        ov = total_overlap([(a + c, b + c) for (a, b) in naive_intervals], runs)
        if ov > best_ov:
            best_ov, best_c = ov, c
    return best_c, best_ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default=os.path.join(WADI, "attack_list.csv"))
    ap.add_argument("--out-json", default=os.path.join(WADI, "attack_targets.json"))
    a = ap.parse_args()

    meta = json.load(open(os.path.join(WADI, "prep_meta.json")))
    anchor = pd.to_datetime(meta["test_anchor"]).to_pydatetime()
    test = pd.read_csv(os.path.join(WADI, "test.csv"), index_col=0)
    sensors = [c for c in test.columns if c != "attack"]
    row_label = test["attack"].to_numpy().astype(np.int8)
    label = row_label[SLIDE_WIN:]          # windowed coords (window-end label)
    runs = label_runs(label)
    print(f"[wadi-att] anchor {anchor}, test rows {len(test)}, "
          f"windowed label rows {len(label)}, runs {len(runs)}", flush=True)

    def to_naive_idx(dt: datetime) -> int:
        return round((dt - anchor).total_seconds() / DOWNSAMPLE) - SLIDE_WIN

    records, naive_intervals = [], []
    unmapped = []
    for aid, s, e, devices, actual, impacts, note in ATTACKS:
        sdt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        edt = datetime.strptime(e, "%Y-%m-%d %H:%M:%S")
        ns, ne = to_naive_idx(sdt), to_naive_idx(edt)
        if ne <= ns:
            ne = ns + 1
        naive_intervals.append((ns, ne))

        targets, in_model = [], []
        for dev in devices:
            cols = match_columns(dev, sensors)
            if not cols:
                unmapped.append((aid, dev))
            targets += cols
            in_model += [True] * len(cols)
        impact_cols = []
        for dev in impacts:
            impact_cols += [c for c in match_columns(dev, sensors) if c not in targets]

        stages = sorted({stage_of(d) for d in devices if stage_of(d) is not None})
        n_points, n_stages = len(devices), len(stages)
        category = ("M" if n_stages > 1 else "S") + "S" + ("MP" if n_points > 1 else "SP")

        records.append(dict(
            attack_id=aid,
            start_time=s, end_time=e,
            raw_attack_point=", ".join(devices),
            actual_change=bool(actual),
            no_physical_impact=False,
            targets=targets, in_model=in_model, impact_sensors=impact_cols,
            n_points=n_points, n_stages=n_stages, category=category,
            naive_start_idx=ns, naive_end_idx=ne, note=note,
        ))

    if unmapped:
        print(f"[wadi-att] WARNING unmapped devices: {unmapped}", flush=True)

    impactful = [naive_intervals[i] for i, r in enumerate(records) if r["actual_change"]]
    c, ov = calibrate_offset(impactful, runs)
    print(f"[wadi-att] calibrated offset c={c} (overlap={ov})", flush=True)

    covs = []
    for rec, (ns, ne) in zip(records, naive_intervals):
        s_idx, e_idx = ns + c, ne + c
        rec["start_idx"], rec["end_idx"] = int(s_idx), int(e_idx)
        s_cl, e_cl = max(0, min(len(label), s_idx)), max(0, min(len(label), e_idx))
        rec["label_coverage"] = float(label[s_cl:e_cl].mean()) if e_cl > s_cl else 0.0
        if rec["actual_change"]:
            covs.append(rec["label_coverage"])
    mean_cov = float(np.mean(covs))
    print(f"[wadi-att] mean label-coverage on actual-change attacks: {mean_cov:.3f}", flush=True)
    if mean_cov < 0.8:
        print("[wadi-att] WARNING coverage < 0.8: time->index mapping suspect", flush=True)

    os.makedirs(WADI, exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(dict(
            anchor=anchor.strftime("%Y-%m-%d %H:%M:%S"),
            downsample=DOWNSAMPLE, slide_win=SLIDE_WIN,
            calibrated_offset=int(c), mean_coverage_actualchange=mean_cov,
            n_attacks=len(records), aliases=ALIASES, attacks=records,
        ), f, indent=2)
    cols = ["attack_id", "start_time", "end_time", "raw_attack_point",
            "actual_change", "no_physical_impact", "targets", "in_model",
            "impact_sensors", "n_points", "n_stages", "category",
            "start_idx", "end_idx", "label_coverage"]
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in records:
            row = {k: rec[k] for k in cols}
            row["targets"] = ";".join(rec["targets"])
            row["in_model"] = ";".join(str(x) for x in rec["in_model"])
            row["impact_sensors"] = ";".join(rec["impact_sensors"])
            w.writerow(row)
    print(f"[wadi-att] wrote {a.out_csv} and {a.out_json}", flush=True)

    cats = {}
    for r in records:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"[wadi-att] categories: {cats}; actual_change "
          f"Y={sum(r['actual_change'] for r in records)} "
          f"N={sum(not r['actual_change'] for r in records)}", flush=True)
    print("\n  aid  idx-range          cat   chg  cov   targets")
    for r in records:
        print(f"  {r['attack_id']:>3d}  [{r['start_idx']:>6d},{r['end_idx']:>6d})  "
              f"{r['category']:>4s}  {'Y' if r['actual_change'] else 'N'}   "
              f"{r['label_coverage']:.2f}  {','.join(r['targets']) or '(none)'}", flush=True)


if __name__ == "__main__":
    main()
