"""Build the SWaT attack ground-truth manifest for the CF cross-check.

Source of truth: the user-designated `List_of_attacks_Final.pdf`. We
read its machine-readable twin `List_of_attacks_Final.xlsx` (verified
row-by-row identical to the PDF), normalize sensor names, derive the
attack category, and map each attack's wall-clock interval to the
array-index space of `test_attack_label` via an *empirically calibrated*
global offset (robust to the stabilization-trim constant).

Outputs:
  data/swat/attack_list.csv      — clean normalized table (41 attacks)
  data/swat/attack_targets.json  — per-attack records with calibrated
                                    index ranges, targets, category, etc.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

ANCHOR = datetime(2015, 12, 28, 10, 0, 0)   # util/iostream.py::save_attack_infos
DOWNSAMPLE = 10
SLIDE_WIN = 60

DEFAULT_XLSX = (REPO_ROOT / 'data/raw/swat_a1_a2_dec2015/'
                'SWaT.A1 & A2_Dec 2015/List_of_attacks_Final.xlsx')
DEFAULT_ARRAYS = (REPO_ROOT / 'results/swat_gdeltauq_sw60_paper_protocol_K100/'
                  '0516-031655/arrays.npz')


def model_sensors() -> list[str]:
    df = pd.read_csv(REPO_ROOT / 'data/swat/test.csv', index_col=0)
    return [c for c in df.columns if c != 'attack']


def normalize_point(raw: str) -> str:
    """`MV-101`->`MV101`, `Mv-303`->`MV303`, `DIT-301`->`DPIT301`."""
    s = raw.strip().upper().replace('-', '').replace(' ', '')
    if s == 'DIT301':           # manifest typo for DPIT-301 (attack 23)
        s = 'DPIT301'
    return s


def stage_of(sensor: str) -> int | None:
    """Process stage = first digit of the trailing number (LIT101->1)."""
    m = re.search(r'(\d)\d\d$', sensor)
    return int(m.group(1)) if m else None


_SENSOR_TOKEN = re.compile(
    r'\b(LIT|FIT|AIT|DPIT|DIT|PIT|MV|UV|P)[-\s]?(\d{3})\b', re.IGNORECASE)
_TANK_TOKEN = re.compile(r'\btank\s*(\d)0?1?\b', re.IGNORECASE)


def extract_impact_sensors(text: str, sensors: list[str]) -> list[str]:
    """Pull sensor codes mentioned in the 'Expected Impact'/'Unexpected
    Outcome' free text — the sensors the attack is documented to affect
    *downstream* (a softer target than the manipulated point). Also maps
    'tank NNN' -> LITNNN (the tank's level transmitter)."""
    if not isinstance(text, str):
        return []
    found = set()
    for m in _SENSOR_TOKEN.finditer(text):
        s = normalize_point(m.group(1) + m.group(2))
        if s in sensors:
            found.add(s)
    for m in _TANK_TOKEN.finditer(text):
        stage = m.group(1)
        cand = f'LIT{stage}01'
        if cand in sensors:
            found.add(cand)
    return sorted(found)


def parse_targets(attack_point: str, sensors: list[str]):
    """Split a multi-target cell, normalize, mark in_model, derive category."""
    parts = re.split(r'[;,]', str(attack_point))
    targets = [normalize_point(p) for p in parts if p.strip()]
    in_model = [t in sensors for t in targets]
    stages = sorted({stage_of(t) for t in targets if stage_of(t) is not None})
    n_points = len(targets)
    n_stages = len(stages)
    stage_tag = 'M' if n_stages > 1 else 'S'        # Multi/Single stage
    point_tag = 'MP' if n_points > 1 else 'SP'       # Multi/Single point
    category = f'{stage_tag}S{point_tag}'            # SSSP/SSMP/MSSP/MSMP
    return targets, in_model, n_points, n_stages, category


def to_naive_idx(dt: datetime) -> int:
    return round((dt - ANCHOR).total_seconds() / DOWNSAMPLE) - SLIDE_WIN


def label_runs(label: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0
    n = len(label)
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


def total_overlap(intervals: list[tuple[int, int]],
                  runs: list[tuple[int, int]]) -> int:
    """Sum of intersection lengths between intervals and label runs."""
    tot = 0
    for (a, b) in intervals:
        for (c, d) in runs:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                tot += hi - lo
    return tot


def calibrate_offset(naive_intervals: list[tuple[int, int]],
                     runs: list[tuple[int, int]],
                     search=range(-600, 601)) -> int:
    """Pick the global shift c that maximizes overlap with the label runs."""
    best_c, best_ov = 0, -1
    for c in search:
        shifted = [(a + c, b + c) for (a, b) in naive_intervals]
        ov = total_overlap(shifted, runs)
        if ov > best_ov:
            best_ov, best_c = ov, c
    return best_c, best_ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', default=str(DEFAULT_XLSX))
    ap.add_argument('--arrays', default=str(DEFAULT_ARRAYS))
    ap.add_argument('--out-json', default=str(REPO_ROOT / 'data/swat/attack_targets.json'))
    ap.add_argument('--out-csv', default=str(REPO_ROOT / 'data/swat/attack_list.csv'))
    args = ap.parse_args()

    sensors = model_sensors()
    print(f'[manifest] model sensors: {len(sensors)}', flush=True)

    df = pd.read_excel(args.xlsx)
    # keep only rows whose 'Attack #' is an integer 1..41
    rows = []
    for _, r in df.iterrows():
        aid = r['Attack #']
        if pd.isna(aid):
            continue
        try:
            aid = int(aid)
        except (ValueError, TypeError):
            continue
        rows.append(r)
    print(f'[manifest] parsed {len(rows)} attack rows', flush=True)

    label = np.load(args.arrays)['test_attack_label'].astype(np.int8)
    runs = label_runs(label)
    print(f'[manifest] test_attack_label runs: {len(runs)}', flush=True)

    # ---- first pass: parse + naive index ----
    records = []
    naive_intervals = []
    for r in rows:
        aid = int(r['Attack #'])
        start_dt = pd.to_datetime(r['Start Time']).to_pydatetime()
        # Manifest typo: attacks 37-41 are dated 2015-01-02 but the
        # experiment ran Dec 2015 -> Jan 2016, so a January-2015 date is
        # impossible and must be 2016.
        if start_dt.year == 2015 and start_dt.month == 1:
            start_dt = start_dt.replace(year=2016)
        ap_raw = r['Attack Point']
        actual = r['Actual Change']
        no_impact = (isinstance(ap_raw, float) and pd.isna(ap_raw)) or \
                    (isinstance(ap_raw, str) and 'No Physical Impact' in ap_raw)

        # End time -> datetime (combine with start date; roll a day if needed)
        end_raw = r['End Time']
        if pd.isna(end_raw):
            end_dt = start_dt
        else:
            if hasattr(end_raw, 'hour'):           # datetime.time
                end_t = end_raw
            else:
                end_t = pd.to_datetime(str(end_raw)).time()
            end_dt = datetime.combine(start_dt.date(), end_t)
            if end_dt < start_dt:
                end_dt += timedelta(days=1)

        ns, ne = to_naive_idx(start_dt), to_naive_idx(end_dt)
        if ne <= ns:
            ne = ns + 1
        naive_intervals.append((ns, ne))

        if no_impact:
            targets, in_model = [], []
            n_points = n_stages = 0
            category = 'NONE'
        else:
            targets, in_model, n_points, n_stages, category = \
                parse_targets(ap_raw, sensors)

        impact_text = ' '.join(
            str(r.get(col, '')) for col in
            ('Expected Impact or attacker intent', 'Unexpected Outcome'))
        impact_sensors = [s for s in extract_impact_sensors(impact_text, sensors)
                          if s not in set(targets)]

        records.append(dict(
            attack_id=aid,
            start_time=start_dt.strftime('%Y-%m-%d %H:%M:%S'),
            end_time=end_dt.strftime('%Y-%m-%d %H:%M:%S'),
            raw_attack_point=('' if no_impact else str(ap_raw)),
            actual_change=(None if pd.isna(actual) else (str(actual).strip() == 'Yes')),
            no_physical_impact=bool(no_impact),
            targets=targets,
            in_model=in_model,
            impact_sensors=impact_sensors,
            n_points=n_points,
            n_stages=n_stages,
            category=category,
            naive_start_idx=ns,
            naive_end_idx=ne,
        ))

    # ---- calibrate global offset against the label ----
    # Only attacks with a physical footprint should align; calibrate on the
    # ones that produced actual change (most likely labelled).
    impactful = [naive_intervals[i] for i, rec in enumerate(records)
                 if rec['actual_change'] is True]
    c, ov = calibrate_offset(impactful if impactful else naive_intervals, runs)
    print(f'[manifest] calibrated global offset c={c}  (overlap={ov})', flush=True)

    # apply offset, compute per-attack coverage vs label
    coverages = []
    for rec, (ns, ne) in zip(records, naive_intervals):
        s_idx, e_idx = ns + c, ne + c
        rec['start_idx'] = int(s_idx)
        rec['end_idx'] = int(e_idx)
        # coverage = fraction of [s_idx,e_idx) that is label==1
        s_cl = max(0, min(len(label), s_idx))
        e_cl = max(0, min(len(label), e_idx))
        if e_cl > s_cl:
            cov = float(label[s_cl:e_cl].mean())
        else:
            cov = 0.0
        rec['label_coverage'] = cov
        if rec['actual_change'] is True:
            coverages.append(cov)
    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    print(f'[manifest] mean label-coverage on Actual-Change=Yes attacks: '
          f'{mean_cov:.3f}', flush=True)
    if mean_cov < 0.8:
        print('[manifest] WARNING: coverage < 0.8 — time->index mapping may be '
              'off; treat overlap matching with caution (consider order-based '
              'fallback).', flush=True)

    # ---- write outputs ----
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(dict(
            anchor=ANCHOR.strftime('%Y-%m-%d %H:%M:%S'),
            downsample=DOWNSAMPLE, slide_win=SLIDE_WIN,
            calibrated_offset=int(c),
            mean_coverage_actualchange=mean_cov,
            n_attacks=len(records),
            attacks=records,
        ), f, indent=2)
    print(f'[manifest] wrote {args.out_json}', flush=True)

    csv_cols = ['attack_id', 'start_time', 'end_time', 'raw_attack_point',
                'actual_change', 'no_physical_impact', 'targets', 'in_model',
                'impact_sensors', 'n_points', 'n_stages', 'category',
                'start_idx', 'end_idx', 'label_coverage']
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for rec in records:
            row = {k: rec[k] for k in csv_cols}
            row['targets'] = ';'.join(rec['targets'])
            row['in_model'] = ';'.join(str(x) for x in rec['in_model'])
            row['impact_sensors'] = ';'.join(rec['impact_sensors'])
            w.writerow(row)
    print(f'[manifest] wrote {args.out_csv}', flush=True)

    # ---- console summary ----
    n_yes = sum(1 for r in records if r['actual_change'] is True)
    n_no = sum(1 for r in records if r['actual_change'] is False)
    n_imp = sum(1 for r in records if r['no_physical_impact'])
    cats = {}
    for r in records:
        if not r['no_physical_impact']:
            cats[r['category']] = cats.get(r['category'], 0) + 1
    print(f'[manifest] Actual-Change Yes={n_yes}  No={n_no}  '
          f'NoPhysicalImpact={n_imp}', flush=True)
    print(f'[manifest] categories (targeted attacks): {cats}', flush=True)
    n_targetable = sum(1 for r in records if any(r['in_model']))
    print(f'[manifest] attacks with >=1 in-model target: {n_targetable}', flush=True)

    print('\n  attack_id  idx-range          cat   chg  cov   targets')
    for r in records:
        chg = {True: 'Y', False: 'N', None: '-'}[r['actual_change']]
        print(f'  {r["attack_id"]:>2d}  [{r["start_idx"]:>6d},{r["end_idx"]:>6d})  '
              f'{r["category"]:>4s}  {chg}   {r["label_coverage"]:.2f}  '
              f'{",".join(r["targets"]) if r["targets"] else "(none)"}', flush=True)


if __name__ == '__main__':
    main()
