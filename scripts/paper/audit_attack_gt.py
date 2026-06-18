#!/usr/bin/env python3
"""Deterministic ground-truth audit: data/swat/attack_targets.json vs the
OFFICIAL iTrust List_of_attacks_Final.xlsx (provided 2026-06-07).

Parses the official sheet (header row 0; End Time is time-of-day only and is
combined with the start date, rolling past midnight when needed; the five
no-physical-impact entries carry 'No Physical Impact Attack' in the Attack
Point column with no end time), normalizes attack points, and diffs every
attack 1-41 field by field. Writes the frozen canonical ground truth to
data/swat/attack_gt_canonical.json with file provenance (sha256).

Exit 0 always; the report is the product.
"""
import datetime as dt
import hashlib
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX = os.path.join(ROOT, "data/swat/List_of_attacks_Final.xlsx")
REPO = os.path.join(ROOT, "data/swat/attack_targets.json")
OUT = os.path.join(ROOT, "data/swat/attack_gt_canonical.json")


def norm_points(s):
    if not isinstance(s, str) or "no physical impact" in s.lower():
        return []
    parts = re.split(r"[,;]|\band\b", s)
    return [p.strip().upper().replace(" ", "") for p in parts if p.strip()]


def parse_official():
    df = pd.read_excel(XLSX, sheet_name=0, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        aid = r.get("Attack #")
        if pd.isna(aid):
            continue
        try:
            aid = int(aid)
        except (TypeError, ValueError):
            continue
        start = r.get("Start Time")
        start = pd.to_datetime(start) if not pd.isna(start) else None
        end_raw = r.get("End Time")
        end = None
        if start is not None and not pd.isna(end_raw):
            if isinstance(end_raw, dt.time):
                end = dt.datetime.combine(start.date(), end_raw)
            else:
                t = pd.to_datetime(str(end_raw)).time()
                end = dt.datetime.combine(start.date(), t)
            if end < start:                      # crossed midnight
                end += dt.timedelta(days=1)
        point_raw = r.get("Attack Point")
        ac_raw = r.get("Actual Change")
        ac = None
        if isinstance(ac_raw, str):
            ac = ac_raw.strip().lower().startswith("y")
        no_impact_entry = isinstance(point_raw, str) and \
            "no physical impact" in point_raw.lower()
        rows.append(dict(
            attack_id=aid,
            start_time=None if start is None else start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=None if end is None else end.strftime("%Y-%m-%d %H:%M:%S"),
            attack_point_raw=None if pd.isna(point_raw) else str(point_raw).strip(),
            points=norm_points(point_raw if isinstance(point_raw, str) else ""),
            actual_change=ac,
            no_physical_impact_entry=no_impact_entry,
            start_state=None if pd.isna(r.get("Start State")) else str(r.get("Start State")).strip(),
            attack_desc=None if pd.isna(r.get("Attack")) else str(r.get("Attack")).strip(),
            expected_impact=None if pd.isna(r.get("Expected Impact or attacker intent")) else str(r.get("Expected Impact or attacker intent")).strip(),
            unexpected_outcome=None if pd.isna(r.get("Unexpected Outcome")) else str(r.get("Unexpected Outcome")).strip(),
        ))
    return rows


def main():
    official = {a["attack_id"]: a for a in parse_official()}
    repo = {a["attack_id"]: a for a in json.load(open(REPO))["attacks"]}
    print(f"official rows: {len(official)} | repo rows: {len(repo)}")
    ids = sorted(set(official) | set(repo))
    diffs, matches = [], 0
    for i in ids:
        o, r = official.get(i), repo.get(i)
        if o is None or r is None:
            diffs.append(f"A{i}: present only in {'repo' if o is None else 'official'}")
            continue
        d = []
        if o["start_time"] != r.get("start_time"):
            d.append(f"start {r.get('start_time')!r} vs OFFICIAL {o['start_time']!r}")
        if o["end_time"] is not None and o["end_time"] != r.get("end_time"):
            d.append(f"end {r.get('end_time')!r} vs OFFICIAL {o['end_time']!r}")
        rp = norm_points(r.get("raw_attack_point") or "")
        if o["points"] and sorted(o["points"]) != sorted(rp):
            d.append(f"points {rp} vs OFFICIAL {o['points']}")
        if o["actual_change"] is not None and bool(r.get("actual_change")) != o["actual_change"]:
            d.append(f"actual_change {r.get('actual_change')} vs OFFICIAL {o['actual_change']}")
        if o["no_physical_impact_entry"] != bool(r.get("no_physical_impact")):
            d.append(f"no_physical_impact {r.get('no_physical_impact')} vs OFFICIAL-entry {o['no_physical_impact_entry']}")
        if d:
            diffs.append(f"A{i}: " + " | ".join(d))
        else:
            matches += 1
    print(f"\nFULL MATCH: {matches}/{len(ids)} attacks")
    print(f"DIFFS ({len(diffs)}):")
    for x in diffs:
        print("  " + x)

    sha = hashlib.sha256(open(XLSX, "rb").read()).hexdigest()
    canon = dict(
        provenance=dict(source_file="data/swat/List_of_attacks_Final.xlsx",
                        sha256=sha, parsed="2026-06-07",
                        parser="scripts/paper/audit_attack_gt.py",
                        note=("Official iTrust attack documentation. End times are "
                              "time-of-day in the source, combined with the start "
                              "date (midnight rollover handled). actual_change is "
                              "the official 'Actual Change' column (Yes/No -> "
                              "bool, None for the five no-physical-impact "
                              "entries). Spoof semantics: actual_change==False "
                              "means the reported value was spoofed without the "
                              "physical quantity changing.")),
        attacks=[official[i] for i in sorted(official)],
    )
    json.dump(canon, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT} (sha256 of source: {sha[:16]}...)")


if __name__ == "__main__":
    main()
