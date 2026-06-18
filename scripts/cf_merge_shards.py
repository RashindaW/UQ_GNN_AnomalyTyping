"""Merge per-shard CSVs from cf_graph_static.py runs into the unified
top-level CSVs at the run dir.

Inputs:  <run_dir>/shard_<I>/{cf_per_anchor.csv, sensor_votes_per_anchor.csv,
         segments_index.csv}  for I in 0..N-1
Outputs: <run_dir>/{cf_per_anchor.csv, sensor_votes_per_anchor.csv,
         segments_index.csv}

per_segment/<seg_idx>.json files are written directly by the shards
into the shared per_segment/ dir; no merge needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def merge(run_dir: Path):
    shards = sorted(run_dir.glob('shard_*'))
    if not shards:
        sys.exit(f'no shard_* dirs found in {run_dir}')
    print(f'[merge] found {len(shards)} shards: {[s.name for s in shards]}')

    # check all shards have DONE markers
    missing = [s for s in shards if not (s / 'DONE').exists()]
    if missing:
        sys.exit(f'[merge] missing DONE markers in: '
                  f'{[s.name for s in missing]} — wait for those shards.')

    # cf_per_anchor.csv
    dfs = [pd.read_csv(s / 'cf_per_anchor.csv') for s in shards
            if (s / 'cf_per_anchor.csv').exists()]
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sort_values(['seg_idx', 'anchor_label', 'cf_idx'])
        merged.to_csv(run_dir / 'cf_per_anchor.csv', index=False)
        print(f'[merge] wrote cf_per_anchor.csv  ({len(merged)} rows)')

    # sensor_votes_per_anchor.csv
    dfs = [pd.read_csv(s / 'sensor_votes_per_anchor.csv') for s in shards
            if (s / 'sensor_votes_per_anchor.csv').exists()]
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sort_values(['seg_idx', 'anchor_label', 'sensor'])
        merged.to_csv(run_dir / 'sensor_votes_per_anchor.csv', index=False)
        print(f'[merge] wrote sensor_votes_per_anchor.csv  ({len(merged)} rows)')

    # segments_index.csv (identical across shards; take first)
    si_files = [s / 'segments_index.csv' for s in shards
                 if (s / 'segments_index.csv').exists()]
    if si_files:
        df = pd.read_csv(si_files[0])
        df = df.sort_values('seg_idx')
        df.to_csv(run_dir / 'segments_index.csv', index=False)
        print(f'[merge] wrote segments_index.csv  ({len(df)} rows)')

    # Sanity: count per_segment JSONs
    n_seg_json = len(list((run_dir / 'per_segment').glob('*.json')))
    print(f'[merge] per_segment/*.json count: {n_seg_json}')

    print('[merge] done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    merge(Path(args.run_dir))


if __name__ == '__main__':
    main()
