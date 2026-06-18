"""M8 (Weighted linear sum) with chunked HP grid for parallel execution.

The full M8 grid has 5^4 = 625 (β_par, β_σ, β_str, β_dist) combinations.
This script lets you split that grid into N chunks and run only chunk I,
so several copies can execute in parallel on different CPU cores.

Each chunk writes its best-of-chunk HP + F1 to:
    <out_root>/chunk_{I}_of_{N}/best.json

A merge script consolidates the chunks afterwards by picking the global
argmax F1 across all chunks.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from fusion_sweep_K100_full import (
    setup_context,
    score_linear_sum,
    eval_score_full,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-arrays', required=True)
    parser.add_argument('-split', required=True)
    parser.add_argument('-bundle', default=None)
    parser.add_argument('-slide_win', type=int, default=60)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-chunk_idx', type=int, required=True,
                        help='0-indexed chunk number')
    parser.add_argument('-n_chunks', type=int, required=True,
                        help='total number of chunks')
    parser.add_argument('-out_root', required=True)
    parser.add_argument('-methods', nargs='*', default=['M8'],
                        help='ignored; kept for argparse compatibility')
    args = parser.parse_args()

    ctx = setup_context(args)
    label = ctx['label']
    agg_z = ctx['agg_z']
    signals = ctx['signals']
    uq_keys = ['U_par_max_v', 'sigma_ale_max_v', 'U_str_mean_e', 'U_dist']
    grid = (0.0, 0.1, 0.3, 1.0, 2.0)

    all_configs = list(itertools.product(grid, grid, grid, grid))
    n_total = len(all_configs)
    chunk_lo = (n_total * args.chunk_idx) // args.n_chunks
    chunk_hi = (n_total * (args.chunk_idx + 1)) // args.n_chunks
    my_configs = all_configs[chunk_lo:chunk_hi]

    print(f"Chunk {args.chunk_idx+1}/{args.n_chunks}: "
          f"HP configs [{chunk_lo}, {chunk_hi}) of {n_total}", flush=True)
    print(f"  ({len(my_configs)} configs in this chunk)", flush=True)

    rows = []
    best = None
    t0 = time.time()
    for i, (bp, bsig, bst, bd) in enumerate(my_configs):
        betas = dict(zip(uq_keys, (bp, bsig, bst, bd)))
        s = score_linear_sum(agg_z, signals, betas)
        res = eval_score_full(s, label)
        row = dict(method='M8', chunk_idx=args.chunk_idx,
                   b_U_par_max_v=bp, b_sigma_ale_max_v=bsig,
                   b_U_str_mean_e=bst, b_U_dist=bd,
                   **res)
        rows.append(row)
        if best is None or row['F1'] > best['F1']:
            best = dict(row)
        if (i + 1) % 20 == 0 or i == len(my_configs) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(my_configs) - i - 1) / rate
            print(f"  [{i+1}/{len(my_configs)}] best F1={best['F1']:.4f} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    out_dir = Path(args.out_root) / f"chunk_{args.chunk_idx}_of_{args.n_chunks}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / 'per_hp.csv', index=False)
    with open(out_dir / 'best.json', 'w') as f:
        json.dump(best, f, indent=2)
    print(f"\nChunk {args.chunk_idx+1} done. "
          f"Best F1 in chunk = {best['F1']:.4f}", flush=True)
    print(f"  -> {out_dir/'best.json'}", flush=True)


if __name__ == '__main__':
    main()
