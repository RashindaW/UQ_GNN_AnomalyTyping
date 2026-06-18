"""Audit attack temporal distribution in data/swat/test.csv and write split boundaries.

Produces `pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json`
with three contiguous index ranges:
  - 'C_indices'        : nominal calibration set (rows where attack==0 in the front)
  - 'labeled_val_range': (start, end) row indices for alpha selection
  - 'final_test_range' : (start, end) row indices for final F1 reporting

Strategy:
  - Inspect attack mass per chronological tertile (T0/T1/T2).
  - 𝒞 = first --c-target-fraction (default 0.375) of attack==0 rows in test.csv.
    The fraction default matches Stage-1's 30,000 / 79,919 ≈ 0.375 of nominal
    rows so the calibration set scales with whatever test.csv size
    prepare_swat.py produces. Use --c-target-rows to override with an absolute
    count (legacy behaviour).
  - The remainder is split between labeled-val and final-test at the row that
    puts ~half of the remaining attack mass in each side (NOT row midpoint —
    attacks are heavily clustered in T1).

Idempotent — rerun produces the same JSON given the same test.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = REPO_ROOT / 'data' / 'swat' / 'test.csv'
DEFAULT_OUT = REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'calibration_bundle' / 'calibration_set_indices.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-csv', type=str, default=str(TEST_CSV))
    parser.add_argument('--out', type=str, default=str(DEFAULT_OUT))
    parser.add_argument(
        '--c-target-fraction', type=float, default=0.375,
        help='Fraction of nominal (attack==0) rows to include in 𝒞 from the '
             'front of test.csv. Default 0.375 mirrors Stage-1 proportions on '
             'whatever test.csv size prepare_swat produces.',
    )
    parser.add_argument(
        '--c-target-rows', type=int, default=None,
        help='[Override] Absolute count of nominal rows for 𝒞. If set, takes '
             'precedence over --c-target-fraction. None by default.',
    )
    args = parser.parse_args()

    test_csv = Path(args.test_csv)
    out = Path(args.out)
    if not test_csv.is_file():
        print(f'[build_test_split] missing {test_csv}', file=sys.stderr)
        sys.exit(1)

    print(f'[build_test_split] reading {test_csv}')
    df = pd.read_csv(test_csv, sep=',', index_col=0, usecols=lambda c: c == 'attack' or c == 0)
    # Re-read more permissively if usecols=lambda failed to pick anything sane:
    if 'attack' not in df.columns:
        df = pd.read_csv(test_csv, sep=',', index_col=0)
    n_rows = len(df)
    n_attack = int(df['attack'].sum())
    print(f'[build_test_split] total rows: {n_rows:,}  attack rows: {n_attack:,}  ({100*n_attack/n_rows:.2f}%)')

    # Tertile audit.
    print(f'[build_test_split] attack mass by chronological tertile:')
    tertile_size = n_rows // 3
    for i in range(3):
        s = i * tertile_size
        e = (i + 1) * tertile_size if i < 2 else n_rows
        a = int(df['attack'].iloc[s:e].sum())
        print(f'   T{i}  [{s:>7,}, {e:>7,})  rows={e-s:>7,}  attack={a:>6,}  ({100*a/max(1,e-s):.2f}%)')

    # 𝒞 = first c_target_rows where attack == 0. Either an absolute count
    # (--c-target-rows, legacy) or a fraction of nominal rows in test.csv
    # (--c-target-fraction, default). The fraction-based default scales
    # automatically with whatever test.csv size prepare_swat.py produces and
    # avoids the Stage-2 bug where the absolute 30,000 default consumed 76 %
    # of nominal rows in the smaller (10× median downsampled) test file.
    n_nominal = int((df['attack'] == 0).sum())
    if args.c_target_rows is not None:
        c_target = int(args.c_target_rows)
        c_target_source = f'--c-target-rows={c_target}'
    else:
        c_target = int(round(n_nominal * args.c_target_fraction))
        c_target_source = (f'--c-target-fraction={args.c_target_fraction:.3f} '
                           f'× n_nominal={n_nominal:,} = {c_target:,}')
    print(f'[build_test_split] 𝒞 target nominal rows: {c_target_source}')
    cum = (df['attack'] == 0).cumsum()
    c_end_pos = int(cum.searchsorted(c_target, side='left'))
    if c_end_pos >= n_rows:
        c_end_pos = n_rows
    c_indices = df.index[:c_end_pos][df['attack'].iloc[:c_end_pos] == 0].tolist()
    print(f'[build_test_split] 𝒞 spans rows [0, {c_end_pos}); {len(c_indices):,} attack==0 rows kept '
          f'({100*len(c_indices)/n_nominal:.1f}% of all nominal rows).')

    # Remaining rows -> labeled-val and final-test.
    # Split at the row that puts ~half of the remaining attacks in each side
    # (NOT at the row midpoint — attacks are heavily clustered in time, so a
    # row-midpoint split leaves one side with almost no attacks).
    remaining_start = c_end_pos
    remaining_end = n_rows
    remaining_attack_cum = df['attack'].iloc[remaining_start:remaining_end].cumsum().to_numpy()
    if remaining_attack_cum.size > 0 and remaining_attack_cum[-1] > 0:
        target_attacks = remaining_attack_cum[-1] // 2
        relative_pivot = int((remaining_attack_cum >= target_attacks).argmax())  # first index meeting target
        mid = remaining_start + relative_pivot
    else:
        mid = remaining_start + (remaining_end - remaining_start) // 2
    labeled_val_range = (remaining_start, mid)
    final_test_range = (mid, remaining_end)

    # Sanity print attack mass in each.
    a_lv = int(df['attack'].iloc[labeled_val_range[0]:labeled_val_range[1]].sum())
    a_ft = int(df['attack'].iloc[final_test_range[0]:final_test_range[1]].sum())
    n_lv = labeled_val_range[1] - labeled_val_range[0]
    n_ft = final_test_range[1] - final_test_range[0]
    print(f'[build_test_split] labeled-val: rows [{labeled_val_range[0]:,}, {labeled_val_range[1]:,})  '
          f'n={n_lv:,}  attack={a_lv:,}  ({100*a_lv/max(1,n_lv):.2f}%)')
    print(f'[build_test_split] final-test : rows [{final_test_range[0]:,}, {final_test_range[1]:,})  '
          f'n={n_ft:,}  attack={a_ft:,}  ({100*a_ft/max(1,n_ft):.2f}%)')

    if a_lv == 0:
        print('[build_test_split] WARN: labeled-val has 0 attacks — alpha selection will be ill-defined.', file=sys.stderr)
    if a_ft == 0:
        print('[build_test_split] WARN: final-test has 0 attacks — F1 will be undefined.', file=sys.stderr)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump({
            'test_csv': str(test_csv),
            'n_rows': n_rows,
            'n_attack': n_attack,
            'C_row_range': (0, c_end_pos),
            'C_attack_zero_indices': c_indices,
            'labeled_val_range': list(labeled_val_range),
            'final_test_range': list(final_test_range),
            'c_target_rows': c_target,
            'c_target_fraction': (None if args.c_target_rows is not None
                                   else float(args.c_target_fraction)),
            'n_nominal_in_test': n_nominal,
        }, f, indent=2)
    print(f'[build_test_split] wrote {out}')


if __name__ == '__main__':
    main()
