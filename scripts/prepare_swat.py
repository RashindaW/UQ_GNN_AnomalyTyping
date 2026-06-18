"""Build data/swat/{train.csv,test.csv,list.txt} from the SWaT.A1 & A2 (Dec 2015) xlsx files.

The two source files must be downloaded from iTrust@SUTD (see README) and dropped into
SWaT/manual/ first.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
NORMAL_PATH = REPO_ROOT / 'SWaT' / 'manual' / 'SWaT_Dataset_Normal_v1.xlsx'
ATTACK_PATH = REPO_ROOT / 'SWaT' / 'manual' / 'SWaT_Dataset_Attack_v0.xlsx'
OUT_DIR = REPO_ROOT / 'data' / 'swat'

SHAREPOINT_URL = (
    'https://sutdapac-my.sharepoint.com/:f:/g/personal/itrust_sutd_edu_sg/'
    'EijnugJpDP1Km9yR1enq4igB-02WFY46LCcLA4tqjmMB3g?e=oSKGYh'
)


def fail_missing(path: Path) -> None:
    print(f'[prepare_swat] missing: {path}')
    print(
        '[prepare_swat] Drop the original SWaT.A1 & A2 (Dec 2015) xlsx files there.\n'
        f'[prepare_swat] Source (browser auth required): {SHAREPOINT_URL}\n'
        '[prepare_swat] Required files:\n'
        '                 SWaT_Dataset_Normal_v1.xlsx\n'
        '                 SWaT_Dataset_Attack_v0.xlsx'
    )
    sys.exit(1)


def load_swat_xlsx(path: Path) -> pd.DataFrame:
    # The A1/A2 sheets have a one-line preamble; the real header is on the second row.
    df = pd.read_excel(path, engine='openpyxl', header=1)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def split_columns(df: pd.DataFrame):
    label_col = None
    for cand in ('Normal/Attack', 'Normal / Attack', 'Normal/Attack '):
        if cand in df.columns:
            label_col = cand
            break

    drop_cols = set()
    for cand in ('Timestamp', ' Timestamp', 'Time'):
        if cand in df.columns:
            drop_cols.add(cand)
    if label_col is not None:
        drop_cols.add(label_col)

    sensor_cols = [c for c in df.columns if c not in drop_cols]
    return sensor_cols, label_col


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--downsample', type=int, default=1,
        help='Aggregate every N rows into one. The GDN paper uses 10. With '
             '--downsample-mode median, sensor columns are aggregated by median '
             'and the attack column by mode (most common). With stride (legacy), '
             'every Nth row is kept (no aggregation).',
    )
    parser.add_argument(
        '--downsample-mode', type=str, default='stride',
        choices=['stride', 'median'],
        help='median = paper-faithful (groupby median + mode); stride = legacy.',
    )
    parser.add_argument(
        '--stabilization-trim', type=int, default=0,
        help='Drop the first N rows of each split AFTER downsampling. Paper §4 '
             'reports ~5h SWaT stabilisation; the GDN paper uses 2160 (36 min @ '
             '1 Hz). After 10× median downsampling, equivalent value is 216.',
    )
    args = parser.parse_args()

    if not NORMAL_PATH.exists():
        fail_missing(NORMAL_PATH)
    if not ATTACK_PATH.exists():
        fail_missing(ATTACK_PATH)

    print(f'[prepare_swat] reading {NORMAL_PATH.name}')
    normal_df = load_swat_xlsx(NORMAL_PATH)
    print(f'[prepare_swat]   rows={len(normal_df)} cols={len(normal_df.columns)}')

    print(f'[prepare_swat] reading {ATTACK_PATH.name}')
    attack_df = load_swat_xlsx(ATTACK_PATH)
    print(f'[prepare_swat]   rows={len(attack_df)} cols={len(attack_df.columns)}')

    sensor_cols, normal_label_col = split_columns(normal_df)
    sensor_cols_a, attack_label_col = split_columns(attack_df)

    if sensor_cols != sensor_cols_a:
        only_in_normal = [c for c in sensor_cols if c not in sensor_cols_a]
        only_in_attack = [c for c in sensor_cols_a if c not in sensor_cols]
        print(
            '[prepare_swat] WARN sensor columns differ between Normal and Attack files; '
            'using their intersection in the order from Normal.'
        )
        if only_in_normal:
            print(f'[prepare_swat]   only in Normal: {only_in_normal}')
        if only_in_attack:
            print(f'[prepare_swat]   only in Attack: {only_in_attack}')
        sensor_cols = [c for c in sensor_cols if c in sensor_cols_a]

    if attack_label_col is None:
        print('[prepare_swat] ERROR: no Normal/Attack column found in attack file.')
        sys.exit(2)

    train_out = normal_df[sensor_cols].reset_index(drop=True)

    test_sensors = attack_df[sensor_cols].reset_index(drop=True)
    raw_labels = attack_df[attack_label_col].astype(str).str.strip()
    attack_flag = (raw_labels.str.lower() != 'normal').astype(int)
    test_out = test_sensors.copy()
    test_out['attack'] = attack_flag.values

    if args.downsample > 1:
        n = args.downsample
        if args.downsample_mode == 'stride':
            train_out = train_out.iloc[::n, :].reset_index(drop=True)
            test_out = test_out.iloc[::n, :].reset_index(drop=True)
            print(f'[prepare_swat] downsampled to every {n}th row (stride)')
        else:
            # Paper-faithful: median over each block of N rows for sensor cols,
            # mode (most-common) for the binary `attack` column.
            def _agg(df, has_attack: bool):
                block = (np.arange(len(df)) // n)
                sensor_aggs = {c: 'median' for c in sensor_cols}
                if has_attack:
                    # mode is multi-valued; pick first mode (lowest by argmax of 0/1).
                    out = df.groupby(block).agg({**sensor_aggs,
                                                 'attack': lambda s: int(s.mode().iloc[0])})
                else:
                    out = df.groupby(block).agg(sensor_aggs)
                return out.reset_index(drop=True)
            train_out = _agg(train_out, has_attack=False)
            test_out = _agg(test_out, has_attack=True)
            print(f'[prepare_swat] {n}x median+mode downsample → '
                  f'train={len(train_out)} rows, test={len(test_out)} rows')

    if args.stabilization_trim > 0:
        n_trim = args.stabilization_trim
        train_out = train_out.iloc[n_trim:].reset_index(drop=True)
        test_out = test_out.iloc[n_trim:].reset_index(drop=True)
        print(f'[prepare_swat] stabilisation-trim: dropped first {n_trim} rows '
              f'from train and test')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUT_DIR / 'train.csv'
    test_path = OUT_DIR / 'test.csv'
    list_path = OUT_DIR / 'list.txt'

    train_out.to_csv(train_path)
    test_out.to_csv(test_path)
    list_path.write_text('\n'.join(sensor_cols) + '\n')

    attack_frac = float(test_out['attack'].mean())
    print(
        f'[prepare_swat] wrote\n'
        f'                 {train_path} ({len(train_out)} rows)\n'
        f'                 {test_path} ({len(test_out)} rows, attack_frac={attack_frac:.3f})\n'
        f'                 {list_path} ({len(sensor_cols)} sensors)'
    )


if __name__ == '__main__':
    main()
