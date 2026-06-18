"""Partition data/{dataset}/train.csv chronologically into three slices.

Layout: [0, 70%) -> GDN train; [70%, 80%) -> GDN val / G-DeltaUQ calibration /
conformal q_v; [80%, 100%) -> aleatoric-head train. Output JSON has row ranges
(in raw-row indices into train.csv) so downstream consumers can build their own
TimeDataset subsets.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='swat')
    parser.add_argument('-train_ratio', type=float, default=0.70)
    parser.add_argument('-val_ratio', type=float, default=0.10)
    parser.add_argument('-aleatoric_ratio', type=float, default=0.20)
    parser.add_argument('-out_path', type=str,
                        default='data/swat/gdeltauq_split.json')
    args = parser.parse_args()

    eps = 1e-6
    total = args.train_ratio + args.val_ratio + args.aleatoric_ratio
    if abs(total - 1.0) > eps:
        raise ValueError(f'ratios must sum to 1, got {total}')

    train_csv_path = f'./data/{args.dataset}/train.csv'
    df = pd.read_csv(train_csv_path, sep=',', index_col=0)
    if 'attack' in df.columns:
        df = df.drop(columns=['attack'])
    N = len(df)

    n_train = int(round(N * args.train_ratio))
    n_val = int(round(N * args.val_ratio))
    n_ale = N - n_train - n_val
    train_rows = (0, n_train)
    val_rows = (n_train, n_train + n_val)
    aleatoric_rows = (n_train + n_val, N)

    payload = {
        'dataset': args.dataset,
        'total_rows': N,
        'train_rows': list(train_rows),
        'val_rows': list(val_rows),
        'aleatoric_rows': list(aleatoric_rows),
        'ratios': {
            'train': args.train_ratio,
            'val': args.val_ratio,
            'aleatoric': args.aleatoric_ratio,
        },
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(payload, f, indent=2)
    print(f'split written to {out_path}: {payload}', flush=True)


if __name__ == '__main__':
    main()
