"""Q2-(1): Temporal sharpening of UQ signals.

Replace each per-timestep UQ signal s(t) with its causal first difference
Δ_W(t) = s(t) − s(t−W) for several W values, then re-run the marginal,
PTaRP, and permutation analyses from `analyze_uq_attack_association.py`.

Tests the hypothesis that the loose alignment of U_par / U_str / U_dist
(L_95 > 1000 windows at level) is dominated by slow regime drift rather
than per-attack onset — i.e., the underlying *change* is more
attack-aligned than the *level*.

Operates entirely on cached `arrays.npz`. ~5 minutes CPU-only.

Outputs `sharpen_sweep.csv` with one row per (signal, W):
  - marginal AUROC
  - best (L, tau)
  - PTaRP at best
  - L_95 = smallest L at which PTaRP ≥ 0.95 × PTaRP@L_max
  - permutation p-value at best (L, tau)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from analyze_uq_attack_association import (
    per_sensor_zscore, renorm_aggregate,
    attack_runs, union_runs, ptarp_one,
)


def causal_diff(x, W):
    """Causal first difference along axis 0: y[t] = x[t] - x[t-W].
    Edge pad: y[:W] = 0. Returns same shape and dtype as x.
    """
    if W <= 0:
        return x
    y = np.zeros_like(x)
    y[W:] = x[W:] - x[:-W]
    return y


def build_signals_for_W(d, W):
    """Reproduces analyze_uq_attack_association.build_signals with a Δ_W
    step inserted between per-component zscore and aggregation. W=0
    reduces to the unsharpened baseline.
    """
    label = d['test_attack_label'].astype(np.int8)

    # Per-component z-scoring on un-aggregated arrays (matches baseline).
    U_par_z = per_sensor_zscore(d['test_U_par'], label)            # (T, V)
    U_str_z = per_sensor_zscore(d['test_U_str'], label)            # (T, E)
    sigma2_ale_z = per_sensor_zscore(d['test_sigma2_ale'], label)  # (T, V)
    U_dist_z = per_sensor_zscore(d['test_U_dist'][:, None], label)[:, 0]  # (T,)

    # Apply causal first difference on the per-component zscored arrays.
    U_par_d = causal_diff(U_par_z, W)
    U_str_d = causal_diff(U_str_z, W)
    sigma2_ale_d = causal_diff(sigma2_ale_z, W)
    U_dist_d = causal_diff(U_dist_z, W)

    raw = {
        'U_par_max_v': U_par_d.max(axis=1),
        'U_par_mean_v': U_par_d.mean(axis=1),
        'U_str_mean_e': U_str_d.mean(axis=1),
        'U_dist': U_dist_d,
        'sigma_ale_max_v': sigma2_ale_d.max(axis=1),
        'sigma_ale_mean_v': sigma2_ale_d.mean(axis=1),
    }
    return {k: renorm_aggregate(v, label) for k, v in raw.items()}


def evaluate_signal(s, label, runs, mass_mask, L_grid, tau_grid):
    """Return DataFrame of PTaRP at each (L, tau), plus marginal stats."""
    T = label.shape[0]
    rows = []
    for L in L_grid:
        for tau in tau_grid:
            spike = s > tau
            d, r, p, ptarp = ptarp_one(spike, label, runs, L, mass_mask)
            rows.append({'L': L, 'tau': tau, 'n_spikes': int(spike.sum()),
                         'ptard': d, 'ptarr': r, 'ptap': p, 'ptarp': ptarp})
    return pd.DataFrame(rows)


def compute_L95(grid_df, L_max):
    """Smallest L at which max-over-tau PTaRP at that L is >= 0.95 *
    max-over-tau PTaRP at L_max. Returns L (int) or L_max + 1 sentinel
    if not satisfied. We pick best tau per L (not a fixed tau) because
    the optimal tau itself may shift with L.
    """
    per_L = grid_df.groupby('L')['ptarp'].max().sort_index()
    if L_max not in per_L.index:
        L_max = per_L.index.max()
    ceiling = float(per_L.loc[L_max])
    threshold = 0.95 * ceiling
    qualifying = per_L[per_L >= threshold]
    if len(qualifying) == 0:
        return int(L_max) + 1, ceiling, threshold
    return int(qualifying.index.min()), ceiling, threshold


def permutation_pvalue(s, label, L, tau, mass_mask, observed, N=1000, seed=0):
    rng = np.random.default_rng(seed)
    T = label.shape[0]
    spike = s > tau
    null_vals = np.empty(N, dtype=np.float64)
    for i in range(N):
        offset = int(rng.integers(1, T))
        null_label = np.roll(label, offset)
        null_runs = attack_runs(null_label)
        null_mass = union_runs(null_runs, T)
        _, _, _, null_ptarp = ptarp_one(spike, null_label, null_runs, L,
                                        null_mass)
        null_vals[i] = null_ptarp
    p_value = (np.sum(null_vals >= observed) + 1) / (N + 1)
    return float(p_value), float(null_vals.mean()), float(null_vals.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-arrays',
        default='results/swat_gdeltauq_paper_protocol/0511-222735/arrays.npz',
    )
    parser.add_argument('-out_root',
                        default='results/uq_attack_assoc/sharpen')
    parser.add_argument('-W_grid', type=int, nargs='+',
                        default=[0, 1, 3, 5, 10, 30])
    parser.add_argument('-L_grid', type=int, nargs='+',
                        default=[0, 3, 10, 30, 100, 200, 500, 1000])
    parser.add_argument('-tau_grid', type=float, nargs='+',
                        default=[0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument('-permutations', type=int, default=1000)
    args = parser.parse_args()

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    out_dir = Path(args.out_root) / datestr
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output: {out_dir}', flush=True)

    print(f'loading {args.arrays}', flush=True)
    d = np.load(args.arrays)
    label = d['test_attack_label'].astype(np.int8)
    T = label.shape[0]
    runs = attack_runs(label)
    mass_mask = union_runs(runs, T)
    print(f'T={T}, attack_rate={label.mean():.4f}, runs={len(runs)}',
          flush=True)

    L_max = max(args.L_grid)
    rows = []
    all_grids = []

    for W in args.W_grid:
        print(f'\n== W = {W} ==', flush=True)
        signals = build_signals_for_W(d, W)
        for name, s in signals.items():
            # Marginal stats
            s_a = s[label == 1]
            s_n = s[label == 0]
            mu_a, mu_n = float(s_a.mean()), float(s_n.mean())
            lift = mu_a / mu_n if abs(mu_n) > 1e-12 else float('nan')
            ks = ks_2samp(s_a, s_n, alternative='two-sided')
            # AUROC: handle constant-signal edge case (e.g., W>=T)
            try:
                auroc = float(roc_auc_score(label, s))
            except ValueError:
                auroc = float('nan')

            grid = evaluate_signal(s, label, runs, mass_mask,
                                   args.L_grid, args.tau_grid)
            grid['signal'] = name
            grid['W'] = W
            all_grids.append(grid)

            # Best (L, tau) by PTaRP
            best_row = grid.iloc[grid['ptarp'].idxmax()].to_dict()
            best_L = int(best_row['L'])
            best_tau = float(best_row['tau'])
            best_ptarp = float(best_row['ptarp'])

            # L_95 saturation
            L_95, ceiling, thresh95 = compute_L95(grid, L_max)

            # Permutation at best (L, tau)
            p_value, null_mean, null_std = permutation_pvalue(
                s, label, best_L, best_tau, mass_mask, best_ptarp,
                N=args.permutations,
            )

            print(f'  {name:18s} AUROC={auroc:.4f} '
                  f'best(L={best_L:4d}, tau={best_tau:.1f}) '
                  f'PTaRP={best_ptarp:.4f} L_95={L_95:4d} '
                  f'p={p_value:.4f}', flush=True)

            rows.append({
                'W': W,
                'signal': name,
                'mu_attack': mu_a,
                'mu_nominal': mu_n,
                'lift_ratio': lift,
                'ks_stat': float(ks.statistic),
                'ks_pvalue': float(ks.pvalue),
                'auroc': auroc,
                'best_L': best_L,
                'best_tau': best_tau,
                'ptarp_best': best_ptarp,
                'ceiling_Lmax': ceiling,
                'L_95': L_95,
                'p_value': p_value,
                'null_mean': null_mean,
                'null_std': null_std,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'sharpen_sweep.csv', index=False)

    all_grid_df = pd.concat(all_grids, ignore_index=True)
    all_grid_df.to_csv(out_dir / 'sharpen_ptarp_grid.csv', index=False)

    # Pivot views
    L95_pivot = df.pivot(index='signal', columns='W', values='L_95')
    auroc_pivot = df.pivot(index='signal', columns='W', values='auroc')
    ptarp_pivot = df.pivot(index='signal', columns='W', values='ptarp_best')
    L95_pivot.to_csv(out_dir / 'L95_vs_W.csv')
    auroc_pivot.to_csv(out_dir / 'auroc_vs_W.csv')
    ptarp_pivot.to_csv(out_dir / 'ptarp_vs_W.csv')

    # SUMMARY.md
    fmt = lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)
    summary = [
        f'# Temporal sharpening sweep: {datestr}',
        '',
        f'Inputs: `{args.arrays}`',
        f'W_grid: {args.W_grid}',
        f'L_grid: {args.L_grid}',
        f'tau_grid: {args.tau_grid}',
        f'permutations: {args.permutations}',
        '',
        '## L_95 alignment slop (smaller = tighter alignment)',
        '',
        L95_pivot.to_string(),
        '',
        '## AUROC',
        '',
        auroc_pivot.to_string(float_format=fmt),
        '',
        '## PTaRP at best (L, tau) per (signal, W)',
        '',
        ptarp_pivot.to_string(float_format=fmt),
        '',
        '## Key findings',
        '',
    ]

    # Find signals where sharpening helped
    best_per_signal = (df.loc[df.groupby('signal')['ptarp_best'].idxmax()]
                         .sort_values('ptarp_best', ascending=False))
    summary.append('### Best W per signal by PTaRP')
    summary.append('')
    summary.append(best_per_signal[['signal', 'W', 'auroc', 'best_L',
                                    'best_tau', 'ptarp_best', 'L_95',
                                    'p_value']].to_string(
        index=False, float_format=fmt))
    summary.append('')

    # Did any W reduce L_95?
    baseline = df[df['W'] == 0].set_index('signal')['L_95']
    sharpened_min = df[df['W'] > 0].groupby('signal')['L_95'].min()
    delta = (sharpened_min - baseline).rename('delta_L95')
    summary.append('### Δ L_95 (best sharpened − W=0)')
    summary.append('')
    summary.append(pd.DataFrame({
        'L_95@W=0': baseline,
        'min_L_95@W>0': sharpened_min,
        'delta': delta,
    }).to_string(float_format=fmt))
    summary.append('')

    (out_dir / 'SUMMARY.md').write_text('\n'.join(summary))
    print(f'\nSUMMARY.md → {out_dir}/SUMMARY.md', flush=True)
    print('=== TEMPORAL SHARPENING DONE ===', flush=True)


if __name__ == '__main__':
    main()
