"""Thread B: UQ-vs-attack-window association analysis for G-DeltaUQ on SWaT.

Four sections, all post-hoc on cached arrays.npz from
`results/swat_gdeltauq_paper_protocol/0511-222735/`:

  §1 Marginal coincidence stats (lift, KS test, AUROC).
  §2 PTaRP-adapted temporal-tolerance alignment metric (Kang et al. 2026,
     adapted: lead-time reward replaced with symmetric tolerance L).
  §3 Permutation test via circular shift of the attack label.
  §4 Hybrid residual + UQ detector under OR and AND-gate rules.

Outputs land under `results/uq_attack_assoc/<datestr>/`.

Normalization choice: per-sensor median + MAD-from-IQR computed on
test-nominal timesteps (label == 0). The val slice doesn't carry the U_*
signals in cached arrays.npz, and this avoids re-running inference. The
mild scale-leak is acceptable: median/IQR over the dominant nominal mass
is structurally robust, and we always normalize the AGGREGATE for the
per-signal z-score.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.data import get_attack_interval


# ---------------------------------------------------------------------------
# Signal construction


def per_sensor_zscore(x, label, eps=1e-6):
    """Robust per-column z-score: (x - median) / IQR, fit on rows where
    label == 0. Returns same shape as x.
    """
    nominal = x[label == 0]
    med = np.median(nominal, axis=0)
    q25 = np.quantile(nominal, 0.25, axis=0)
    q75 = np.quantile(nominal, 0.75, axis=0)
    iqr = q75 - q25
    return (x - med) / (iqr + eps)


def renorm_aggregate(s, label, eps=1e-6):
    """Re-z-score a 1D aggregate signal on its test-nominal statistics. Without
    this, mean-of-N aggregates have stddev shrunk by ~1/sqrt(N), making a
    universal tau grid (0.5-3 z-units) wildly miscalibrated across aggregates.
    """
    s = s.astype(np.float64)
    nominal = s[label == 0]
    med = float(np.median(nominal))
    q25 = float(np.quantile(nominal, 0.25))
    q75 = float(np.quantile(nominal, 0.75))
    iqr = q75 - q25
    return (s - med) / (iqr + eps)


def build_signals(d):
    """Build the 6 aggregate per-timestep UQ signals + the residual score.

    Returns dict {signal_name: (T,)} and a separate (T,) residual_score
    and (T,) attack_label.

    Per-component z-norm happens first (per sensor / per edge), then we
    aggregate, then we RE-z-score the aggregate so the universal tau grid
    {0.5, 1, 1.5, 2, 3} is interpretable across all 6 signals.
    """
    label = d['test_attack_label'].astype(np.int8)
    T = label.shape[0]

    U_par_z = per_sensor_zscore(d['test_U_par'], label)            # (T, V)
    U_str_z = per_sensor_zscore(d['test_U_str'], label)            # (T, E)
    sigma2_ale_z = per_sensor_zscore(d['test_sigma2_ale'], label)  # (T, V)
    U_dist_z = per_sensor_zscore(d['test_U_dist'][:, None], label)[:, 0]  # (T,)

    raw = {
        'U_par_max_v': U_par_z.max(axis=1),
        'U_par_mean_v': U_par_z.mean(axis=1),
        'U_str_mean_e': U_str_z.mean(axis=1),
        'U_dist': U_dist_z,
        'sigma_ale_max_v': sigma2_ale_z.max(axis=1),
        'sigma_ale_mean_v': sigma2_ale_z.mean(axis=1),
    }
    signals = {k: renorm_aggregate(v, label) for k, v in raw.items()}

    residual = d['full_scores'].max(axis=0).astype(np.float64)
    return signals, residual, label, T


# ---------------------------------------------------------------------------
# §1 Marginal coincidence


def section1_marginal(signals, label, out_dir):
    rows = []
    for name, s in signals.items():
        s_attack = s[label == 1]
        s_nominal = s[label == 0]
        mu_a, mu_n = float(s_attack.mean()), float(s_nominal.mean())
        lift = mu_a / mu_n if mu_n != 0 else float('inf')
        ks = ks_2samp(s_attack, s_nominal, alternative='two-sided')
        auroc = float(roc_auc_score(label, s))
        rows.append({
            'signal': name,
            'mu_attack': mu_a,
            'mu_nominal': mu_n,
            'lift_ratio': lift,
            'ks_stat': float(ks.statistic),
            'ks_pvalue': float(ks.pvalue),
            'auroc': auroc,
        })
        # Density plot
        fig, ax = plt.subplots(figsize=(6, 4))
        lo, hi = np.quantile(s, 0.005), np.quantile(s, 0.995)
        bins = np.linspace(lo, hi, 60)
        ax.hist(s_nominal, bins=bins, density=True, alpha=0.5, label='nominal',
                color='C0')
        ax.hist(s_attack, bins=bins, density=True, alpha=0.5, label='attack',
                color='C3')
        ax.set_xlabel(f'z-normalized {name}')
        ax.set_ylabel('density')
        ax.set_title(f'{name}: lift={lift:.2f}, KS p={ks.pvalue:.2e}, '
                     f'AUROC={auroc:.3f}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f'marginal_{name}.png', dpi=110)
        plt.close(fig)

    df = pd.DataFrame(rows).sort_values('auroc', ascending=False)
    df.to_csv(out_dir / 'marginal.csv', index=False)
    return df


# ---------------------------------------------------------------------------
# §2 PTaRP-adapted temporal-tolerance alignment


def attack_runs(label):
    """Return list of (start, end_inclusive) tuples."""
    return get_attack_interval(label.tolist())


def union_extended(runs, T, L):
    """Mask of size (T,) marking the tolerance-extended union of runs."""
    m = np.zeros(T, dtype=bool)
    for a, b in runs:
        lo = max(0, a - L)
        hi = min(T - 1, b + L)
        m[lo:hi + 1] = True
    return m


def union_runs(runs, T):
    m = np.zeros(T, dtype=bool)
    for a, b in runs:
        m[a:b + 1] = True
    return m


def ptarp_one(spike, label, runs, L, attack_mass_mask):
    """Compute PTaR^d, PTaR^r, PTaP, PTaRP for one (L, threshold) point.
    `spike`: bool array (T,) of UQ spike timesteps after thresholding.
    `runs`: list of (start, end) attack runs.
    `attack_mass_mask`: bool (T,) = union of un-extended runs.
    """
    T = spike.size
    if len(runs) == 0 or not attack_mass_mask.any():
        return 0.0, 0.0, 0.0, 0.0
    ext = union_extended(runs, T, L)

    # PTaR^d: segment-level detection rate
    d_hits = 0
    for a, b in runs:
        lo, hi = max(0, a - L), min(T - 1, b + L)
        if spike[lo:hi + 1].any():
            d_hits += 1
    ptard = d_hits / len(runs)

    # PTaR^r: within-segment coverage (un-extended attack mass)
    in_attack_spikes = spike & attack_mass_mask
    ptarr = in_attack_spikes.sum() / attack_mass_mask.sum()

    # PTaP: fraction of spikes that land in the extended union
    n_spikes = int(spike.sum())
    if n_spikes == 0:
        ptap = 0.0
    else:
        ptap = (spike & ext).sum() / n_spikes

    alpha = 0.5
    detection = alpha * ptard + (1 - alpha) * ptarr
    if (detection + ptap) > 0:
        ptarp = 2 * detection * ptap / (detection + ptap)
    else:
        ptarp = 0.0
    return ptard, ptarr, ptap, ptarp


def section2_ptarp(signals, label, out_dir,
                   L_grid=(0, 3, 10, 30, 100, 200, 500, 1000),
                   tau_grid=(0.5, 1.0, 1.5, 2.0, 3.0)):
    runs = attack_runs(label)
    T = label.shape[0]
    mass_mask = union_runs(runs, T)

    rows = []
    for name, s in signals.items():
        for L in L_grid:
            for tau in tau_grid:
                spike = s > tau
                d, r, p, ptarp = ptarp_one(spike, label, runs, L, mass_mask)
                rows.append({
                    'signal': name,
                    'L': L,
                    'tau': tau,
                    'n_spikes': int(spike.sum()),
                    'ptard': d,
                    'ptarr': r,
                    'ptap': p,
                    'ptarp': ptarp,
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'ptarp_grid.csv', index=False)

    # Best per signal
    best = df.loc[df.groupby('signal')['ptarp'].idxmax()].copy()
    best = best.sort_values('ptarp', ascending=False)
    best.to_csv(out_dir / 'ptarp_best_per_signal.csv', index=False)
    return df, best


# ---------------------------------------------------------------------------
# §3 Permutation test


def section3_permutation(signals, label, best_per_signal, out_dir, N=1000,
                         seed=0):
    rng = np.random.default_rng(seed)
    T = label.shape[0]
    rows = []
    for _, row in best_per_signal.iterrows():
        name = row['signal']
        L = int(row['L'])
        tau = float(row['tau'])
        observed = float(row['ptarp'])
        s = signals[name]
        spike = s > tau

        null_vals = np.empty(N, dtype=np.float64)
        for i in range(N):
            offset = int(rng.integers(1, T))  # exclude 0 shift
            null_label = np.roll(label, offset)
            null_runs = attack_runs(null_label)
            null_mass = union_runs(null_runs, T)
            _, _, _, null_ptarp = ptarp_one(
                spike, null_label, null_runs, L, null_mass,
            )
            null_vals[i] = null_ptarp

        # Phipson-Smyth p-value
        p_value = (np.sum(null_vals >= observed) + 1) / (N + 1)
        rows.append({
            'signal': name,
            'L': L,
            'tau': tau,
            'observed_ptarp': observed,
            'null_mean': float(null_vals.mean()),
            'null_std': float(null_vals.std()),
            'null_max': float(null_vals.max()),
            'p_value': float(p_value),
            'N_permutations': N,
        })

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(null_vals, bins=40, color='C0', alpha=0.7)
        ax.axvline(observed, color='C3', lw=2,
                   label=f'observed = {observed:.4f}')
        ax.set_xlabel('PTaRP under circular-shift null')
        ax.set_ylabel('count')
        ax.set_title(f'{name}: L={L}, tau={tau}, p={p_value:.4f} (N={N})')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f'permutation_{name}.png', dpi=110)
        plt.close(fig)

    df = pd.DataFrame(rows).sort_values('p_value')
    df.to_csv(out_dir / 'permutation.csv', index=False)
    return df


# ---------------------------------------------------------------------------
# §4 Hybrid residual + UQ detector


def f1_pr(pred, label):
    pred = pred.astype(np.int8)
    label = label.astype(np.int8)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r, tp, fp, fn, tn


def section4_hybrid(signals, residual, label, out_dir,
                    n_taur=80, n_taur_prime=30,
                    tau_s_grid=(1.0, 1.5, 2.0, 3.0, 5.0)):
    # Wider, finer residual quantile grids. Rank-based threshold matching
    # util.data.eval_scores would also work; quantile-on-values is close
    # enough at 80 points.
    qs = np.linspace(0.5, 0.9995, n_taur)
    tau_r_vals = np.quantile(residual, qs)
    qs2 = np.linspace(0.05, 0.7, n_taur_prime)
    tau_r_prime_vals = np.quantile(residual, qs2)

    # Residual-only baseline: best F1 over tau_r grid
    best_resid = {'F1': 0.0}
    for tr, q in zip(tau_r_vals, qs):
        pred = (residual > tr).astype(np.int8)
        f1, p, r, *_ = f1_pr(pred, label)
        if f1 > best_resid['F1']:
            best_resid = {'F1': f1, 'P': p, 'R': r, 'tau_r': float(tr),
                          'quantile': float(q)}
    print(f'  residual-only baseline: F1={best_resid["F1"]:.4f} '
          f'P={best_resid["P"]:.4f} R={best_resid["R"]:.4f}', flush=True)

    rows = []

    # OR rule sweep
    for name, s in signals.items():
        for tau_s in tau_s_grid:
            spike = (s > tau_s)
            for tr, q in zip(tau_r_vals, qs):
                pred = ((residual > tr) | spike).astype(np.int8)
                f1, p, r, tp, fp, fn, tn = f1_pr(pred, label)
                rows.append({
                    'rule': 'OR',
                    'signal': name,
                    'tau_r': float(tr),
                    'tau_r_q': float(q),
                    'tau_r_prime': float('nan'),
                    'tau_s': float(tau_s),
                    'F1': f1, 'P': p, 'R': r,
                    'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                })

    # AND-gate sweep (uses lowered residual threshold)
    for name, s in signals.items():
        for tau_s in tau_s_grid:
            spike = (s > tau_s)
            for tr_prime, q in zip(tau_r_prime_vals, qs2):
                pred = ((residual > tr_prime) & spike).astype(np.int8)
                f1, p, r, tp, fp, fn, tn = f1_pr(pred, label)
                rows.append({
                    'rule': 'AND_lowerthr',
                    'signal': name,
                    'tau_r': float('nan'),
                    'tau_r_q': float('nan'),
                    'tau_r_prime': float(tr_prime),
                    'tau_s': float(tau_s),
                    'F1': f1, 'P': p, 'R': r,
                    'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'hybrid.csv', index=False)

    best_idx = df['F1'].idxmax()
    best_row = df.loc[best_idx].to_dict()

    with (out_dir / 'best_hybrid.json').open('w') as f:
        json.dump({
            'residual_only_baseline': best_resid,
            'best_hybrid': best_row,
            'lift_vs_residual_only': best_row['F1'] - best_resid['F1'],
        }, f, indent=2, default=lambda x: None if pd.isna(x) else x)

    return df, best_row, best_resid


# ---------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-arrays',
        default='results/swat_gdeltauq_paper_protocol/0511-222735/arrays.npz',
    )
    parser.add_argument('-out_root', default='results/uq_attack_assoc')
    parser.add_argument('-permutations', type=int, default=1000)
    parser.add_argument('-L_grid', type=int, nargs='+',
                        default=[0, 3, 10, 30, 100, 200, 500, 1000])
    args = parser.parse_args()

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    out_dir = Path(args.out_root) / datestr
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output dir: {out_dir}', flush=True)

    print(f'loading {args.arrays}', flush=True)
    d = np.load(args.arrays)
    signals, residual, label, T = build_signals(d)
    print(f'T={T}, attack_rate={label.mean():.4f}, '
          f'attack_runs={len(attack_runs(label))}', flush=True)
    print(f'signals: {list(signals.keys())}', flush=True)

    print('§1 marginal coincidence ...', flush=True)
    marg = section1_marginal(signals, label, out_dir)
    print(marg.to_string(index=False), flush=True)

    print('§2 PTaRP-adapted alignment ...', flush=True)
    _, best_per = section2_ptarp(signals, label, out_dir,
                                 L_grid=tuple(args.L_grid))
    print('  best PTaRP per signal:', flush=True)
    print(best_per[['signal', 'L', 'tau', 'ptard', 'ptarr', 'ptap',
                    'ptarp']].to_string(index=False), flush=True)

    print(f'§3 permutation test (N={args.permutations}) ...', flush=True)
    perm = section3_permutation(signals, label, best_per, out_dir,
                                N=args.permutations)
    print(perm.to_string(index=False), flush=True)

    print('§4 hybrid residual+UQ detector ...', flush=True)
    hyb_df, hyb_best, resid_baseline = section4_hybrid(
        signals, residual, label, out_dir,
    )
    print(f'  best hybrid: rule={hyb_best["rule"]} signal={hyb_best["signal"]} '
          f'F1={hyb_best["F1"]:.4f} P={hyb_best["P"]:.4f} '
          f'R={hyb_best["R"]:.4f}', flush=True)
    print(f'  vs residual-only baseline F1={resid_baseline["F1"]:.4f} '
          f'(lift = {hyb_best["F1"] - resid_baseline["F1"]:+.4f})', flush=True)
    print(f'  vs cheap-sweep best 0.7855 (lift = '
          f'{hyb_best["F1"] - 0.7855:+.4f})', flush=True)

    # SUMMARY.md
    summary_lines = [
        f'# UQ-vs-attack association: {datestr}',
        '',
        f'Inputs: `{args.arrays}`  ',
        f'Output: `{out_dir}`  ',
        f'T={T}, attack rate={label.mean():.4f}, '
        f'attack runs={len(attack_runs(label))}, '
        f'permutations={args.permutations}',
        '',
        '## §1 Marginal coincidence',
        '',
        marg.to_string(index=False, float_format=lambda x: f'{x:.4f}'),
        '',
        '## §2 Best PTaRP per signal',
        '',
        best_per[['signal', 'L', 'tau', 'ptard', 'ptarr', 'ptap',
                  'ptarp']].to_string(index=False, float_format=lambda x: f'{x:.4f}'),
        '',
        '## §3 Permutation test (circular shift)',
        '',
        perm[['signal', 'L', 'tau', 'observed_ptarp', 'null_mean',
              'null_std', 'p_value']].to_string(index=False, float_format=lambda x: f'{x:.4f}'),
        '',
        '## §4 Hybrid residual + UQ detector',
        '',
        f'**Residual-only baseline (cached top-1, smooth-3)**: F1='
        f'{resid_baseline["F1"]:.4f}, P={resid_baseline["P"]:.4f}, '
        f'R={resid_baseline["R"]:.4f}',
        '',
        f'**Best hybrid**: rule={hyb_best["rule"]}, signal={hyb_best["signal"]}, '
        f'F1={hyb_best["F1"]:.4f}, P={hyb_best["P"]:.4f}, '
        f'R={hyb_best["R"]:.4f}',
        '',
        f'- Lift vs residual-only: '
        f'**{hyb_best["F1"] - resid_baseline["F1"]:+.4f}**',
        f'- Lift vs cheap-sweep best 0.7855: '
        f'**{hyb_best["F1"] - 0.7855:+.4f}**',
        '',
        '### Top-10 hybrid configs',
        '',
        hyb_df.nlargest(10, 'F1')[['rule', 'signal', 'tau_r', 'tau_r_prime',
                                   'tau_s', 'F1', 'P', 'R']].to_string(
            index=False, float_format=lambda x: f'{x:.4f}'),
        '',
        '## Notes',
        '',
        '- Per-sensor z-normalization (median + IQR) fit on test-nominal '
        '(label==0). Val slice lacks cached U_*; this approach is robust to '
        'the ~12% attack contamination since median/IQR ignore tail mass.',
        '- Tolerance L is **symmetric** (`[a-L, b+L]`); not directional '
        'lead-time (Kang et al. drop point — we test association, not '
        'precursor forecasting).',
        '- Circular-shift null preserves both marginal attack rate and '
        'run-length distribution; rejects purely under temporal coincidence.',
    ]
    (out_dir / 'SUMMARY.md').write_text('\n'.join(summary_lines))
    print(f'\nSUMMARY.md written to {out_dir}/SUMMARY.md', flush=True)


if __name__ == '__main__':
    main()
