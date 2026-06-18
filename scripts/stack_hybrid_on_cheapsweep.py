"""Stack hybrid OR(residual, UQ) onto the cheap-sweep best detector.

The prior `scripts/sweep_eval_gdeltauq.py` chose topk=10, smoothing=0,
extend_W=5, merge_G=3 as the F1-optimal residual-only configuration on the
cached arrays.npz (F1 = 0.7855). The earlier `analyze_uq_attack_association.py`
§4 showed that adding sigma_ale_max_v via an OR rule on top of the **raw**
top-1 residual lifts F1 by +0.0120. This script tests whether that lift
compounds when stacked on top of the cheap-sweep best.

Two OR placements are swept:

  A (OR after post-proc):
      residual_alarm = postproc((agg > tau_r), W, G)
      combined       = residual_alarm | (z_signal > tau_s)
      UQ spike is a separate raw recall channel; post-proc only refines the
      residual.

  B (OR before post-proc):
      raw            = (agg > tau_r) | (z_signal > tau_s)
      combined       = postproc(raw, W, G)
      UQ spikes share the dilation + gap-merge with residual alarms.

Sweeps tau_r over 200 quantiles in [0.5, 0.9999] of the per-timestep top-10
aggregate, tau_s over a small z-unit grid, all 6 z-normalized UQ signals,
and both placements. ~17K configs, runs in <2 min.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from sweep_eval_gdeltauq import (
    build_full_err_scores, topk_aggregate, apply_postproc,
)
from analyze_uq_attack_association import per_sensor_zscore, renorm_aggregate


def build_uq_signals(d):
    label = d['test_attack_label'].astype(np.int8)
    U_par_z = per_sensor_zscore(d['test_U_par'], label)
    U_str_z = per_sensor_zscore(d['test_U_str'], label)
    sigma2_ale_z = per_sensor_zscore(d['test_sigma2_ale'], label)
    U_dist_z = per_sensor_zscore(d['test_U_dist'][:, None], label)[:, 0]
    raw = {
        'U_par_max_v': U_par_z.max(axis=1),
        'U_par_mean_v': U_par_z.mean(axis=1),
        'U_str_mean_e': U_str_z.mean(axis=1),
        'U_dist': U_dist_z,
        'sigma_ale_max_v': sigma2_ale_z.max(axis=1),
        'sigma_ale_mean_v': sigma2_ale_z.mean(axis=1),
    }
    return {k: renorm_aggregate(v, label) for k, v in raw.items()}


def metrics(pred, label):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-arrays',
        default='results/swat_gdeltauq_paper_protocol/0511-222735/arrays.npz',
    )
    parser.add_argument('-out_root',
                        default='results/uq_attack_assoc/stacked')
    parser.add_argument('-topk', type=int, default=10)
    parser.add_argument('-smoothing', type=int, default=0)
    parser.add_argument('-W', type=int, default=5)
    parser.add_argument('-G', type=int, default=3)
    parser.add_argument('-tau_r_n', type=int, default=200)
    parser.add_argument('-tau_r_prime_n', type=int, default=30,
                        help='Number of lowered quantiles for the AND-gate '
                             'residual threshold (tau_r_prime).')
    parser.add_argument('-tau_s_grid', type=float, nargs='+',
                        default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    parser.add_argument('-rules', type=str, nargs='+',
                        default=['OR_A', 'OR_B', 'AND_A', 'AND_B'],
                        help='Subset of {OR_A, OR_B, AND_A, AND_B} to sweep.')
    args = parser.parse_args()

    print(f'loading {args.arrays}', flush=True)
    d = np.load(args.arrays)
    test_mu = d['test_mu_bar']
    test_y = d['test_ground_truth']
    val_mu = d['val_mu_bar']
    val_y = d['val_ground_truth']
    label = d['test_attack_label'].astype(np.int8)
    T = label.shape[0]
    print(f'T={T}, attack_rate={label.mean():.4f}', flush=True)
    print(f'config: topk={args.topk}, smoothing={args.smoothing}, '
          f'extend_W={args.W}, merge_G={args.G}', flush=True)

    print('rebuilding full_scores at smoothing=0 ...', flush=True)
    full_scores = build_full_err_scores(test_mu, test_y, val_mu, val_y,
                                        args.smoothing)
    agg = topk_aggregate(full_scores, topk=args.topk)
    print(f'  agg shape={agg.shape}', flush=True)

    # tau_r grid for OR rules: 200 quantiles in (0.5, 0.9999)
    qs = np.linspace(0.5, 0.9999, args.tau_r_n)
    tau_r_vals = np.quantile(agg, qs)
    # tau_r_prime grid for AND-gate rules: lowered quantiles in (0.05, 0.7)
    qs_prime = np.linspace(0.05, 0.7, args.tau_r_prime_n)
    tau_r_prime_vals = np.quantile(agg, qs_prime)

    # Residual-only baseline reproduction (cheap-sweep best)
    print('reproducing cheap-sweep residual-only baseline ...', flush=True)
    best_resid = {'F1': 0.0}
    for tr, q in zip(tau_r_vals, qs):
        post = apply_postproc((agg > tr).astype(np.int8), args.W, args.G)
        f1, p, r, *_ = metrics(post, label)
        if f1 > best_resid['F1']:
            best_resid = {'F1': f1, 'P': p, 'R': r,
                          'tau_r': float(tr), 'quantile': float(q)}
    print(f'  reproduced F1={best_resid["F1"]:.4f}  P={best_resid["P"]:.4f}  '
          f'R={best_resid["R"]:.4f}  (prior cheap-sweep was 0.7855)',
          flush=True)

    print('building z-normalized UQ signals ...', flush=True)
    uq_signals = build_uq_signals(d)
    print(f'  signals: {list(uq_signals.keys())}', flush=True)

    print('sweeping stacked hybrid grid ...', flush=True)
    rows = []
    or_rules = [r for r in args.rules if r.startswith('OR')]
    and_rules = [r for r in args.rules if r.startswith('AND')]
    n_total = (len(uq_signals) * len(args.tau_s_grid)
               * (len(or_rules) * args.tau_r_n
                  + len(and_rules) * args.tau_r_prime_n))
    print(f'  {n_total} configs '
          f'(OR rules: {or_rules}; AND rules: {and_rules})', flush=True)

    for name, s in uq_signals.items():
        for tau_s in args.tau_s_grid:
            spike = (s > tau_s).astype(np.int8)

            # OR rules: residual at high tau_r OR UQ spike.
            for rule in or_rules:
                for tr, q in zip(tau_r_vals, qs):
                    residual = (agg > tr).astype(np.int8)
                    if rule == 'OR_A':
                        residual_post = apply_postproc(residual, args.W,
                                                       args.G)
                        combined = (residual_post | spike).astype(np.int8)
                    elif rule == 'OR_B':
                        raw_or = (residual | spike).astype(np.int8)
                        combined = apply_postproc(raw_or, args.W, args.G)
                    else:
                        raise ValueError(rule)
                    f1, p, r, tp, fp, fn, tn = metrics(combined, label)
                    rows.append({
                        'signal': name, 'rule': rule,
                        'tau_r': float(tr), 'tau_r_q': float(q),
                        'tau_r_prime': float('nan'),
                        'tau_s': float(tau_s),
                        'F1': f1, 'P': p, 'R': r,
                        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                    })

            # AND rules: residual at LOW tau_r_prime AND UQ spike.
            for rule in and_rules:
                for trp, qp in zip(tau_r_prime_vals, qs_prime):
                    residual = (agg > trp).astype(np.int8)
                    if rule == 'AND_A':
                        # post-proc the lowered residual, then AND with spike
                        residual_post = apply_postproc(residual, args.W,
                                                       args.G)
                        combined = (residual_post & spike).astype(np.int8)
                    elif rule == 'AND_B':
                        # AND first, then post-proc the result
                        raw_and = (residual & spike).astype(np.int8)
                        combined = apply_postproc(raw_and, args.W, args.G)
                    else:
                        raise ValueError(rule)
                    f1, p, r, tp, fp, fn, tn = metrics(combined, label)
                    rows.append({
                        'signal': name, 'rule': rule,
                        'tau_r': float('nan'), 'tau_r_q': float('nan'),
                        'tau_r_prime': float(trp),
                        'tau_s': float(tau_s),
                        'F1': f1, 'P': p, 'R': r,
                        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                    })

    df = pd.DataFrame(rows)
    df = df.sort_values('F1', ascending=False).reset_index(drop=True)

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    out_dir = Path(args.out_root) / datestr
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'stacked_sweep.csv', index=False)

    print('\nTop-15 stacked configs:', flush=True)
    print(df.head(15).to_string(
        index=False, float_format=lambda v: f'{v:.4f}',
        columns=['signal', 'rule', 'tau_r', 'tau_r_prime', 'tau_s',
                 'F1', 'P', 'R', 'TP', 'FP', 'FN'],
    ), flush=True)

    # Best per (signal, rule)
    best_per = (df.loc[df.groupby(['signal', 'rule'])['F1'].idxmax()]
                  .sort_values('F1', ascending=False)
                  .reset_index(drop=True))
    best_per.to_csv(out_dir / 'best_per_signal_rule.csv', index=False)

    print('\nBest per (signal, rule):', flush=True)
    print(best_per.to_string(
        index=False, float_format=lambda v: f'{v:.4f}',
        columns=['signal', 'rule', 'tau_r_q', 'tau_r_prime', 'tau_s',
                 'F1', 'P', 'R'],
    ), flush=True)

    best = df.iloc[0].to_dict()
    lift_baseline = best['F1'] - best_resid['F1']
    print(f'\nBest stacked hybrid: signal={best["signal"]} '
          f'rule={best["rule"]} F1={best["F1"]:.4f} '
          f'P={best["P"]:.4f} R={best["R"]:.4f}', flush=True)
    print(f'  lift vs reproduced cheap-sweep '
          f'{best_resid["F1"]:.4f}: {lift_baseline:+.4f}', flush=True)
    print(f'  lift vs prior cheap-sweep 0.7855: '
          f'{best["F1"] - 0.7855:+.4f}', flush=True)

    with (out_dir / 'best_stacked.json').open('w') as f:
        json.dump({
            'cheap_sweep_baseline_reproduced': best_resid,
            'best_stacked': {k: (None if (isinstance(v, float)
                                          and np.isnan(v)) else v)
                             for k, v in best.items()},
            'lift_vs_baseline': float(lift_baseline),
            'lift_vs_prior_0855': float(best['F1'] - 0.7855),
            'config': {'topk': args.topk, 'smoothing': args.smoothing,
                       'extend_W': args.W, 'merge_G': args.G,
                       'tau_r_n': args.tau_r_n,
                       'tau_r_prime_n': args.tau_r_prime_n,
                       'tau_s_grid': list(args.tau_s_grid),
                       'rules': list(args.rules),
                       'arrays': args.arrays},
        }, f, indent=2)
    print(f'\noutputs -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
