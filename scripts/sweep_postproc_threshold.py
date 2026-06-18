"""Fix A: post-proc-aware threshold optimization.

Standard procedure (used by `evaluate.get_best_performance_data` and
`scripts/sweep_eval_gdeltauq.py`):

  1. Pick threshold tau* maximizing F1 on the RAW alarm vector.
  2. Apply post-proc (extend_W, merge_G).
  3. Report F1 of the post-processed alarm.

Issue: tau* is optimal for step 1, not step 3. Once dilation + gap-merge
runs, the F1-optimal threshold generally shifts.

Fix A: sweep tau across a fine grid, but compute F1 of the
POST-PROCESSED alarm at each tau, then pick tau* on that curve.

Operates on a cached arrays.npz with at minimum:
  full_scores (V, T)        per-feature smoothed err scores
  test_attack_label (T,)    binary labels

Usage:
  python scripts/sweep_postproc_threshold.py \
    -arrays results/swat_gdeltauq_paper_protocol/0511-222735/arrays.npz \
    -configs paper cheap                              # named presets
  python scripts/sweep_postproc_threshold.py \
    -arrays ... -topk_grid 1 10 -smoothing_grid 0 3 \
    -W_grid 0 5 -G_grid 0 3
"""
import argparse
import itertools
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
    build_full_err_scores, topk_aggregate, apply_postproc, sweep_threshold,
)


# Named config presets so the CLI is concise.
PRESETS = {
    'paper':   {'topk': 1,  'smoothing': 3, 'W': 0, 'G': 0},
    'cheap':   {'topk': 10, 'smoothing': 0, 'W': 5, 'G': 3},
    'paperW3': {'topk': 1,  'smoothing': 3, 'W': 3, 'G': 1},
    'cheapW3': {'topk': 10, 'smoothing': 0, 'W': 3, 'G': 1},
}


def metrics_from_pred(pred, label):
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


def best_threshold_postproc_aware(agg, label, W, G, n_taus=400):
    """For a given (W, G), sweep n_taus quantile thresholds. At each tau,
    apply post-proc and measure F1. Return the best (F1, tau, P, R) and
    the full curve.
    """
    qs = np.linspace(0.0, 0.9999, n_taus)
    tau_vals = np.quantile(agg, qs)
    best = {'F1': -1.0}
    curve = []
    for tau, q in zip(tau_vals, qs):
        raw = (agg > tau).astype(np.int8)
        if W == 0 and G == 0:
            pred = raw
        else:
            pred = apply_postproc(raw, extend_W=W, merge_G=G)
        f1, p, r, tp, fp, fn, tn = metrics_from_pred(pred, label)
        curve.append((float(q), float(tau), f1, p, r, tp, fp, fn, tn))
        if f1 > best['F1']:
            best = {'F1': f1, 'P': p, 'R': r, 'tau': float(tau),
                    'q': float(q), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
    return best, curve


def legacy_threshold_then_postproc(agg, label, W, G):
    """Reproduce the legacy procedure: pick tau on RAW alarm using the
    400-rank sweep (util.data.eval_scores), then apply post-proc and
    report F1.
    """
    # sweep_threshold uses rank-based eval_scores → returns best F1 + tau
    # on the raw alarm. We then re-apply post-proc at that tau.
    f1_raw, tau = sweep_threshold(agg, label.tolist())
    raw = (agg > tau).astype(np.int8)
    if W == 0 and G == 0:
        pred = raw
    else:
        pred = apply_postproc(raw, extend_W=W, merge_G=G)
    f1, p, r, tp, fp, fn, tn = metrics_from_pred(pred, label)
    return {'F1': f1, 'P': p, 'R': r, 'tau': float(tau),
            'F1_raw_alarm': float(f1_raw),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def evaluate_one_config(arrays, label, topk, smoothing, W, G,
                        full_scores_cache=None, n_taus=400):
    """Build the per-timestep aggregate at this (topk, smoothing), then
    compare legacy vs Fix-A thresholds.
    `full_scores_cache` is a dict keyed by `smoothing` to avoid recomputing
    the smoothed err scores when the same smoothing is reused across rows.
    """
    if smoothing not in full_scores_cache:
        if smoothing == 'cached':
            full_scores_cache['cached'] = arrays['full_scores']
        else:
            # Alias: G-DeltaUQ arrays.npz uses test_mu_bar / val_mu_bar
            # instead of test_predict / val_predict. Both are (T, V) tensors
            # of model predictions and are interchangeable from the err-score
            # standpoint.
            files = arrays.files
            test_pred = arrays['test_predict' if 'test_predict' in files
                               else 'test_mu_bar']
            val_pred = arrays['val_predict' if 'val_predict' in files
                              else 'val_mu_bar']
            full_scores_cache[smoothing] = build_full_err_scores(
                test_pred, arrays['test_ground_truth'],
                val_pred, arrays['val_ground_truth'],
                int(smoothing),
            )
    fs = full_scores_cache[smoothing]
    agg = topk_aggregate(fs, topk=topk)

    legacy = legacy_threshold_then_postproc(agg, label, W, G)
    fix_a, _curve = best_threshold_postproc_aware(agg, label, W, G,
                                                  n_taus=n_taus)
    return legacy, fix_a, agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-arrays', required=True,
                        help='cached arrays.npz with at minimum full_scores '
                             'and test_attack_label. If smoothing_grid != '
                             '[cached], also needs test_predict/_ground_truth '
                             'and val_predict/_ground_truth.')
    parser.add_argument('-out_root',
                        default='results/postproc_threshold_fixA')
    parser.add_argument('-configs', nargs='+', default=['paper', 'cheap'],
                        help=f'named presets to evaluate ({list(PRESETS)}).')
    parser.add_argument('-topk_grid', type=int, nargs='*', default=None,
                        help='If given, ignore -configs and sweep this grid.')
    parser.add_argument('-smoothing_grid', type=int, nargs='*', default=None)
    parser.add_argument('-W_grid', type=int, nargs='*', default=None)
    parser.add_argument('-G_grid', type=int, nargs='*', default=None)
    parser.add_argument('-n_taus', type=int, default=400)
    args = parser.parse_args()

    print(f'loading {args.arrays}', flush=True)
    arrays = np.load(args.arrays)
    label = arrays['test_attack_label'].astype(np.int8)
    print(f'T={label.shape[0]}  attack_rate={label.mean():.4f}', flush=True)

    if args.topk_grid:
        configs = []
        for tk, sm, w, g in itertools.product(
            args.topk_grid, args.smoothing_grid or [3],
            args.W_grid or [0], args.G_grid or [0],
        ):
            configs.append({'name': f'tk{tk}_sm{sm}_W{w}_G{g}',
                            'topk': tk, 'smoothing': sm, 'W': w, 'G': g})
    else:
        configs = []
        for name in args.configs:
            if name not in PRESETS:
                raise ValueError(f'unknown preset {name!r}; pick from '
                                 f'{list(PRESETS)}')
            cfg = dict(PRESETS[name])
            cfg['name'] = name
            configs.append(cfg)

    # Decide whether we have the raw mu / gt / val arrays needed to
    # rebuild full_scores at smoothing != cached. If not, we can only use
    # the smoothing the cached full_scores was built at (likely 3).
    need_rebuild = any(c['smoothing'] != 3 for c in configs)
    has_raw = ('test_predict' in arrays.files
               or 'test_mu_bar' in arrays.files)
    full_scores_cache = {}
    if has_raw and need_rebuild:
        # Fine-grained rebuild for any smoothing.
        pass
    elif need_rebuild and not has_raw:
        print('NOTE: arrays.npz lacks raw test_predict/val_predict; cannot '
              'rebuild err scores at non-default smoothing. Falling back to '
              'cached full_scores for all smoothing values.', flush=True)
        full_scores_cache['cached'] = arrays['full_scores']
        for c in configs:
            c['smoothing'] = 'cached'
    else:
        # smoothing == 3 across the board, just use cache for that key
        full_scores_cache[3] = arrays['full_scores']

    rows = []
    for c in configs:
        print(f"\n=== config={c['name']}  topk={c['topk']} "
              f"smoothing={c['smoothing']} W={c['W']} G={c['G']}", flush=True)
        legacy, fix_a, _ = evaluate_one_config(
            arrays, label, c['topk'], c['smoothing'], c['W'], c['G'],
            full_scores_cache=full_scores_cache, n_taus=args.n_taus,
        )
        lift = fix_a['F1'] - legacy['F1']
        print(f"  legacy : F1={legacy['F1']:.4f}  P={legacy['P']:.4f}  "
              f"R={legacy['R']:.4f}  tau={legacy['tau']:.4f}", flush=True)
        print(f"  fix_A  : F1={fix_a['F1']:.4f}  P={fix_a['P']:.4f}  "
              f"R={fix_a['R']:.4f}  tau={fix_a['tau']:.4f}", flush=True)
        print(f"  lift   : {lift:+.4f}", flush=True)

        rows.append({
            'arrays': args.arrays,
            'config': c['name'],
            'topk': c['topk'], 'smoothing': c['smoothing'],
            'W': c['W'], 'G': c['G'],
            'F1_legacy': legacy['F1'], 'P_legacy': legacy['P'],
            'R_legacy': legacy['R'], 'tau_legacy': legacy['tau'],
            'F1_raw_alarm_legacy': legacy.get('F1_raw_alarm'),
            'F1_fixA': fix_a['F1'], 'P_fixA': fix_a['P'],
            'R_fixA': fix_a['R'], 'tau_fixA': fix_a['tau'],
            'q_fixA': fix_a['q'],
            'lift': lift,
        })

    df = pd.DataFrame(rows)

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    arrays_tag = Path(args.arrays).parent.name
    out_dir = Path(args.out_root) / f'{arrays_tag}_{datestr}'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'fixA_sweep.csv', index=False)

    print('\n=== Summary ===', flush=True)
    print(df[['config', 'F1_legacy', 'F1_fixA', 'lift', 'P_fixA',
              'R_fixA']].to_string(
        index=False, float_format=lambda v: f'{v:.4f}',
    ), flush=True)

    best_idx = df['F1_fixA'].idxmax()
    best = df.loc[best_idx].to_dict()
    with (out_dir / 'best_fixA.json').open('w') as f:
        json.dump({k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                   for k, v in best.items()}, f, indent=2)
    print(f'\nbest config: {best["config"]} F1={best["F1_fixA"]:.4f}', flush=True)
    print(f'output: {out_dir}', flush=True)


if __name__ == '__main__':
    main()
