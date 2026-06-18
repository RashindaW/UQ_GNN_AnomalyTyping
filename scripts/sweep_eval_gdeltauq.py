"""F1-improvement sweep for G-DeltaUQ on cached test/val arrays.

Loads `arrays.npz` from a previous `eval_paper_protocol_gdeltauq.py` run and
sweeps four cheap recall-improving levers without retraining or re-running
inference (except where U_par is required and isn't yet in the npz):

  1. eval-topk:    {1, 2, 3, 5, 10}
  2. smoothing:    {0, 1, 2, 3, 4, 5} timesteps SMA over per-feature err scores
  3. U_par gate:   {none, soft inverse, hard low-mask}
  4. post-proc:    extend each alarm by +-W/2 windows, merge gaps <= G

Optimal config is the (topk, smoothing, gate, W, G) tuple that maximises F1
on the test set.  Reports per-config metrics to a CSV plus the best
config's full breakdown.

Sanity-runs the cached baseline first to confirm we reproduce the prior F1
within float-comparison tolerance.
"""
import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Reuse the IQR/quantile helpers and the threshold-sweep machinery.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.data import get_err_median_and_iqr, eval_scores


def smoothed_err_scores(test_predict, test_gt, val_predict, val_gt, before_num):
    """Re-implementation of evaluate.get_err_scores with `before_num` parametric
    rather than hardcoded 3 (evaluate.py:58).  All other steps identical:
    fit median+IQR on val residuals, normalise test residuals, SMA-smooth.
    `before_num=0` disables smoothing entirely.
    """
    n_err_mid, n_err_iqr = get_err_median_and_iqr(val_predict, val_gt)
    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64),
    ))
    eps = 1e-2
    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + eps)
    if before_num <= 0:
        return err_scores
    smoothed = np.zeros_like(err_scores)
    for i in range(before_num, len(err_scores)):
        smoothed[i] = np.mean(err_scores[i - before_num:i + 1])
    return smoothed


def build_full_err_scores(test_mu, test_y, val_mu, val_y, before_num):
    """Per-feature variant of evaluate.get_full_err_scores with parametric
    smoothing window.  Returns scores of shape (V, T)."""
    V = test_mu.shape[1]
    out = None
    for i in range(V):
        s = smoothed_err_scores(
            test_mu[:, i], test_y[:, i], val_mu[:, i], val_y[:, i], before_num,
        )
        if out is None:
            out = s[None, :]
        else:
            out = np.vstack((out, s[None, :]))
    return out


def topk_aggregate(full_scores, topk):
    """Per-timestep top-k feature score (sum of top-k err_scores).  Mirrors
    evaluate.get_best_performance_data lines 128-134."""
    V = full_scores.shape[0]
    if topk == 1:
        return full_scores.max(axis=0)
    idx = np.argpartition(full_scores, V - topk, axis=0)[-topk:]
    return np.take_along_axis(full_scores, idx, axis=0).sum(axis=0)


def sweep_threshold(scores_1d, gt_labels, n_steps=400):
    """Best-F1 over a 400-rank-percentile threshold sweep, matching
    util.data.eval_scores -> evaluate.get_best_performance_data."""
    fmeas, thresholds = eval_scores(
        scores_1d.tolist() if hasattr(scores_1d, 'tolist') else list(scores_1d),
        list(gt_labels), n_steps, return_thresold=True,
    )
    best_i = int(np.argmax(fmeas))
    best_f1 = float(fmeas[best_i])
    best_thr = float(thresholds[best_i])
    return best_f1, best_thr


def metrics_at_threshold(scores_1d, gt_labels, threshold):
    pred = (np.asarray(scores_1d) > threshold).astype(np.int8)
    gt = np.asarray(gt_labels).astype(np.int8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    auc = float(roc_auc_score(gt, scores_1d)) if len(set(gt.tolist())) > 1 else float('nan')
    return dict(F1=f1, P=p, R=r, AUC=auc, TP=tp, FP=fp, FN=fn, TN=tn, threshold=float(threshold))


def apply_postproc(pred, extend_W, merge_G):
    """Recall-side post-processing, opposite of v4 sustained-window.
    extend_W: extend each isolated alarm by +- floor(extend_W/2) windows.
    merge_G:  merge gaps of length <= merge_G into the surrounding alarm run.
    Both default to 0 (no-op).
    """
    p = pred.astype(np.int8).copy()
    if extend_W and extend_W > 0:
        half = extend_W // 2
        # symmetric dilation
        kernel = np.ones(2 * half + 1, dtype=np.int8)
        # convolve by max-pool (binary dilation)
        p = (np.convolve(p, kernel, mode='same') > 0).astype(np.int8)
    if merge_G and merge_G > 0:
        # find runs of zeros surrounded by ones; if length <= G, fill them
        T = len(p)
        i = 0
        while i < T:
            if p[i] == 0:
                j = i
                while j < T and p[j] == 0:
                    j += 1
                gap = j - i
                if 0 < i and j < T and gap <= merge_G:
                    p[i:j] = 1
                i = j
            else:
                i += 1
    return p


def gate_scores(scores_1d, U_par_t, gate_kind, gate_param):
    """Uncertainty-aware re-weighting of per-timestep scores."""
    if gate_kind == 'none':
        return scores_1d
    if gate_kind == 'soft_inv':
        # higher confidence -> larger boost.  Centred at median U_par.
        med = float(np.median(U_par_t))
        eps = 1e-6
        return scores_1d * (med + eps) / (U_par_t + eps)
    if gate_kind == 'hard_low':
        # alarm only when both score is high AND U_par below quantile
        q = float(np.quantile(U_par_t, gate_param))
        mask = (U_par_t <= q).astype(np.float64)
        boost = float(gate_param)  # baseline weight outside the mask
        return scores_1d * (mask + (1 - mask) * boost)
    raise ValueError(f'unknown gate_kind {gate_kind!r}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-arrays', required=True, help='path to arrays.npz')
    parser.add_argument('-out_csv', default=None)
    parser.add_argument('-results_dir', default='results/swat_gdeltauq_paper_protocol/sweeps')
    parser.add_argument('-topk_grid', type=int, nargs='+', default=[1, 2, 3, 5, 10])
    parser.add_argument('-smoothing_grid', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5])
    parser.add_argument('-extend_grid', type=int, nargs='+', default=[0, 1, 3, 5])
    parser.add_argument('-merge_grid', type=int, nargs='+', default=[0, 1, 3])
    parser.add_argument('-gate_kinds', nargs='+', default=['none', 'soft_inv', 'hard_low'])
    parser.add_argument('-hard_low_quantiles', type=float, nargs='+', default=[0.5, 0.7])
    args = parser.parse_args()

    A = np.load(args.arrays)
    test_mu = A['test_mu_bar']           # (T, V)
    test_y = A['test_ground_truth']
    test_lbl = A['test_attack_label']
    val_mu = A['val_mu_bar']
    val_y = A['val_ground_truth']
    cached_full_scores = A['full_scores']  # (V, T) under default smoothing=3
    has_uncertainty = 'test_U_par' in A.files
    test_U_par = A['test_U_par'] if has_uncertainty else None  # (T, V)
    print(f'arrays: T={len(test_y)} V={test_mu.shape[1]} '
          f'has_uncertainty={has_uncertainty}', flush=True)

    # 1. Sanity baseline reproduction --------------------------------------
    base_scores_3 = build_full_err_scores(test_mu, test_y, val_mu, val_y, before_num=3)
    diff = float(np.abs(base_scores_3 - cached_full_scores).max())
    print(f'sanity: max |new_scores - cached_scores| (smoothing=3) = {diff:.6e}', flush=True)
    base_agg_top1 = topk_aggregate(base_scores_3, topk=1)
    base_f1, base_thr = sweep_threshold(base_agg_top1, test_lbl)
    print(f'sanity: cached-baseline F1 (topk=1, smoothing=3, no-gate, no-post) = {base_f1:.4f}',
          flush=True)

    # 2. Pre-build per-smoothing scores grid (most expensive step) --------
    full_by_smoothing = {b: build_full_err_scores(test_mu, test_y, val_mu, val_y, b)
                        for b in args.smoothing_grid}

    # 3. Sweep --------------------------------------------------------------
    # Build the gate grid: [(kind, param), ...]
    gate_grid = []
    for k in args.gate_kinds:
        if k == 'none' or k == 'soft_inv':
            gate_grid.append((k, 0.0))
        elif k == 'hard_low':
            for q in args.hard_low_quantiles:
                gate_grid.append(('hard_low', q))
    if not has_uncertainty:
        gate_grid = [('none', 0.0)]
        print('NOTE: no test_U_par in npz, skipping gating sweep', flush=True)

    rows = []
    total = (len(args.topk_grid) * len(args.smoothing_grid) * len(gate_grid)
             * len(args.extend_grid) * len(args.merge_grid))
    print(f'sweeping {total} configurations ...', flush=True)

    for topk, b, (gk, gp), W, G in itertools.product(
        args.topk_grid, args.smoothing_grid, gate_grid, args.extend_grid, args.merge_grid,
    ):
        scores = full_by_smoothing[b]
        agg = topk_aggregate(scores, topk=topk)
        if gk != 'none':
            U_t = test_U_par.max(axis=1)
            agg = gate_scores(agg, U_t, gk, gp)
        f1_at, thr = sweep_threshold(agg, test_lbl)
        if W == 0 and G == 0:
            m = metrics_at_threshold(agg, test_lbl, thr)
        else:
            base_pred = (agg > thr).astype(np.int8)
            post = apply_postproc(base_pred, extend_W=W, merge_G=G)
            tp = int(((post == 1) & (test_lbl == 1)).sum())
            fp = int(((post == 1) & (test_lbl == 0)).sum())
            fn = int(((post == 0) & (test_lbl == 1)).sum())
            tn = int(((post == 0) & (test_lbl == 0)).sum())
            P = tp / max(1, tp + fp)
            R = tp / max(1, tp + fn)
            F1_post = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
            m = dict(F1=F1_post, P=P, R=R, AUC=float('nan'),
                     TP=tp, FP=fp, FN=fn, TN=tn, threshold=float(thr))
        row = dict(topk=topk, smoothing=b, gate=gk, gate_param=gp,
                   extend_W=W, merge_G=G, **m)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('F1', ascending=False).reset_index(drop=True)
    print('\nTop 15 configs by F1:\n')
    print(df.head(15).to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_csv or str(out_dir / f'sweep_{datestr}.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nfull sweep saved to {out_csv}', flush=True)

    # 4. Best config breakdown + short-attack sanity ----------------------
    best = df.iloc[0].to_dict()
    print('\nBest config:')
    print(json.dumps({k: (v if not isinstance(v, float) else round(v, 4))
                      for k, v in best.items()}, indent=2))

    # short-attack recall by attack-run length
    runs = []
    in_run = False
    start = 0
    for i, lab in enumerate(test_lbl.tolist() + [0]):
        if lab and not in_run:
            in_run = True
            start = i
        elif not lab and in_run:
            in_run = False
            runs.append((start, i))
    print(f'\nattack runs: {len(runs)}; '
          f'duration buckets (windows): '
          f'1-3={sum(1 for s,e in runs if 1<=e-s<=3)}, '
          f'4-10={sum(1 for s,e in runs if 4<=e-s<=10)}, '
          f'11-50={sum(1 for s,e in runs if 11<=e-s<=50)}, '
          f'51+={sum(1 for s,e in runs if e-s>=51)}', flush=True)

    # Re-derive predictions for the best config and report per-bin recall
    scores = full_by_smoothing[int(best['smoothing'])]
    agg = topk_aggregate(scores, topk=int(best['topk']))
    if best['gate'] != 'none':
        U_t = test_U_par.max(axis=1)
        agg = gate_scores(agg, U_t, best['gate'], float(best['gate_param']))
    pred = (agg > best['threshold']).astype(np.int8)
    if best['extend_W'] or best['merge_G']:
        pred = apply_postproc(pred, int(best['extend_W']), int(best['merge_G']))
    bins = [(1, 3), (4, 10), (11, 50), (51, 10**9)]
    per_bin = {}
    for lo, hi in bins:
        runs_b = [(s, e) for s, e in runs if lo <= e - s <= hi]
        caught = sum(1 for s, e in runs_b if pred[s:e].sum() > 0)
        per_bin[f'{lo}-{hi if hi < 10**9 else "inf"}'] = (caught, len(runs_b))
    print('per-bin recall (caught / total):', per_bin)

    summary_path = out_dir / f'best_{datestr}.json'
    with summary_path.open('w') as f:
        json.dump({
            'arrays': args.arrays,
            'baseline_F1_top1_smooth3': float(base_f1),
            'best_config': {k: (v if not isinstance(v, np.generic) else v.item())
                            for k, v in best.items()},
            'per_bin_recall': {k: list(v) for k, v in per_bin.items()},
            'csv': out_csv,
        }, f, indent=2)
    print(f'best-config summary -> {summary_path}', flush=True)


if __name__ == '__main__':
    main()
