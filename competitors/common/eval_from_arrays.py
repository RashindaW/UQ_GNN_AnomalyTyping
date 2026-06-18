"""Model-agnostic evaluation wrapper for the competitor benchmark.

Given a cached arrays.npz that follows our contract (test_mu_bar,
test_ground_truth, test_attack_label, test_U_par[, test_U_str], test_U_dist,
test_sigma2_ale, val_mu_bar, val_ground_truth), this emits BOTH:

  - baseline (M0, residual-only top-1 aggregate): Fix-A F1/P/R + PA%K-AUC
  - M10 (GBM stacker on residual + uncertainty): Fix-A F1/P/R + PA%K-AUC

All numbers use the SAME protocol as the GDN reference (Fix-A post-proc
W=5/G=5, 400-quantile tau sweep; PA%K over K in [0,100]). This is the single
entry point every competitor's arrays.npz flows through, so cross-method
comparison is apples-to-apples by construction.

Reuses (unchanged): scripts/fusion_sweep_K100_full.py (setup_context,
eval_score_full, build_stacker_features), scripts/compute_M10_PAK.py
(fit_M10_score), scripts/pa_k_metric.py (f1_pa_k_auc).

Module-0 gate: run on the GDN reference arrays.npz and confirm M10 F1
reproduces 0.8391 and M0 ~0.81.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from fusion_sweep_K100_full import setup_context, eval_score_full
from compute_M10_PAK import fit_M10_score
from pa_k_metric import f1_pa_k_auc
from sweep_eval_gdeltauq import build_full_err_scores, topk_aggregate


def _pa_summary(score, label):
    pa = f1_pa_k_auc(score, label, K_grid=np.arange(0, 101, 1), n_thresholds=400)
    return {
        'F1_PA_K0': float(pa['F1_PA_K0']),
        'F1_PA_K50': float(pa['F1_PA_K50']),
        'F1_PA_K100': float(pa['F1_PA_K100']),
        'PA_K_AUC': float(pa['PA_K_AUC']),
    }


def evaluate_baseline_only(arrays, slide_win=60, method_label='competitor'):
    """Baseline (M0 residual-only) F1/P/R + PA%K from a competitor's forecasts,
    WITHOUT the UQ channels (used pre-anchoring). Mirrors setup_context's M0
    score: per-feature smoothed err scores -> top-1 aggregate."""
    d = np.load(arrays)
    test_mu = d['test_mu_bar'].astype(np.float64)
    test_gt = d['test_ground_truth'].astype(np.float64)
    val_mu = d['val_mu_bar'].astype(np.float64)
    val_gt = d['val_ground_truth'].astype(np.float64)
    label = d['test_attack_label'].astype(np.int8)
    full_scores = build_full_err_scores(test_mu, test_gt, val_mu, val_gt, 5)
    agg = topk_aggregate(full_scores, 1).astype(np.float64)
    m0 = eval_score_full(agg, label)
    m0_pa = _pa_summary(agg, label)
    return {
        'method': method_label, 'arrays': arrays, 'slide_win': slide_win,
        'baseline_M0': {
            'F1': float(m0['F1']), 'P': float(m0['P']), 'R': float(m0['R']),
            'tau': float(m0['tau']), **m0_pa,
        },
    }


def evaluate_arrays(arrays, split, bundle, slide_win=60, seed=42,
                    method_label='competitor'):
    ctx_args = argparse.Namespace(
        arrays=arrays, split=split, bundle=bundle,
        slide_win=slide_win, seed=seed,
    )
    ctx = setup_context(ctx_args)
    label = ctx['label']

    # ---- baseline: residual-only top-1 aggregate (M0) ----
    agg = ctx['agg']
    m0 = eval_score_full(agg, label)
    m0_pa = _pa_summary(agg, label)

    # ---- M10: GBM stacker over residual + uncertainty channels ----
    m10_score, m10_hp = fit_M10_score(ctx)
    m10_pa = _pa_summary(m10_score, label)

    return {
        'method': method_label,
        'seed': seed,
        'arrays': arrays,
        'slide_win': slide_win,
        'baseline_M0': {
            'F1': float(m0['F1']), 'P': float(m0['P']), 'R': float(m0['R']),
            'tau': float(m0['tau']), **m0_pa,
        },
        'M10': {
            'F1': float(m10_hp['F1']), 'P': float(m10_hp['P']),
            'R': float(m10_hp['R']), 'tau': float(m10_hp['tau']),
            'max_depth': int(m10_hp['max_depth']),
            'max_iter': int(m10_hp['max_iter']),
            **m10_pa,
        },
        'delta_F1_M10_minus_M0': float(m10_hp['F1'] - m0['F1']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arrays', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--slide_win', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--label', default='competitor', help='method name for the report')
    ap.add_argument('--baseline-only', action='store_true',
                    help='compute only M0 residual-only baseline (no UQ channels needed)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.baseline_only:
        res = evaluate_baseline_only(args.arrays, args.slide_win, args.label)
        b = res['baseline_M0']
        print(json.dumps(res, indent=2), flush=True)
        print(f"\n[{args.label}] baseline M0: F1={b['F1']:.4f} P={b['P']:.4f} "
              f"R={b['R']:.4f} PA%K-AUC={b['PA_K_AUC']:.4f}", flush=True)
    else:
        res = evaluate_arrays(args.arrays, args.split, args.bundle,
                              args.slide_win, args.seed, args.label)
        print(json.dumps(res, indent=2), flush=True)
        b, m = res['baseline_M0'], res['M10']
        print(f"\n[{args.label}] baseline M0: F1={b['F1']:.4f} P={b['P']:.4f} "
              f"R={b['R']:.4f} PA%K-AUC={b['PA_K_AUC']:.4f}", flush=True)
        print(f"[{args.label}] M10       : F1={m['F1']:.4f} P={m['P']:.4f} "
              f"R={m['R']:.4f} PA%K-AUC={m['PA_K_AUC']:.4f}  "
              f"(ΔF1={res['delta_F1_M10_minus_M0']:+.4f})", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
