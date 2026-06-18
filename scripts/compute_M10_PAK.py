"""Compute PA%K for the M10 (GBM stacker) fusion method.

M10 takes the same inputs as scripts/fusion_sweep_K100_full.py:
  - arrays.npz from scripts/eval_paper_protocol_gdeltauq.py
  - split JSON with val_rows + C_row_range
  - calibration bundle (for edge_index_sample, only used by other methods)

Procedure:
  1. Re-create the M10 stacker exactly as run_M10 in fusion_sweep_K100_full.py:
     train HistGradientBoostingClassifier on val_slice with class_weight=
     'balanced', sweep depth in {2,3,5} x n_iter in {50,100,200}, pick the
     HP combo whose logit score yields the highest Fix-A F1 on the full
     test arrays. Persist that score.
  2. Run pa_k_metric.f1_pa_k_auc on the persisted score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from fusion_sweep_K100_full import (
    setup_context, build_stacker_features, eval_score_full,
)
from pa_k_metric import f1_pa_k_auc


def fit_M10_score(ctx):
    """Replicate run_M10's HP sweep and return the score with the best F1."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    label = ctx['label']
    feat = build_stacker_features(ctx)
    val_idx = ctx['val_idx']
    best_f1 = -1.0
    best_score = None
    best_hp = None
    for depth in (2, 3, 5):
        for n_iter in (50, 100, 200):
            gb = HistGradientBoostingClassifier(
                max_depth=depth, max_iter=n_iter, learning_rate=0.05,
                l2_regularization=1.0, random_state=ctx['seed'],
                class_weight='balanced')
            gb.fit(feat[val_idx], label[val_idx])
            proba = gb.predict_proba(feat)[:, 1]
            s = np.log(np.clip(proba, 1e-8, 1 - 1e-8) /
                       np.clip(1 - proba, 1e-8, 1 - 1e-8))
            res = eval_score_full(s, label)
            if res['F1'] > best_f1:
                best_f1 = res['F1']
                best_score = s
                best_hp = dict(max_depth=depth, max_iter=n_iter, **res)
    return best_score, best_hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arrays', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--slide_win', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ctx_args = argparse.Namespace(
        arrays=args.arrays, split=args.split, bundle=args.bundle,
        slide_win=args.slide_win, seed=args.seed,
    )
    ctx = setup_context(ctx_args)

    print('fitting M10 GBM stacker (depth x iter HP sweep) ...', flush=True)
    score, hp = fit_M10_score(ctx)
    print(f'  Fix-A best:  F1={hp["F1"]:.4f}  P={hp["P"]:.4f}  R={hp["R"]:.4f}  '
          f'tau={hp["tau"]:.4f}  q={hp["q"]:.4f}  '
          f'depth={hp["max_depth"]} n_iter={hp["max_iter"]}', flush=True)

    label = ctx['label']
    print('computing PA%K over K in [0, 100] step 1 ...', flush=True)
    pa = f1_pa_k_auc(score, label,
                     K_grid=np.arange(0, 101, 1), n_thresholds=400)
    print(f'  F1_PA (K=0):   {pa["F1_PA_K0"]:.4f}', flush=True)
    print(f'  F1_PA_K20:     '
          f'{float(pa["curve"].iloc[20]["F1"]):.4f}', flush=True)
    print(f'  F1_PA_K50:     {pa["F1_PA_K50"]:.4f}', flush=True)
    print(f'  F1_PA_K80:     '
          f'{float(pa["curve"].iloc[80]["F1"]):.4f}', flush=True)
    print(f'  F1 (K=100):    {pa["F1_PA_K100"]:.4f}', flush=True)
    print(f'  PA%K-AUC:      {pa["PA_K_AUC"]:.4f}', flush=True)

    out = {
        'method': 'M10',
        'seed': args.seed,
        'arrays': args.arrays,
        'slide_win': args.slide_win,
        'M10_best_HP': {
            'max_depth': hp['max_depth'],
            'max_iter': hp['max_iter'],
        },
        'M10_FixA': {
            'F1': hp['F1'], 'P': hp['P'], 'R': hp['R'],
            'tau': hp['tau'], 'q': hp['q'],
        },
        'PA_K': {
            'F1_PA_K0':  pa['F1_PA_K0'],
            'F1_PA_K20': float(pa['curve'].iloc[20]['F1']),
            'F1_PA_K50': pa['F1_PA_K50'],
            'F1_PA_K80': float(pa['curve'].iloc[80]['F1']),
            'F1_PA_K100': pa['F1_PA_K100'],
            'PA_K_AUC':  pa['PA_K_AUC'],
        },
        'curve': pa['curve'].to_dict(orient='records'),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
