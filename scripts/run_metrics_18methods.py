"""Unified metrics driver — Fix-A F1 + PTaPR + PA%K for all 18 fusion methods.

For each method M0-M17 (plus two baselines), re-derive the continuous
score `s(t)` at its frozen best hyperparameters, then apply:
  - Fix-A best-F1 (the protocol of the published 0.8109)
  - PTaPR (Kang et al. 2026) with AUC over θ ∈ [0, 1]
  - PA%K (Kim et al. 2022) with AUC over K ∈ [0, 100]

Baselines (Kim et al. §4.1):
  - Random uniform anomaly score
  - Input L2-norm score (Eq. 8)

For methods that natively produce a binary alarm (M1 OR / M2 AND / M3
4-Placement / M12 Triple-OR), we RECONSTRUCT a continuous score as a
weighted sum of the underlying continuous channels at the method's
winning thresholds. This allows PA%K to operate at non-degenerate K.

Output: a single CSV ranking all methods by all three metrics, plus per-
method PA%K and PTaPR curves.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

# Reuse setup + helpers from the main sweep
from fusion_sweep_K100_full import (
    setup_context,
    score_per_sensor_sigma,
    score_uq_weighted_topk,
    score_mult_aggregate_gate,
    score_linear_sum,
    _pool_U_str_to_sensor,
    EPS,
    SIGMA_FLOOR,
    POST_W,
    POST_G,
    N_TAUS,
)
from sweep_eval_gdeltauq import topk_aggregate, apply_postproc
from sweep_postproc_threshold import (
    best_threshold_postproc_aware,
    metrics_from_pred,
)
import ptapr_metric as ptapr
import pa_k_metric as pak


# ============================================================================
# Frozen best-HP per method
#
# These come from the standalone smoke-test outputs persisted under
# results/fusion_K100_full_M14/, results/fusion_K100_full_M15-17/, and the
# main sweep log. M8-M13 are placeholders to be filled once the main
# sweep finishes. The driver gracefully skips methods with HPs=None.
# ============================================================================

FROZEN_HPS: Dict[str, Optional[Dict]] = {
    'M0': {},
    'M1': {'signal': 'U_dist', 'tau_r': 80.727, 'tau_s': 5.0},
    'M2': {'signal': 'U_par_mean_v', 'tau_r_prime': 4.393, 'tau_s': 3.0},
    'M3': {'rule': 'OR_A', 'signal': 'U_dist', 'tau_r': 80.727, 'tau_s': 4.0},
    'M4': {'topk': 1, 'smoothing': 3},
    'M5': {'topk': 2, 'smoothing': 5},
    'M6': {'lam': 0.5, 'topk': 1},
    'M7': {'lam_par': 0.0, 'lam_str': 0.0, 'lam_dist': 0.0},
    'M8': None,    # not yet finished in main sweep
    'M9': None,
    'M10': None,
    'M11': None,
    'M12': None,
    'M13': None,
    'M14': {'lam': 0.5},
    'M15': {'agg': 'top2'},
    'M16': {'tau_epi_q': 0.7, 'tau_z_q': 0.95, 'beta': 1.0},
    'M17': {'C': 0.1},
}


# ============================================================================
# Score builders at frozen HPs
# ============================================================================

def build_score_M0(ctx, hp):
    return ctx['agg']


def build_score_M1(ctx, hp):
    """Simple OR: reconstruct continuous score as agg / tau_r + sig / tau_s.

    This is a defensible continuous proxy whose threshold-1 binarization
    recovers the OR rule (alarm where agg > tau_r OR sig > tau_s)."""
    agg = ctx['agg']
    sig = ctx['signals'][hp['signal']]
    s = (agg / hp['tau_r']) + (sig / max(hp['tau_s'], 1e-6))
    return s


def build_score_M2(ctx, hp):
    """Simple AND: continuous score that is high where BOTH conditions hold.

    Use min(agg/tau_r', sig/tau_s) — large when both terms are large.
    Threshold-1 binarization recovers AND rule."""
    agg = ctx['agg']
    sig = ctx['signals'][hp['signal']]
    s = np.minimum(agg / max(hp['tau_r_prime'], 1e-6),
                   sig / max(hp['tau_s'], 1e-6))
    return s


def build_score_M3(ctx, hp):
    """4-Placement Stacked OR/AND: reconstruct continuous score by the rule.

    OR_A: postproc(residual_alarm) ∨ spike → continuous proxy via union
    OR_B: postproc(residual_alarm ∨ spike)
    AND_A/B analogous.

    For continuous reconstruction we use the same proxy as M1 (sum of
    normalized channels) before any post-proc."""
    agg = ctx['agg']
    sig = ctx['signals'][hp['signal']]
    if hp['rule'].startswith('OR'):
        tau_r = hp['tau_r']
    else:
        tau_r = hp.get('tau_r_prime', hp.get('tau_r', 1.0))
    s = (agg / max(tau_r, 1e-6)) + (sig / max(hp['tau_s'], 1e-6))
    return s


def build_score_M4(ctx, hp):
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])
    sigma_ale = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR))
    return score_per_sensor_sigma(r_abs, sigma_ale, hp['topk'], hp['smoothing'])


def build_score_M5(ctx, hp):
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])
    sigma_tot = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR)
                         + np.maximum(ctx['test_U_par'], 0.0))
    return score_per_sensor_sigma(r_abs, sigma_tot, hp['topk'], hp['smoothing'])


def build_score_M6(ctx, hp):
    return score_uq_weighted_topk(ctx['full_scores'], ctx['z_U_par_TxV'],
                                   hp['lam'], hp['topk'])


def build_score_M7(ctx, hp):
    agg = ctx['agg']
    z_par = ctx['signals']['U_par_max_v']
    z_str = ctx['signals']['U_str_mean_e']
    z_dist = ctx['signals']['U_dist']
    return score_mult_aggregate_gate(agg, z_par, z_str, z_dist,
                                      hp['lam_par'], hp['lam_str'], hp['lam_dist'])


def build_score_M14(ctx, hp):
    """M14 = M4 base + λ · max(z_U_par, z_U_str, z_U_dist) epistemic boost."""
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])
    sigma_ale = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR))
    s_resid = score_per_sensor_sigma(r_abs, sigma_ale, tk=1, sm=3)
    z_par = ctx['signals']['U_par_max_v']
    z_str = ctx['signals']['U_str_mean_e']
    z_dist = ctx['signals']['U_dist']
    epi_max = np.maximum.reduce([
        np.clip(z_par, 0.0, 5.0),
        np.clip(z_str, 0.0, 5.0),
        np.clip(z_dist, 0.0, 5.0),
    ])
    return s_resid + hp['lam'] * epi_max


def build_score_M15(ctx, hp):
    """M15 Mahalanobis at the frozen aggregation HP."""
    full_scores = ctx['full_scores']
    test_U_par = ctx['test_U_par']
    test_U_str = ctx['test_U_str']
    c_mask = ctx['c_mask']
    edge_index = ctx.get('edge_index')
    V_ = full_scores.shape[0]
    T_ = full_scores.shape[1]
    U_str_v = _pool_U_str_to_sensor(test_U_str, edge_index, V_)
    r_per_sensor = full_scores.T
    s_stack = np.stack([r_per_sensor, test_U_par, U_str_v], axis=-1)
    mu0 = np.zeros((V_, 3))
    sigma0_inv = np.zeros((V_, 3, 3))
    reg = 1e-6
    for v in range(V_):
        x = s_stack[c_mask, v, :]
        mu0[v] = x.mean(axis=0)
        cov = np.cov(x.T) + reg * np.eye(3)
        try:
            sigma0_inv[v] = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            sigma0_inv[v] = np.linalg.pinv(cov)
    centered = s_stack - mu0[None, :, :]
    D2 = np.einsum('tvi,vij,tvj->tv', centered, sigma0_inv, centered)
    agg_kind = hp['agg']
    if agg_kind == 'max':
        return D2.max(axis=1)
    elif agg_kind == 'mean':
        return D2.mean(axis=1)
    elif agg_kind.startswith('top'):
        k = int(agg_kind[3:])
        return np.sort(D2, axis=1)[:, -k:].sum(axis=1)
    return D2.max(axis=1)


def build_score_M16(ctx, hp):
    """M16 regime-routing at the frozen HP."""
    full_scores = ctx['full_scores']
    test_U_par = ctx['test_U_par']
    c_mask = ctx['c_mask']
    z = full_scores.T
    U_par = test_U_par
    tau_epi = float(np.quantile(U_par[c_mask], hp['tau_epi_q']))
    tau_z = float(np.quantile(z[c_mask], hp['tau_z_q']))
    precision_term = z * (U_par < tau_epi).astype(np.float64)
    recall_term = hp['beta'] * U_par * (z < tau_z).astype(np.float64)
    A = precision_term + recall_term
    return A.max(axis=1)


def build_score_M17(ctx, hp):
    """M17 per-sensor LogReg at the frozen HP."""
    from sklearn.linear_model import LogisticRegression
    full_scores = ctx['full_scores']
    test_U_par = ctx['test_U_par']
    test_U_str = ctx['test_U_str']
    test_sigma2_ale = ctx['test_sigma2_ale']
    val_idx = ctx['val_idx']
    edge_index = ctx.get('edge_index')
    label = ctx['label']
    V_ = full_scores.shape[0]
    T_ = label.shape[0]
    U_str_v = _pool_U_str_to_sensor(test_U_str, edge_index, V_)
    sigma_ale = np.sqrt(np.maximum(test_sigma2_ale, SIGMA_FLOOR))
    feat = np.stack([full_scores.T, test_U_par, U_str_v, sigma_ale],
                    axis=-1)
    feat_flat = feat.reshape(-1, 4)
    label_flat = np.tile(label[:, None], (1, V_)).reshape(-1)
    val_mask = np.zeros(T_, dtype=bool)
    val_mask[val_idx] = True
    val_mask_flat = np.tile(val_mask[:, None], (1, V_)).reshape(-1)
    lr = LogisticRegression(class_weight='balanced', C=hp['C'],
                             max_iter=1000, solver='lbfgs')
    lr.fit(feat_flat[val_mask_flat], label_flat[val_mask_flat])
    proba = lr.predict_proba(feat_flat)[:, 1]
    return proba.reshape(T_, V_).max(axis=1)


SCORE_BUILDERS = {
    'M0': build_score_M0,
    'M1': build_score_M1,
    'M2': build_score_M2,
    'M3': build_score_M3,
    'M4': build_score_M4,
    'M5': build_score_M5,
    'M6': build_score_M6,
    'M7': build_score_M7,
    'M14': build_score_M14,
    'M15': build_score_M15,
    'M16': build_score_M16,
    'M17': build_score_M17,
}


# ============================================================================
# Driver
# ============================================================================

def evaluate_score(method_name: str, scores: np.ndarray, label: np.ndarray,
                   K_grid: np.ndarray, theta_grid: np.ndarray,
                   compute_ptapr: bool = True,
                   compute_pa_k: bool = True,
                   n_thresholds: int = 200) -> Dict:
    """Run Fix-A best-F1 + PTaPR-AUC + PA%K-AUC for one (method, scores)."""
    out = {'method': method_name}

    # ---- Fix-A best F1 (post-proc-aware) ----
    fixA, _ = best_threshold_postproc_aware(scores, label, POST_W, POST_G,
                                              n_taus=n_thresholds)
    out['F1_fixA'] = float(fixA['F1'])
    out['P_fixA'] = float(fixA['P'])
    out['R_fixA'] = float(fixA['R'])
    out['tau_fixA'] = float(fixA['tau'])

    # Binary alarm from Fix-A for PTaPR (uses binary pred, not continuous)
    alarm = apply_postproc((scores > fixA['tau']).astype(np.int8),
                            POST_W, POST_G)

    # ---- PTaPR ----
    if compute_ptapr:
        pp = ptapr.ptapr_auc(label, alarm, theta_grid=theta_grid)
        out['PTaPR_F1_0'] = float(pp['F1_0'])
        out['PTaPR_F1_1'] = float(pp['F1_1'])
        out['PTaPR_AUC'] = float(pp['auc'])

    # ---- PA%K ----
    if compute_pa_k:
        pk = pak.f1_pa_k_auc(scores, label, K_grid=K_grid,
                              n_thresholds=n_thresholds)
        out['F1_PA_K0'] = float(pk['F1_PA_K0'])      # standard PA
        out['F1_PA_K50'] = float(pk['F1_PA_K50'])     # midpoint
        out['F1_PA_K100'] = float(pk['F1_PA_K100'])   # standard F1
        out['PA_K_AUC'] = float(pk['PA_K_AUC'])
        # Keep curve for downstream plotting
        out['_pa_k_curve'] = pk['curve']
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-arrays', required=True)
    parser.add_argument('-split', required=True)
    parser.add_argument('-bundle', default=None)
    parser.add_argument('-slide_win', type=int, default=60)
    parser.add_argument('-methods', nargs='+', default=None,
                        help='subset of M0..M17 to evaluate; default is all '
                             'with available frozen HPs')
    parser.add_argument('-n_thresholds', type=int, default=200,
                        help='quantile-grid points for δ-sweep (default 200 '
                             'for speed; standard is 400)')
    parser.add_argument('-theta_grid_n', type=int, default=11,
                        help='PTaPR θ grid points (default 11 → step 0.1)')
    parser.add_argument('-K_grid_n', type=int, default=11,
                        help='PA%K K grid points (default 11 → step 10)')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-out_root', default='results/metrics_18methods')
    parser.add_argument('-skip_baselines', action='store_true')
    args = parser.parse_args()

    ctx = setup_context(args)

    theta_grid = np.linspace(0.0, 1.0, args.theta_grid_n)
    K_grid = np.linspace(0.0, 100.0, args.K_grid_n)
    label = ctx['label']

    requested = args.methods or [
        m for m in SCORE_BUILDERS.keys() if FROZEN_HPS.get(m) is not None
    ]
    print(f"\nEvaluating {len(requested)} methods on {len(label)} timesteps", flush=True)
    print(f"PTaPR θ-grid: {args.theta_grid_n} points, "
          f"PA%K K-grid: {args.K_grid_n} points, "
          f"δ-grid: {args.n_thresholds} points\n", flush=True)

    rows = []
    curves = []
    for method in requested:
        hp = FROZEN_HPS.get(method)
        if hp is None and method not in ('M0',):
            print(f"  SKIP {method} — frozen HP not available yet", flush=True)
            continue
        if method not in SCORE_BUILDERS:
            print(f"  SKIP {method} — no score builder registered", flush=True)
            continue
        print(f"=== {method} (HP={hp}) ===", flush=True)
        t0 = time.time()
        scores = SCORE_BUILDERS[method](ctx, hp or {})
        scores = np.asarray(scores).astype(np.float64)
        if scores.ndim != 1 or scores.shape[0] != label.shape[0]:
            print(f"  ERR: score shape {scores.shape} not (T,); skipping",
                  flush=True)
            continue
        res = evaluate_score(method, scores, label, K_grid, theta_grid,
                              n_thresholds=args.n_thresholds)
        res['wall_sec'] = time.time() - t0
        curve = res.pop('_pa_k_curve', None)
        if curve is not None:
            curve['method'] = method
            curves.append(curve)
        rows.append(res)
        print(f"  F1_fixA={res['F1_fixA']:.4f}  "
              f"PA_K_AUC={res['PA_K_AUC']:.4f}  "
              f"PTaPR_AUC={res['PTaPR_AUC']:.4f}  "
              f"PA_K0={res['F1_PA_K0']:.4f}  "
              f"PA_K100={res['F1_PA_K100']:.4f}  "
              f"wall={res['wall_sec']:.1f}s", flush=True)

    # ---- Baselines ----
    if not args.skip_baselines:
        print("\n=== baseline: random uniform ===", flush=True)
        rand_scores = pak.random_baseline_score(label.shape[0], seed=args.seed)
        t0 = time.time()
        res = evaluate_score('baseline_random', rand_scores, label,
                              K_grid, theta_grid,
                              n_thresholds=args.n_thresholds)
        res['wall_sec'] = time.time() - t0
        curve = res.pop('_pa_k_curve', None)
        if curve is not None:
            curve['method'] = 'baseline_random'
            curves.append(curve)
        rows.append(res)
        print(f"  random: F1_fixA={res['F1_fixA']:.4f}  "
              f"PA_K0={res['F1_PA_K0']:.4f}  "
              f"PA_K100={res['F1_PA_K100']:.4f}  "
              f"PA_K_AUC={res['PA_K_AUC']:.4f}", flush=True)

        print("\n=== baseline: input L2-norm ===", flush=True)
        inorm = pak.input_norm_baseline_score(ctx['test_gt'],
                                                slide_win=args.slide_win)
        t0 = time.time()
        res = evaluate_score('baseline_input_norm', inorm, label,
                              K_grid, theta_grid,
                              n_thresholds=args.n_thresholds)
        res['wall_sec'] = time.time() - t0
        curve = res.pop('_pa_k_curve', None)
        if curve is not None:
            curve['method'] = 'baseline_input_norm'
            curves.append(curve)
        rows.append(res)
        print(f"  input-norm: F1_fixA={res['F1_fixA']:.4f}  "
              f"PA_K0={res['F1_PA_K0']:.4f}  "
              f"PA_K100={res['F1_PA_K100']:.4f}  "
              f"PA_K_AUC={res['PA_K_AUC']:.4f}", flush=True)

    # ---- Write outputs ----
    datestr = datetime.now().strftime('%m%d-%H%M%S')
    out_dir = Path(args.out_root) / datestr
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Reorder columns sensibly
    front = ['method', 'F1_fixA', 'PA_K_AUC', 'PTaPR_AUC',
             'F1_PA_K0', 'F1_PA_K50', 'F1_PA_K100',
             'PTaPR_F1_0', 'PTaPR_F1_1',
             'P_fixA', 'R_fixA', 'tau_fixA', 'wall_sec']
    cols = [c for c in front if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    df.to_csv(out_dir / 'metrics_18methods.csv', index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(
            out_dir / 'pa_k_curves.csv', index=False)

    # ---- SUMMARY.md ----
    df_sorted = df.sort_values('PA_K_AUC', ascending=False)
    md = [
        f"# Unified metrics: 18 methods × 3 protocols ({datestr})",
        "",
        f"Inputs: `{args.arrays}`",
        f"T={len(label)}, attack_rate={float(label.mean()):.4f}",
        "",
        "## Method ranking by PA%K-AUC (Kim et al. 2022, rigorous protocol)",
        "",
        "| Method | Fix-A F1 | PA%K AUC | PTaPR AUC | F1_PA (K=0) | F1 (K=100) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_sorted.iterrows():
        md.append(
            f"| {r['method']} | {r.get('F1_fixA', float('nan')):.4f} | "
            f"{r.get('PA_K_AUC', float('nan')):.4f} | "
            f"{r.get('PTaPR_AUC', float('nan')):.4f} | "
            f"{r.get('F1_PA_K0', float('nan')):.4f} | "
            f"{r.get('F1_PA_K100', float('nan')):.4f} |"
        )
    md.append("")
    md.append("## Notes")
    md.append("- **F1_PA (K=0)**: standard PA-adjusted F1 — KNOWN TO OVERESTIMATE.")
    md.append("- **F1 (K=100)**: standard F1 with no PA — most rigorous.")
    md.append("- **PA%K AUC**: K-integral, the headline rigorous number per Kim et al.")
    md.append("- **PTaPR AUC**: θ-integral, Kang et al. 2026 precursor-aware metric.")
    md.append("- **Fix-A F1**: existing project metric for backward comparison.")
    md.append("")
    md.append("## Baselines (should be DOMINATED by real methods)")
    base_rows = df[df['method'].str.startswith('baseline')]
    for _, r in base_rows.iterrows():
        md.append(f"- **{r['method']}**: F1_fixA={r['F1_fixA']:.4f}, "
                  f"F1_PA (K=0)={r['F1_PA_K0']:.4f} — note the F1_PA inflation.")

    (out_dir / 'SUMMARY.md').write_text('\n'.join(md))

    print(f"\n=== OUTPUTS ===", flush=True)
    print(f"  {out_dir/'metrics_18methods.csv'}", flush=True)
    print(f"  {out_dir/'pa_k_curves.csv'}", flush=True)
    print(f"  {out_dir/'SUMMARY.md'}", flush=True)


if __name__ == '__main__':
    main()
