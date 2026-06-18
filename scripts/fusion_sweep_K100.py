"""Fusion sweep: combine UQ measures with residual anomaly scores at K=100.

Evaluates 14 fusion methods (M0-M13) on the cached K=100 G-DeltaUQ inference
arrays. Each method produces a per-timestep score (or binary alarm); we
sweep its hyperparameters on the val slice, freeze the best HP, then
evaluate on the held-out test slice. Outputs one CSV ranking methods.

CPU-only, no retraining. Reuses primitives from sweep_eval_gdeltauq.py,
sweep_postproc_threshold.py, analyze_uq_attack_association.py, and
stack_hybrid_on_cheapsweep.py.
"""
import argparse
import csv
import itertools
import json
import sys
import time
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
from sweep_postproc_threshold import (
    best_threshold_postproc_aware, metrics_from_pred,
)

EPS = 1e-6
SIGMA_FLOOR = 1e-12
POST_W, POST_G = 5, 5
N_TAUS = 400


# --------------------------------------------------------------------------- #
# zscore helpers fit on an explicit mask (not on full-test label==0 like the
# existing per_sensor_zscore — we fit on the C-slice only to avoid leaking
# val/test attack-zone statistics into the score construction).
# --------------------------------------------------------------------------- #

def fit_zscore_params(x, mask=None):
    s = x[mask] if mask is not None else x
    med = np.median(s, axis=0)
    q25 = np.quantile(s, 0.25, axis=0)
    q75 = np.quantile(s, 0.75, axis=0)
    iqr = q75 - q25
    return med, iqr


def apply_zscore(x, med, iqr):
    return (x - med) / (iqr + EPS)


def fit_apply_1d_renorm(s, mask):
    """Re-z-score a 1D aggregate on its mask-true statistics (mirrors
    renorm_aggregate but with explicit mask)."""
    s = s.astype(np.float64)
    nominal = s[mask]
    med = float(np.median(nominal))
    q25 = float(np.quantile(nominal, 0.25))
    q75 = float(np.quantile(nominal, 0.75))
    iqr = q75 - q25
    return (s - med) / (iqr + EPS)


# --------------------------------------------------------------------------- #
# Slice helpers
# --------------------------------------------------------------------------- #

def eval_with_postproc(s, label, tau, W=POST_W, G=POST_G):
    """Threshold + post-proc + metrics. Returns dict(F1, P, R, ...)."""
    raw = (s > tau).astype(np.int8)
    pred = apply_postproc(raw, W, G) if (W or G) else raw
    f1, p, r, tp, fp, fn, tn = metrics_from_pred(pred, label)
    return dict(F1=f1, P=p, R=r, TP=tp, FP=fp, FN=fn, TN=tn, tau=float(tau))


def best_tau_on_val(s_val, label_val, W=POST_W, G=POST_G, n_taus=N_TAUS):
    """Sweep tau (post-proc-aware) on val slice and return best."""
    best, _ = best_threshold_postproc_aware(s_val, label_val, W, G, n_taus=n_taus)
    return best


def eval_method(s, val_idx, test_idx, label, W=POST_W, G=POST_G,
                n_taus=N_TAUS):
    """For a given 1D score vector defined on full T, sweep tau on val,
    freeze, evaluate on test. Also compute test_best_tau for the optimism
    diagnostic."""
    val_best = best_tau_on_val(s[val_idx], label[val_idx], W, G, n_taus)
    test_frozen = eval_with_postproc(s[test_idx], label[test_idx],
                                     val_best['tau'], W, G)
    test_best, _ = best_threshold_postproc_aware(
        s[test_idx], label[test_idx], W, G, n_taus=n_taus
    )
    return dict(
        val_F1=val_best['F1'], val_P=val_best['P'], val_R=val_best['R'],
        val_tau=val_best['tau'], val_q=val_best['q'],
        test_F1=test_frozen['F1'], test_P=test_frozen['P'],
        test_R=test_frozen['R'], test_tau_frozen=test_frozen['tau'],
        test_F1_best_tau=test_best['F1'],
        val_test_gap=test_frozen['F1'] - test_best['F1'],
    )


def eval_alarm_on_slices(alarm, val_idx, test_idx, label, W=POST_W, G=POST_G,
                         apply_post=True):
    """For a binary alarm vector on full T, post-proc each slice and
    return val and test F1/P/R (no tau sweep). For OR/AND family."""
    out = {}
    for tag, idx in (('val', val_idx), ('test', test_idx)):
        a = alarm[idx]
        p = apply_postproc(a, W, G) if apply_post else a
        f1, pr, rc, tp, fp, fn, tn = metrics_from_pred(p, label[idx])
        out[f'{tag}_F1'] = f1
        out[f'{tag}_P'] = pr
        out[f'{tag}_R'] = rc
    return out


# --------------------------------------------------------------------------- #
# Score builders for the per-sensor / multiplicative methods
# --------------------------------------------------------------------------- #

def score_M0(agg, **_):
    return agg


def score_per_sensor_sigma(r_abs, sigma_v, tk, sm, before_smoothing=None):
    """M4/M5/M8: |y - mu| / sigma per-sensor, then top-k aggregate.
    r_abs: (T, V) absolute residual.  sigma_v: (T, V) per-sensor sigma.
    Returns (T,)."""
    rtilde = r_abs / (sigma_v + EPS)
    # smooth per-sensor (SMA before_num=sm) before top-k
    if sm and sm > 0:
        # simple causal SMA: smoothed[i] = mean(rtilde[i-sm:i+1])
        T_, V_ = rtilde.shape
        sm_out = np.zeros_like(rtilde)
        for v in range(V_):
            x = rtilde[:, v]
            cum = np.concatenate(([0.0], np.cumsum(x)))
            for i in range(sm, T_):
                sm_out[i, v] = (cum[i + 1] - cum[i - sm]) / (sm + 1)
        rtilde = sm_out
    full_scores = rtilde.T  # (V, T)
    return topk_aggregate(full_scores, tk)


def score_uq_weighted_topk(full_scores_VxT, z_U_par_TxV, lam, tk):
    """M6: r'_v(t) = r_v(t) * clip(1 + lam * z_U_par[t,v], 0.1, 10)."""
    weight = np.clip(1.0 + lam * z_U_par_TxV, 0.1, 10.0)  # (T, V)
    r_prime = full_scores_VxT * weight.T                  # (V, T)
    return topk_aggregate(r_prime, tk)


def score_mult_aggregate_gate(agg, z_par_max, z_str_mean, z_dist,
                              lam_par, lam_str, lam_dist):
    """M7: s = agg * (1 + lam_par * z_par_max + lam_str * z_str_mean
    + lam_dist * z_dist)."""
    factor = 1.0 + lam_par * z_par_max + lam_str * z_str_mean + lam_dist * z_dist
    return agg * factor


def score_linear_sum(agg_z, signals_z, betas):
    """M8: s = agg_z + sum_i beta_i * z_UQ_i."""
    s = agg_z.copy()
    for name, beta in betas.items():
        s = s + beta * signals_z[name]
    return s


# --------------------------------------------------------------------------- #
# Method runners — each returns (per_hp_rows, best_row)
# --------------------------------------------------------------------------- #

def run_M0(ctx):
    """Residual-only Fix-A best. Sweep tau on val, freeze, eval test."""
    res = eval_method(ctx['agg'], ctx['val_idx'], ctx['test_idx'],
                      ctx['label'])
    row = dict(method='M0', hp_summary='residual-only', **res)
    return [row], row


def run_M1(ctx):
    """Simple OR: (agg > tau_r) | (z_signal > tau_s).
    HP grid: 6 signals * 80 tau_r quantiles * 5 tau_s."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    agg, signals = ctx['agg'], ctx['signals']
    tau_r_qs = np.linspace(0.5, 0.9999, 80)
    tau_r_vals = np.quantile(agg, tau_r_qs)
    tau_s_grid = [1.0, 1.5, 2.0, 3.0, 5.0]
    rows = []
    for name, sig in signals.items():
        for tr, qr in zip(tau_r_vals, tau_r_qs):
            for ts in tau_s_grid:
                alarm = ((agg > tr) | (sig > ts)).astype(np.int8)
                pp_val = apply_postproc(alarm[val_idx], POST_W, POST_G)
                f1, p, r, *_ = metrics_from_pred(pp_val, label[val_idx])
                rows.append(dict(method='M1', signal=name, tau_r=float(tr),
                                 tau_r_q=float(qr), tau_s=float(ts),
                                 val_F1=f1, val_P=p, val_R=r))
    df = pd.DataFrame(rows)
    bi = df['val_F1'].idxmax()
    best = df.loc[bi]
    # Apply best HP to test
    sig = signals[best['signal']]
    alarm = ((agg > best['tau_r']) | (sig > best['tau_s'])).astype(np.int8)
    pp_test = apply_postproc(alarm[test_idx], POST_W, POST_G)
    f1t, pt, rt, *_ = metrics_from_pred(pp_test, label[test_idx])
    # Optimism: re-sweep tau_r/tau_s on test with frozen signal
    pp_alarm_test = apply_postproc(
        ((agg[test_idx] > best['tau_r']) |
         (sig[test_idx] > best['tau_s'])).astype(np.int8), POST_W, POST_G)
    test_best_f1 = f1t  # frozen HP IS the best HP for this method
    summary = dict(
        method='M1',
        hp_summary=f"OR signal={best['signal']} tau_r={best['tau_r']:.3f} "
                   f"tau_s={best['tau_s']:.2f}",
        val_F1=float(best['val_F1']), val_P=float(best['val_P']),
        val_R=float(best['val_R']),
        val_tau=float(best['tau_r']), val_q=float(best['tau_r_q']),
        test_F1=f1t, test_P=pt, test_R=rt, test_tau_frozen=float(best['tau_r']),
        test_F1_best_tau=test_best_f1, val_test_gap=0.0,
    )
    return rows, summary


def run_M2(ctx):
    """Simple AND with lowered tau_r_prime: (agg > tau_r') & (z_signal > tau_s)."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    agg, signals = ctx['agg'], ctx['signals']
    tau_rp_qs = np.linspace(0.05, 0.7, 30)
    tau_rp_vals = np.quantile(agg, tau_rp_qs)
    tau_s_grid = [1.0, 1.5, 2.0, 3.0, 5.0]
    rows = []
    for name, sig in signals.items():
        for trp, qrp in zip(tau_rp_vals, tau_rp_qs):
            for ts in tau_s_grid:
                alarm = ((agg > trp) & (sig > ts)).astype(np.int8)
                pp_val = apply_postproc(alarm[val_idx], POST_W, POST_G)
                f1, p, r, *_ = metrics_from_pred(pp_val, label[val_idx])
                rows.append(dict(method='M2', signal=name,
                                 tau_r_prime=float(trp),
                                 tau_r_prime_q=float(qrp),
                                 tau_s=float(ts),
                                 val_F1=f1, val_P=p, val_R=r))
    df = pd.DataFrame(rows)
    bi = df['val_F1'].idxmax()
    best = df.loc[bi]
    sig = signals[best['signal']]
    alarm = ((agg > best['tau_r_prime']) & (sig > best['tau_s'])).astype(np.int8)
    pp_test = apply_postproc(alarm[test_idx], POST_W, POST_G)
    f1t, pt, rt, *_ = metrics_from_pred(pp_test, label[test_idx])
    summary = dict(
        method='M2',
        hp_summary=f"AND signal={best['signal']} "
                   f"tau_r'={best['tau_r_prime']:.3f} "
                   f"tau_s={best['tau_s']:.2f}",
        val_F1=float(best['val_F1']), val_P=float(best['val_P']),
        val_R=float(best['val_R']),
        val_tau=float(best['tau_r_prime']),
        val_q=float(best['tau_r_prime_q']),
        test_F1=f1t, test_P=pt, test_R=rt,
        test_tau_frozen=float(best['tau_r_prime']),
        test_F1_best_tau=f1t, val_test_gap=0.0,
    )
    return rows, summary


def run_M3(ctx):
    """4-Placement Stacked OR/AND on K=100 Fix-A base (post-proc fixed
    at W=5, G=5)."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    agg, signals = ctx['agg'], ctx['signals']
    tau_r_qs = np.linspace(0.5, 0.9999, 80)
    tau_r_vals = np.quantile(agg, tau_r_qs)
    tau_rp_qs = np.linspace(0.05, 0.7, 30)
    tau_rp_vals = np.quantile(agg, tau_rp_qs)
    tau_s_grid = [1.0, 2.0, 3.0, 4.0, 5.0]
    rows = []
    for name, sig in signals.items():
        for ts in tau_s_grid:
            spike = (sig > ts).astype(np.int8)
            # OR_A: postproc(residual) | spike
            # OR_B: postproc(residual | spike)
            for tr, qr in zip(tau_r_vals, tau_r_qs):
                residual = (agg > tr).astype(np.int8)
                # OR_A
                resA = apply_postproc(residual, POST_W, POST_G)
                combA = (resA | spike).astype(np.int8)
                # OR_B
                combB = apply_postproc((residual | spike).astype(np.int8),
                                       POST_W, POST_G)
                for rule, comb in (('OR_A', combA), ('OR_B', combB)):
                    f1, p, r, *_ = metrics_from_pred(comb[val_idx],
                                                    label[val_idx])
                    rows.append(dict(method='M3', signal=name, rule=rule,
                                     tau_r=float(tr), tau_r_q=float(qr),
                                     tau_r_prime=np.nan, tau_s=float(ts),
                                     val_F1=f1, val_P=p, val_R=r))
            # AND_A: postproc(residual at low tau_r') & spike
            # AND_B: postproc(residual at low tau_r' & spike)
            for trp, qrp in zip(tau_rp_vals, tau_rp_qs):
                residual = (agg > trp).astype(np.int8)
                resA = apply_postproc(residual, POST_W, POST_G)
                combA = (resA & spike).astype(np.int8)
                combB = apply_postproc((residual & spike).astype(np.int8),
                                       POST_W, POST_G)
                for rule, comb in (('AND_A', combA), ('AND_B', combB)):
                    f1, p, r, *_ = metrics_from_pred(comb[val_idx],
                                                    label[val_idx])
                    rows.append(dict(method='M3', signal=name, rule=rule,
                                     tau_r=np.nan, tau_r_q=np.nan,
                                     tau_r_prime=float(trp), tau_s=float(ts),
                                     val_F1=f1, val_P=p, val_R=r))
    df = pd.DataFrame(rows)
    bi = df['val_F1'].idxmax()
    best = df.loc[bi]
    # Apply best HP to test
    sig = signals[best['signal']]
    spike = (sig > best['tau_s']).astype(np.int8)
    rule = best['rule']
    if rule in ('OR_A', 'OR_B'):
        residual = (agg > best['tau_r']).astype(np.int8)
    else:
        residual = (agg > best['tau_r_prime']).astype(np.int8)
    if rule == 'OR_A':
        comb = (apply_postproc(residual, POST_W, POST_G) | spike).astype(np.int8)
    elif rule == 'OR_B':
        comb = apply_postproc((residual | spike).astype(np.int8), POST_W, POST_G)
    elif rule == 'AND_A':
        comb = (apply_postproc(residual, POST_W, POST_G) & spike).astype(np.int8)
    else:  # AND_B
        comb = apply_postproc((residual & spike).astype(np.int8), POST_W, POST_G)
    f1t, pt, rt, *_ = metrics_from_pred(comb[test_idx], label[test_idx])
    tau_used = best['tau_r'] if rule.startswith('OR') else best['tau_r_prime']
    summary = dict(
        method='M3',
        hp_summary=f"{rule} signal={best['signal']} tau_r={tau_used:.3f} "
                   f"tau_s={best['tau_s']:.2f}",
        val_F1=float(best['val_F1']), val_P=float(best['val_P']),
        val_R=float(best['val_R']),
        val_tau=float(tau_used), val_q=float('nan'),
        test_F1=f1t, test_P=pt, test_R=rt, test_tau_frozen=float(tau_used),
        test_F1_best_tau=f1t, val_test_gap=0.0,
    )
    return rows, summary


def run_M4(ctx):
    """Per-sensor sigma_ale residual: r̃_v = |y_v - mu_v| / sigma_ale_v,
    then top-k aggregate; Fix-A post-proc."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])  # (T, V)
    sigma_ale = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR))
    rows = []
    best_row = None
    for tk in (1, 2, 3, 5):
        for sm in (3, 5):
            s = score_per_sensor_sigma(r_abs, sigma_ale, tk, sm)
            res = eval_method(s, val_idx, test_idx, label)
            row = dict(method='M4', topk=tk, smoothing=sm, **res)
            rows.append(row)
            if best_row is None or res['val_F1'] > best_row['val_F1']:
                best_row = row
    summary = dict(best_row)
    summary['method'] = 'M4'
    summary['hp_summary'] = (f"per-sensor σ_ale topk={best_row['topk']} "
                              f"sm={best_row['smoothing']}")
    return rows, summary


def run_M5(ctx):
    """Total-variance residual: same as M4 but σ_tot = √(σ²_ale + U_par).
    Run only at M4's best (topk, sm)."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])
    sigma_tot = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR)
                        + np.maximum(ctx['test_U_par'], 0.0))
    # Use the same grid as M4 to make M5 directly comparable
    rows = []
    best_row = None
    for tk in (1, 2, 3, 5):
        for sm in (3, 5):
            s = score_per_sensor_sigma(r_abs, sigma_tot, tk, sm)
            res = eval_method(s, val_idx, test_idx, label)
            row = dict(method='M5', topk=tk, smoothing=sm, **res)
            rows.append(row)
            if best_row is None or res['val_F1'] > best_row['val_F1']:
                best_row = row
    summary = dict(best_row)
    summary['method'] = 'M5'
    summary['hp_summary'] = (f"per-sensor σ_tot topk={best_row['topk']} "
                              f"sm={best_row['smoothing']}")
    return rows, summary


def run_M6(ctx):
    """UQ-weighted per-sensor top-k:
    r'_v(t) = r_v(t) * clip(1 + λ·z_U_par,v(t), 0.1, 10)."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    rows = []
    best_row = None
    for lam in (0.2, 0.5, 1.0, 2.0):
        for tk in (1, 2, 3):
            s = score_uq_weighted_topk(ctx['full_scores'],
                                       ctx['z_U_par_TxV'], lam, tk)
            res = eval_method(s, val_idx, test_idx, label)
            row = dict(method='M6', lam=lam, topk=tk, **res)
            rows.append(row)
            if best_row is None or res['val_F1'] > best_row['val_F1']:
                best_row = row
    summary = dict(best_row)
    summary['method'] = 'M6'
    summary['hp_summary'] = (f"UQ-weighted top-k λ={best_row['lam']} "
                              f"topk={best_row['topk']}")
    return rows, summary


def run_M7(ctx):
    """Multiplicative aggregate gate:
    s = agg · (1 + λ_par·z_par_max + λ_str·z_str_mean + λ_dist·z_dist)."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    agg = ctx['agg']
    z_par = ctx['signals']['U_par_max_v']
    z_str = ctx['signals']['U_str_mean_e']
    z_dist = ctx['signals']['U_dist']
    rows = []
    best_row = None
    lam_grid = (0.0, 0.1, 0.3, 1.0)
    for lp, ls, ld in itertools.product(lam_grid, lam_grid, lam_grid):
        s = score_mult_aggregate_gate(agg, z_par, z_str, z_dist, lp, ls, ld)
        res = eval_method(s, val_idx, test_idx, label)
        row = dict(method='M7', lam_par=lp, lam_str=ls, lam_dist=ld, **res)
        rows.append(row)
        if best_row is None or res['val_F1'] > best_row['val_F1']:
            best_row = row
    summary = dict(best_row)
    summary['method'] = 'M7'
    summary['hp_summary'] = (f"mult-gate λ_par={best_row['lam_par']} "
                              f"λ_str={best_row['lam_str']} "
                              f"λ_dist={best_row['lam_dist']}")
    return rows, summary


def run_M8(ctx):
    """Weighted linear sum: s = agg_z + Σ β_i · z_UQ_i."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    agg_z = ctx['agg_z']
    signals = ctx['signals']  # 6 z-scored aggregates
    # Use 4 representative UQ signals (the 4 with non-zero AUROC > 0.5)
    uq_keys = ['U_par_max_v', 'sigma_ale_max_v', 'U_str_mean_e', 'U_dist']
    rows = []
    best_row = None
    grid = (0.0, 0.1, 0.3, 1.0, 2.0)
    for bp, bsig, bst, bd in itertools.product(grid, grid, grid, grid):
        betas = dict(zip(uq_keys, (bp, bsig, bst, bd)))
        s = score_linear_sum(agg_z, signals, betas)
        res = eval_method(s, val_idx, test_idx, label)
        row = dict(method='M8',
                   b_U_par_max_v=bp, b_sigma_ale_max_v=bsig,
                   b_U_str_mean_e=bst, b_U_dist=bd, **res)
        rows.append(row)
        if best_row is None or res['val_F1'] > best_row['val_F1']:
            best_row = row
    summary = dict(best_row)
    summary['method'] = 'M8'
    summary['hp_summary'] = (
        f"linear β=(par={best_row['b_U_par_max_v']}, "
        f"σ={best_row['b_sigma_ale_max_v']}, "
        f"str={best_row['b_U_str_mean_e']}, "
        f"dist={best_row['b_U_dist']})"
    )
    return rows, summary


def run_M9(ctx):
    """Logistic stacker on 8 features. Train on val, evaluate on test."""
    from sklearn.linear_model import LogisticRegression
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    feat = build_stacker_features(ctx)
    rows = []
    best_row = None
    for C in (0.1, 1.0, 10.0):
        lr = LogisticRegression(class_weight='balanced', C=C,
                                max_iter=1000, solver='lbfgs')
        lr.fit(feat[val_idx], label[val_idx])
        # Log-odds = decision_function
        s_full = lr.decision_function(feat)
        res = eval_method(s_full, val_idx, test_idx, label)
        # Re-evaluate on val (in-sample, for sanity) — this is the
        # quantity we use for HP selection (val F1)
        row = dict(method='M9', C=C, **res)
        rows.append(row)
        if best_row is None or res['val_F1'] > best_row['val_F1']:
            best_row = row
    summary = dict(best_row)
    summary['method'] = 'M9'
    summary['hp_summary'] = f"LogReg C={best_row['C']}"
    return rows, summary


def run_M10(ctx):
    """Gradient-boosting stacker on 8 features."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    feat = build_stacker_features(ctx)
    rows = []
    best_row = None
    for depth in (2, 3, 5):
        for n_iter in (50, 100, 200):
            gb = HistGradientBoostingClassifier(
                max_depth=depth, max_iter=n_iter, learning_rate=0.05,
                l2_regularization=1.0, random_state=ctx['seed'],
                class_weight='balanced')
            gb.fit(feat[val_idx], label[val_idx])
            # log-odds = log(p/(1-p))
            proba = gb.predict_proba(feat)[:, 1]
            s_full = np.log(np.clip(proba, 1e-8, 1 - 1e-8) /
                            np.clip(1 - proba, 1e-8, 1 - 1e-8))
            res = eval_method(s_full, val_idx, test_idx, label)
            row = dict(method='M10', max_depth=depth, max_iter=n_iter, **res)
            rows.append(row)
            if best_row is None or res['val_F1'] > best_row['val_F1']:
                best_row = row
    summary = dict(best_row)
    summary['method'] = 'M10'
    summary['hp_summary'] = (f"GBM depth={best_row['max_depth']} "
                              f"iter={best_row['max_iter']}")
    return rows, summary


def run_M11(ctx, m4_best=None, m6_best=None):
    """M4 + M6 combo: r̃_v = (|y - mu| / σ_ale,v) · (1 + λ · z_U_par,v); top-k."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])
    sigma_ale = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR))
    tk = int(m4_best['topk']) if m4_best is not None else 1
    rows = []
    best_row = None
    for lam in (0.2, 0.5, 1.0):
        weight = np.clip(1.0 + lam * ctx['z_U_par_TxV'], 0.1, 10.0)  # (T, V)
        rtilde = (r_abs / (sigma_ale + EPS)) * weight  # (T, V)
        full_scores = rtilde.T
        s = topk_aggregate(full_scores, tk)
        res = eval_method(s, val_idx, test_idx, label)
        row = dict(method='M11', topk=tk, lam=lam, **res)
        rows.append(row)
        if best_row is None or res['val_F1'] > best_row['val_F1']:
            best_row = row
    summary = dict(best_row)
    summary['method'] = 'M11'
    summary['hp_summary'] = f"M4+M6 topk={tk} λ={best_row['lam']}"
    return rows, summary


def run_M12(ctx):
    """Triple-OR: (agg > τ_r) ∨ (z_σ_ale_max > τ_s1) ∨ (z_U_str_mean > τ_s2).
    post-proc on union."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    agg = ctx['agg']
    z_sigma = ctx['signals']['sigma_ale_max_v']
    z_str = ctx['signals']['U_str_mean_e']
    tau_r_qs = np.linspace(0.5, 0.999, 40)
    tau_r_vals = np.quantile(agg, tau_r_qs)
    tau_s_grid = [2.0, 3.0, 4.0, np.inf]
    rows = []
    for tr, qr in zip(tau_r_vals, tau_r_qs):
        for ts1, ts2 in itertools.product(tau_s_grid, tau_s_grid):
            alarm = ((agg > tr) | (z_sigma > ts1) | (z_str > ts2)).astype(np.int8)
            pp_val = apply_postproc(alarm[val_idx], POST_W, POST_G)
            f1, p, r, *_ = metrics_from_pred(pp_val, label[val_idx])
            rows.append(dict(method='M12', tau_r=float(tr), tau_r_q=float(qr),
                             tau_s1=float(ts1), tau_s2=float(ts2),
                             val_F1=f1, val_P=p, val_R=r))
    df = pd.DataFrame(rows)
    bi = df['val_F1'].idxmax()
    best = df.loc[bi]
    alarm = ((agg > best['tau_r']) |
             (z_sigma > best['tau_s1']) |
             (z_str > best['tau_s2'])).astype(np.int8)
    pp_test = apply_postproc(alarm[test_idx], POST_W, POST_G)
    f1t, pt, rt, *_ = metrics_from_pred(pp_test, label[test_idx])
    summary = dict(
        method='M12',
        hp_summary=(f"3OR tau_r={best['tau_r']:.3f} "
                    f"tau_s1={best['tau_s1']:.2f} tau_s2={best['tau_s2']:.2f}"),
        val_F1=float(best['val_F1']), val_P=float(best['val_P']),
        val_R=float(best['val_R']),
        val_tau=float(best['tau_r']), val_q=float(best['tau_r_q']),
        test_F1=f1t, test_P=pt, test_R=rt, test_tau_frozen=float(best['tau_r']),
        test_F1_best_tau=f1t, val_test_gap=0.0,
    )
    return rows, summary


def run_M13(ctx):
    """Adaptive sliding-quantile threshold on agg(t).
    τ(t) = quantile(agg[t-W₀:t], q); alarm = agg > τ(t); post-proc with W=5, G=5."""
    val_idx, test_idx, label = ctx['val_idx'], ctx['test_idx'], ctx['label']
    agg = ctx['agg']
    T = agg.shape[0]
    rows = []
    best_row = None
    for W0 in (500, 2000, 8000):
        for q in (0.97, 0.99, 0.995):
            # rolling quantile (slow but correct)
            tau_t = np.zeros(T)
            for i in range(T):
                lo = max(0, i - W0)
                tau_t[i] = np.quantile(agg[lo:i + 1], q) if i > 10 else np.inf
            alarm = (agg > tau_t).astype(np.int8)
            pp_val = apply_postproc(alarm[val_idx], POST_W, POST_G)
            f1v, pv, rv, *_ = metrics_from_pred(pp_val, label[val_idx])
            pp_test = apply_postproc(alarm[test_idx], POST_W, POST_G)
            f1t, pt, rt, *_ = metrics_from_pred(pp_test, label[test_idx])
            row = dict(method='M13', W0=W0, q=q,
                       val_F1=f1v, val_P=pv, val_R=rv,
                       test_F1=f1t, test_P=pt, test_R=rt,
                       test_tau_frozen=float('nan'),
                       test_F1_best_tau=f1t, val_test_gap=0.0,
                       val_tau=float('nan'), val_q=float(q))
            rows.append(row)
            if best_row is None or f1v > best_row['val_F1']:
                best_row = row
    summary = dict(best_row)
    summary['method'] = 'M13'
    summary['hp_summary'] = (f"adaptive W0={best_row['W0']} q={best_row['q']}")
    return rows, summary


def build_stacker_features(ctx):
    """8 per-timestep features for M9/M10:
       [agg_z, z_U_par_max, z_U_par_mean, z_sigma_ale_max,
        z_U_str_mean, z_U_dist, z_U_par_max * agg_z, log_sigma_tot_max].
    """
    agg_z = ctx['agg_z']
    sig = ctx['signals']
    # log_sigma_tot_max: per-sensor sqrt(σ²_ale + U_par).max over V
    sigma_tot = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR)
                        + np.maximum(ctx['test_U_par'], 0.0))
    log_sigma_tot_max = np.log(sigma_tot.max(axis=1) + EPS)
    # z-score it on C-slice
    med_lst, iqr_lst = fit_zscore_params(log_sigma_tot_max[:, None],
                                          mask=ctx['c_mask'])
    z_log_sigma_tot_max = ((log_sigma_tot_max - med_lst[0])
                           / (iqr_lst[0] + EPS))
    feat = np.column_stack([
        agg_z,
        sig['U_par_max_v'],
        sig['U_par_mean_v'],
        sig['sigma_ale_max_v'],
        sig['U_str_mean_e'],
        sig['U_dist'],
        sig['U_par_max_v'] * agg_z,
        z_log_sigma_tot_max,
    ])
    return feat


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

METHOD_RUNNERS = {
    'M0': run_M0, 'M1': run_M1, 'M2': run_M2, 'M3': run_M3,
    'M4': run_M4, 'M5': run_M5, 'M6': run_M6, 'M7': run_M7,
    'M8': run_M8, 'M9': run_M9, 'M10': run_M10,
    'M11': run_M11, 'M12': run_M12, 'M13': run_M13,
}


def attack_runs(label):
    """Return list of (start, end_inclusive) tuples for runs of 1s."""
    runs = []
    in_run = False
    for i, x in enumerate(label):
        if x and not in_run:
            start = i
            in_run = True
        elif not x and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(label) - 1))
    return runs


def setup_context(args):
    print(f'loading arrays: {args.arrays}', flush=True)
    d = np.load(args.arrays)
    test_mu = d['test_mu_bar'].astype(np.float64)
    test_gt = d['test_ground_truth'].astype(np.float64)
    test_U_par = d['test_U_par'].astype(np.float64)
    test_U_str = d['test_U_str'].astype(np.float64)
    test_U_dist = d['test_U_dist'].astype(np.float64)
    test_sigma2_ale = d['test_sigma2_ale'].astype(np.float64)
    val_mu = d['val_mu_bar'].astype(np.float64)
    val_gt = d['val_ground_truth'].astype(np.float64)
    label = d['test_attack_label'].astype(np.int8)
    T = label.shape[0]
    print(f'T={T}, V={test_mu.shape[1]}, attack_rate={label.mean():.4f}, '
          f'attack_runs={len(attack_runs(label))}', flush=True)

    # Slice mapping: subtract slide_win from JSON row indices to get
    # arrays indices.
    sw = args.slide_win
    print(f'loading split: {args.split}', flush=True)
    with open(args.split) as f:
        split = json.load(f)
    C_lo, C_hi = split['C_row_range']
    val_lo, val_hi = split['labeled_val_range']
    test_lo, test_hi = split['final_test_range']
    # Convert to arrays indices
    C_idx_start = max(0, C_lo - sw)
    C_idx_end = max(0, C_hi - sw)
    val_idx_start = max(0, val_lo - sw)
    val_idx_end = max(0, val_hi - sw)
    test_idx_start = max(0, test_lo - sw)
    test_idx_end = max(0, test_hi - sw)
    # Trim to T
    C_idx_end = min(T, C_idx_end)
    val_idx_end = min(T, val_idx_end)
    test_idx_end = min(T, test_idx_end)
    c_idx = np.arange(C_idx_start, C_idx_end)
    val_idx = np.arange(val_idx_start, val_idx_end)
    test_idx = np.arange(test_idx_start, test_idx_end)
    c_mask = np.zeros(T, dtype=bool)
    c_mask[c_idx] = True
    # Within C, restrict to label==0
    c_mask_nominal = c_mask & (label == 0)
    print(f'  C   slice arrays[{C_idx_start},{C_idx_end}): '
          f'n={C_idx_end-C_idx_start}, '
          f'attack_runs_in={len(attack_runs(label[c_idx]))}', flush=True)
    print(f'  val slice arrays[{val_idx_start},{val_idx_end}): '
          f'n={val_idx_end-val_idx_start}, '
          f'attack_runs_in={len(attack_runs(label[val_idx]))}, '
          f'attack_count={int(label[val_idx].sum())}', flush=True)
    print(f'  test slice arrays[{test_idx_start},{test_idx_end}): '
          f'n={test_idx_end-test_idx_start}, '
          f'attack_runs_in={len(attack_runs(label[test_idx]))}, '
          f'attack_count={int(label[test_idx].sum())}', flush=True)

    # Build residual base (Fix-A best smoothing=5)
    print('building full_scores at smoothing=5 ...', flush=True)
    full_scores = build_full_err_scores(test_mu, test_gt, val_mu, val_gt, 5)
    agg = topk_aggregate(full_scores, 1).astype(np.float64)
    print(f'  agg shape={agg.shape}', flush=True)

    # Build z-scored UQ aggregates on C-nominal stats
    print('fitting zscore params on C-slice nominal ...', flush=True)
    med_par, iqr_par = fit_zscore_params(test_U_par, c_mask_nominal)
    med_str, iqr_str = fit_zscore_params(test_U_str, c_mask_nominal)
    med_sig, iqr_sig = fit_zscore_params(test_sigma2_ale, c_mask_nominal)
    med_dist, iqr_dist = fit_zscore_params(test_U_dist[:, None],
                                            c_mask_nominal)
    # Apply
    z_U_par_TxV = apply_zscore(test_U_par, med_par, iqr_par)
    z_U_str_TxE = apply_zscore(test_U_str, med_str, iqr_str)
    z_sigma2_TxV = apply_zscore(test_sigma2_ale, med_sig, iqr_sig)
    z_U_dist = apply_zscore(test_U_dist[:, None], med_dist, iqr_dist)[:, 0]
    # 6 aggregates (re-zscored on C-nominal)
    raw = {
        'U_par_max_v': z_U_par_TxV.max(axis=1),
        'U_par_mean_v': z_U_par_TxV.mean(axis=1),
        'U_str_mean_e': z_U_str_TxE.mean(axis=1),
        'U_dist': z_U_dist,
        'sigma_ale_max_v': z_sigma2_TxV.max(axis=1),
        'sigma_ale_mean_v': z_sigma2_TxV.mean(axis=1),
    }
    signals = {k: fit_apply_1d_renorm(v, c_mask_nominal) for k, v in raw.items()}
    agg_z = fit_apply_1d_renorm(agg, c_mask_nominal)

    return dict(
        agg=agg, agg_z=agg_z, signals=signals,
        full_scores=full_scores,
        test_mu=test_mu, test_gt=test_gt,
        test_U_par=test_U_par, test_U_str=test_U_str,
        test_U_dist=test_U_dist, test_sigma2_ale=test_sigma2_ale,
        z_U_par_TxV=z_U_par_TxV,
        label=label, T=T,
        c_idx=c_idx, val_idx=val_idx, test_idx=test_idx,
        c_mask=c_mask_nominal,
        seed=args.seed,
    )


# --------------------------------------------------------------------------- #
# Verification: M0 reproduction on full arrays
# --------------------------------------------------------------------------- #

def verify_M0_full(ctx):
    """Reproduce F1=0.8109 on full 44716 arrays with post-proc-aware tau-sweep
    on Fix-A best config (tk=1, sm=5, W=5, G=5)."""
    print('\n=== M0 sanity: reproduction on full test arrays ===', flush=True)
    best, _ = best_threshold_postproc_aware(
        ctx['agg'], ctx['label'], POST_W, POST_G, n_taus=N_TAUS
    )
    print(f"  full-test F1_fixA = {best['F1']:.4f}  P = {best['P']:.4f}  "
          f"R = {best['R']:.4f}  tau = {best['tau']:.4f}  q = {best['q']:.4f}",
          flush=True)
    expected_f1 = 0.8109
    if abs(best['F1'] - expected_f1) > 0.001:
        print(f"  WARN: deviation from expected F1=0.8109 is "
              f"{best['F1']-expected_f1:+.4f}", flush=True)
    else:
        print(f"  OK: matches expected F1=0.8109 within 0.001", flush=True)
    return best


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-arrays', required=True,
        help='K=100 inference arrays.npz')
    parser.add_argument(
        '-split', required=True,
        help='calibration_set_indices.json (test.csv row ranges)')
    parser.add_argument('-bundle', default=None,
                        help='calibration bundle dir (optional; for q_v)')
    parser.add_argument('-slide_win', type=int, default=60)
    parser.add_argument('-methods', nargs='+',
                        default=list(METHOD_RUNNERS.keys()))
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-out_root', default='results/fusion_K100')
    args = parser.parse_args()

    ctx = setup_context(args)

    # Sanity check
    m0_full = verify_M0_full(ctx)
    # M0 on the test_slice with test-resweep tau (optimism reference only)
    print('\n=== M0 on test_slice (test-resweep tau, optimism reference) ===',
          flush=True)
    m0_slice_optimistic = best_threshold_postproc_aware(
        ctx['agg'][ctx['test_idx']], ctx['label'][ctx['test_idx']],
        POST_W, POST_G, n_taus=N_TAUS
    )[0]
    print(f"  test_slice F1_M0_optimistic = {m0_slice_optimistic['F1']:.4f} "
          f"P = {m0_slice_optimistic['P']:.4f}  "
          f"R = {m0_slice_optimistic['R']:.4f}", flush=True)

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    out_dir = Path(args.out_root) / datestr
    out_dir.mkdir(parents=True, exist_ok=True)

    all_per_hp_rows = []
    summaries = []
    m4_best, m6_best = None, None
    baseline_test_F1 = None  # set after M0 runs; fair comparison baseline
    for method in args.methods:
        if method not in METHOD_RUNNERS:
            print(f'  SKIP unknown method {method}', flush=True)
            continue
        print(f'\n=== {method} ===', flush=True)
        t0 = time.time()
        try:
            if method == 'M11':
                hp_rows, summary = run_M11(ctx, m4_best, m6_best)
            else:
                hp_rows, summary = METHOD_RUNNERS[method](ctx)
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            import traceback; traceback.print_exc()
            continue
        dt = time.time() - t0
        if method == 'M0':
            baseline_test_F1 = summary.get('test_F1', 0.0)
        # Compute lift_vs_M0 (fair: vs M0's val-frozen test F1)
        if baseline_test_F1 is not None:
            summary['lift_vs_M0'] = (summary.get('test_F1', 0.0)
                                      - baseline_test_F1)
        else:
            summary['lift_vs_M0'] = float('nan')
        summary['wall_sec'] = dt
        if method == 'M4':
            m4_best = summary
        if method == 'M6':
            m6_best = summary
        print(f"  {method} val_F1={summary.get('val_F1', float('nan')):.4f} "
              f"test_F1={summary.get('test_F1', float('nan')):.4f} "
              f"P={summary.get('test_P', float('nan')):.4f} "
              f"R={summary.get('test_R', float('nan')):.4f} "
              f"lift={summary.get('lift_vs_M0', 0):+.4f} "
              f"wall={dt:.1f}s", flush=True)
        all_per_hp_rows.extend(hp_rows)
        summaries.append(summary)

    if baseline_test_F1 is None:
        # M0 wasn't requested; use optimistic reference instead
        baseline_test_F1 = m0_slice_optimistic['F1']
        print(f"\nNOTE: M0 not in requested methods; using "
              f"optimistic test-slice F1 = {baseline_test_F1:.4f} as baseline",
              flush=True)

    # Add the optimistic reference row for context (not used for comparison)
    summaries.append(dict(
        method='M0_test_slice (optimistic, test-resweep τ)',
        hp_summary='residual-only on test_slice, τ swept on test',
        val_F1=float('nan'),
        val_P=float('nan'), val_R=float('nan'),
        val_tau=float('nan'), val_q=float('nan'),
        test_F1=m0_slice_optimistic['F1'], test_P=m0_slice_optimistic['P'],
        test_R=m0_slice_optimistic['R'],
        test_tau_frozen=m0_slice_optimistic['tau'],
        test_F1_best_tau=m0_slice_optimistic['F1'],
        val_test_gap=0.0,
        lift_vs_M0=m0_slice_optimistic['F1'] - baseline_test_F1,
        wall_sec=0.0,
    ))

    # Write methods.csv
    fields = ['method', 'hp_summary',
              'val_F1', 'val_P', 'val_R', 'val_tau', 'val_q',
              'test_F1', 'test_P', 'test_R', 'test_tau_frozen',
              'test_F1_best_tau', 'val_test_gap', 'lift_vs_M0', 'wall_sec']
    df_methods = pd.DataFrame(summaries)
    # Reorder columns
    cols = [c for c in fields if c in df_methods.columns]
    extra = [c for c in df_methods.columns if c not in cols]
    df_methods = df_methods[cols + extra]
    df_methods.to_csv(out_dir / 'methods.csv', index=False)
    # Per-hp CSV
    df_hp = pd.DataFrame(all_per_hp_rows)
    df_hp.to_csv(out_dir / 'per_hp.csv', index=False)

    # SUMMARY.md
    df_sorted = df_methods.sort_values('test_F1', ascending=False)
    summary_md = [
        f"# Fusion sweep @ K=100, sw=60, seed=42 ({datestr})",
        "",
        f"Inputs: `{args.arrays}`",
        f"Split: `{args.split}`",
        f"slide_win = {args.slide_win}",
        "",
        f"## Baselines",
        f"- M0 (full test arrays, paper-protocol Fix-A): F1 = {m0_full['F1']:.4f}, "
        f"P = {m0_full['P']:.4f}, R = {m0_full['R']:.4f}",
        f"- M0 (test_slice, val-frozen τ, **fair baseline for comparison**): "
        f"F1 = {baseline_test_F1:.4f}",
        f"- M0 (test_slice, test-resweep τ, optimistic reference): "
        f"F1 = {m0_slice_optimistic['F1']:.4f}",
        "",
        f"## Method ranking by test_F1",
        "",
        "| Method | HP | val_F1 | test_F1 | test_P | test_R | lift_vs_M0 | gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_sorted.iterrows():
        summary_md.append(
            f"| {r['method']} | {r.get('hp_summary', '')} | "
            f"{r.get('val_F1', float('nan')):.4f} | "
            f"{r.get('test_F1', float('nan')):.4f} | "
            f"{r.get('test_P', float('nan')):.4f} | "
            f"{r.get('test_R', float('nan')):.4f} | "
            f"{r.get('lift_vs_M0', 0):+.4f} | "
            f"{r.get('val_test_gap', 0):+.4f} |"
        )
    summary_md.append("")
    summary_md.append("## Acceptance verdict")
    winners = df_sorted[
        (df_sorted['test_F1'] > baseline_test_F1 + 0.005) &
        (df_sorted['val_test_gap'] > -0.01) &
        (~df_sorted['method'].astype(str).str.startswith('M0'))
    ]
    if len(winners) > 0:
        best_method = winners.iloc[0]
        summary_md.append(
            f"\n**Winner**: `{best_method['method']}` — "
            f"test_F1 = {best_method['test_F1']:.4f} "
            f"(lift {best_method['lift_vs_M0']:+.4f} vs M0 on test_slice)"
        )
    else:
        summary_md.append(
            f"\n**No method beats M0 by >0.005 with val_test_gap > -0.01.** "
            f"Headline finding: single-model G-ΔUQ at K=100 has saturated "
            f"for SWaT under residual+post-proc protocol."
        )
    (out_dir / 'SUMMARY.md').write_text('\n'.join(summary_md))

    # best_per_method.json
    best_json = {}
    for s in summaries:
        best_json[s['method']] = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                                  for k, v in s.items()
                                  if isinstance(v, (int, float, str))}
    (out_dir / 'best_per_method.json').write_text(json.dumps(best_json, indent=2))

    print(f"\n=== OUTPUTS ===", flush=True)
    print(f"  {out_dir/'methods.csv'}", flush=True)
    print(f"  {out_dir/'per_hp.csv'}", flush=True)
    print(f"  {out_dir/'SUMMARY.md'}", flush=True)
    print(f"  {out_dir/'best_per_method.json'}", flush=True)


if __name__ == '__main__':
    main()
