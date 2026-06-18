"""Paper-protocol full-arrays fusion sweep at K=100.

Same 14 methods (M0-M13) as fusion_sweep_K100.py, but each method is
evaluated on the FULL 44,716-timestep arrays with τ swept on the full
arrays — matching the paper-protocol used for the published F1=0.8109
baseline.

The val/test split semantics here are different from the slice script:
- C-slice is still used to fit per-sensor zscore params (median, IQR) for
  the 6 UQ aggregate signals — these params are NOT label-dependent past
  the C-slice nominal mask.
- val_slice is used ONLY as the training set for learned stackers
  (M9 LogReg, M10 GBM). All other methods don't need a training set.
- Final F1 is reported on the full arrays with τ chosen via
  best_threshold_postproc_aware on the full arrays — matching the
  protocol of the published Fix-A best at F1=0.8109.

This is "optimistic" in the sense that τ is selected with access to the
full eval labels, but it is the same protocol the published baseline
uses, so M1-M13 are now apples-to-apples comparable against F1=0.8109.

CPU-only, no retraining. Reuses primitives.
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
# zscore helpers
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
    s = s.astype(np.float64)
    nominal = s[mask]
    med = float(np.median(nominal))
    q25 = float(np.quantile(nominal, 0.25))
    q75 = float(np.quantile(nominal, 0.75))
    iqr = q75 - q25
    return (s - med) / (iqr + EPS)


# --------------------------------------------------------------------------- #
# Eval helpers (full-arrays protocol)
# --------------------------------------------------------------------------- #

def eval_score_full(s, label, W=POST_W, G=POST_G, n_taus=N_TAUS):
    """Score-based method: sweep τ on full arrays, return best."""
    best, _ = best_threshold_postproc_aware(s, label, W, G, n_taus=n_taus)
    return dict(F1=best['F1'], P=best['P'], R=best['R'],
                tau=float(best['tau']), q=float(best['q']))


def eval_alarm_full(alarm, label, W=POST_W, G=POST_G):
    """Alarm-based method (OR/AND): apply post-proc, compute F1 on full."""
    pp = apply_postproc(alarm, W, G)
    f1, p, r, *_ = metrics_from_pred(pp, label)
    return dict(F1=f1, P=p, R=r, tau=float('nan'), q=float('nan'))


# --------------------------------------------------------------------------- #
# Score builders
# --------------------------------------------------------------------------- #

def score_per_sensor_sigma(r_abs, sigma_v, tk, sm):
    rtilde = r_abs / (sigma_v + EPS)
    if sm and sm > 0:
        T_, V_ = rtilde.shape
        sm_out = np.zeros_like(rtilde)
        for v in range(V_):
            x = rtilde[:, v]
            cum = np.concatenate(([0.0], np.cumsum(x)))
            for i in range(sm, T_):
                sm_out[i, v] = (cum[i + 1] - cum[i - sm]) / (sm + 1)
        rtilde = sm_out
    return topk_aggregate(rtilde.T, tk)


def score_uq_weighted_topk(full_scores_VxT, z_U_par_TxV, lam, tk):
    weight = np.clip(1.0 + lam * z_U_par_TxV, 0.1, 10.0)
    r_prime = full_scores_VxT * weight.T
    return topk_aggregate(r_prime, tk)


def score_mult_aggregate_gate(agg, z_par_max, z_str_mean, z_dist,
                              lam_par, lam_str, lam_dist):
    factor = 1.0 + lam_par * z_par_max + lam_str * z_str_mean + lam_dist * z_dist
    return agg * factor


def score_linear_sum(agg_z, signals_z, betas):
    s = agg_z.copy()
    for name, beta in betas.items():
        s = s + beta * signals_z[name]
    return s


def build_stacker_features(ctx):
    """8 per-timestep features for M9/M10."""
    agg_z = ctx['agg_z']
    sig = ctx['signals']
    sigma_tot = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR)
                        + np.maximum(ctx['test_U_par'], 0.0))
    log_sigma_tot_max = np.log(sigma_tot.max(axis=1) + EPS)
    med, iqr = fit_zscore_params(log_sigma_tot_max[:, None],
                                  mask=ctx['c_mask'])
    z_log_sigma_tot_max = ((log_sigma_tot_max - med[0]) / (iqr[0] + EPS))
    cols = [
        agg_z,
        sig['U_par_max_v'],
        sig['U_par_mean_v'],
        sig['sigma_ale_max_v'],
    ]
    if 'U_str_mean_e' in sig:          # attention-free methods: 7/8 features
        cols.append(sig['U_str_mean_e'])
    cols += [
        sig['U_dist'],
        sig['U_par_max_v'] * agg_z,
        z_log_sigma_tot_max,
    ]
    feat = np.column_stack(cols)
    return feat


# --------------------------------------------------------------------------- #
# Method runners — full-arrays protocol
# --------------------------------------------------------------------------- #

def run_M0(ctx):
    """Residual-only Fix-A best on full arrays."""
    res = eval_score_full(ctx['agg'], ctx['label'])
    row = dict(method='M0', hp_summary='residual-only', **res)
    return [row], row


def run_M1(ctx):
    """Simple OR on full arrays."""
    agg, signals, label = ctx['agg'], ctx['signals'], ctx['label']
    tau_r_qs = np.linspace(0.5, 0.9999, 80)
    tau_r_vals = np.quantile(agg, tau_r_qs)
    tau_s_grid = [1.0, 1.5, 2.0, 3.0, 5.0]
    rows = []
    best_row = None
    for name, sig in signals.items():
        for tr, qr in zip(tau_r_vals, tau_r_qs):
            for ts in tau_s_grid:
                alarm = ((agg > tr) | (sig > ts)).astype(np.int8)
                res = eval_alarm_full(alarm, label)
                row = dict(method='M1', signal=name,
                            tau_r=float(tr), tau_r_q=float(qr),
                            tau_s=float(ts), **res)
                rows.append(row)
                if best_row is None or row['F1'] > best_row['F1']:
                    best_row = row
    summary = dict(method='M1',
                   hp_summary=(f"OR signal={best_row['signal']} "
                                f"tau_r={best_row['tau_r']:.3f} "
                                f"tau_s={best_row['tau_s']:.2f}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau_r'], q=best_row['tau_r_q'])
    return rows, summary


def run_M2(ctx):
    """Simple AND with lowered tau_r' on full arrays."""
    agg, signals, label = ctx['agg'], ctx['signals'], ctx['label']
    tau_rp_qs = np.linspace(0.05, 0.7, 30)
    tau_rp_vals = np.quantile(agg, tau_rp_qs)
    tau_s_grid = [1.0, 1.5, 2.0, 3.0, 5.0]
    rows = []
    best_row = None
    for name, sig in signals.items():
        for trp, qrp in zip(tau_rp_vals, tau_rp_qs):
            for ts in tau_s_grid:
                alarm = ((agg > trp) & (sig > ts)).astype(np.int8)
                res = eval_alarm_full(alarm, label)
                row = dict(method='M2', signal=name,
                            tau_r_prime=float(trp),
                            tau_r_prime_q=float(qrp),
                            tau_s=float(ts), **res)
                rows.append(row)
                if best_row is None or row['F1'] > best_row['F1']:
                    best_row = row
    summary = dict(method='M2',
                   hp_summary=(f"AND signal={best_row['signal']} "
                                f"tau_r'={best_row['tau_r_prime']:.3f} "
                                f"tau_s={best_row['tau_s']:.2f}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau_r_prime'], q=best_row['tau_r_prime_q'])
    return rows, summary


def run_M3(ctx):
    """4-Placement Stacked OR/AND on full arrays."""
    agg, signals, label = ctx['agg'], ctx['signals'], ctx['label']
    tau_r_qs = np.linspace(0.5, 0.9999, 80)
    tau_r_vals = np.quantile(agg, tau_r_qs)
    tau_rp_qs = np.linspace(0.05, 0.7, 30)
    tau_rp_vals = np.quantile(agg, tau_rp_qs)
    tau_s_grid = [1.0, 2.0, 3.0, 4.0, 5.0]
    rows = []
    best_row = None
    for name, sig in signals.items():
        for ts in tau_s_grid:
            spike = (sig > ts).astype(np.int8)
            for tr, qr in zip(tau_r_vals, tau_r_qs):
                residual = (agg > tr).astype(np.int8)
                resA = apply_postproc(residual, POST_W, POST_G)
                combA = (resA | spike).astype(np.int8)
                combB = apply_postproc((residual | spike).astype(np.int8),
                                       POST_W, POST_G)
                for rule, comb in (('OR_A', combA), ('OR_B', combB)):
                    f1, p, r, *_ = metrics_from_pred(comb, label)
                    row = dict(method='M3', signal=name, rule=rule,
                                tau_r=float(tr), tau_r_q=float(qr),
                                tau_r_prime=np.nan, tau_s=float(ts),
                                F1=f1, P=p, R=r)
                    rows.append(row)
                    if best_row is None or f1 > best_row['F1']:
                        best_row = row
            for trp, qrp in zip(tau_rp_vals, tau_rp_qs):
                residual = (agg > trp).astype(np.int8)
                resA = apply_postproc(residual, POST_W, POST_G)
                combA = (resA & spike).astype(np.int8)
                combB = apply_postproc((residual & spike).astype(np.int8),
                                       POST_W, POST_G)
                for rule, comb in (('AND_A', combA), ('AND_B', combB)):
                    f1, p, r, *_ = metrics_from_pred(comb, label)
                    row = dict(method='M3', signal=name, rule=rule,
                                tau_r=np.nan, tau_r_q=np.nan,
                                tau_r_prime=float(trp), tau_s=float(ts),
                                F1=f1, P=p, R=r)
                    rows.append(row)
                    if best_row is None or f1 > best_row['F1']:
                        best_row = row
    tau_used = (best_row.get('tau_r')
                if best_row['rule'].startswith('OR')
                else best_row.get('tau_r_prime'))
    summary = dict(method='M3',
                   hp_summary=(f"{best_row['rule']} signal={best_row['signal']} "
                                f"tau={tau_used:.3f} tau_s={best_row['tau_s']:.2f}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=tau_used, q=float('nan'))
    return rows, summary


def run_M4(ctx):
    """Per-sensor σ_ale residual on full arrays."""
    label = ctx['label']
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])
    sigma_ale = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR))
    rows = []
    best_row = None
    for tk in (1, 2, 3, 5):
        for sm in (3, 5):
            s = score_per_sensor_sigma(r_abs, sigma_ale, tk, sm)
            res = eval_score_full(s, label)
            row = dict(method='M4', topk=tk, smoothing=sm, **res)
            rows.append(row)
            if best_row is None or row['F1'] > best_row['F1']:
                best_row = row
    summary = dict(method='M4',
                   hp_summary=(f"per-sensor σ_ale topk={best_row['topk']} "
                                f"sm={best_row['smoothing']}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M5(ctx):
    """Total-variance residual on full arrays."""
    label = ctx['label']
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])
    sigma_tot = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR)
                        + np.maximum(ctx['test_U_par'], 0.0))
    rows = []
    best_row = None
    for tk in (1, 2, 3, 5):
        for sm in (3, 5):
            s = score_per_sensor_sigma(r_abs, sigma_tot, tk, sm)
            res = eval_score_full(s, label)
            row = dict(method='M5', topk=tk, smoothing=sm, **res)
            rows.append(row)
            if best_row is None or row['F1'] > best_row['F1']:
                best_row = row
    summary = dict(method='M5',
                   hp_summary=(f"per-sensor σ_tot topk={best_row['topk']} "
                                f"sm={best_row['smoothing']}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M6(ctx):
    """UQ-weighted per-sensor top-k on full arrays."""
    label = ctx['label']
    rows = []
    best_row = None
    for lam in (0.2, 0.5, 1.0, 2.0):
        for tk in (1, 2, 3):
            s = score_uq_weighted_topk(ctx['full_scores'],
                                       ctx['z_U_par_TxV'], lam, tk)
            res = eval_score_full(s, label)
            row = dict(method='M6', lam=lam, topk=tk, **res)
            rows.append(row)
            if best_row is None or row['F1'] > best_row['F1']:
                best_row = row
    summary = dict(method='M6',
                   hp_summary=(f"UQ-weighted top-k λ={best_row['lam']} "
                                f"topk={best_row['topk']}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M7(ctx):
    """Multiplicative aggregate gate on full arrays."""
    label = ctx['label']
    agg = ctx['agg']
    z_par = ctx['signals']['U_par_max_v']
    z_str = ctx['signals']['U_str_mean_e']
    z_dist = ctx['signals']['U_dist']
    rows = []
    best_row = None
    lam_grid = (0.0, 0.1, 0.3, 1.0)
    for lp, ls, ld in itertools.product(lam_grid, lam_grid, lam_grid):
        s = score_mult_aggregate_gate(agg, z_par, z_str, z_dist, lp, ls, ld)
        res = eval_score_full(s, label)
        row = dict(method='M7', lam_par=lp, lam_str=ls, lam_dist=ld, **res)
        rows.append(row)
        if best_row is None or row['F1'] > best_row['F1']:
            best_row = row
    summary = dict(method='M7',
                   hp_summary=(f"mult-gate λ_par={best_row['lam_par']} "
                                f"λ_str={best_row['lam_str']} "
                                f"λ_dist={best_row['lam_dist']}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M8(ctx):
    """Weighted linear sum on full arrays."""
    label = ctx['label']
    agg_z = ctx['agg_z']
    signals = ctx['signals']
    uq_keys = ['U_par_max_v', 'sigma_ale_max_v', 'U_str_mean_e', 'U_dist']
    rows = []
    best_row = None
    grid = (0.0, 0.1, 0.3, 1.0, 2.0)
    for bp, bsig, bst, bd in itertools.product(grid, grid, grid, grid):
        betas = dict(zip(uq_keys, (bp, bsig, bst, bd)))
        s = score_linear_sum(agg_z, signals, betas)
        res = eval_score_full(s, label)
        row = dict(method='M8',
                   b_U_par_max_v=bp, b_sigma_ale_max_v=bsig,
                   b_U_str_mean_e=bst, b_U_dist=bd, **res)
        rows.append(row)
        if best_row is None or row['F1'] > best_row['F1']:
            best_row = row
    summary = dict(method='M8',
                   hp_summary=(
                       f"linear β=(par={best_row['b_U_par_max_v']}, "
                       f"σ={best_row['b_sigma_ale_max_v']}, "
                       f"str={best_row['b_U_str_mean_e']}, "
                       f"dist={best_row['b_U_dist']})"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M9(ctx):
    """LogReg stacker: train on val_slice, eval on full arrays."""
    from sklearn.linear_model import LogisticRegression
    label = ctx['label']
    feat = build_stacker_features(ctx)
    val_idx = ctx['val_idx']
    rows = []
    best_row = None
    for C in (0.1, 1.0, 10.0):
        lr = LogisticRegression(class_weight='balanced', C=C,
                                max_iter=1000, solver='lbfgs')
        lr.fit(feat[val_idx], label[val_idx])
        s_full = lr.decision_function(feat)
        res = eval_score_full(s_full, label)
        row = dict(method='M9', C=C, **res)
        rows.append(row)
        if best_row is None or row['F1'] > best_row['F1']:
            best_row = row
    summary = dict(method='M9',
                   hp_summary=f"LogReg C={best_row['C']} (trained on val_slice)",
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M10(ctx):
    """GBM stacker: train on val_slice, eval on full arrays."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    label = ctx['label']
    feat = build_stacker_features(ctx)
    val_idx = ctx['val_idx']
    rows = []
    best_row = None
    for depth in (2, 3, 5):
        for n_iter in (50, 100, 200):
            gb = HistGradientBoostingClassifier(
                max_depth=depth, max_iter=n_iter, learning_rate=0.05,
                l2_regularization=1.0, random_state=ctx['seed'],
                class_weight='balanced')
            gb.fit(feat[val_idx], label[val_idx])
            proba = gb.predict_proba(feat)[:, 1]
            s_full = np.log(np.clip(proba, 1e-8, 1 - 1e-8) /
                            np.clip(1 - proba, 1e-8, 1 - 1e-8))
            res = eval_score_full(s_full, label)
            row = dict(method='M10', max_depth=depth, max_iter=n_iter, **res)
            rows.append(row)
            if best_row is None or row['F1'] > best_row['F1']:
                best_row = row
    summary = dict(method='M10',
                   hp_summary=(f"GBM depth={best_row['max_depth']} "
                                f"iter={best_row['max_iter']} "
                                f"(trained on val_slice)"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M11(ctx, m4_best=None, m6_best=None):
    """M4 + M6 combo on full arrays."""
    label = ctx['label']
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])
    sigma_ale = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR))
    tk_default = 1
    if m4_best is not None:
        for r in m4_best.get('hp_rows', []):
            if r['method'] == 'M4':
                tk_default = int(r.get('topk', 1))
                break
    rows = []
    best_row = None
    for lam in (0.2, 0.5, 1.0):
        for tk in (1, 2, 3):
            weight = np.clip(1.0 + lam * ctx['z_U_par_TxV'], 0.1, 10.0)
            rtilde = (r_abs / (sigma_ale + EPS)) * weight
            full_scores = rtilde.T
            s = topk_aggregate(full_scores, tk)
            res = eval_score_full(s, label)
            row = dict(method='M11', topk=tk, lam=lam, **res)
            rows.append(row)
            if best_row is None or row['F1'] > best_row['F1']:
                best_row = row
    summary = dict(method='M11',
                   hp_summary=(f"M4+M6 topk={best_row['topk']} "
                                f"λ={best_row['lam']}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M12(ctx):
    """Triple-OR on full arrays."""
    agg, signals, label = ctx['agg'], ctx['signals'], ctx['label']
    z_sigma = signals['sigma_ale_max_v']
    z_str = signals['U_str_mean_e']
    tau_r_qs = np.linspace(0.5, 0.999, 40)
    tau_r_vals = np.quantile(agg, tau_r_qs)
    tau_s_grid = [2.0, 3.0, 4.0, np.inf]
    rows = []
    best_row = None
    for tr, qr in zip(tau_r_vals, tau_r_qs):
        for ts1, ts2 in itertools.product(tau_s_grid, tau_s_grid):
            alarm = ((agg > tr) | (z_sigma > ts1) |
                     (z_str > ts2)).astype(np.int8)
            res = eval_alarm_full(alarm, label)
            row = dict(method='M12', tau_r=float(tr), tau_r_q=float(qr),
                        tau_s1=float(ts1), tau_s2=float(ts2), **res)
            rows.append(row)
            if best_row is None or row['F1'] > best_row['F1']:
                best_row = row
    summary = dict(method='M12',
                   hp_summary=(f"3OR tau_r={best_row['tau_r']:.3f} "
                                f"tau_s1={best_row['tau_s1']:.2f} "
                                f"tau_s2={best_row['tau_s2']:.2f}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau_r'], q=best_row['tau_r_q'])
    return rows, summary


def run_M13(ctx):
    """Adaptive sliding-quantile on full arrays."""
    label = ctx['label']
    agg = ctx['agg']
    T = agg.shape[0]
    rows = []
    best_row = None
    for W0 in (500, 2000, 8000):
        for q in (0.97, 0.99, 0.995):
            tau_t = np.zeros(T)
            for i in range(T):
                lo = max(0, i - W0)
                tau_t[i] = np.quantile(agg[lo:i + 1], q) if i > 10 else np.inf
            alarm = (agg > tau_t).astype(np.int8)
            res = eval_alarm_full(alarm, label)
            row = dict(method='M13', W0=W0, q=q, **res)
            rows.append(row)
            if best_row is None or row['F1'] > best_row['F1']:
                best_row = row
    summary = dict(method='M13',
                   hp_summary=(f"adaptive W0={best_row['W0']} "
                                f"q={best_row['q']}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=float('nan'), q=best_row['q'])
    return rows, summary


def run_M14(ctx):
    """M4 base + additive multi-mode epistemic boost + CF routing labels.

    Score:
        r̃_v(t) = |y_v - μ̄_v| / (σ_ale,v + ε)                # M4 base, tk=1, sm=3
        resid_max(t) = max_v r̃_v(t)
        epi_max(t) = max( clip(z_U_par_max(t), 0, 5),
                          clip(z_U_str_mean(t), 0, 5),
                          clip(z_U_dist(t), 0, 5) )
        epi_arg(t) ∈ {par, str, dist}  — argmax over modes
        s(t) = resid_max(t) + λ · epi_max(t)

    HP swept: λ ∈ {0, 0.5, 1.0, 2.0, 5.0}. λ=0 reduces to M4 — sanity check.

    Side product: per-alarm-run CF routing label.
        - data-anomaly       : residual contribution > 0.7 of total score
        - param-epistemic    : epi_arg dominant = par
        - struct-epistemic   : epi_arg dominant = str
        - dist-epistemic     : epi_arg dominant = dist
    """
    label = ctx['label']
    r_abs = np.abs(ctx['test_gt'] - ctx['test_mu'])
    sigma_ale = np.sqrt(np.maximum(ctx['test_sigma2_ale'], SIGMA_FLOOR))

    # M4 base score (tk=1, sm=3 — matches M4's winning HP)
    s_resid = score_per_sensor_sigma(r_abs, sigma_ale, tk=1, sm=3)

    # Epistemic z-aggregates (already re-z-scored on C-nominal)
    z_par = ctx['signals']['U_par_max_v']
    z_str = ctx['signals']['U_str_mean_e']
    z_dist = ctx['signals']['U_dist']
    # Only positive z-deviations count; cap at 5 to avoid extreme outliers
    z_par_clip = np.clip(z_par, 0.0, 5.0)
    z_str_clip = np.clip(z_str, 0.0, 5.0)
    z_dist_clip = np.clip(z_dist, 0.0, 5.0)
    epi_stack = np.stack([z_par_clip, z_str_clip, z_dist_clip], axis=1)
    epi_max = epi_stack.max(axis=1)        # (T,)
    epi_arg = epi_stack.argmax(axis=1)     # (T,) ∈ {0:par, 1:str, 2:dist}

    rows = []
    best_row = None
    for lam in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = s_resid + lam * epi_max
        res = eval_score_full(s, label)
        row = dict(method='M14', lam=lam, **res)
        rows.append(row)
        if best_row is None or row['F1'] > best_row['F1']:
            best_row = row

    # --- CF routing analysis at best λ ---
    best_lam = best_row['lam']
    tau_best = best_row['tau']
    s_best = s_resid + best_lam * epi_max
    raw_alarm = (s_best > tau_best).astype(np.int8)

    # Per-timestep dominant contributor (raw alarm only)
    # 0 = data-anomaly, 1 = param, 2 = struct, 3 = dist, -1 = no raw alarm
    cf_label = np.full(len(s_best), -1, dtype=np.int8)
    pos_mask = raw_alarm == 1
    if best_lam > 0:
        epi_contrib = best_lam * epi_max
        resid_share = np.where(s_best > 0, s_resid / np.maximum(s_best, EPS), 1.0)
        is_data = resid_share >= 0.7
        cf_label[pos_mask & is_data] = 0
        cf_label[pos_mask & ~is_data] = epi_arg[pos_mask & ~is_data] + 1
    else:
        # λ=0 ≡ M4: everything is data-anomaly
        cf_label[pos_mask] = 0

    # Post-proc the raw alarm + assign each pp-run the majority raw label
    pp_alarm = apply_postproc(raw_alarm, POST_W, POST_G)
    final_label = np.full(len(pp_alarm), -1, dtype=np.int8)
    in_run = False
    run_start = 0
    runs_info = []
    for t in range(len(pp_alarm)):
        if pp_alarm[t] and not in_run:
            run_start = t
            in_run = True
        elif (not pp_alarm[t]) and in_run:
            sub = cf_label[run_start:t]
            valid = sub[sub >= 0]
            if len(valid) > 0:
                vals, counts = np.unique(valid, return_counts=True)
                lbl = int(vals[counts.argmax()])
            else:
                lbl = -2  # postproc-only (dilation), no raw evidence
            final_label[run_start:t] = lbl
            runs_info.append(dict(start=int(run_start), end=int(t - 1),
                                   len=int(t - run_start), label=lbl,
                                   true_attack=bool(label[run_start:t].sum() > 0)))
            in_run = False
    if in_run:
        sub = cf_label[run_start:]
        valid = sub[sub >= 0]
        if len(valid) > 0:
            vals, counts = np.unique(valid, return_counts=True)
            lbl = int(vals[counts.argmax()])
        else:
            lbl = -2
        final_label[run_start:] = lbl
        runs_info.append(dict(start=int(run_start), end=int(len(pp_alarm) - 1),
                               len=int(len(pp_alarm) - run_start), label=lbl,
                               true_attack=bool(label[run_start:].sum() > 0)))

    # Summarise CF label distribution over alarm timesteps + runs
    label_names = {0: 'data-anomaly', 1: 'param-epistemic',
                   2: 'struct-epistemic', 3: 'dist-epistemic',
                   -2: 'postproc-only'}
    n_alarm_ts = int(pp_alarm.sum())
    n_alarm_runs = len(runs_info)
    ts_counts = {name: int((final_label == code).sum())
                  for code, name in label_names.items()}
    run_counts = {name: sum(1 for r in runs_info if r['label'] == code)
                   for code, name in label_names.items()}
    # TP/FP breakdown per CF label
    tp_per = {name: 0 for name in label_names.values()}
    fp_per = {name: 0 for name in label_names.values()}
    for r in runs_info:
        name = label_names.get(r['label'], 'unknown')
        if r['true_attack']:
            tp_per[name] = tp_per.get(name, 0) + 1
        else:
            fp_per[name] = fp_per.get(name, 0) + 1
    print(f"  CF routing distribution (best λ={best_lam}):", flush=True)
    for name in ['data-anomaly', 'param-epistemic', 'struct-epistemic',
                 'dist-epistemic', 'postproc-only']:
        ts = ts_counts.get(name, 0)
        rn = run_counts.get(name, 0)
        tp = tp_per.get(name, 0)
        fp = fp_per.get(name, 0)
        pct_ts = 100.0 * ts / max(n_alarm_ts, 1)
        print(f"    {name:>18s}: {rn:>3d} alarm-runs "
              f"({tp} TP / {fp} FP), {ts:>5d} alarm-ts ({pct_ts:>5.1f}%)",
              flush=True)

    summary = dict(method='M14',
                   hp_summary=(f"M4 + λ·max-epi boost, λ={best_row['lam']} "
                                f"(tk=1, sm=3)"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'],
                   cf_label_ts_counts=json.dumps(ts_counts),
                   cf_label_run_counts=json.dumps(run_counts),
                   cf_label_tp_per=json.dumps(tp_per),
                   cf_label_fp_per=json.dumps(fp_per),
                   cf_runs_info=runs_info)
    return rows, summary


def _pool_U_str_to_sensor(test_U_str, edge_index, V):
    """Pool per-edge U_str to per-sensor by averaging over incoming edges.

    edge_index: (2, E) with row 0 = sources, row 1 = targets (PyG convention).
    Returns (T, V) of per-sensor structural uncertainty.
    """
    T_ = test_U_str.shape[0]
    if edge_index is not None:
        targets = edge_index[1]
        out = np.zeros((T_, V), dtype=np.float64)
        for v in range(V):
            inc = np.where(targets == v)[0]
            if len(inc) > 0:
                out[:, v] = test_U_str[:, inc].mean(axis=1)
        return out
    # Fallback if no edge_index: broadcast the per-edge mean
    return np.tile(test_U_str.mean(axis=1, keepdims=True), (1, V))


def run_M15(ctx):
    """Mahalanobis on joint per-sensor signal vector [r_v, U_par_v, U_str_v].

    Fit per-sensor (μ₀, Σ₀) on C-nominal timesteps (label==0 within C-slice),
    score D²_v(t) = (s_v(t) - μ₀_v)ᵀ Σ₀_v⁻¹ (s_v(t) - μ₀_v).

    Captures correlations between residual and uncertainty channels on clean
    data — anomalous if the joint point breaks the *correlation structure*,
    not just the marginals (unlike OR/AND which only see marginals).
    """
    label = ctx['label']
    full_scores = ctx['full_scores']        # (V, T)
    test_U_par = ctx['test_U_par']          # (T, V)
    test_U_str = ctx['test_U_str']          # (T, E)
    c_mask = ctx['c_mask']                  # (T,) bool — C-nominal
    edge_index = ctx.get('edge_index')
    V_ = full_scores.shape[0]
    T_ = full_scores.shape[1]
    U_str_v = _pool_U_str_to_sensor(test_U_str, edge_index, V_)   # (T, V)

    r_per_sensor = full_scores.T                                  # (T, V)
    s = np.stack([r_per_sensor, test_U_par, U_str_v], axis=-1)    # (T, V, 3)

    # Fit (μ₀, Σ₀⁻¹) per sensor on C-nominal
    mu0 = np.zeros((V_, 3))
    sigma0_inv = np.zeros((V_, 3, 3))
    reg = 1e-6
    for v in range(V_):
        x = s[c_mask, v, :]
        mu0[v] = x.mean(axis=0)
        cov = np.cov(x.T) + reg * np.eye(3)
        try:
            sigma0_inv[v] = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            sigma0_inv[v] = np.linalg.pinv(cov)

    # D²_v(t) via einsum
    centered = s - mu0[None, :, :]
    D2 = np.einsum('tvi,vij,tvj->tv', centered, sigma0_inv, centered)
    print(f"  D2 stats: median={np.median(D2):.3f}, "
          f"q95={np.quantile(D2, 0.95):.3f}, "
          f"max={D2.max():.3f}", flush=True)

    rows = []
    best_row = None
    for agg_kind in ['max', 'mean', 'top2', 'top3', 'top5']:
        if agg_kind == 'max':
            s_t = D2.max(axis=1)
        elif agg_kind == 'mean':
            s_t = D2.mean(axis=1)
        else:
            k = int(agg_kind[3:])
            s_t = np.sort(D2, axis=1)[:, -k:].sum(axis=1)
        res = eval_score_full(s_t, label)
        row = dict(method='M15', agg=agg_kind, **res)
        rows.append(row)
        if best_row is None or row['F1'] > best_row['F1']:
            best_row = row
    summary = dict(method='M15',
                   hp_summary=(f"Mahalanobis [r,U_par,U_str] per-sensor, "
                                f"agg={best_row['agg']}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M16(ctx):
    """Decomposition-aware composite: regime-routing rule.

    A_v(t) = z_v · 1[U_par_v < τ_epi]            (precision lever)
             + β · U_par_v · 1[z_v < τ_z]         (recall lever)
    s(t) = max_v A_v(t)

    - First term fires only when the model is CONFIDENT (low U_par) AND the
      residual is high — suppresses high-epistemic FPs.
    - Second term fires only when the residual is QUIET (low z) AND the
      model is uncertain — recovers stealthy/OOD attacks the residual missed.
    The two regimes are mutually exclusive in z_v × U_par_v plane.
    """
    label = ctx['label']
    full_scores = ctx['full_scores']
    test_U_par = ctx['test_U_par']
    c_mask = ctx['c_mask']
    V_ = full_scores.shape[0]
    z = full_scores.T                  # (T, V) — per-sensor smoothed err score
    U_par = test_U_par                 # (T, V)

    tau_epi_qs = [0.5, 0.7, 0.9, 0.95]
    tau_z_qs = [0.5, 0.7, 0.9, 0.95]
    beta_grid = [0.5, 1.0, 2.0, 5.0]

    rows = []
    best_row = None
    for tau_epi_q, tau_z_q, beta in itertools.product(
            tau_epi_qs, tau_z_qs, beta_grid):
        tau_epi = float(np.quantile(U_par[c_mask], tau_epi_q))
        tau_z = float(np.quantile(z[c_mask], tau_z_q))
        precision_term = z * (U_par < tau_epi).astype(np.float64)
        recall_term = beta * U_par * (z < tau_z).astype(np.float64)
        A = precision_term + recall_term     # (T, V)
        s_t = A.max(axis=1)
        res = eval_score_full(s_t, label)
        row = dict(method='M16',
                   tau_epi_q=tau_epi_q, tau_epi=tau_epi,
                   tau_z_q=tau_z_q, tau_z=tau_z,
                   beta=beta, **res)
        rows.append(row)
        if best_row is None or row['F1'] > best_row['F1']:
            best_row = row
    summary = dict(method='M16',
                   hp_summary=(f"regime-routing τ_epi_q={best_row['tau_epi_q']} "
                                f"τ_z_q={best_row['tau_z_q']} "
                                f"β={best_row['beta']}"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


def run_M17(ctx):
    """Calibrated logistic regression on per-sensor 4D feature vector.

    Features per (t, v): [r_v(t), U_par_v(t), U_str_v(t), σ_ale,v(t)].
    Train on val_slice with broadcast labels (every sensor gets the
    timestep-level attack label since SWaT has no per-sensor ground truth).
    Predict on full arrays; per-timestep score = max_v predict_proba(t, v).

    Caveat: broadcast label is a strong assumption — only some sensors are
    actually targeted in each SWaT attack. The classifier learns
    "is this (t, v) drawn from the attack-marginal distribution" rather
    than "is sensor v under attack at t".
    """
    from sklearn.linear_model import LogisticRegression
    label = ctx['label']
    full_scores = ctx['full_scores']
    test_U_par = ctx['test_U_par']
    test_U_str = ctx['test_U_str']
    test_sigma2_ale = ctx['test_sigma2_ale']
    val_idx = ctx['val_idx']
    edge_index = ctx.get('edge_index')
    V_ = full_scores.shape[0]
    T_ = label.shape[0]

    U_str_v = _pool_U_str_to_sensor(test_U_str, edge_index, V_)
    sigma_ale = np.sqrt(np.maximum(test_sigma2_ale, SIGMA_FLOOR))
    feat = np.stack([full_scores.T, test_U_par, U_str_v, sigma_ale],
                    axis=-1)                          # (T, V, 4)
    feat_flat = feat.reshape(-1, 4)
    label_flat = np.tile(label[:, None], (1, V_)).reshape(-1)
    val_mask = np.zeros(T_, dtype=bool)
    val_mask[val_idx] = True
    val_mask_flat = np.tile(val_mask[:, None], (1, V_)).reshape(-1)

    rows = []
    best_row = None
    for C in (0.1, 1.0, 10.0):
        lr = LogisticRegression(class_weight='balanced', C=C,
                                max_iter=1000, solver='lbfgs')
        lr.fit(feat_flat[val_mask_flat], label_flat[val_mask_flat])
        coef = lr.coef_[0]
        print(f"  LogReg C={C} coef [r, U_par, U_str, σ_ale] = "
              f"{coef[0]:.3f}, {coef[1]:.3f}, {coef[2]:.3f}, {coef[3]:.3f}",
              flush=True)
        proba_per = lr.predict_proba(feat_flat)[:, 1].reshape(T_, V_)
        s_t = proba_per.max(axis=1)
        res = eval_score_full(s_t, label)
        row = dict(method='M17', C=C, mode='val-train',
                   coef_r=float(coef[0]), coef_U_par=float(coef[1]),
                   coef_U_str=float(coef[2]), coef_sigma_ale=float(coef[3]),
                   **res)
        rows.append(row)
        if best_row is None or row['F1'] > best_row['F1']:
            best_row = row
    summary = dict(method='M17',
                   hp_summary=(f"per-sensor LogReg C={best_row['C']} "
                                f"(broadcast labels, trained on val_slice)"),
                   F1=best_row['F1'], P=best_row['P'], R=best_row['R'],
                   tau=best_row['tau'], q=best_row['q'])
    return rows, summary


METHOD_RUNNERS = {
    'M0': run_M0, 'M1': run_M1, 'M2': run_M2, 'M3': run_M3,
    'M4': run_M4, 'M5': run_M5, 'M6': run_M6, 'M7': run_M7,
    'M8': run_M8, 'M9': run_M9, 'M10': run_M10,
    'M11': run_M11, 'M12': run_M12, 'M13': run_M13, 'M14': run_M14,
    'M15': run_M15, 'M16': run_M16, 'M17': run_M17,
}


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def attack_runs(label):
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
    has_U_str = 'test_U_str' in d.files          # attention-free methods omit it
    test_U_str = d['test_U_str'].astype(np.float64) if has_U_str else None
    test_U_dist = d['test_U_dist'].astype(np.float64)
    test_sigma2_ale = d['test_sigma2_ale'].astype(np.float64)
    val_mu = d['val_mu_bar'].astype(np.float64)
    val_gt = d['val_ground_truth'].astype(np.float64)
    label = d['test_attack_label'].astype(np.int8)
    T = label.shape[0]
    print(f'T={T}, V={test_mu.shape[1]}, attack_rate={label.mean():.4f}, '
          f'attack_runs={len(attack_runs(label))}', flush=True)

    sw = args.slide_win
    print(f'loading split: {args.split}', flush=True)
    with open(args.split) as f:
        split = json.load(f)
    C_lo, C_hi = split['C_row_range']
    val_lo, val_hi = split['labeled_val_range']
    C_idx_start = max(0, C_lo - sw)
    C_idx_end = max(0, C_hi - sw)
    val_idx_start = max(0, val_lo - sw)
    val_idx_end = max(0, val_hi - sw)
    C_idx_end = min(T, C_idx_end)
    val_idx_end = min(T, val_idx_end)
    c_idx = np.arange(C_idx_start, C_idx_end)
    val_idx = np.arange(val_idx_start, val_idx_end)
    c_mask = np.zeros(T, dtype=bool)
    c_mask[c_idx] = True
    c_mask_nominal = c_mask & (label == 0)
    print(f'  C   slice arrays[{C_idx_start},{C_idx_end}): '
          f'n={C_idx_end-C_idx_start} (zscore fit only)', flush=True)
    print(f'  val slice arrays[{val_idx_start},{val_idx_end}): '
          f'n={val_idx_end-val_idx_start}, '
          f'attack_count={int(label[val_idx].sum())} (stacker train only)',
          flush=True)
    print(f'  Full arrays for τ-sweep + F1 reporting on all methods',
          flush=True)

    print('building full_scores at smoothing=5 ...', flush=True)
    full_scores = build_full_err_scores(test_mu, test_gt, val_mu, val_gt, 5)
    agg = topk_aggregate(full_scores, 1).astype(np.float64)
    print(f'  agg shape={agg.shape}', flush=True)

    print('fitting zscore params on C-slice nominal ...', flush=True)
    med_par, iqr_par = fit_zscore_params(test_U_par, c_mask_nominal)
    med_sig, iqr_sig = fit_zscore_params(test_sigma2_ale, c_mask_nominal)
    med_dist, iqr_dist = fit_zscore_params(test_U_dist[:, None],
                                            c_mask_nominal)
    z_U_par_TxV = apply_zscore(test_U_par, med_par, iqr_par)
    z_sigma2_TxV = apply_zscore(test_sigma2_ale, med_sig, iqr_sig)
    z_U_dist = apply_zscore(test_U_dist[:, None], med_dist, iqr_dist)[:, 0]
    raw = {
        'U_par_max_v': z_U_par_TxV.max(axis=1),
        'U_par_mean_v': z_U_par_TxV.mean(axis=1),
        'U_dist': z_U_dist,
        'sigma_ale_max_v': z_sigma2_TxV.max(axis=1),
        'sigma_ale_mean_v': z_sigma2_TxV.mean(axis=1),
    }
    if has_U_str:
        med_str, iqr_str = fit_zscore_params(test_U_str, c_mask_nominal)
        z_U_str_TxE = apply_zscore(test_U_str, med_str, iqr_str)
        raw['U_str_mean_e'] = z_U_str_TxE.mean(axis=1)
    signals = {k: fit_apply_1d_renorm(v, c_mask_nominal) for k, v in raw.items()}
    agg_z = fit_apply_1d_renorm(agg, c_mask_nominal)

    # Load edge_index from calibration bundle for U_str → per-sensor pooling
    edge_index = None
    if args.bundle:
        edge_path = Path(args.bundle) / 'edge_index_sample.npz'
        if edge_path.exists():
            ed = np.load(edge_path)
            key = 'edge_index_sample' if 'edge_index_sample' in ed.files else ed.files[0]
            edge_index = ed[key]
            print(f'  loaded edge_index from {edge_path} shape={edge_index.shape}',
                  flush=True)

    return dict(
        agg=agg, agg_z=agg_z, signals=signals,
        full_scores=full_scores,
        test_mu=test_mu, test_gt=test_gt,
        test_U_par=test_U_par, test_U_str=test_U_str,
        test_U_dist=test_U_dist, test_sigma2_ale=test_sigma2_ale,
        z_U_par_TxV=z_U_par_TxV,
        label=label, T=T,
        c_idx=c_idx, val_idx=val_idx,
        c_mask=c_mask_nominal,
        edge_index=edge_index,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-arrays', required=True)
    parser.add_argument('-split', required=True)
    parser.add_argument('-bundle', default=None)
    parser.add_argument('-slide_win', type=int, default=60)
    parser.add_argument('-methods', nargs='+',
                        default=list(METHOD_RUNNERS.keys()))
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-out_root', default='results/fusion_K100_full')
    args = parser.parse_args()

    ctx = setup_context(args)

    # M0 sanity (always run, regardless of args.methods)
    print('\n=== M0 sanity: paper-protocol on full arrays ===', flush=True)
    m0_full_res = eval_score_full(ctx['agg'], ctx['label'])
    print(f"  full-test F1 = {m0_full_res['F1']:.4f}  P = {m0_full_res['P']:.4f}  "
          f"R = {m0_full_res['R']:.4f}  tau = {m0_full_res['tau']:.4f}  "
          f"q = {m0_full_res['q']:.4f}", flush=True)
    if abs(m0_full_res['F1'] - 0.8109) > 0.001:
        print(f"  WARN: deviation from expected F1=0.8109 is "
              f"{m0_full_res['F1']-0.8109:+.4f}", flush=True)
    else:
        print(f"  OK: matches expected F1=0.8109 within 0.001", flush=True)
    M0_F1 = m0_full_res['F1']

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    out_dir = Path(args.out_root) / datestr
    out_dir.mkdir(parents=True, exist_ok=True)

    all_per_hp_rows = []
    summaries = []
    m4_best, m6_best = None, None
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
        summary['lift_vs_M0'] = summary.get('F1', 0.0) - M0_F1
        summary['wall_sec'] = dt
        if method == 'M4':
            m4_best = summary
        if method == 'M6':
            m6_best = summary
        print(f"  {method} F1={summary.get('F1', float('nan')):.4f} "
              f"P={summary.get('P', float('nan')):.4f} "
              f"R={summary.get('R', float('nan')):.4f} "
              f"lift_vs_M0={summary.get('lift_vs_M0', 0):+.4f} "
              f"wall={dt:.1f}s", flush=True)
        all_per_hp_rows.extend(hp_rows)
        # Extract M14's per-run CF labels (list, not CSV-friendly) and
        # write to a dedicated file
        if method == 'M14' and 'cf_runs_info' in summary:
            cf_runs = summary.pop('cf_runs_info')
            df_cf = pd.DataFrame(cf_runs)
            df_cf.to_csv(out_dir / 'cf_routing_M14.csv', index=False)
            print(f"  CF routing per-run written to {out_dir/'cf_routing_M14.csv'}",
                  flush=True)
        summaries.append(summary)

    # Build output table
    fields = ['method', 'hp_summary', 'F1', 'P', 'R', 'tau', 'q',
              'lift_vs_M0', 'wall_sec']
    df_methods = pd.DataFrame(summaries)
    cols = [c for c in fields if c in df_methods.columns]
    extra = [c for c in df_methods.columns if c not in cols]
    df_methods = df_methods[cols + extra]
    df_methods.to_csv(out_dir / 'methods_full.csv', index=False)
    df_hp = pd.DataFrame(all_per_hp_rows)
    df_hp.to_csv(out_dir / 'per_hp_full.csv', index=False)

    # SUMMARY.md
    df_sorted = df_methods.sort_values('F1', ascending=False)
    md = [
        f"# Fusion sweep PAPER-PROTOCOL @ K=100, sw=60, seed=42 ({datestr})",
        "",
        f"Inputs: `{args.arrays}`",
        f"Protocol: τ swept on FULL 44,716-timestep arrays per method "
        f"(matches the published F1=0.8109 protocol).",
        "Stackers (M9/M10) trained on val_slice; eval on full arrays.",
        "",
        f"## M0 sanity reproduction: F1 = {M0_F1:.4f} "
        f"(expected 0.8109)",
        "",
        f"## Method ranking by full-arrays F1",
        "",
        "| Method | HP | F1 | P | R | lift vs M0 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in df_sorted.iterrows():
        md.append(
            f"| {r['method']} | {r.get('hp_summary', '')} | "
            f"{r.get('F1', float('nan')):.4f} | "
            f"{r.get('P', float('nan')):.4f} | "
            f"{r.get('R', float('nan')):.4f} | "
            f"{r.get('lift_vs_M0', 0):+.4f} |"
        )
    md.append("")
    md.append("## Verdict")
    winners = df_sorted[df_sorted['lift_vs_M0'] > 0.005]
    if len(winners) > 0:
        best = winners.iloc[0]
        md.append(f"\n**Winner**: `{best['method']}` — "
                  f"F1 = {best['F1']:.4f} (lift {best['lift_vs_M0']:+.4f} "
                  f"vs M0={M0_F1:.4f})")
    else:
        md.append(f"\n**No method beats M0={M0_F1:.4f} by >0.005 on the "
                  f"full-arrays paper-protocol.** Headline finding: "
                  f"single-model G-ΔUQ at K=100 has saturated for SWaT under "
                  f"residual+post-proc protocol.")
    (out_dir / 'SUMMARY.md').write_text('\n'.join(md))

    best_json = {}
    for s in summaries:
        best_json[s['method']] = {k: (None if (isinstance(v, float)
                                                and np.isnan(v)) else v)
                                  for k, v in s.items()
                                  if isinstance(v, (int, float, str))}
    (out_dir / 'best_per_method.json').write_text(json.dumps(best_json,
                                                             indent=2))

    print(f"\n=== OUTPUTS ===", flush=True)
    print(f"  {out_dir/'methods_full.csv'}", flush=True)
    print(f"  {out_dir/'per_hp_full.csv'}", flush=True)
    print(f"  {out_dir/'SUMMARY.md'}", flush=True)
    print(f"  {out_dir/'best_per_method.json'}", flush=True)


if __name__ == '__main__':
    main()
