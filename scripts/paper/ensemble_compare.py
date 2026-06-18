"""Compare ANCHOR vs ENSEMBLE-5 vs ENSEMBLE-10 on epistemic-UQ quality + calibration.

Consumes the emitted arrays.npz (anchor + the two ensemble configs) and the eval
harness JSONs (detection), and computes:

  item2  epistemic attack-AUROC : AUROC of the epistemic channel (sigma2_epistemic,
         aggregated over sensors by mean and by max) vs test_attack_label.
  item2  AUSE / sparsification   : on NOMINAL test timesteps, sort by predicted sigma_tot,
         drop the most-uncertain fraction, track RMSE; AUSE_norm = area between the
         prediction-sparsification curve and the oracle (error-sorted) curve, divided by
         the area under the oracle curve. Lower is better. Computed PER SENSOR (each sensor
         standardized to its own RMSE@0) then averaged, so no single large-scale SWaT
         sensor dominates.
  item3  regression calibration  : N(mu, sigma_tot) central-interval coverage on nominal
         test (t,v) pairs at alpha in {0.5,0.8,0.9,0.95}; cal_err = mean |alpha - emp|.

Writes results/paper/ensemble/ensemble_vs_anchor.csv (one row per config).

Notes on channel sourcing:
  ANCHOR arrays   : epistemic = test_U_par ; aleatoric = test_sigma2_ale ;
                    sigma2_total = test_U_par + test_sigma2_ale.
  ENSEMBLE arrays : epistemic = test_U_par (== sigma2_epistemic) ;
                    aleatoric = test_sigma2_ale ; sigma2_total = test_sigma2_total.
Each config's calibration/AUSE use that config's own predictive, so they are
internally consistent quality measures and comparable across configs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'competitors' / 'common'))

from extra_metrics import auroc  # noqa: E402  (numpy ROC-AUC, ties-aware)

ALPHAS = [0.5, 0.8, 0.9, 0.95]
EPS = 1e-8


def load_config(arrays_path, is_anchor):
    d = np.load(arrays_path)
    mu = d['test_mu_bar'].astype(np.float64)
    gt = d['test_ground_truth'].astype(np.float64)
    lab = d['test_attack_label'].astype(np.int8)
    epi = d['test_U_par'].astype(np.float64)               # sigma2_epistemic (variance)
    ale = d['test_sigma2_ale'].astype(np.float64)
    if 'test_sigma2_total' in d.files:
        tot = d['test_sigma2_total'].astype(np.float64)
    else:
        tot = epi + ale                                     # anchor: build total
    return dict(mu=mu, gt=gt, lab=lab, epi=epi, ale=ale, tot=tot)


def epistemic_attack_auroc(epi_var, lab):
    """AUROC of the epistemic channel vs binary attack label, aggregated over sensors."""
    s_mean = epi_var.mean(axis=1)
    s_max = epi_var.max(axis=1)
    return {
        'epi_attackAUROC_meanV': float(auroc(s_mean, lab)),
        'epi_attackAUROC_maxV': float(auroc(s_max, lab)),
    }


def sparsification_ause(mu, gt, sigma_tot, nominal_mask, n_steps=21):
    """Per-sensor sparsification AUSE on nominal timesteps, averaged over sensors.

    For each sensor v:
      - take nominal rows; abs error e = |gt-mu|, predicted sigma s = sqrt(sigma_tot).
      - prediction curve: drop the top-f fraction by s (keep least-uncertain), RMSE of the kept.
      - oracle curve     : drop the top-f fraction by e itself (best possible), RMSE of the kept.
      - normalize each curve by RMSE at f=0 (so it starts at 1.0).
      - AUSE_v = area(pred_norm - oracle_norm) over f in [0,1) (trapezoid).
    Return mean AUSE over sensors with >= 50 nominal rows.
    """
    fracs = np.linspace(0.0, 0.95, n_steps)
    T, V = mu.shape
    err = np.abs(gt - mu)
    sig = np.sqrt(np.maximum(sigma_tot, EPS))
    ause_per_v = []
    for v in range(V):
        m = nominal_mask
        e = err[m, v]
        s = sig[m, v]
        n = e.shape[0]
        if n < 50:
            continue
        order_pred = np.argsort(-s)     # most-uncertain first (to drop)
        order_orac = np.argsort(-e)     # largest-error first (oracle drop)
        e_pred = e[order_pred]
        e_orac = e[order_orac]
        rmse_pred = np.empty_like(fracs)
        rmse_orac = np.empty_like(fracs)
        for i, f in enumerate(fracs):
            k = int(np.floor(f * n))
            kept_p = e_pred[k:]
            kept_o = e_orac[k:]
            rmse_pred[i] = np.sqrt(np.mean(kept_p ** 2)) if kept_p.size else 0.0
            rmse_orac[i] = np.sqrt(np.mean(kept_o ** 2)) if kept_o.size else 0.0
        base = rmse_pred[0]
        if base < EPS:
            continue
        pred_n = rmse_pred / base
        orac_n = rmse_orac / base
        diff = np.clip(pred_n - orac_n, 0.0, None)
        ause_v = float(np.trapz(diff, fracs)) if hasattr(np, 'trapz') else float(np.trapezoid(diff, fracs))
        ause_per_v.append(ause_v)
    return {
        'AUSE_norm_mean': float(np.mean(ause_per_v)) if ause_per_v else float('nan'),
        'AUSE_norm_median': float(np.median(ause_per_v)) if ause_per_v else float('nan'),
        'AUSE_n_sensors': int(len(ause_per_v)),
    }


def regression_calibration(mu, gt, sigma_tot, nominal_mask):
    """Central-interval coverage of N(mu, sigma_tot) on nominal (t,v) pairs.

    For each alpha, central interval half-width z = ppf(0.5 + alpha/2) in sigma units.
    Empirical coverage = fraction of nominal (t,v) with |gt-mu|/sigma <= z.
    cal_err = mean_alpha |alpha - emp_coverage|.
    """
    sig = np.sqrt(np.maximum(sigma_tot, EPS))
    z_abs = np.abs(gt - mu) / sig                 # (T,V) standardized residual
    z_nom = z_abs[nominal_mask, :].ravel()         # all nominal (t,v)
    n = z_nom.size
    per_alpha = {}
    errs = []
    for a in ALPHAS:
        z = norm.ppf(0.5 + a / 2.0)
        emp = float(np.mean(z_nom <= z))
        per_alpha[f'cov_a{a}'] = emp
        errs.append(abs(a - emp))
    return {
        'cal_err_mean': float(np.mean(errs)),
        'n_nominal_tv': int(n),
        **per_alpha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', default=str(REPO / 'results/gdn/ref_seed42/arrays.npz'))
    ap.add_argument('--ens5', default=str(REPO / 'results/paper/ensemble/arrays_ensemble5.npz'))
    ap.add_argument('--ens10', default=str(REPO / 'results/paper/ensemble/arrays_ensemble10.npz'))
    ap.add_argument('--eval-anchor', default=str(REPO / 'results/paper/ensemble/eval_anchor.json'))
    ap.add_argument('--eval-ens5', default=str(REPO / 'results/paper/ensemble/eval_ensemble5.json'))
    ap.add_argument('--eval-ens10', default=str(REPO / 'results/paper/ensemble/eval_ensemble10.json'))
    ap.add_argument('--emit-summary', default=str(REPO / 'results/paper/ensemble/emit_summary.json'))
    ap.add_argument('--out-csv', default=str(REPO / 'results/paper/ensemble/ensemble_vs_anchor.csv'))
    args = ap.parse_args()

    emit = json.load(open(args.emit_summary)) if Path(args.emit_summary).exists() else {}
    test_wall_M10 = emit.get('test_inference_wall_s_M10', float('nan'))
    peak_M10 = emit.get('test_peak_mem_MB_M10', float('nan'))

    configs = [
        ('ANCHOR_K100', args.anchor, args.eval_anchor, True, 100, 1),
        ('ENSEMBLE5', args.ens5, args.eval_ens5, False, None, 5),
        ('ENSEMBLE10', args.ens10, args.eval_ens10, False, None, 10),
    ]

    rows = []
    for name, arr_path, eval_path, is_anchor, K, M in configs:
        c = load_config(arr_path, is_anchor)
        nominal = (c['lab'] == 0)
        n_nom = int(nominal.sum())
        det = json.load(open(eval_path))['baseline_M0']
        epi = epistemic_attack_auroc(c['epi'], c['lab'])
        ause = sparsification_ause(c['mu'], c['gt'], c['tot'], nominal)
        cal = regression_calibration(c['mu'], c['gt'], c['tot'], nominal)

        # forward-pass count (epistemic budget) + model-storage cost
        if is_anchor:
            fwd_passes = K          # K anchors of ONE model
            n_models_stored = 1
            budget_label = f'K={K} (1 model)'
        else:
            fwd_passes = M          # M forward passes of M models
            n_models_stored = M
            budget_label = f'M={M} models'

        row = {
            'config': name,
            'budget': budget_label,
            'fwd_passes': fwd_passes,
            'n_models_stored': n_models_stored,
            # detection
            'M0_F1': round(det['F1'], 4),
            'M0_P': round(det['P'], 4),
            'M0_R': round(det['R'], 4),
            'PA_K_AUC': round(det['PA_K_AUC'], 4),
            # epistemic quality
            'epi_attackAUROC_meanV': round(epi['epi_attackAUROC_meanV'], 4),
            'epi_attackAUROC_maxV': round(epi['epi_attackAUROC_maxV'], 4),
            'AUSE_norm_mean': round(ause['AUSE_norm_mean'], 4),
            'AUSE_norm_median': round(ause['AUSE_norm_median'], 4),
            'AUSE_n_sensors': ause['AUSE_n_sensors'],
            # calibration
            'cal_err_mean': round(cal['cal_err_mean'], 4),
            'cov_a0.5': round(cal['cov_a0.5'], 4),
            'cov_a0.8': round(cal['cov_a0.8'], 4),
            'cov_a0.9': round(cal['cov_a0.9'], 4),
            'cov_a0.95': round(cal['cov_a0.95'], 4),
            'n_nominal_t': n_nom,
            'n_nominal_tv': cal['n_nominal_tv'],
        }
        rows.append(row)
        print(f"[cmp] {name:12s} F1={row['M0_F1']:.4f} PA={row['PA_K_AUC']:.4f} | "
              f"epiAUROC(mean/max)={row['epi_attackAUROC_meanV']:.3f}/{row['epi_attackAUROC_maxV']:.3f} | "
              f"AUSE={row['AUSE_norm_mean']:.4f} | cal_err={row['cal_err_mean']:.4f}", flush=True)

    # compute row: add ensemble-10 wall/peak; anchor cost noted in report (K=100 of 1 model)
    fieldnames = list(rows[0].keys())
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'[cmp] wrote {args.out_csv}', flush=True)

    # stash compute facts for the report
    compute = {
        'ensemble10_test_wall_s': test_wall_M10,
        'ensemble10_peak_mem_MB': peak_M10,
        'ensemble10_fwd_passes': 10,
        'anchor_fwd_passes': 100,
        'note': ('anchor cost = K=100 forward passes of ONE model; '
                 'ensemble cost = M forward passes of M stored models.'),
    }
    json.dump({'rows': rows, 'compute': compute},
              open(str(Path(args.out_csv).with_suffix('.json')), 'w'), indent=2)
    print('[cmp] DONE', flush=True)


if __name__ == '__main__':
    main()
