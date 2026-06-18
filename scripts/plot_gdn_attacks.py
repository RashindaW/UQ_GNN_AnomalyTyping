"""Per-attack plotly plots for the GDN test set, variant 4 (sustained-window).

Identifies attack events in data/swat/test.csv, groups attacks that occur close
together into a single plot window, and emits one HTML file per attack group
showing:

  1. Ground-truth attack flag (red, filled).
  2. v4 sustained-window alarm (dark blue, primary) + v3 reference (faded
     orange).
  3. System anomaly score A_s(t) = SMA(max_v r_v(t), 4) with τ_v3 line.
  4. All 51 standardised residuals r_v(t) — 51 toggleable traces. The K_TOP
     most-anomalous sensors during this attack window are visible by default;
     the rest start in legend-only mode (click to show).
  5. All 51 total predictive uncertainties σ_tot,v(t) — same traces, same
     colours, same default-visibility rules.

Attack grouping: contiguous attack==1 runs are first identified; runs whose
edges are within --gap-threshold timesteps are merged into a single plot
window. Each window then gets a --buffer of timesteps on either side for
context.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference import (  # noqa: E402
    apply_lambda,
    apply_sigma_floor,
    build_dataset_from_csv,
    load_ensemble,
    run_inference,
    sma_smooth,
    standardised_residual,
)


# Plotly Tableau-20 + extras — 51 distinguishable colours for sensor traces.
def _palette(n: int) -> list[str]:
    """Return n distinguishable hex colour strings."""
    base = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    ]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def find_attack_events(attack: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end_inclusive) for each contiguous attack run."""
    a = attack.astype(np.int8)
    diff = np.diff(a, prepend=np.int8(0), append=np.int8(0))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def group_events(
    events: list[tuple[int, int]],
    gap_threshold: int,
) -> list[list[tuple[int, int]]]:
    """Group events whose inter-event gap (next_start - prev_end) ≤ gap_threshold."""
    if not events:
        return []
    groups: list[list[tuple[int, int]]] = [[events[0]]]
    for ev in events[1:]:
        prev_end = groups[-1][-1][1]
        if ev[0] - prev_end <= gap_threshold:
            groups[-1].append(ev)
        else:
            groups.append([ev])
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=str,
                        default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'manifest.json'))
    parser.add_argument('--bundle-dir', type=str,
                        default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'calibration_bundle'))
    parser.add_argument('--test-csv', type=str,
                        default=str(REPO_ROOT / 'data' / 'swat' / 'test.csv'))
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Default: results/swat_ensemble/<datestr>/per_attack_plots/.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu by default to avoid GPU contention.')
    parser.add_argument('--gap-threshold', type=int, default=200,
                        help='Two contiguous attack runs whose edges are within '
                             'this many timesteps get grouped into one plot. '
                             'Default 200 (≈ 33 min after 10× downsample).')
    parser.add_argument('--buffer', type=int, default=50,
                        help='Pre/post context timesteps included in the plot '
                             'window around the grouped attack span.')
    parser.add_argument('--top-k', type=int, default=5,
                        help='How many sensors are shown by default per plot. '
                             'The remaining sensors are in the legend with '
                             'visible="legendonly" — click to enable.')
    parser.add_argument('--max-points', type=int, default=4000,
                        help='Decimate inside each window to this many points '
                             '(should rarely trigger — windows are short).')
    args = parser.parse_args()

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print('[plot_gdn_attacks] plotly not installed; pip install plotly', file=sys.stderr)
        sys.exit(2)

    bundle_dir = Path(args.bundle_dir)
    if not (bundle_dir / 'bundle.json').is_file():
        raise SystemExit(f'[plot_gdn_attacks] calibration bundle missing at {bundle_dir}; '
                         'run scripts/calibrate.sh first.')

    with (bundle_dir / 'bundle.json').open() as f:
        bundle = json.load(f)
    taus_npz = np.load(bundle_dir / 'taus.npz')
    sigma_floor_v = taus_npz['sigma_floor_v']
    if 'lam_v' in taus_npz.files:
        lam_for_apply = taus_npz['lam_v']
        lam_summary = f'λ_v median={float(np.median(lam_for_apply)):.4f}'
    else:
        lam_for_apply = float(bundle['lambda'])
        lam_summary = f'λ (scalar)={lam_for_apply:.4f}'
    sma_window = int(bundle.get('sma_window', 4))
    tau_v3 = float(bundle['variant3_maxv_paper_sweep']['threshold'])
    v4_cfg = bundle.get('variant4_sustained_window')
    if v4_cfg is None:
        raise SystemExit('[plot_gdn_attacks] bundle has no variant4_sustained_window; '
                         're-run scripts/calibrate.sh first.')
    v4_W = int(v4_cfg['W'])
    v4_Kw = int(v4_cfg['K_w'])

    print(f'[plot_gdn_attacks] loaded bundle: {lam_summary}  '
          f'σ_floor median={float(np.median(sigma_floor_v)):.4e}  sma={sma_window}')
    print(f'[plot_gdn_attacks] τ_v3={tau_v3:.4f}  v4: W={v4_W} K_w={v4_Kw}')

    print(f'[plot_gdn_attacks] loading ensemble (device={args.device})...')
    ensemble = load_ensemble(args.manifest, device=args.device, repo_root=REPO_ROOT)
    feature_map = ensemble.feature_map
    V = len(feature_map)
    print(f'[plot_gdn_attacks] M={ensemble.cfg.M} V={V} slide_win={ensemble.cfg.slide_win}')

    print(f'[plot_gdn_attacks] preparing dataset from {args.test_csv}...')
    ds = build_dataset_from_csv(
        args.test_csv, feature_map, ensemble.fc_edge_index,
        slide_win=ensemble.cfg.slide_win, slide_stride=1, mode='test',
    )
    print(f'[plot_gdn_attacks] {len(ds):,} test windows')

    t0 = time.time()
    out = run_inference(ensemble, ds, batch_size=ensemble.cfg.batch)
    print(f'[plot_gdn_attacks] inference done in {time.time()-t0:.1f}s')

    sigma2_floored = apply_sigma_floor(out.sigma2_total, sigma_floor_v)
    sigma2_cal = apply_lambda(sigma2_floored, lam_for_apply)
    r = standardised_residual(out.ground_truth, out.mu_bar, sigma2_cal)        # (T, V)
    sigma_tot = np.sqrt(np.maximum(sigma2_cal, 0.0))                            # (T, V)

    T = r.shape[0]
    attack = out.attack_label.astype(np.int8)

    # System-level v3 / v4 quantities (same as in plot_gdn_test.py).
    A_max = r.max(axis=1)
    A_s = sma_smooth(A_max, sma_window)
    pred_alarm_v3 = (A_s > tau_v3).astype(np.int8)
    kernel = np.ones(v4_W, dtype=np.int32)
    rolling_sum = np.convolve(pred_alarm_v3.astype(np.int32), kernel, mode='full')[:T]
    pred_alarm_v4 = (rolling_sum >= v4_Kw).astype(np.int8)

    # ------------------------------------------------------------------
    # Identify attack events and group nearby ones.
    # ------------------------------------------------------------------
    events = find_attack_events(attack)
    print(f'[plot_gdn_attacks] found {len(events)} attack runs in {T:,} test windows')
    groups = group_events(events, gap_threshold=args.gap_threshold)
    print(f'[plot_gdn_attacks] grouped into {len(groups)} attack windows '
          f'(gap_threshold={args.gap_threshold} timesteps)')

    # ------------------------------------------------------------------
    # Build four system-level uncertainty signals from the literature.
    # ------------------------------------------------------------------
    # All operate on per-sensor σ_tot,v(t) of shape (T, V). The first three
    # are order-statistics with progressively more studentisation; the fourth
    # is a multivariate Mahalanobis distance on raw residuals (Johnstone &
    # Ndiaye 2024 — multivariate split conformal score).
    raw_res = out.ground_truth - out.mu_bar              # (T, V)
    eps = 1e-9
    sigma_floor_safe = np.maximum(sigma_floor_v.astype(np.float32), eps)
    sigma_tot_st = sigma_tot / sigma_floor_safe[None, :]  # studentised by σ_floor

    # 1) U_max — max_v σ_tot,v(t). The current default; biased toward
    #    naturally noisy sensors. Diquigiovanni et al. 2022 (max-of-V).
    U_max = sigma_tot.max(axis=1)

    # 2) U_max_st — max_v σ_tot,v(t) / σ_floor,v. Studentised max.
    #    Hawkins 1980 (max studentised residual).
    U_max_st = sigma_tot_st.max(axis=1)

    # 3) U_top5_st — mean of top-5 studentised σ. Aggarwal & Sathe 2017
    #    (Outlier Ensembles), trades single-spike sensitivity for robustness.
    K = min(5, V)
    U_top5_st = np.partition(sigma_tot_st, V - K, axis=1)[:, V - K:].mean(axis=1)

    # 4) U_mahal — sqrt((y - μ̄)ᵀ Σ̂⁻¹ (y - μ̄)). Empirical residual
    #    covariance fitted on the calibration set 𝒞 (attack==0 rows in the
    #    front of test.csv). Johnstone & Ndiaye 2024.
    indices = json.load(open(bundle_dir / 'calibration_set_indices.json'))
    c_end_pos = int(indices['C_row_range'][1])
    n_C_windows = max(0, c_end_pos - ensemble.cfg.slide_win + 1)
    mask_C = np.zeros(T, dtype=bool)
    mask_C[:n_C_windows] = True
    mask_C &= (attack == 0)
    if mask_C.sum() < 10:
        # Fallback: use first attack==0 windows up to 10000 if 𝒞 mask too narrow.
        first_clean = np.where(attack == 0)[0][:10000]
        mask_C = np.zeros(T, dtype=bool); mask_C[first_clean] = True
    res_C = raw_res[mask_C]                           # (n_clean, V)
    mu_C = res_C.mean(axis=0)
    cov_C = np.cov(res_C, rowvar=False)
    ridge = 1e-3 * np.trace(cov_C) / V                # Tikhonov regularisation
    cov_C_reg = cov_C + ridge * np.eye(V)
    inv_cov_C = np.linalg.inv(cov_C_reg)
    delta = raw_res - mu_C[None, :]
    U_mahal = np.sqrt(np.maximum(np.einsum('ti,ij,tj->t', delta, inv_cov_C, delta), 0.0))
    print(f'[plot_gdn_attacks] Mahalanobis fit: |𝒞|={int(mask_C.sum()):,} clean windows, '
          f'ridge={ridge:.3e}, U_mahal range=[{U_mahal.min():.2f}, {U_mahal.max():.2f}]')

    # Aggregator catalogue.
    aggregators = {
        'U_max':      ('max_v σ_tot,v(t)',                                  U_max,      '#1f77b4'),
        'U_max_st':   ('max_v σ_tot,v(t) / σ_floor,v',                      U_max_st,   '#d62728'),
        'U_top5_st':  ('mean(top-5 of σ_tot,v(t)/σ_floor,v)',               U_top5_st,  '#9467bd'),
        'U_mahal':    ('sqrt((y-μ̄)ᵀΣ̂⁻¹(y-μ̄))   [Johnstone-Ndiaye 2024]', U_mahal,    '#2ca02c'),
    }

    # Per-aggregator τ (95th-pct on full test set) for spike counts.
    tau_per_agg = {name: float(np.quantile(sig, 0.95))
                   for name, (_, sig, _) in aggregators.items()}
    print('[plot_gdn_attacks] per-aggregator τ_U (95th-pctl on full test):')
    for name, t in tau_per_agg.items():
        print(f'    {name:<11} τ = {t:.4f}')

    def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
        """Trapezoidal-rule ROC-AUC; numerically stable, no sklearn dep."""
        y = np.asarray(y_true).astype(np.int8)
        s = np.asarray(score, dtype=np.float64)
        if y.sum() == 0 or y.sum() == len(y):
            return float('nan')
        order = np.argsort(-s, kind='stable')
        y_sorted = y[order]
        # Standard rank-based AUC using positive-vs-negative pair counts.
        n_pos = int(y_sorted.sum())
        n_neg = int(len(y_sorted) - n_pos)
        # Mann-Whitney U via ranks.
        ranks = np.empty(len(s), dtype=np.float64)
        order_asc = np.argsort(s, kind='stable')
        ranks[order_asc] = np.arange(1, len(s) + 1)
        sum_ranks_pos = ranks[y == 1].sum()
        return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def best_lag(g: np.ndarray, u: np.ndarray, max_lag: int = 30) -> int:
        """Return k in [-max_lag, max_lag] maximising corr(u[t], g[t-k])."""
        g = g - g.mean(); u = u - u.mean()
        if g.std() == 0 or u.std() == 0:
            return 0
        best_k, best_c = 0, -np.inf
        for k in range(-max_lag, max_lag + 1):
            if k == 0:
                gg, uu = g, u
            elif k > 0:
                gg, uu = g[k:], u[:-k]
            else:
                gg, uu = g[:k], u[-k:]
            if len(gg) < 3:
                continue
            c = float(np.dot(gg, uu) / (gg.std() * uu.std() * len(gg)))
            if c > best_c:
                best_c = c; best_k = k
        return best_k

    def score_one(gflag: np.ndarray, U: np.ndarray, tau: float) -> dict:
        """Compute AUC, spike_alignment, rise_ratio, lag, composite."""
        auc = roc_auc(gflag, U)
        attack_mask = gflag == 1
        spikes_in = int(((U > tau) & attack_mask).sum())
        spikes_out = int(((U > tau) & ~attack_mask).sum())
        if spikes_in + spikes_out == 0:
            spike_alignment = float('nan')
        else:
            spike_alignment = spikes_in / float(spikes_in + spikes_out)
        if attack_mask.any() and (~attack_mask).any():
            med_in = float(np.median(U[attack_mask]))
            med_out = float(np.median(U[~attack_mask]))
            rise_ratio = med_in / max(1e-6, med_out)
        else:
            med_in = med_out = float('nan')
            rise_ratio = float('nan')
        lag = best_lag(gflag.astype(float), U)
        comps = []
        if not np.isnan(auc): comps.append(auc)
        if not np.isnan(spike_alignment): comps.append(spike_alignment)
        composite = float(np.mean(comps)) if comps else float('nan')
        return {
            'auc': auc, 'spike_alignment': spike_alignment,
            'spikes_in': spikes_in, 'spikes_out': spikes_out,
            'rise_ratio': rise_ratio,
            'median_U_in_attack': med_in, 'median_U_out_attack': med_out,
            'best_lag': lag, 'composite': composite,
        }

    # Score each attack window with each of the four aggregators.
    scores = []          # one row per attack
    for g_idx, group in enumerate(groups):
        g_start = max(0, group[0][0] - args.buffer)
        g_end = min(T - 1, group[-1][1] + args.buffer)
        sl = slice(g_start, g_end + 1)
        gflag = attack[sl].astype(np.int8)
        n_attack = int((gflag == 1).sum())
        n_buffer = int((gflag == 0).sum())
        per_agg = {}
        for name, (_, sig, _) in aggregators.items():
            per_agg[name] = score_one(gflag, sig[sl], tau_per_agg[name])
        # Best aggregator for this attack and its composite.
        best_name = max(per_agg.keys(),
                        key=lambda k: (per_agg[k]['composite']
                                       if not np.isnan(per_agg[k]['composite']) else -1.0))
        scores.append({
            'idx': g_idx,
            'rows_start': group[0][0], 'rows_end': group[-1][1],
            'n_runs': len(group),
            'n_attack_in_window': n_attack,
            'n_buffer_in_window': n_buffer,
            'per_agg': per_agg,
            'best_agg': best_name,
            'best_composite': per_agg[best_name]['composite'],
        })

    # Sort by best-aggregator composite, descending.
    scores_sorted = sorted(scores, key=lambda d: (-(d['best_composite']
                                                     if not np.isnan(d['best_composite']) else -1.0),
                                                   d['idx']))
    # Print summary to stdout — one row per attack with all four aggregators.
    print('\n[plot_gdn_attacks] per-attack alignment scores (best → worst).')
    print('  Each cell is "AUC|spike_align|composite" for the named aggregator.')
    print(f"{'rank':>4} {'idx':>3} {'rows':>15}  "
          f"{'U_max':>20}  {'U_max_st':>20}  {'U_top5_st':>20}  {'U_mahal':>20}  "
          f"{'best':>11}  {'comp':>6}")
    for rank, s in enumerate(scores_sorted):
        rows_str = f"{s['rows_start']:,}-{s['rows_end']:,}"
        cells = []
        for name in aggregators:
            p = s['per_agg'][name]
            auc_s = '   nan' if np.isnan(p['auc']) else f"{p['auc']:>5.3f}"
            sa_s  = '   nan' if np.isnan(p['spike_alignment']) else f"{p['spike_alignment']:>5.3f}"
            cp_s  = '   nan' if np.isnan(p['composite']) else f"{p['composite']:>5.3f}"
            cells.append(f"{auc_s}|{sa_s}|{cp_s}")
        print(f"{rank:>4} {s['idx']:>3} {rows_str:>15}  "
              f"{cells[0]:>20}  {cells[1]:>20}  {cells[2]:>20}  {cells[3]:>20}  "
              f"{s['best_agg']:>11}  {s['best_composite']:>6.3f}")
    score_by_idx = {s['idx']: s for s in scores}

    if args.out_dir is None:
        from datetime import datetime
        datestr = datetime.now().strftime('%m%d-%H%M%S')
        out_dir = REPO_ROOT / 'results' / 'swat_ensemble' / datestr / 'per_attack_plots'
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[plot_gdn_attacks] writing HTML files to {out_dir}')

    palette = _palette(V)

    for g_idx, group in enumerate(groups):
        # Plot window: pre/post buffer around the grouped attack span.
        g_start = max(0, group[0][0] - args.buffer)
        g_end = min(T - 1, group[-1][1] + args.buffer)
        window = np.arange(g_start, g_end + 1)
        n_pts = len(window)
        if n_pts > args.max_points:
            step = (n_pts + args.max_points - 1) // args.max_points
            window = window[::step]

        # Per-sensor anomaly score within window: max |r_v(t)| over the
        # ATTACK rows only (not the buffer), so default-visible sensors are
        # the ones the model thought were most anomalous during the attack.
        attack_rows = np.concatenate([np.arange(s, e + 1) for s, e in group])
        attack_rows = attack_rows[(attack_rows >= 0) & (attack_rows < T)]
        per_sensor_score = np.abs(r[attack_rows, :]).max(axis=0)               # (V,)
        topk_idx = set(np.argsort(-per_sensor_score)[:args.top_k].tolist())

        # ----- Build figure -----
        n_runs = len(group)
        s = score_by_idx[g_idx]
        best_p = s['per_agg'][s['best_agg']]
        title = (f'Attack window #{g_idx:02d}: rows [{group[0][0]:,} – '
                 f'{group[-1][1]:,}]   ({n_runs} attack run{"s" if n_runs > 1 else ""}, '
                 f'{int((attack[group[0][0]:group[-1][1]+1] == 1).sum()):,} attack timesteps) '
                 f'  |  best aggregator: <b>{s["best_agg"]}</b> '
                 f'(AUC={best_p["auc"]:.3f}, spike={best_p["spike_alignment"]:.2f}, '
                 f'rise={best_p["rise_ratio"]:.2f}×, comp={s["best_composite"]:.3f})')

        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True,
            row_heights=[0.08, 0.14, 0.20, 0.29, 0.29],
            subplot_titles=(
                'Ground truth (red) + v4 alarm (dark blue) + v3 alarm (faded orange)',
                f'System anomaly score A_s(t) = SMA(max_v r_v(t), 4) with τ = {tau_v3:.2f}',
                ('System uncertainty signals — 4 aggregators normalised by their own '
                 '95th-pctl (compare which agrees best with ground truth)'),
                'Standardised residual r_v(t) — all 51 sensors (click legend to toggle)',
                'Total predictive uncertainty σ_tot,v(t) — all 51 sensors (click legend to toggle)',
            ),
            vertical_spacing=0.045,
        )

        # ----- Subplot 1: ground truth + v4 + v3 alarms -----
        fig.add_trace(go.Scatter(
            x=window, y=attack[window], mode='lines',
            line=dict(width=1, color='rgba(220, 20, 60, 1.0)', shape='hv'),
            name='ground truth', fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.3)',
            legendgroup='system', showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=window, y=pred_alarm_v3[window], mode='lines',
            line=dict(width=1, color='rgba(255, 140, 0, 0.6)', shape='hv'),
            name='v3 alarm (ref)', fill='tozeroy',
            fillcolor='rgba(255, 140, 0, 0.15)',
            legendgroup='system', showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=window, y=pred_alarm_v4[window], mode='lines',
            line=dict(width=1.6, color='rgba(20, 60, 180, 1.0)', shape='hv'),
            name='v4 alarm', fill='tozeroy',
            fillcolor='rgba(20, 60, 180, 0.25)',
            legendgroup='system', showlegend=True,
        ), row=1, col=1)

        # ----- Subplot 2: system score A_s with τ -----
        fig.add_trace(go.Scatter(
            x=window, y=A_s[window], mode='lines',
            line=dict(width=1.5, color='rgba(200, 30, 30, 0.9)'),
            name='A_s(t)', legendgroup='system', showlegend=True,
        ), row=2, col=1)
        fig.add_hline(y=tau_v3, line_dash='dot',
                      line=dict(color='black', width=1.5),
                      annotation_text=f'τ = {tau_v3:.2f}',
                      annotation_position='top right',
                      row=2, col=1)

        # Highlight each individual attack run with a faint vertical band
        # (helps when multiple runs are grouped into one figure).
        for grp_s, grp_e in group:
            fig.add_vrect(x0=grp_s, x1=grp_e,
                          fillcolor='rgba(220, 20, 60, 0.06)',
                          line_width=0,
                          row='all', col=1)

        # ----- Subplot 3: 4 system-level uncertainty aggregators -----
        # Normalise each by its own global 95th-pctl so the four traces are
        # directly comparable on a single y-axis (each ≈ 1 at its 95th-pctl).
        for name, (label, sig, colour) in aggregators.items():
            denom = max(tau_per_agg[name], 1e-9)
            fig.add_trace(go.Scatter(
                x=window, y=sig[window] / denom, mode='lines',
                line=dict(width=1.6, color=colour),
                name=f'{name}',
                hovertext=[label] * len(window),
                hoverinfo='x+y+name+text',
                legendgroup=f'agg_{name}',
                showlegend=True,
            ), row=3, col=1)
            # 1.0-line (= each aggregator's 95th-pctl).
            # Drawn once via add_hline below.
        fig.add_hline(y=1.0, line_dash='dot',
                      line=dict(color='black', width=1),
                      annotation_text='95th-pctl on full test',
                      annotation_position='top right',
                      row=3, col=1)

        # ----- Subplots 4 & 5: all 51 sensors, toggleable legend -----
        for v in range(V):
            sensor = feature_map[v]
            in_topk = v in topk_idx
            visible = True if in_topk else 'legendonly'
            colour = palette[v]
            badge = ' ★' if in_topk else ''  # mark default-visible sensors

            # Standardised residual r_v(t).
            fig.add_trace(go.Scatter(
                x=window, y=r[window, v], mode='lines',
                line=dict(width=1.0, color=colour),
                name=f'{v:02d} {sensor}{badge}',
                legendgroup=f'sensor_{v:02d}',
                showlegend=True,
                visible=visible,
            ), row=4, col=1)

            # σ_tot,v(t) — same colour, same legendgroup so legend toggles
            # both panels at once.
            fig.add_trace(go.Scatter(
                x=window, y=sigma_tot[window, v], mode='lines',
                line=dict(width=1.0, color=colour, dash='solid'),
                name=f'{v:02d} {sensor}{badge}',
                legendgroup=f'sensor_{v:02d}',
                showlegend=False,                # legend entry already in r-row
                visible=visible,
            ), row=5, col=1)

        fig.update_layout(
            title=title,
            height=1500, hovermode='x unified',
            margin=dict(l=60, r=20, t=80, b=50),
            legend=dict(
                title='aggregators (top) + sensors (★ = top-{} during attack)'.format(args.top_k),
                tracegroupgap=5,
                font=dict(size=10),
            ),
        )
        fig.update_xaxes(title_text='timestep (windowed)', row=5, col=1)
        fig.update_yaxes(title_text='attack / alarm', row=1, col=1,
                         range=[-0.1, 1.1], tickvals=[0, 1])
        fig.update_yaxes(title_text='A_s(t)', row=2, col=1)
        fig.update_yaxes(title_text='U(t) / 95th-pctl', row=3, col=1)
        fig.update_yaxes(title_text='r_v(t)', row=4, col=1)
        fig.update_yaxes(title_text='σ_tot,v(t)', row=5, col=1)

        out_path = out_dir / f'attack_{g_idx:02d}_rows_{group[0][0]:06d}-{group[-1][1]:06d}.html'
        fig.write_html(out_path, include_plotlyjs='cdn')

    # ------------------------------------------------------------------
    # Write CSV + sorted index.html with one column per aggregator.
    # ------------------------------------------------------------------
    csv_path = out_dir / 'alignment_scores.csv'
    base_cols = ['rank', 'idx', 'rows_start', 'rows_end', 'n_runs',
                 'n_attack_in_window', 'n_buffer_in_window',
                 'best_agg', 'best_composite']
    metric_cols = ['auc', 'spike_alignment', 'spikes_in', 'spikes_out',
                   'rise_ratio', 'median_U_in_attack', 'median_U_out_attack',
                   'best_lag', 'composite']
    csv_header = base_cols + [f'{a}_{m}' for a in aggregators for m in metric_cols]
    def fmt(v):
        if isinstance(v, float):
            return 'NaN' if np.isnan(v) else f'{v:.6f}'
        return str(v)
    with csv_path.open('w') as f:
        f.write(','.join(csv_header) + '\n')
        for rank, s in enumerate(scores_sorted):
            row = [rank, s['idx'], s['rows_start'], s['rows_end'], s['n_runs'],
                   s['n_attack_in_window'], s['n_buffer_in_window'],
                   s['best_agg'], s['best_composite']]
            for a in aggregators:
                p = s['per_agg'][a]
                row.extend([p[m] for m in metric_cols])
            f.write(','.join(fmt(v) for v in row) + '\n')
    print(f'[plot_gdn_attacks] wrote {csv_path}')

    # Sorted index — best agreeing first, with per-aggregator composite cells.
    def comp_cell(p: dict) -> str:
        if np.isnan(p['composite']):
            return '<td style="color:#999">—</td>'
        pct = p['composite'] * 100
        bar_color = ('#28a745' if pct >= 70
                     else ('#ffc107' if pct >= 50 else '#dc3545'))
        bar = (f'<div title="AUC={p["auc"]:.3f} spike={p["spike_alignment"]:.2f} '
               f'rise={p["rise_ratio"]:.2f} in/out={p["spikes_in"]}/{p["spikes_out"]} '
               f'lag={p["best_lag"]:+d}" '
               f'style="background:#eee;width:80px;height:13px;border-radius:3px;'
               f'overflow:hidden;display:inline-block;vertical-align:middle">'
               f'<div style="width:{pct:.1f}%;background:{bar_color};'
               f'height:100%"></div></div>&nbsp;{pct:.1f}%')
        return f'<td>{bar}</td>'

    rows_html = []
    for rank, s in enumerate(scores_sorted):
        g_idx = s['idx']
        group = groups[g_idx]
        fname = f'attack_{g_idx:02d}_rows_{group[0][0]:06d}-{group[-1][1]:06d}.html'
        cells = ''.join(comp_cell(s['per_agg'][a]) for a in aggregators)
        best_str = f'{s["best_agg"]} ({s["best_composite"]*100:.1f}%)' \
                   if not np.isnan(s['best_composite']) else '—'
        rows_html.append(
            f'<tr><td>{rank}</td>'
            f'<td>{g_idx:02d}</td>'
            f'<td><a href="{fname}">{fname}</a></td>'
            f'<td>{group[0][0]:,} – {group[-1][1]:,}</td>'
            f'<td>{len(group)}</td>'
            f'<td>{s["n_attack_in_window"]:,}</td>'
            + cells
            + f'<td><b>{best_str}</b></td></tr>'
        )

    legend_text = (
        'Each cell shows the <b>composite alignment score</b> (50/50 mix of AUC '
        'and spike_alignment) for one of the four system-uncertainty aggregators. '
        'Hover any bar to see the underlying AUC, spike_in/out, rise_ratio, and lag. '
        'Sorted best → worst by the <b>best aggregator</b> per attack.'
    )
    agg_definitions = '<dl style="font-size:13px;margin-top:8px">'
    for name, (label, _, colour) in aggregators.items():
        agg_definitions += (f'<dt style="color:{colour};font-weight:600">{name}</dt>'
                             f'<dd style="margin:0 0 6px 18px">{label}</dd>')
    agg_definitions += '</dl>'

    agg_header = ''.join(
        f'<th style="color:{aggregators[a][2]}">{a}</th>' for a in aggregators
    )
    index_path = out_dir / 'index.html'
    index_path.write_text(
        '<!doctype html><meta charset="utf-8"><title>GDN per-attack plots</title>'
        '<style>'
        'body{font-family:system-ui,sans-serif;margin:24px;max-width:1400px}'
        'table{border-collapse:collapse;font-size:14px}'
        'td,th{padding:6px 10px;border:1px solid #ddd}'
        'th{background:#f5f5f5;text-align:left}'
        'tr:hover{background:#fafafa}'
        '.legend{margin:12px 0;font-size:13px;color:#444}'
        '.legend code{background:#f0f0f0;padding:1px 5px;border-radius:3px}'
        'dt{font-family:ui-monospace,monospace}'
        '</style>'
        f'<h1>GDN_UQ per-attack plots — {len(groups)} attack windows</h1>'
        f'<p>Source bundle: <code>{bundle_dir}</code><br>'
        f'τ_v3 = {tau_v3:.2f}, v4 W = {v4_W}, K_w = {v4_Kw}, '
        f'gap_threshold = {args.gap_threshold}, buffer = {args.buffer}</p>'
        f'<div class="legend">{legend_text}{agg_definitions}'
        '<p>Per-aggregator τ_U (95th-pctl on full test set, used for spike counts):</p><ul>'
        + ''.join(f'<li><code>{a}</code>: τ = {tau_per_agg[a]:.4f}</li>'
                  for a in aggregators)
        + '</ul></div>'
        '<table><thead><tr>'
        '<th>rank</th><th>#</th><th>file</th><th>row range</th>'
        '<th>runs</th><th>attack ts</th>'
        + agg_header
        + '<th>best</th>'
        '</tr></thead><tbody>'
        + ''.join(rows_html)
        + '</tbody></table>'
        f'<p style="margin-top:18px;font-size:13px;color:#666">'
        f'Raw scores in <a href="alignment_scores.csv">alignment_scores.csv</a>.</p>'
    )
    print(f'[plot_gdn_attacks] wrote {len(groups)} attack-window HTML files + index.html in {out_dir}')


if __name__ == '__main__':
    main()
