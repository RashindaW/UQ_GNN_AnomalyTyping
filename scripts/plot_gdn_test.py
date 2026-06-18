"""Per-node plotly plots for the GDN test set, variant 4 (sustained-window).

v4 is the sustained-window detector built on top of v3 (max-of-V): it fires
only when v3 has already fired in at least K_w of the previous W timesteps.
The Stage-2 calibrated values are W=10, K_w=8, τ=263.07. v4 trades a small
recall hit for a huge precision gain (final-test FP drops from 361 → 8).

For each of the 51 sensors, generate an HTML file with four vertically-stacked
subplots:
  1. Ground-truth attack flag (0 / 1) over time.
  2. **System** predicted anomaly label — v4 alarm (prominent, dark blue)
     overlaid with v3 alarm (faded orange) so the difference between the two
     decisions is visible. Both traces are identical across all 51 files
     (same system-level decisions), but they're plotted in each per-sensor
     file so you can see how each sensor's local signals correspond to the
     system decisions.
  3. System anomaly score A_s(t) = SMA(max_v r_v(t), 4) (red, prominent) with
     τ_v3 as a dashed horizontal line. The current sensor's per-node r_v(t)
     is overlaid as a lighter trace for context — when r_v ≈ A_s, this sensor
     was responsible for the v3/v4 decision at that timestep.
  4. Total predictive uncertainty σ_tot,v(t) (sqrt of σ²_aleatoric + σ²_epistemic),
     with the aleatoric and epistemic components shown as lighter overlay lines.

Runs over the FULL data/swat/test.csv so the user can see the model behaviour
across the whole test period (not only the final-test slice used for F1
reporting).
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=str,
                        default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'manifest.json'))
    parser.add_argument('--bundle-dir', type=str,
                        default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'calibration_bundle'))
    parser.add_argument('--test-csv', type=str,
                        default=str(REPO_ROOT / 'data' / 'swat' / 'test.csv'))
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Default: results/swat_ensemble/<datestr>/per_node_plots/.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu by default to avoid contending with concurrent training.')
    parser.add_argument('--max-points', type=int, default=20000,
                        help='Decimate to this many points per trace (plotly perf).')
    args = parser.parse_args()

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print('[plot_gdn_test] plotly not installed; pip install plotly', file=sys.stderr)
        sys.exit(2)

    bundle_dir = Path(args.bundle_dir)
    if not (bundle_dir / 'bundle.json').is_file():
        raise SystemExit(f'[plot_gdn_test] calibration bundle missing at {bundle_dir}; '
                         'run scripts/calibrate.sh first.')

    with (bundle_dir / 'bundle.json').open() as f:
        bundle = json.load(f)
    taus_npz = np.load(bundle_dir / 'taus.npz')
    sigma_floor_v = taus_npz['sigma_floor_v']
    # Per-sensor λ_v if available (Stage-2+), else fall back to scalar (legacy).
    if 'lam_v' in taus_npz.files:
        lam_for_apply = taus_npz['lam_v']
        lam_summary = f'λ_v median={float(np.median(lam_for_apply)):.4f}'
    else:
        lam_for_apply = float(bundle['lambda'])
        lam_summary = f'λ (scalar)={lam_for_apply:.4f}'
    sma_window = int(bundle.get('sma_window', 4))
    # v3 max-of-V threshold from the paper-style F1 sweep on labeled-val.
    tau_v3 = float(bundle['variant3_maxv_paper_sweep']['threshold'])
    # v4 sustained-window (built on top of v3).
    v4_cfg = bundle.get('variant4_sustained_window')
    if v4_cfg is None:
        raise SystemExit('[plot_gdn_test] bundle has no variant4_sustained_window; '
                         're-run scripts/calibrate.sh first.')
    v4_W = int(v4_cfg['W'])
    v4_Kw = int(v4_cfg['K_w'])

    print(f'[plot_gdn_test] loaded bundle: {lam_summary}  '
          f'σ_floor median={float(np.median(sigma_floor_v)):.4e}  sma={sma_window}')
    print(f'[plot_gdn_test] variant 3 threshold τ_v3 = {tau_v3:.4f}')
    print(f'[plot_gdn_test] variant 4 (sustained): W={v4_W}, K_w={v4_Kw}')

    print(f'[plot_gdn_test] loading ensemble (device={args.device})...')
    ensemble = load_ensemble(args.manifest, device=args.device, repo_root=REPO_ROOT)
    feature_map = ensemble.feature_map
    V = len(feature_map)
    print(f'[plot_gdn_test] M={ensemble.cfg.M} V={V} slide_win={ensemble.cfg.slide_win}')

    print(f'[plot_gdn_test] preparing dataset from {args.test_csv}...')
    ds = build_dataset_from_csv(
        args.test_csv, feature_map, ensemble.fc_edge_index,
        slide_win=ensemble.cfg.slide_win, slide_stride=1, mode='test',
    )
    print(f'[plot_gdn_test] {len(ds):,} test windows')

    t0 = time.time()
    out = run_inference(ensemble, ds, batch_size=ensemble.cfg.batch)
    print(f'[plot_gdn_test] inference done in {time.time()-t0:.1f}s '
          f'({"cpu" if args.device == "cpu" else args.device})')

    # Apply σ-floor and λ correction (matches detect.py).
    sigma2_floored = apply_sigma_floor(out.sigma2_total, sigma_floor_v)
    sigma2_cal = apply_lambda(sigma2_floored, lam_for_apply)
    r = standardised_residual(out.ground_truth, out.mu_bar, sigma2_cal)        # (T, V)
    sigma_tot = np.sqrt(np.maximum(sigma2_cal, 0.0))
    sigma_a = np.sqrt(np.maximum(out.sigma2_aleatoric, 0.0))
    sigma_e = np.sqrt(np.maximum(out.sigma2_epistemic, 0.0))

    T = r.shape[0]
    if args.out_dir is None:
        from datetime import datetime
        datestr = datetime.now().strftime('%m%d-%H%M%S')
        out_dir = REPO_ROOT / 'results' / 'swat_ensemble' / datestr / 'per_node_plots'
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[plot_gdn_test] writing {V} HTML files to {out_dir}')

    # Decimation for plotly responsiveness.
    if T > args.max_points:
        step = (T + args.max_points - 1) // args.max_points
        sub = np.arange(0, T, step)
        print(f'[plot_gdn_test] decimating from {T:,} to {len(sub):,} points (step={step})')
    else:
        sub = np.arange(T)

    attack = out.attack_label.astype(np.int8)

    # ------------------------------------------------------------------
    # System-level v3 + v4 quantities (computed once, reused across all 51 plots).
    # ------------------------------------------------------------------
    A_max = r.max(axis=1)                                # (T,)
    A_s = sma_smooth(A_max, sma_window)                  # (T,) smoothed system score
    pred_alarm_v3 = (A_s > tau_v3).astype(np.int8)       # (T,)  v3 system alarm
    # v4 sustained-window: rolling sum over W timesteps; alarm if ≥ K_w hits.
    kernel = np.ones(v4_W, dtype=np.int32)
    rolling_sum = np.convolve(pred_alarm_v3.astype(np.int32), kernel, mode='full')[:len(pred_alarm_v3)]
    pred_alarm_v4 = (rolling_sum >= v4_Kw).astype(np.int8)
    argmax_v = r.argmax(axis=1)                          # (T,)  which sensor argmaxed

    print(f'[plot_gdn_test] system-level metrics on full test set: '
          f'v3 alarm_rate={float(pred_alarm_v3.mean()):.4f}  '
          f'v4 alarm_rate={float(pred_alarm_v4.mean()):.4f}  '
          f'attack_rate={float(attack.mean()):.4f}')

    for v in range(V):
        sensor = feature_map[v]
        # Mark timesteps where THIS sensor was the argmax (responsible for v3 score).
        is_blame = (argmax_v == v).astype(np.int8)
        blame_count_total = int(is_blame.sum())
        blame_count_alarm_v4 = int(((argmax_v == v) & (pred_alarm_v4 == 1)).sum())

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=(
                'Ground-truth attack flag',
                (f'Predicted anomaly labels: v4 sustained-window (dark blue, primary) '
                 f'+ v3 (faded orange, reference) — '
                 f'v4: τ={tau_v3:.2f}, W={v4_W}, K_w={v4_Kw}; '
                 f'v3: 1[A_s(t) > τ]'),
                (f'System anomaly score A_s(t) (red) and this sensor r_v(t) (light blue) — '
                 f'τ = {tau_v3:.4f}; sensor blame: {blame_count_total} ts '
                 f'({blame_count_alarm_v4} during v4 alarms)'),
                'Predictive uncertainty (σ_tot, with σ_aleatoric + σ_epistemic decomposition)',
            ),
            vertical_spacing=0.06,
        )
        # Subplot 1 — ground-truth attack labels (red, filled)
        fig.add_trace(
            go.Scatter(
                x=sub, y=attack[sub], mode='lines',
                line=dict(width=1, color='rgba(220, 20, 60, 1.0)', shape='hv'),
                name='ground truth', fill='tozeroy',
                fillcolor='rgba(220, 20, 60, 0.25)',
            ),
            row=1, col=1,
        )
        # Subplot 2 — system alarms. v4 (sustained-window) is the primary
        # detector (dark blue, prominent). v3 is overlaid as a faded reference
        # so the reader can see which v3 alarms got filtered out by the
        # sustained-window rule.
        # Plot v3 first (so v4 overlays on top).
        fig.add_trace(
            go.Scatter(
                x=sub, y=pred_alarm_v3[sub], mode='lines',
                line=dict(width=1, color='rgba(255, 140, 0, 0.6)', shape='hv'),
                name='v3 alarm (reference)', fill='tozeroy',
                fillcolor='rgba(255, 140, 0, 0.15)',
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub, y=pred_alarm_v4[sub], mode='lines',
                line=dict(width=1.6, color='rgba(20, 60, 180, 1.0)', shape='hv'),
                name='v4 alarm (primary)', fill='tozeroy',
                fillcolor='rgba(20, 60, 180, 0.25)',
            ),
            row=2, col=1,
        )
        # Subplot 3 — A_s(t) + this sensor's r_v(t) + τ_v3
        # Light overlay: per-sensor r_v
        fig.add_trace(
            go.Scatter(
                x=sub, y=r[sub, v], mode='lines',
                line=dict(width=1, color='rgba(30, 90, 200, 0.4)'),
                name=f'r_{sensor}(t)',
            ),
            row=3, col=1,
        )
        # Highlight where this sensor was the argmax (helps see when this sensor was "the blame")
        blame_mask = is_blame[sub] == 1
        if blame_mask.any():
            blame_x = sub[blame_mask]
            blame_y = r[blame_x, v]
            fig.add_trace(
                go.Scatter(
                    x=blame_x, y=blame_y, mode='markers',
                    marker=dict(size=4, color='rgba(30, 90, 200, 0.85)',
                                line=dict(width=0)),
                    name='argmax',
                ),
                row=3, col=1,
            )
        # Primary trace: system A_s
        fig.add_trace(
            go.Scatter(
                x=sub, y=A_s[sub], mode='lines',
                line=dict(width=1.5, color='rgba(200, 30, 30, 0.9)'),
                name='A_s(t)',
            ),
            row=3, col=1,
        )
        fig.add_hline(y=tau_v3, line_dash='dot',
                      line=dict(color='black', width=1.5),
                      annotation_text=f'τ = {tau_v3:.3f}',
                      annotation_position='top right',
                      row=3, col=1)
        # Subplot 4 — σ_tot, σ_a, σ_e
        fig.add_trace(
            go.Scatter(
                x=sub, y=sigma_a[sub, v], mode='lines',
                line=dict(width=1, color='rgba(160, 160, 160, 0.7)', dash='dot'),
                name='σ_aleatoric',
            ),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub, y=sigma_e[sub, v], mode='lines',
                line=dict(width=1, color='rgba(180, 90, 200, 0.7)', dash='dot'),
                name='σ_epistemic',
            ),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub, y=sigma_tot[sub, v], mode='lines',
                line=dict(width=1.5, color='rgba(20, 130, 80, 1.0)'),
                name='σ_tot',
            ),
            row=4, col=1,
        )

        fig.update_layout(
            title=(f'Sensor {v:02d}: {sensor} — GDN test-set diagnostic '
                   f'(variant 4 sustained, τ={tau_v3:.2f}, W={v4_W}, K_w={v4_Kw})'),
            height=1100, hovermode='x unified',
            margin=dict(l=60, r=20, t=80, b=50),
        )
        fig.update_xaxes(title_text='timestep (windowed)', row=4, col=1)
        fig.update_yaxes(title_text='ground truth', row=1, col=1,
                         range=[-0.1, 1.1], tickvals=[0, 1])
        fig.update_yaxes(title_text='alarm (v4 primary, v3 ref)', row=2, col=1,
                         range=[-0.1, 1.1], tickvals=[0, 1])
        fig.update_yaxes(title_text='A_s(t) / r_v(t)', row=3, col=1)
        fig.update_yaxes(title_text='σ', row=4, col=1)

        out_path = out_dir / f'node_{v:02d}_{sensor}.html'
        fig.write_html(out_path, include_plotlyjs='cdn')

    print(f'[plot_gdn_test] wrote {V} HTML files to {out_dir}')
    print(f'[plot_gdn_test] open one with: xdg-open {out_dir / f"node_00_{feature_map[0]}.html"}')


if __name__ == '__main__':
    main()
