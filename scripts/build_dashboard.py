"""Build a single-page HTML dashboard summarising the latest GDN_UQ pipeline run.

Reads:
  - pretrained/swat_ensemble/manifest.json
  - pretrained/swat_ensemble/calibration_bundle/{bundle.json, reliability.png}
  - results/swat_ensemble/<latest>/report.json
  - results/swat_ensemble/<latest>/per_node_plots/  (most recent)
  - results/swat_ensemble/<latest>/per_attack_plots/{index.html, alignment_scores.csv}
  - (optional) pretrained/swat_ensemble/calibration_bundle.before_<TS>/bundle.json
    for M=N vs previous-M comparison

Writes a single self-contained HTML file dashboard.html in --out-dir, plus
embedded copies of links to per-node and per-attack indices.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_json(p: Path) -> dict | None:
    if not p.is_file():
        return None
    with p.open() as f:
        return json.load(f)


def latest_subdir_with(parent: Path, child_name: str) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = []
    for d in parent.iterdir():
        if not d.is_dir():
            continue
        target = d / child_name
        if target.exists():
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        return None
    return max(candidates)[1]


def encode_image(p: Path) -> str | None:
    if not p.is_file():
        return None
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode('ascii')
    suffix = p.suffix.lstrip('.').lower()
    mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'svg': 'image/svg+xml', 'gif': 'image/gif'}.get(suffix, 'image/png')
    return f'data:{mime};base64,{b64}'


def fmt_int(x) -> str:
    return f'{int(x):,}' if isinstance(x, (int, float)) else str(x)


def fmt_float(x, places: int = 4) -> str:
    if x is None:
        return '—'
    if isinstance(x, float) and (x != x):  # NaN
        return '—'
    return f'{float(x):.{places}f}'


def fmt_pct(x, places: int = 2) -> str:
    if x is None:
        return '—'
    if isinstance(x, float) and (x != x):
        return '—'
    return f'{100*float(x):.{places}f}%'


def section(title: str, body: str, anchor: str | None = None) -> str:
    aid = f' id="{anchor}"' if anchor else ''
    return f'<section{aid}><h2>{title}</h2>{body}</section>'


def kv_table(rows: list[tuple[str, str]]) -> str:
    body = ''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in rows)
    return f'<table class="kv">{body}</table>'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=str, required=True,
                    help='Destination directory for dashboard.html.')
    ap.add_argument('--manifest', type=str,
                    default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'manifest.json'))
    ap.add_argument('--bundle-dir', type=str,
                    default=str(REPO_ROOT / 'pretrained' / 'swat_ensemble' / 'calibration_bundle'))
    ap.add_argument('--results-root', type=str,
                    default=str(REPO_ROOT / 'results' / 'swat_ensemble'))
    ap.add_argument('--m', type=int, default=None,
                    help='Annotated M value for the title (defaults to manifest M).')
    ap.add_argument('--seeds', type=str, default='')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(Path(args.manifest))
    if manifest is None:
        print(f'[dashboard] manifest missing: {args.manifest}', file=sys.stderr)
        sys.exit(1)

    bundle_dir = Path(args.bundle_dir)
    bundle = read_json(bundle_dir / 'bundle.json')
    if bundle is None:
        print(f'[dashboard] bundle.json missing: {bundle_dir}', file=sys.stderr)
        sys.exit(1)

    # Latest detect output (has report.json).
    results_root = Path(args.results_root)
    latest_detect = latest_subdir_with(results_root, 'report.json')
    report = read_json(latest_detect / 'report.json') if latest_detect else None
    if report is None:
        print('[dashboard] no detect output (results/swat_ensemble/<datestr>/report.json)',
              file=sys.stderr)
        sys.exit(1)

    # Latest per-node and per-attack plot dirs.
    per_node_dir = latest_subdir_with(results_root, 'per_node_plots')
    per_node_dir = (per_node_dir / 'per_node_plots') if per_node_dir else None
    per_attack_dir = latest_subdir_with(results_root, 'per_attack_plots')
    per_attack_dir = (per_attack_dir / 'per_attack_plots') if per_attack_dir else None

    # Optional: previous bundle (one of pretrained/swat_ensemble/calibration_bundle.*).
    prev_bundle = None
    prev_label = None
    for cand in sorted(bundle_dir.parent.glob('calibration_bundle.*')):
        if cand.is_dir() and (cand / 'bundle.json').is_file() \
                and 'before_' in cand.name:
            prev_bundle = read_json(cand / 'bundle.json')
            prev_label = cand.name
    # Fallback: any earlier bundle backup
    if prev_bundle is None:
        for cand in sorted(bundle_dir.parent.glob('calibration_bundle.*')):
            if cand.is_dir() and (cand / 'bundle.json').is_file():
                prev_bundle = read_json(cand / 'bundle.json')
                prev_label = cand.name
                break

    M = args.m or manifest.get('M', len(manifest.get('members', [])))
    seeds_str = args.seeds or ' '.join(str(m['seed']) for m in manifest.get('members', []))
    hp = manifest.get('hyperparameters', {})
    members = manifest.get('members', [])
    datestr = report.get('datestr', '?')
    final_test_range = report.get('final_test_range', [None, None])
    n_windows = report.get('n_windows', None)
    metrics = report.get('metrics_final_test', {})
    triage = report.get('triage_distribution', {})

    # Per-attack alignment top/bottom (CSV).
    attack_top, attack_bottom = [], []
    csv_path = (per_attack_dir / 'alignment_scores.csv') if per_attack_dir else None
    if csv_path and csv_path.is_file():
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            attack_top = rows[:5]
            attack_bottom = rows[-5:]

    # Reliability diagram (embed as base64).
    rel_b64 = encode_image(bundle_dir / 'reliability.png')

    # ----------------------------------------------------------------------
    # Build HTML.
    # ----------------------------------------------------------------------
    html = []
    html.append('<!doctype html><meta charset="utf-8">')
    html.append(f'<title>GDN_UQ M={M} — pipeline dashboard</title>')
    html.append('<style>')
    html.append('''
      body {font-family: system-ui, -apple-system, sans-serif; max-width: 1280px;
            margin: 24px auto; padding: 0 24px; color: #222; line-height: 1.5}
      h1 {margin-top: 0; font-size: 28px}
      h2 {margin-top: 36px; padding-bottom: 6px; border-bottom: 2px solid #eee;
          font-size: 20px}
      h3 {margin-top: 24px; font-size: 16px; color: #444}
      table {border-collapse: collapse; font-size: 14px; margin: 8px 0 16px}
      td, th {padding: 6px 12px; border: 1px solid #ddd; text-align: left}
      th {background: #f5f5f5; font-weight: 600}
      table.kv th {background: #fafafa; width: 280px}
      table.kv td {font-family: ui-monospace, monospace}
      table.scoreboard td.num, table.scoreboard th.num {text-align: right;
            font-variant-numeric: tabular-nums; font-family: ui-monospace, monospace}
      .pill {display: inline-block; padding: 2px 8px; border-radius: 12px;
             font-size: 12px; font-weight: 600; margin-right: 4px}
      .pill.green {background: #e6f4ea; color: #137333}
      .pill.amber {background: #fef7e0; color: #9d6d04}
      .pill.red {background: #fce8e6; color: #b3261e}
      .pill.blue {background: #e8f0fe; color: #1a73e8}
      .nav {background: #f8f9fa; padding: 12px 20px; border-radius: 8px;
            margin-bottom: 20px}
      .nav a {margin-right: 16px}
      .small {font-size: 12px; color: #666}
      .twocol {display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
               align-items: start}
      img {max-width: 100%; height: auto}
      .footer {margin-top: 60px; padding-top: 16px; border-top: 1px solid #eee;
               font-size: 12px; color: #888}
      a {color: #1a73e8; text-decoration: none}
      a:hover {text-decoration: underline}
      .bar {display: inline-block; vertical-align: middle; background: #eee;
            width: 100px; height: 12px; border-radius: 3px; overflow: hidden;
            margin-right: 8px}
      .bar > div {height: 100%}
    ''')
    html.append('</style>')

    # ---- Header ----
    html.append(f'<h1>GDN_UQ pipeline dashboard <span class="pill blue">M = {M}</span></h1>')
    html.append(f'<p class="small">Run timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '
                f'· Detect output: <code>{datestr}</code> '
                f'(rows {fmt_int(final_test_range[0])} – {fmt_int(final_test_range[1])}, '
                f'{fmt_int(n_windows)} windows)</p>')

    # ---- Nav ----
    html.append('<div class="nav">')
    html.append('<b>Sections:</b> ')
    for anchor, label in [
        ('summary', 'Run summary'),
        ('scoreboard', 'Performance scoreboard'),
        ('triage', 'Triage distribution'),
        ('calibration', 'Calibration scalars'),
        ('reliability', 'Reliability'),
        ('attacks', 'Attack alignment'),
        ('plots', 'Plot indices'),
        ('comparison', 'M=prev comparison'),
    ]:
        html.append(f'<a href="#{anchor}">{label}</a>')
    html.append('</div>')

    # ---- Run summary ----
    member_lines = '<br>'.join(
        f'<code>member_{m["index"]:02d}_seed_{m["seed"]}</code> → '
        f'<code>{m["checkpoint"]}</code>'
        f'{" <span class=pill,red>missing</span>" if not m.get("checkpoint_exists", True) else ""}'
        for m in members
    )
    summary_table = kv_table([
        ('M (ensemble size)', f'<b>{M}</b>'),
        ('Seeds', f'<code>{seeds_str}</code>'),
        ('Backbone', f'{manifest.get("model", "gdn_uq")} on {manifest.get("dataset", "swat")}'),
        ('Hyperparameters', '<br>'.join(
            f'<code>{k}={v}</code>' for k, v in sorted(hp.items())
            if k not in ('model', 'dataset', 'optimizer', 'betas')
        )),
        ('Member checkpoints', member_lines),
    ])
    html.append(section('Run summary', summary_table, anchor='summary'))

    # ---- Performance scoreboard ----
    variants_order = ['v1_pernode_or', 'v2_maxv_valmax', 'v3_maxv_paper', 'v4_sustained']
    variant_labels = {
        'v1_pernode_or': 'v1 — per-node OR (Outline §9)',
        'v2_maxv_valmax': 'v2 — max-of-V, val-max τ',
        'v3_maxv_paper': 'v3 — max-of-V, paper F1 sweep',
        'v4_sustained': 'v4 — sustained-window (W, K_w)',
    }
    rows = []
    for vk in variants_order:
        m = metrics.get(vk, {})
        if not m:
            continue
        rows.append(
            f'<tr><th>{variant_labels[vk]}</th>'
            f'<td class="num">{fmt_float(m.get("precision"), 4)}</td>'
            f'<td class="num">{fmt_float(m.get("recall"), 4)}</td>'
            f'<td class="num"><b>{fmt_float(m.get("f1"), 4)}</b></td>'
            f'<td class="num">{fmt_int(m.get("tp"))}</td>'
            f'<td class="num">{fmt_int(m.get("fp"))}</td>'
            f'<td class="num">{fmt_int(m.get("fn"))}</td>'
            f'<td class="num">{fmt_int(m.get("tn"))}</td>'
            f'<td class="num">{fmt_pct(m.get("alarm_rate"), 2)}</td></tr>'
        )
    scoreboard = (
        '<table class="scoreboard">'
        '<thead><tr><th>Variant</th>'
        '<th class="num">Precision</th><th class="num">Recall</th>'
        '<th class="num">F1</th>'
        '<th class="num">TP</th><th class="num">FP</th>'
        '<th class="num">FN</th><th class="num">TN</th>'
        '<th class="num">Alarm rate</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    # v4 config detail
    v4_cfg = bundle.get('variant4_sustained_window', {})
    v4_extra = (f'<p class="small">v4 calibrated on labeled-val: '
                f'τ={fmt_float(v4_cfg.get("threshold"), 2)}, '
                f'W={v4_cfg.get("W")}, K_w={v4_cfg.get("K_w")}; '
                f'GDN paper baseline: F1=0.81 / precision=0.99 / recall=0.68 (Deng &amp; Hooi 2021).</p>')
    html.append(section(f'Performance scoreboard — final-test ({fmt_int(n_windows)} windows)',
                        scoreboard + v4_extra, anchor='scoreboard'))

    # ---- Triage distribution ----
    if triage:
        total_t = sum(triage.values()) or 1
        triage_rows = ''.join(
            f'<tr><th>{k}</th><td class="num">{fmt_int(v)}</td>'
            f'<td class="num">{fmt_pct(v/total_t, 2)}</td></tr>'
            for k, v in sorted(triage.items(), key=lambda kv: -kv[1])
        )
        triage_table = (
            f'<table><thead><tr><th>Triage label</th>'
            f'<th class="num">Count</th><th class="num">% of windows</th></tr></thead>'
            f'<tbody>{triage_rows}</tbody></table>'
            f'<p class="small">Canonical alarm = v3 (max-of-V paper sweep). Aleatoric / '
            f'epistemic / OOD-dominant labels indicate which uncertainty regime triggered '
            f'(if any). False-negative-candidate = attack rows with no v3 alarm; '
            f'false-positive-candidate = v3 alarms on nominal rows.</p>'
        )
        html.append(section('Triage distribution', triage_table, anchor='triage'))

    # ---- Calibration scalars ----
    lam_v = bundle.get('lambda_v_summary', {})
    sigma_health = bundle.get('sigma_health', {})
    cal_rows = [
        ('λ_v median (per-sensor mult. correction)',
         f'{fmt_float(lam_v.get("median"))} '
         f'(min {fmt_float(lam_v.get("min"))}, max {fmt_float(lam_v.get("max"))}, '
         f'{lam_v.get("n_clamped_to_1", 0)}/51 clamped to 1.0)'),
        ('σ_floor median',
         f'{fmt_float(sigma_health.get("sigma_floor_median"))} '
         f'(min {fmt_float(sigma_health.get("sigma_floor_min"))}, '
         f'max {fmt_float(sigma_health.get("sigma_floor_max"))})'),
        ('σ-health (training)',
         f'sat_low_max = {fmt_float(sigma_health.get("sat_low_max"), 4)}, '
         f'sat_high_max = {fmt_float(sigma_health.get("sat_high_max"), 4)}, '
         f'median(log_var) range '
         f'[{fmt_float(sigma_health.get("median_logvar_per_v_min"), 2)}, '
         f'{fmt_float(sigma_health.get("median_logvar_per_v_max"), 2)}]'),
        ('Ω_thresh (99th-pct max-Ω on 𝒞)',
         fmt_float(bundle.get('omega_thresh'))),
        ('θ_e (Variant A, capped at 1)',
         fmt_float(bundle.get('theta_e_A'))),
        ('θ_e (Variant B, capped at 1)',
         fmt_float(bundle.get('theta_e_B'))),
        ('sensitivity_threshold',
         fmt_float(bundle.get('sensitivity_threshold'))),
        ('SMA window',
         str(bundle.get('sma_window', '—'))),
    ]
    html.append(section('Calibration scalars (from bundle.json)',
                        kv_table(cal_rows), anchor='calibration'))

    # ---- Reliability diagram ----
    rel_inner = ''
    rel_csv_path = bundle_dir / 'reliability.csv'
    if rel_csv_path.is_file():
        with rel_csv_path.open() as f:
            reader = csv.DictReader(f)
            rel_rows = list(reader)
        rel_table = '<table><thead><tr><th>Nominal</th><th class="num">Empirical</th>' \
                    '<th class="num">Gap</th></tr></thead><tbody>'
        for r in rel_rows:
            rel_table += (f'<tr><th>{r["nominal"]}</th>'
                          f'<td class="num">{fmt_float(float(r["empirical"]), 4)}</td>'
                          f'<td class="num">{fmt_float(float(r["gap"]), 4)}</td></tr>')
        rel_table += '</tbody></table>'
        rel_inner += '<div class="twocol">'
        rel_inner += '<div>' + rel_table + '</div>'
        if rel_b64:
            rel_inner += f'<div><img src="{rel_b64}" alt="reliability diagram"></div>'
        rel_inner += '</div>'
        rel_inner += ('<p class="small">Coverage at nominal 0.95 is anchored by per-sensor '
                      'λ_v calibration; under-coverage at lower nominal levels reflects '
                      'heavier-than-Gaussian tails in the standardised residual distribution '
                      'on 𝒞.</p>')
    html.append(section('Reliability diagram (after σ_floor + λ_v on 𝒞)',
                        rel_inner or '<p>(reliability.csv missing)</p>',
                        anchor='reliability'))

    # ---- Attack alignment table (top + bottom) ----
    attacks_inner = ''
    if attack_top:
        cols = list(attack_top[0].keys())
        # We only show the most useful columns inline.
        keep_cols = ['rank', 'idx', 'rows_start', 'rows_end',
                     'n_attack_in_window', 'best_agg', 'best_composite']
        keep_cols = [c for c in keep_cols if c in cols]

        def render_table(rows: list[dict], heading: str) -> str:
            head = '<tr>' + ''.join(f'<th>{c}</th>' for c in keep_cols) + '</tr>'
            body_rows = []
            for r in rows:
                cells = []
                for c in keep_cols:
                    val = r.get(c, '')
                    if c == 'best_composite':
                        try:
                            v = float(val)
                            pct = v * 100
                            color = ('#28a745' if pct >= 70
                                     else '#ffc107' if pct >= 50 else '#dc3545')
                            val = (f'<span class="bar"><div style="width:{pct:.1f}%;'
                                   f'background:{color}"></div></span>{pct:.1f}%')
                        except (ValueError, TypeError):
                            pass
                    cells.append(f'<td>{val}</td>')
                body_rows.append('<tr>' + ''.join(cells) + '</tr>')
            return f'<h3>{heading}</h3><table><thead>{head}</thead>' \
                   f'<tbody>{"".join(body_rows)}</tbody></table>'

        attacks_inner += render_table(attack_top, 'Best-agreeing attacks (top 5)')
        attacks_inner += render_table(attack_bottom, 'Worst-agreeing attacks (bottom 5)')
        if per_attack_dir:
            link_target = (per_attack_dir / 'index.html').resolve()
            attacks_inner += (f'<p>Full sortable index (all attacks, all 4 aggregators): '
                              f'<a href="file://{link_target}">{link_target}</a></p>')
        attacks_inner += ('<p class="small">Aggregators compared: '
                          '<code>U_max</code> (max σ_tot), '
                          '<code>U_max_st</code> (max studentised by σ_floor), '
                          '<code>U_top5_st</code> (top-5 mean of studentised σ), '
                          '<code>U_mahal</code> (multivariate Mahalanobis on residuals). '
                          'best_composite = ½·AUC + ½·spike_alignment of the winning aggregator.'
                          '</p>')
    else:
        attacks_inner = '<p>(per-attack alignment_scores.csv not found)</p>'
    html.append(section('Attack–uncertainty alignment scoring', attacks_inner, anchor='attacks'))

    # ---- Plot indices ----
    plot_html = '<ul>'
    if per_node_dir:
        link = (per_node_dir / f'node_00_{manifest["members"][0]["index"]:02d}').resolve()
        idx = per_node_dir.resolve()
        plot_html += (f'<li><b>Per-node plots ({len(list(per_node_dir.glob("node_*.html")))} files)</b> — '
                      f'<a href="file://{idx}">{idx}</a><br>'
                      f'<span class="small">One HTML per sensor over the full test timeline. '
                      f'Subplots: ground-truth, v4+v3 alarms, A_s(t), σ decomposition.</span></li>')
    if per_attack_dir:
        idx = (per_attack_dir / 'index.html').resolve()
        plot_html += (f'<li><b>Per-attack plots ({len(list(per_attack_dir.glob("attack_*.html")))} files)</b> — '
                      f'<a href="file://{idx}">{idx}</a><br>'
                      f'<span class="small">One HTML per attack window (nearby attacks grouped). '
                      f'Subplots: ground-truth+alarms, A_s(t), 4 system-uncertainty aggregators '
                      f'(toggleable), 51 sensor residuals, 51 sensor σ_tot.</span></li>')
    plot_html += '</ul>'
    html.append(section('Plot indices', plot_html, anchor='plots'))

    # ---- Comparison with previous bundle ----
    if prev_bundle is not None:
        prev_v3 = (prev_bundle.get('variant3_maxv_paper_sweep') or {}).get('metrics_labeled_val', {})
        prev_v4 = (prev_bundle.get('variant4_sustained_window') or {}).get('metrics_labeled_val', {})
        prev_v3_thr = (prev_bundle.get('variant3_maxv_paper_sweep') or {}).get('threshold')
        cur_v3 = (bundle.get('variant3_maxv_paper_sweep') or {}).get('metrics_labeled_val', {})
        cur_v4 = (bundle.get('variant4_sustained_window') or {}).get('metrics_labeled_val', {})
        cur_v3_thr = (bundle.get('variant3_maxv_paper_sweep') or {}).get('threshold')
        prev_lam = (prev_bundle.get('lambda_v_summary') or {}).get('median')
        cur_lam = (bundle.get('lambda_v_summary') or {}).get('median')
        prev_M = ((prev_bundle.get('hyperparameters') or {}).get('M', '?'))
        comp_rows = [
            ('Previous bundle', f'<code>{prev_label}</code>'),
            ('Previous M (manifest hint)', str(prev_M)),
            ('λ_v median', f'{fmt_float(prev_lam)} → {fmt_float(cur_lam)}'),
            ('v3 τ (labeled-val sweep)',
             f'{fmt_float(prev_v3_thr)} → {fmt_float(cur_v3_thr)}'),
            ('v3 F1 (labeled-val)',
             f'{fmt_float(prev_v3.get("f1"))} → {fmt_float(cur_v3.get("f1"))}'),
            ('v3 precision (labeled-val)',
             f'{fmt_float(prev_v3.get("precision"))} → {fmt_float(cur_v3.get("precision"))}'),
            ('v4 F1 (labeled-val)',
             f'{fmt_float(prev_v4.get("f1"))} → {fmt_float(cur_v4.get("f1"))}'),
            ('v4 precision (labeled-val)',
             f'{fmt_float(prev_v4.get("precision"))} → {fmt_float(cur_v4.get("precision"))}'),
        ]
        html.append(section('Comparison with previous bundle',
                            kv_table(comp_rows), anchor='comparison'))
    else:
        html.append(section('Comparison with previous bundle',
                            '<p>(no previous calibration_bundle.* backup found)</p>',
                            anchor='comparison'))

    # ---- Footer ----
    html.append('<div class="footer">')
    html.append(f'Generated by <code>scripts/build_dashboard.py</code> '
                f'on {datetime.now().isoformat(timespec="seconds")}')
    html.append('</div>')

    out_path = out_dir / 'dashboard.html'
    out_path.write_text('\n'.join(html), encoding='utf-8')
    print(f'[dashboard] wrote {out_path}')

    # Also copy/symlink the run log into the same dir if not already there.
    print(f'[dashboard] read manifest M={M}, '
          f'detect={datestr}, '
          f'per_node={per_node_dir}, per_attack={per_attack_dir}, '
          f'prev_bundle={prev_label}')


if __name__ == '__main__':
    main()
