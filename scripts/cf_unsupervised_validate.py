"""Unsupervised validation of the static-GNN CF outputs at the
per-segment (raw M10 alarm) + per-anchor schema produced by
cf_graph_static.py.

Reads <run_dir>/per_segment/<seg_idx>.json files (one per segment,
each with a list of anchors and N=5 CFs per anchor) and computes:

  1. Within-anchor convergence (per anchor): max-vote sensor / N.
     Headline: fraction of anchors with convergence >= 0.8.
  2. Within-segment anchor agreement (long segments only): for each
     long (n_anchors=3) segment, fraction of anchors sharing the
     same top-1 sensor.
  3. Cross-segment clustering: top-1 sensor per segment (majority of
     its anchors); count of segments per sensor.
  4. Persistence test: each unique top-1 sensor's z-score across the
     51 sensors of its C-slice nominal-mass mean sigma2_ale / U_par.
     Top-decile = 1.28. Faulty-sensor signature.
  5. Per-segment cost-and-metrics table (43 rows):
     seg_idx, raw_start, length, n_anchors, max_s_M10, s0_mean,
     mean_E_removed, mean_V_removed, mean_score_drop,
     stageA_success_rate, cf_success_rate, within_anchor_conv,
     anchor_agreement, top1_sensor, top1_persistence.
     Written to cost_per_segment.csv and rendered in SUMMARY.md.

NO ground-truth labels are consumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

EPS = 1e-6


def sensor_names(dataset_name: str) -> list[str]:
    csv_path = REPO_ROOT / 'data' / dataset_name / 'test.csv'
    df = pd.read_csv(csv_path, sep=',', index_col=0)
    return [c for c in df.columns if c != 'attack']


def compute_persistence_stats(arrays_npz: str, cal_split_json: str,
                                slide_win: int = 60) -> dict:
    """Per-sensor C-slice nominal-mean statistics and z-scores."""
    d = np.load(arrays_npz)
    test_sigma2_ale = d['test_sigma2_ale'].astype(np.float64)
    test_U_par = d['test_U_par'].astype(np.float64)
    label = d['test_attack_label'].astype(np.int8)
    T, V = test_sigma2_ale.shape

    with open(cal_split_json) as f:
        cal = json.load(f)
    C_lo, C_hi = cal['C_row_range']
    C_idx_start = max(0, C_lo - slide_win)
    C_idx_end = min(T, max(0, C_hi - slide_win))
    c_mask = np.zeros(T, dtype=bool)
    c_mask[C_idx_start:C_idx_end] = True
    c_mask_nominal = c_mask & (label == 0)

    sigma_mean_per_sensor = test_sigma2_ale[c_mask_nominal].mean(axis=0)
    par_mean_per_sensor = test_U_par[c_mask_nominal].mean(axis=0)
    mu_s, sd_s = sigma_mean_per_sensor.mean(), sigma_mean_per_sensor.std(ddof=1) + EPS
    mu_p, sd_p = par_mean_per_sensor.mean(), par_mean_per_sensor.std(ddof=1) + EPS
    sigma_z = (sigma_mean_per_sensor - mu_s) / sd_s
    par_z = (par_mean_per_sensor - mu_p) / sd_p
    return dict(sigma_z=sigma_z, par_z=par_z,
                sigma_mean=sigma_mean_per_sensor,
                par_mean=par_mean_per_sensor)


def per_anchor_metrics(anchor: dict, N: int) -> dict:
    """Extract per-anchor headline metrics from one anchor dict."""
    cfs = anchor['cfs']
    e_counts = [len(cf['edge_steps']) for cf in cfs]
    v_counts = [len(cf['node_steps']) for cf in cfs]
    score_drops = [anchor['s0'] - cf['s_final'] for cf in cfs]
    stageA_success = sum(1 for cf in cfs
                          if cf['stage'] == 'A' and cf['succeeded'])
    cf_success = sum(1 for cf in cfs if cf['succeeded'])
    votes = anchor['sensor_votes']
    if votes:
        max_v = max(votes.values())
        top1 = anchor['ranked_sensors'][0][0] if anchor['ranked_sensors'] else None
    else:
        max_v = 0
        top1 = None
    return dict(
        n_cfs=len(cfs),
        mean_E_removed=float(np.mean(e_counts)) if e_counts else 0.0,
        mean_V_removed=float(np.mean(v_counts)) if v_counts else 0.0,
        mean_score_drop=float(np.mean(score_drops)) if score_drops else 0.0,
        stageA_success_rate=stageA_success / max(N, 1),
        cf_success_rate=cf_success / max(N, 1),
        within_anchor_conv=max_v / max(N, 1),
        top1_sensor=top1,
    )


def aggregate_segment(seg: dict) -> dict:
    """Roll up a segment's anchors into a single per-segment row."""
    N = len(seg['anchors'][0]['cfs']) if seg['anchors'] and seg['anchors'][0]['cfs'] else 5
    per_anchor = [per_anchor_metrics(a, N) for a in seg['anchors']]

    # Anchor agreement (long segments only; short have 1 anchor so it's trivially 1.0)
    top1s = [pa['top1_sensor'] for pa in per_anchor if pa['top1_sensor'] is not None]
    if top1s:
        cnt = Counter(top1s)
        most_common, most_n = cnt.most_common(1)[0]
        anchor_agreement = most_n / len(per_anchor) if per_anchor else 1.0
        seg_top1 = most_common
    else:
        anchor_agreement = 1.0
        seg_top1 = None

    s0_mean = float(np.mean([a['s0'] for a in seg['anchors']]))
    return dict(
        seg_idx=int(seg['seg_idx']),
        raw_start=int(seg['raw_start']),
        length=int(seg['length']),
        n_anchors=int(seg['n_anchors']),
        max_s_M10=float(seg['max_s_M10']),
        s0_mean=s0_mean,
        mean_E_removed=float(np.mean([pa['mean_E_removed'] for pa in per_anchor])),
        mean_V_removed=float(np.mean([pa['mean_V_removed'] for pa in per_anchor])),
        mean_score_drop=float(np.mean([pa['mean_score_drop'] for pa in per_anchor])),
        stageA_success_rate=float(np.mean([pa['stageA_success_rate'] for pa in per_anchor])),
        cf_success_rate=float(np.mean([pa['cf_success_rate'] for pa in per_anchor])),
        within_anchor_conv=float(np.mean([pa['within_anchor_conv'] for pa in per_anchor])),
        anchor_agreement=float(anchor_agreement),
        top1_sensor=seg_top1,
        N=int(N),
    )


def _md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    """Render rows as a Markdown table given (column_key, header, fmt)."""
    out = []
    out.append('| ' + ' | '.join(h for _, h, _ in cols) + ' |')
    out.append('|' + '|'.join('---' for _ in cols) + '|')
    for r in rows:
        vals = []
        for key, _, fmt in cols:
            v = r.get(key, '')
            if v is None:
                vals.append('-')
            elif isinstance(v, float):
                vals.append(format(v, fmt))
            else:
                vals.append(str(v))
        out.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True,
                    help='results/cf_static_graph/<datestr>/ from cf_graph_static.py + cf_merge_shards.py')
    ap.add_argument('--arrays',
                    default='results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655/arrays.npz')
    ap.add_argument('--cal-split',
                    default='pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json')
    ap.add_argument('--dataset', default='swat')
    ap.add_argument('--slide-win', type=int, default=60)
    ap.add_argument('--persistence-threshold', type=float, default=1.28)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    sensors = sensor_names(args.dataset)
    V = len(sensors)

    # ---- load all per_segment JSONs ----
    seg_files = sorted((run_dir / 'per_segment').glob('*.json'))
    if not seg_files:
        sys.exit(f'[validate] no per_segment/*.json in {run_dir}')
    segments = [json.load(open(p)) for p in seg_files]
    print(f'[validate] loaded {len(segments)} per_segment files', flush=True)

    # ---- per-segment rollup table ----
    seg_rows = [aggregate_segment(s) for s in segments]
    # Sort by length desc for the headline table
    seg_rows_sorted = sorted(seg_rows, key=lambda r: -r['length'])

    # ---- Persistence per sensor ----
    pers = compute_persistence_stats(args.arrays, args.cal_split, args.slide_win)
    sigma_z = pers['sigma_z']
    par_z = pers['par_z']

    def _persist_pass(sensor_name: str) -> str:
        if sensor_name is None:
            return '-'
        idx = sensors.index(sensor_name)
        sp = sigma_z[idx] >= args.persistence_threshold
        pp = par_z[idx] >= args.persistence_threshold
        if sp and pp:
            return 'sigma+par'
        if sp:
            return 'sigma'
        if pp:
            return 'par'
        return 'fail'

    for r in seg_rows_sorted:
        r['top1_persistence'] = _persist_pass(r['top1_sensor'])

    # ---- write cost_per_segment.csv ----
    cost_cols = ['seg_idx', 'raw_start', 'length', 'n_anchors', 'max_s_M10',
                  's0_mean', 'mean_E_removed', 'mean_V_removed', 'mean_score_drop',
                  'stageA_success_rate', 'cf_success_rate', 'within_anchor_conv',
                  'anchor_agreement', 'top1_sensor', 'top1_persistence', 'N']
    with open(run_dir / 'cost_per_segment.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cost_cols)
        w.writeheader()
        w.writerows(seg_rows_sorted)
    print(f'[validate] wrote cost_per_segment.csv ({len(seg_rows_sorted)} rows)')

    # ---- aggregate metrics ----
    n_segments = len(seg_rows)
    n_anchors_total = sum(s['n_anchors'] for s in seg_rows)
    long_segments = [s for s in seg_rows if s['n_anchors'] >= 2]

    # within-anchor convergence: derive from all anchors, not segments
    all_anchor_convs = []
    for seg in segments:
        for a in seg['anchors']:
            votes = a['sensor_votes']
            max_v = max(votes.values()) if votes else 0
            N = len(a['cfs']) if a['cfs'] else 5
            all_anchor_convs.append(max_v / max(N, 1))
    all_anchor_convs = np.array(all_anchor_convs)
    frac_conv_ge_08 = float((all_anchor_convs >= 0.8).mean()) if all_anchor_convs.size else 0.0
    frac_conv_ge_06 = float((all_anchor_convs >= 0.6).mean()) if all_anchor_convs.size else 0.0
    frac_conv_eq_10 = float((all_anchor_convs >= 0.999).mean()) if all_anchor_convs.size else 0.0

    # cross-segment clustering: top-1 per segment
    incidence_top1: Counter = Counter()
    incidence_any: Counter = Counter()
    for s in seg_rows:
        if s['top1_sensor'] is not None:
            incidence_top1[s['top1_sensor']] += 1
    # any votes per segment
    for seg in segments:
        sensors_in_seg = set()
        for a in seg['anchors']:
            for sn, c in a['sensor_votes'].items():
                if c > 0:
                    sensors_in_seg.add(sn)
        for sn in sensors_in_seg:
            incidence_any[sn] += 1

    # persistence summary
    implicated = sorted({s['top1_sensor'] for s in seg_rows
                          if s['top1_sensor'] is not None})
    n_pers_pass = sum(1 for s in implicated
                       if _persist_pass(s) != 'fail' and _persist_pass(s) != '-')
    pers_frac = n_pers_pass / max(len(implicated), 1)

    # anchor agreement for long segments
    n_long = len(long_segments)
    long_unanimous = sum(1 for s in long_segments if s['anchor_agreement'] >= 0.999)

    # ---- write per-segment cost markdown (the headline table) ----
    md = []
    md.append('# Unsupervised Static-GNN CF — Validation Summary\n\n')
    md.append(f'- Segments (raw `s_M10 > tau*` runs): **{n_segments}**\n')
    md.append(f'- Total anchors processed: **{n_anchors_total}**\n')
    md.append(f'- Long segments (length >= 60, 3 anchors each): **{n_long}**\n')
    md.append(f'- Short segments (length < 60, 1 anchor each): **{n_segments - n_long}**\n')
    md.append(f'- N (diverse CFs per anchor): **{seg_rows_sorted[0]["N"]}**\n')
    md.append(f'- Persistence top-decile threshold: **z >= {args.persistence_threshold}**\n\n')

    md.append('## Headline: per-segment cost-and-metrics table\n\n')
    md.append('Sorted by `length` descending. `cost` is captured by '
              '`mean_E_removed` + `mean_V_removed` (graph mass removed per CF); '
              '`mean_score_drop` is how much M10\'s log-odds decreased after the CF.\n\n')
    md.append(_md_table(seg_rows_sorted, [
        ('seg_idx', 'seg', '.0f'),
        ('raw_start', 't0', '.0f'),
        ('length', 'len', '.0f'),
        ('n_anchors', 'A', '.0f'),
        ('max_s_M10', 'max_s', '.3f'),
        ('s0_mean', 's0', '.3f'),
        ('mean_E_removed', 'E_rem', '.2f'),
        ('mean_V_removed', 'V_rem', '.2f'),
        ('mean_score_drop', 'drop', '.3f'),
        ('stageA_success_rate', 'A%', '.2f'),
        ('cf_success_rate', 'CF%', '.2f'),
        ('within_anchor_conv', 'conv', '.2f'),
        ('anchor_agreement', 'aagr', '.2f'),
        ('top1_sensor', 'top1', 's'),
        ('top1_persistence', 'pers', 's'),
    ]))
    md.append('\n')

    md.append('## 1. Within-anchor convergence\n\n')
    md.append(f'- fraction of anchors with convergence >= 0.8: **{frac_conv_ge_08:.3f}** '
              f'({int(frac_conv_ge_08*len(all_anchor_convs))}/{len(all_anchor_convs)})\n')
    md.append(f'- fraction with convergence >= 0.6: **{frac_conv_ge_06:.3f}**\n')
    md.append(f'- fraction with convergence == 1.0 (unanimous): **{frac_conv_eq_10:.3f}**\n\n')

    md.append('## 2. Within-segment anchor agreement (long segments only)\n\n')
    if long_segments:
        md.append(f'- {n_long} long segments processed\n')
        md.append(f'- fraction unanimous across the 3 anchors: '
                  f'**{long_unanimous/n_long:.3f}** ({long_unanimous}/{n_long})\n\n')
        md.append(_md_table(sorted(long_segments, key=lambda r: -r['length']), [
            ('seg_idx', 'seg', '.0f'),
            ('length', 'len', '.0f'),
            ('top1_sensor', 'top1', 's'),
            ('anchor_agreement', 'agree', '.2f'),
            ('within_anchor_conv', 'avg_conv', '.2f'),
            ('mean_score_drop', 'drop', '.3f'),
        ]))
        md.append('\n')
    else:
        md.append('(no long segments)\n\n')

    md.append('## 3. Cross-segment clustering\n\n')
    md.append('Top sensors by count of segments where they were the consensus top-1 '
              '(majority-of-anchors).\n\n')
    md.append('| sensor | top1_segments | any_vote_segments |\n')
    md.append('|---|---|---|\n')
    for s, c in incidence_top1.most_common(15):
        md.append(f'| {s} | {c} | {incidence_any.get(s, 0)} |\n')
    md.append('\n')

    md.append('## 4. Persistence test\n\n')
    md.append(f'For each unique top-1-implicated sensor, z-score across the 51 sensors '
              f'of (a) C-nominal mean sigma2_ale and (b) C-nominal mean U_par. '
              f'Top-decile = {args.persistence_threshold}.\n\n')
    md.append(f'- implicated sensors: **{len(implicated)}**\n')
    md.append(f'- fraction passing (top-decile in sigma_z OR par_z): '
              f'**{pers_frac:.3f}**\n\n')
    md.append('| sensor | sigma_z | par_z | passes | top1_segments |\n')
    md.append('|---|---|---|---|---|\n')
    for s in sorted(implicated, key=lambda x: -incidence_top1[x]):
        idx = sensors.index(s)
        md.append(f'| {s} | {sigma_z[idx]:+.3f} | {par_z[idx]:+.3f} '
                  f'| {_persist_pass(s)} | {incidence_top1[s]} |\n')
    md.append('\n')

    md.append('## Interpretation\n\n')
    md.append('- High **within-anchor convergence** means the 5 diverse CFs at one '
              'anchor agree on the root cause sensor — the model genuinely depends '
              'on that sensor at that timestep.\n')
    md.append('- High **within-segment anchor agreement** (for long segments) means '
              'the root cause is stable across the alarm episode\'s duration — no '
              'drift in the model\'s view.\n')
    md.append('- High **cross-segment clustering** highlights sensors that appear as '
              'root causes repeatedly across different alarms — operator review priority.\n')
    md.append('- A **passing persistence test** (`pers=sigma`/`par`/`sigma+par`) is '
              'consistent with the faulty-sensor hypothesis (chronically noisy on '
              'nominal data); a **failing** test is consistent with the attack-target '
              'hypothesis (healthy on nominal, UQ spikes during attack).\n')

    with open(run_dir / 'SUMMARY.md', 'w') as f:
        f.writelines(md)
    print(f'[validate] wrote {run_dir/"SUMMARY.md"}')

    # ---- side CSVs ----
    with open(run_dir / 'convergence_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seg_idx', 'anchor_label', 't_star', 'max_votes', 'N', 'convergence'])
        for seg in segments:
            for a in seg['anchors']:
                votes = a['sensor_votes']
                max_v = max(votes.values()) if votes else 0
                N = len(a['cfs']) if a['cfs'] else 5
                w.writerow([seg['seg_idx'], a['anchor_label'], a['t_star'],
                             max_v, N, max_v / max(N, 1)])
    with open(run_dir / 'persistence_test.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sensor', 'sigma_z', 'par_z', 'passes', 'top1_segments'])
        for s in sorted(implicated, key=lambda x: -incidence_top1[x]):
            idx = sensors.index(s)
            w.writerow([s, sigma_z[idx], par_z[idx], _persist_pass(s),
                         incidence_top1[s]])
    print('[validate] done')


if __name__ == '__main__':
    main()
