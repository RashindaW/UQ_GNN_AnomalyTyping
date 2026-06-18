"""Post-hoc GT cross-check: do the CF root-cause sensors match the
documented SWaT attack targets?

Consumes a CF run-dir (per_segment/*.json) + data/swat/attack_targets.json.
For each alarm segment, finds the overlapping documented attack(s),
compares the CF's segment-level top-1 / top-3 implicated sensor(s) to
the attack target(s), and reports hit@1 / hit@3 / MRR stratified by
Actual-Change, point-count, and in-model. Also builds the
persistence x attack cross-tab (the attack-vs-fault payoff).

NO GT is used in CF generation; this is strictly downstream validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from cf_unsupervised_validate import compute_persistence_stats, sensor_names

PERS_THRESHOLD = 1.28


def build_undirected_adjacency(edge_index: np.ndarray, V: int) -> list[set]:
    """Undirected neighbor sets from the learned graph's edge_index (2,E)."""
    adj = [set() for _ in range(V)]
    for u, v in zip(edge_index[0], edge_index[1]):
        u, v = int(u), int(v)
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def bfs_dist(adj: list[set], src: int, dst: int) -> float:
    """Undirected shortest-path hop count; inf if unreachable."""
    if src == dst:
        return 0
    seen = {src}
    frontier = [src]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if w == dst:
                    return d
                if w not in seen:
                    seen.add(w)
                    nxt.append(w)
        frontier = nxt
    return float('inf')


def overlap(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0, hi - lo)


def segment_ranking(seg: dict) -> list[tuple[str, int]]:
    """Aggregate per-anchor sensor_votes across the segment into one
    descending (sensor, total_votes) ranking."""
    tally: Counter = Counter()
    for a in seg['anchors']:
        for s, c in a['sensor_votes'].items():
            tally[s] += int(c)
    return [(s, c) for s, c in tally.most_common() if c > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attack-json',
                    default=str(REPO_ROOT / 'data/swat/attack_targets.json'))
    ap.add_argument('--arrays',
                    default='results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655/arrays.npz')
    ap.add_argument('--cal-split',
                    default='pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json')
    ap.add_argument('--dataset', default='swat')
    ap.add_argument('--bundle',
                    default='pretrained/swat_gdeltauq_sw60/calibration_bundle_K100')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    sensors = sensor_names(args.dataset)
    s2i = {s: i for i, s in enumerate(sensors)}
    attacks = json.load(open(args.attack_json))['attacks']

    # learned-graph adjacency for hop-distance proximity
    eis = np.load(Path(args.bundle) / 'edge_index_sample.npz')['edge_index_sample']
    adj = build_undirected_adjacency(eis, len(sensors))

    def min_hops(top1_name, target_names):
        if not top1_name or not target_names:
            return float('inf')
        su = s2i[top1_name]
        return min(bfs_dist(adj, su, s2i[t]) for t in target_names if t in s2i)

    def null_min_hops(target_names):
        """Expected min-hop to targets from a sensor picked uniformly at
        random — the chance baseline for the proximity claim (the learned
        graph is dense, so most sensors are 1-2 hops from anything)."""
        if not target_names:
            return float('inf')
        vals = [min_hops(sn, target_names) for sn in sensors]
        vals = [v for v in vals if v != float('inf')]
        return float(np.mean(vals)) if vals else float('inf')

    # persistence verdict per sensor
    pers = compute_persistence_stats(args.arrays, args.cal_split)
    sigma_z, par_z = pers['sigma_z'], pers['par_z']
    def persist_pass(s):
        i = sensors.index(s)
        return (sigma_z[i] >= PERS_THRESHOLD) or (par_z[i] >= PERS_THRESHOLD)

    seg_files = sorted((run_dir / 'per_segment').glob('*.json'))
    segments = [json.load(open(p)) for p in seg_files]
    print(f'[gt] loaded {len(segments)} CF segments, {len(attacks)} attacks',
          flush=True)

    rows = []
    for seg in segments:
        a0, a1 = seg['raw_start'], seg['raw_end_excl']
        ranking = segment_ranking(seg)
        ranked_sensors = [s for s, _ in ranking]
        top1 = ranked_sensors[0] if ranked_sensors else None
        top3 = ranked_sensors[:3]

        # overlapping attacks
        matched = []
        for atk in attacks:
            ov = overlap(a0, a1, atk['start_idx'], atk['end_idx'])
            if ov > 0:
                matched.append((atk, ov))
        matched.sort(key=lambda x: -x[1])

        if matched:
            atk_ids = [m[0]['attack_id'] for m in matched]
            all_targets, in_model_targets, impact_in_model = set(), set(), set()
            actual_changes, n_pts, cats = [], [], []
            for m, _ in matched:
                for t, im in zip(m['targets'], m['in_model']):
                    all_targets.add(t)
                    if im:
                        in_model_targets.add(t)
                for s in m.get('impact_sensors', []):
                    if s in s2i:
                        impact_in_model.add(s)
                actual_changes.append(m['actual_change'])
                n_pts.append(m['n_points'])
                cats.append(m['category'])
            relaxed = in_model_targets | impact_in_model  # targets + downstream impact
            best = matched[0][0]
            ov_frac = matched[0][1] / max(1, (a1 - a0))
            hit1 = top1 in in_model_targets if top1 else False
            hit3 = bool(set(top3) & in_model_targets)
            hit1_relax = top1 in relaxed if top1 else False
            hit3_relax = bool(set(top3) & relaxed)
            mrr = 0.0
            for rank, s in enumerate(ranked_sensors, 1):
                if s in in_model_targets:
                    mrr = 1.0 / rank
                    break
            hops = min_hops(top1, in_model_targets)
            hops_relax = min_hops(top1, relaxed)
            null_hops = null_min_hops(in_model_targets)
            rows.append(dict(
                seg_idx=seg['seg_idx'], raw_start=a0, length=a1 - a0,
                cf_top1=top1, cf_top3=';'.join(top3),
                matched_attack_ids=';'.join(str(x) for x in atk_ids),
                attack_targets=';'.join(sorted(all_targets)),
                in_model_targets=';'.join(sorted(in_model_targets)),
                impact_sensors=';'.join(sorted(impact_in_model)),
                target_in_model=bool(in_model_targets),
                actual_change=best['actual_change'],
                category=best['category'], n_points=best['n_points'],
                overlap_frac=round(ov_frac, 3),
                hit1=int(hit1), hit3=int(hit3), mrr=round(mrr, 3),
                hit1_relax=int(hit1_relax), hit3_relax=int(hit3_relax),
                hops_to_target=(None if hops == float('inf') else int(hops)),
                hops_to_relaxed=(None if hops_relax == float('inf') else int(hops_relax)),
                null_hops_to_target=(None if null_hops == float('inf') else round(null_hops, 3)),
                matched=1,
                top1_persist=('pass' if (top1 and persist_pass(top1)) else 'fail'),
            ))
        else:
            rows.append(dict(
                seg_idx=seg['seg_idx'], raw_start=a0, length=a1 - a0,
                cf_top1=top1, cf_top3=';'.join(top3),
                matched_attack_ids='', attack_targets='', in_model_targets='',
                impact_sensors='',
                target_in_model=False, actual_change=None, category='',
                n_points=0, overlap_frac=0.0, hit1=0, hit3=0, mrr=0.0,
                hit1_relax=0, hit3_relax=0, hops_to_target=None,
                hops_to_relaxed=None, null_hops_to_target=None,
                matched=0,
                top1_persist=('pass' if (top1 and persist_pass(top1)) else 'fail'),
            ))

    # ---- write per-segment CSV ----
    cols = ['seg_idx', 'raw_start', 'length', 'cf_top1', 'cf_top3',
            'matched', 'matched_attack_ids', 'attack_targets',
            'in_model_targets', 'impact_sensors', 'target_in_model',
            'actual_change', 'category', 'n_points', 'overlap_frac',
            'hit1', 'hit3', 'mrr', 'hit1_relax', 'hit3_relax',
            'hops_to_target', 'hops_to_relaxed', 'null_hops_to_target',
            'top1_persist']
    with open(run_dir / 'gt_crosscheck.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'[gt] wrote {run_dir/"gt_crosscheck.csv"} ({len(rows)} rows)', flush=True)

    # ---- stratified hit rates ----
    def stratum(pred):
        sub = [r for r in rows if r['matched'] and pred(r)]
        if not sub:
            return (0, 0.0, 0.0, 0.0)
        n = len(sub)
        return (n,
                float(np.mean([r['hit1'] for r in sub])),
                float(np.mean([r['hit3'] for r in sub])),
                float(np.mean([r['mrr'] for r in sub])))

    strata = {
        'all matched':                 stratum(lambda r: True),
        'in-model target':             stratum(lambda r: r['target_in_model']),
        'Actual-Change=Yes':           stratum(lambda r: r['actual_change'] is True),
        'AC=Yes & in-model':           stratum(lambda r: r['actual_change'] is True and r['target_in_model']),
        'AC=Yes & in-model & single':  stratum(lambda r: r['actual_change'] is True and r['target_in_model'] and r['n_points'] == 1),
    }
    n_matched = sum(r['matched'] for r in rows)
    n_no = len(rows) - n_matched

    # ---- persistence x attack cross-tab ----
    # bucket each segment: (top1 persist pass/fail) x (matched&hit / matched&miss / no_overlap)
    ctab = defaultdict(int)
    for r in rows:
        pv = r['top1_persist']
        if not r['matched']:
            col = 'no_overlap'
        elif r['hit1']:
            col = 'matched&hit'
        else:
            col = 'matched&miss'
        ctab[(pv, col)] += 1

    # ---- console + markdown ----
    md = []
    md.append('\n## GT cross-check (post-hoc, vs List_of_attacks_Final)\n\n')
    md.append(f'- CF segments: **{len(rows)}**  |  overlap a documented attack: '
              f'**{n_matched}**  |  no-overlap (candidate FP / fault): **{n_no}**\n\n')
    md.append('### Hit rates by stratum (top-1 vs documented attack POINT)\n\n')
    md.append('| stratum | n | hit@1 | hit@3 | MRR |\n|---|---|---|---|---|\n')
    for name, (n, h1, h3, mrr) in strata.items():
        md.append(f'| {name} | {n} | {h1:.3f} | {h3:.3f} | {mrr:.3f} |\n')
    md.append('\n')

    # relaxed: target POINT or documented downstream IMPACT sensor
    def stratum_relax(pred):
        sub = [r for r in rows if r['matched'] and pred(r)]
        if not sub:
            return (0, 0.0, 0.0)
        return (len(sub),
                float(np.mean([r['hit1_relax'] for r in sub])),
                float(np.mean([r['hit3_relax'] for r in sub])))
    md.append('### Relaxed hit (top-1 vs target POINT or documented downstream IMPACT sensor)\n\n')
    md.append('| stratum | n | hit@1 | hit@3 |\n|---|---|---|---|\n')
    for name, pred in [
        ('all matched', lambda r: True),
        ('Actual-Change=Yes', lambda r: r['actual_change'] is True),
        ('AC=Yes & in-model', lambda r: r['actual_change'] is True and r['target_in_model']),
    ]:
        n, h1, h3 = stratum_relax(pred)
        md.append(f'| {name} | {n} | {h1:.3f} | {h3:.3f} |\n')
    md.append('\n')

    # graph hop-distance from CF top-1 to the nearest in-model true target
    hopvals = [r['hops_to_target'] for r in rows
               if r['matched'] and r['hops_to_target'] is not None]
    if hopvals:
        hist = {f'{k} hop': sum(1 for h in hopvals if h == k) for k in range(0, 4)}
        hist['>=4 / unreachable'] = (
            sum(1 for h in hopvals if h >= 4)
            + sum(1 for r in rows if r['matched']
                  and r['target_in_model'] and r['hops_to_target'] is None))
        nullvals = [r['null_hops_to_target'] for r in rows
                    if r['matched'] and r['null_hops_to_target'] is not None]
        null_mean = float(np.mean(nullvals)) if nullvals else float('nan')
        md.append('### Graph hop-distance: CF top-1 -> nearest true target '
                  '(learned graph, undirected)\n\n')
        md.append(f'- CF top-1 mean hops = **{np.mean(hopvals):.2f}** '
                  f'(median {int(np.median(hopvals))}, n={len(hopvals)})\n')
        md.append(f'- **random-sensor null** (chance baseline) mean hops = '
                  f'**{null_mean:.2f}**\n')
        md.append(f'- CF advantage over null = **{null_mean - np.mean(hopvals):+.2f}** '
                  f'hops (>0 means CF is closer than chance)\n')
        md.append(f'- NOTE: the learned graph is dense (avg degree ~21 of 50), '
                  f'so most sensors are 1-2 hops from anything — proximity here '
                  f'carries little signal unless it beats the null.\n')
        md.append('| distance | count |\n|---|---|\n')
        for k, c in hist.items():
            md.append(f'| {k} | {c} |\n')
        md.append('\n')
    md.append('### Persistence × attack cross-tab (top-1 sensor)\n\n')
    md.append('Rows = top-1 persistence verdict; cols = GT outcome. '
              'Attack-target hypothesis predicts persistence-**fail** sensors '
              'cluster in *matched&hit*; faulty-sensor hypothesis predicts '
              'persistence-**pass** sensors cluster in *no_overlap*.\n\n')
    md.append('| persist | matched&hit | matched&miss | no_overlap |\n|---|---|---|---|\n')
    for pv in ('fail', 'pass'):
        md.append(f'| {pv} | {ctab[(pv,"matched&hit")]} | '
                  f'{ctab[(pv,"matched&miss")]} | {ctab[(pv,"no_overlap")]} |\n')
    md.append('\n')
    md.append('### Per-segment detail\n\n')
    md.append('| seg | len | cf_top1 | persist | attack | targets(in-model) | AC | hit@1 |\n')
    md.append('|---|---|---|---|---|---|---|---|\n')
    for r in sorted(rows, key=lambda x: -x['length']):
        ac = {True: 'Y', False: 'N', None: '-'}[r['actual_change']]
        md.append(f"| {r['seg_idx']} | {r['length']} | {r['cf_top1']} | "
                  f"{r['top1_persist']} | {r['matched_attack_ids'] or '-'} | "
                  f"{r['in_model_targets'] or '-'} | {ac} | "
                  f"{'Y' if r['hit1'] else ''} |\n")
    md.append('\n')
    md_text = ''.join(md)

    with open(run_dir / 'gt_crosscheck.md', 'w') as f:
        f.write(md_text)
    print(f'[gt] wrote {run_dir/"gt_crosscheck.md"}', flush=True)

    # guarded append to SUMMARY.md (replace any prior GT section)
    summ = run_dir / 'SUMMARY.md'
    if summ.exists():
        base = summ.read_text()
        marker = '\n## GT cross-check (post-hoc'
        if marker in base:
            base = base[:base.index(marker)]
        summ.write_text(base.rstrip() + '\n' + md_text)
        print(f'[gt] appended GT-validity section to {summ}', flush=True)

    # console headline
    n, h1, h3, mrr = strata['AC=Yes & in-model']
    print(f'\n[gt] HEADLINE (Actual-Change=Yes & in-model, n={n}): '
          f'hit@1={h1:.3f} hit@3={h3:.3f} MRR={mrr:.3f}', flush=True)
    print(f'[gt] cross-tab: {dict(ctab)}', flush=True)


if __name__ == '__main__':
    main()
