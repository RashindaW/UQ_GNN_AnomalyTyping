"""Static-GNN CF orchestrator over all raw M10 alarm segments.

Pipeline:
  1. Load cf_engine context (frozen model + M10).
  2. Segment the raw `s_M10 > tau*` signal into contiguous runs (no
     Fix-A post-proc) — produces 43 segments on the SWaT eval set.
  3. For each segment seg of length L, pick anchors:
       L >= 60      -> 3 anchors  (start+5, argmax, end-6)
       L <  60      -> 1 anchor   (argmax)
  4. For each (seg_idx, anchor_label, t_star), generate N=5 diverse CFs
     via cf_diverse_greedy.diverse_cf.
  5. Write per-anchor outputs.

Supports sharding via --shard-idx / --num-shards so the whole pipeline
can run across multiple GPUs in parallel. Each shard processes the
anchors of segments where `seg_idx % num_shards == shard_idx`, writes
its own CSVs into `<out>/shard_<shard_idx>/` and the shared
`<out>/per_segment/<seg_idx>.json` files. `scripts/cf_merge_shards.py`
concats the shard CSVs after all shards finish.

NO ground-truth labels are consumed; CF generation and ranking are
fully unsupervised.
"""
from __future__ import annotations

import argparse
import csv
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

from cf_engine import build_cf_context, CFContext
from cf_diverse_greedy import diverse_cf, SingleCF

DEFAULT_TAU_STAR = 2.0914
LONG_THRESHOLD = 60     # segments >= this length get 3 anchors
ANCHOR_PAD = 5          # 'start' anchor = seg_start + 5; 'end' = seg_end - 6


def extract_alarm_runs(alarm: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end_exclusive) for contiguous runs of 1s."""
    runs = []
    i = 0
    n = len(alarm)
    while i < n:
        if alarm[i]:
            j = i
            while j < n and alarm[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def pick_anchors(seg_start: int, seg_end_excl: int, s_M10: np.ndarray,
                  long_threshold: int = LONG_THRESHOLD,
                  edge_pad: int = ANCHOR_PAD) -> list[tuple[str, int]]:
    """Return list of (anchor_label, t_star) for a single segment.

    Short segments (length < long_threshold): one 'mid' anchor at the
    argmax of s_M10 inside the segment.

    Long segments (length >= long_threshold): three anchors at
    (seg_start + edge_pad, argmax, seg_end_excl - edge_pad - 1).
    All anchors are clipped to [seg_start, seg_end_excl).
    """
    length = seg_end_excl - seg_start
    sub = s_M10[seg_start:seg_end_excl]
    t_mid = int(seg_start + np.argmax(sub))
    if length < long_threshold:
        return [('mid', t_mid)]
    t_start = max(seg_start, seg_start + edge_pad)
    t_end = min(seg_end_excl - 1, seg_end_excl - edge_pad - 1)
    # Guarantee distinct & ordered
    anchors_unsorted = {('start', int(t_start)), ('mid', int(t_mid)),
                         ('end', int(t_end))}
    by_label = {lbl: t for (lbl, t) in anchors_unsorted}
    return [(lbl, by_label[lbl]) for lbl in ('start', 'mid', 'end')
            if lbl in by_label]


def sensor_names(dataset_name: str) -> list[str]:
    csv_path = REPO_ROOT / 'data' / dataset_name / 'test.csv'
    df = pd.read_csv(csv_path, sep=',', index_col=0)
    return [c for c in df.columns if c != 'attack']


def _write_per_segment_json(per_segment_dir: Path, seg_idx: int,
                              raw_start: int, raw_end_excl: int,
                              length: int, max_s_M10: float,
                              anchors_payload: list[dict]) -> None:
    payload = dict(
        seg_idx=int(seg_idx),
        raw_start=int(raw_start),
        raw_end_excl=int(raw_end_excl),
        length=int(length),
        max_s_M10=float(max_s_M10),
        n_anchors=int(len(anchors_payload)),
        anchors=anchors_payload,
    )
    with open(per_segment_dir / f'{seg_idx:03d}.json', 'w') as f:
        json.dump(payload, f, indent=2)


def run_pipeline(args):
    if args.out_dir_fixed:
        out_dir = Path(args.out_dir_fixed)
    else:
        out_dir = Path(args.out_dir) / datetime.now().strftime('%m%d-%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    per_segment_dir = out_dir / 'per_segment'
    per_segment_dir.mkdir(exist_ok=True)
    shard_dir = out_dir / f'shard_{args.shard_idx}'
    shard_dir.mkdir(exist_ok=True)
    logs_dir = out_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    print(f'[cf_orch] output dir: {out_dir}', flush=True)
    print(f'[cf_orch] shard {args.shard_idx} / {args.num_shards} '
          f'writing CSVs to {shard_dir}', flush=True)

    ctx = build_cf_context(
        arrays_path=args.arrays, checkpoint_path=args.checkpoint,
        hp_path=args.hp, bundle_dir=args.bundle,
        cal_split_path=args.cal_split, dataset_name=args.dataset,
        device=args.device, m10_tau_star=args.tau_star,
    )
    sensors = sensor_names(args.dataset)

    # ---- Segment raw alarm (no post-proc) ----
    alarm = (ctx.s_M10_full > ctx.m10_tau_star).astype(np.int8)
    segments = extract_alarm_runs(alarm)
    print(f'[cf_orch] raw alarm segments: {len(segments)}', flush=True)
    if args.max_runs > 0:
        segments = segments[:args.max_runs]
        print(f'[cf_orch] --max-runs truncated to {len(segments)}', flush=True)
    if not segments:
        print('[cf_orch] no alarm segments — aborting', flush=True)
        return

    # ---- Build anchor plan ----
    anchor_plan: list[tuple[int, int, int, int, str, int]] = []
    # (seg_idx, raw_start, raw_end_excl, length, anchor_label, t_star)
    seg_anchor_count = {}
    segments_index_rows = []
    for seg_idx, (a, b) in enumerate(segments):
        length = b - a
        max_s = float(ctx.s_M10_full[a:b].max())
        anchors = pick_anchors(a, b, ctx.s_M10_full)
        seg_anchor_count[seg_idx] = len(anchors)
        if (seg_idx % args.num_shards) == args.shard_idx:
            for (lbl, t) in anchors:
                anchor_plan.append((seg_idx, a, b, length, lbl, t))
        segments_index_rows.append(dict(
            seg_idx=int(seg_idx), raw_start=int(a),
            raw_end_excl=int(b), length=int(length),
            n_anchors=int(len(anchors)),
            max_s_M10=float(max_s),
        ))
    n_owned_segments = sum(1 for r in segments_index_rows
                            if (r['seg_idx'] % args.num_shards) == args.shard_idx)
    print(f'[cf_orch] shard owns {n_owned_segments} segments / '
          f'{len(anchor_plan)} anchors', flush=True)

    # Always write segments_index.csv (shard 0 only, or every shard — fine either way
    # since the file is identical across shards; we let every shard rewrite it idempotently)
    with open(shard_dir / 'segments_index.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(segments_index_rows[0].keys()))
        w.writeheader()
        w.writerows(segments_index_rows)

    # ---- Per-anchor CF generation ----
    cf_rows = []
    sensor_vote_rows = []
    t0 = time.time()
    # Accumulator: collect per-segment anchor payloads for the shared
    # per_segment/<seg_idx>.json (only segments owned by this shard).
    seg_payloads: dict[int, list[dict]] = {}
    seg_meta: dict[int, tuple[int, int, int, float]] = {}

    for i, (seg_idx, a, b, length, label, t_star) in enumerate(anchor_plan):
        elapsed = time.time() - t0
        s0_hint = float(ctx.s_M10_full[t_star])
        print(f'[cf_orch] anchor {i+1}/{len(anchor_plan)}  seg={seg_idx} '
              f'len={length} label={label} t*={t_star}  s0~={s0_hint:.3f}  '
              f'elapsed={elapsed:.1f}s', flush=True)
        res = diverse_cf(ctx, t_star, N=args.N,
                          K_edge_max=args.K_edge_max,
                          K_node_max=args.K_node_max,
                          diversity_mode=args.diversity_mode,
                          faithful_smoothing=args.faithful_smoothing)

        # accumulate per-segment payload
        anchor_payload = dict(
            anchor_label=label,
            t_star=int(t_star),
            s0=float(res['s0']),
            tau_star=float(ctx.m10_tau_star),
            sensor_votes={sensors[v]: int(c) for v, c in res['sensor_votes'].items()},
            ranked_sensors=[(sensors[v], int(c)) for v, c in res['ranked_sensors']],
            cfs=[cf.to_dict() for cf in res['cfs']],
        )
        seg_payloads.setdefault(seg_idx, []).append(anchor_payload)
        seg_meta[seg_idx] = (int(a), int(b), int(length),
                              float(ctx.s_M10_full[a:b].max()))

        # per-CF rows
        for ci, cf in enumerate(res['cfs']):
            cf_rows.append(dict(
                seg_idx=int(seg_idx),
                anchor_label=label,
                t_star=int(t_star),
                cf_idx=int(ci),
                E_removed_count=int(len(cf.edge_steps)),
                V_removed_count=int(len(cf.node_steps)),
                stage=cf.stage,
                succeeded=int(cf.succeeded),
                s0=float(res['s0']),
                s_final=float(cf.s_final),
                edge_seq=';'.join(str(e) for e, _ in cf.edge_steps),
                node_seq=';'.join(sensors[v] for v, _ in cf.node_steps),
            ))
        # per-sensor vote rows
        for v, votes in res['sensor_votes'].items():
            if votes == 0:
                continue
            sensor_vote_rows.append(dict(
                seg_idx=int(seg_idx),
                anchor_label=label,
                t_star=int(t_star),
                sensor=sensors[v],
                votes_in_5_diverse_CFs=int(votes),
                U_par_t_star=float(ctx.test_U_par[t_star, v]),
                sigma2_ale_t_star=float(ctx.test_sigma2_ale[t_star, v]),
            ))

    # ---- write per-segment JSONs (shared dir; this shard's segments only) ----
    for seg_idx, anchors_payload in seg_payloads.items():
        a, b, length, max_s = seg_meta[seg_idx]
        _write_per_segment_json(per_segment_dir, seg_idx, a, b, length,
                                  max_s, anchors_payload)
    print(f'[cf_orch] wrote {len(seg_payloads)} per_segment/*.json files',
          flush=True)

    # ---- write shard CSVs ----
    if cf_rows:
        with open(shard_dir / 'cf_per_anchor.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(cf_rows[0].keys()))
            w.writeheader()
            w.writerows(cf_rows)
        print(f'[cf_orch] wrote {shard_dir/"cf_per_anchor.csv"} ({len(cf_rows)} rows)',
              flush=True)
    if sensor_vote_rows:
        with open(shard_dir / 'sensor_votes_per_anchor.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sensor_vote_rows[0].keys()))
            w.writeheader()
            w.writerows(sensor_vote_rows)
        print(f'[cf_orch] wrote {shard_dir/"sensor_votes_per_anchor.csv"} '
              f'({len(sensor_vote_rows)} rows)', flush=True)

    # marker file for the launcher (signals shard completion)
    (shard_dir / 'DONE').write_text(
        f'{datetime.now().isoformat()}  anchors={len(anchor_plan)}\n')
    elapsed = time.time() - t0
    print(f'[cf_orch] shard {args.shard_idx} done in {elapsed:.1f}s', flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arrays',
                    default='results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655/arrays.npz')
    ap.add_argument('--checkpoint',
                    default='pretrained/swat_gdeltauq_sw60/best_0513-211014.pt')
    ap.add_argument('--hp',
                    default='pretrained/swat_gdeltauq_sw60/hyperparameters_0513-211014.json')
    ap.add_argument('--bundle',
                    default='pretrained/swat_gdeltauq_sw60/calibration_bundle_K100')
    ap.add_argument('--cal-split',
                    default='pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json')
    ap.add_argument('--dataset', default='swat')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tau-star', type=float, default=None,
                    help='M10 log-odds threshold. If omitted, build_cf_context '
                         'auto-computes the Fix-A best tau* on this model\'s arrays '
                         '(do NOT reuse the baseline 2.0914 for a different model).')
    ap.add_argument('--N', type=int, default=5)
    ap.add_argument('--K-edge-max', type=int, default=15)
    ap.add_argument('--K-node-max', type=int, default=5)
    ap.add_argument('--out-dir', default='results/cf_static_graph',
                    help='Parent dir; a datestr subdir is created. Ignored if --out-dir-fixed is set.')
    ap.add_argument('--out-dir-fixed', default='',
                    help='If set, use this exact dir (no datestr append). Used by the parallel launcher.')
    ap.add_argument('--max-runs', type=int, default=0,
                    help='If >0, only process the first N alarm segments (smoke).')
    ap.add_argument('--shard-idx', type=int, default=0,
                    help='Shard index (0..num_shards-1). Processes segments where seg_idx %% num_shards == shard_idx.')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--diversity-mode', choices=['first_edge', 'all_edges'],
                    default='first_edge',
                    help="Diverse-CF edge-forbidding strategy.")
    ap.add_argument('--faithful-smoothing', action='store_true',
                    help='Re-run the 5 prior smoothing windows under the mask (~6x cost).')
    args = ap.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
