"""Diverse static-GNN counterfactual generation.

Builds N=5 structurally distinct counterfactuals per alarm by forbidding
the first removed edge of each successive CF. Each CF runs the staged
greedy search:
  STAGE A — edge-level CF, greedy U_str-ranked (up to K_edge_max edges)
  STAGE B — node-level escalation, greedy U_par-ranked (up to K_node_max
            nodes), only if Stage A failed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cf_engine import CFContext, cf_engine


@dataclass
class SingleCF:
    """One counterfactual: ordered list of (edge_idx, score_after) tuples
    for Stage A, and (node_idx, score_after) for Stage B."""
    edge_steps: list[tuple[int, float]] = field(default_factory=list)
    node_steps: list[tuple[int, float]] = field(default_factory=list)
    s0: float = float('nan')              # baseline score (empty mask)
    s_final: float = float('nan')         # score after the last step
    stage: str = 'A'                       # 'A', 'B', or 'failed'
    succeeded: bool = False                # alarm dropped below tau*
    forbidden_edges_used: tuple[int, ...] = ()  # what was excluded from candidate ordering

    @property
    def removed_edges(self) -> list[int]:
        return [e for e, _ in self.edge_steps]

    @property
    def removed_nodes(self) -> list[int]:
        return [v for v, _ in self.node_steps]

    def to_dict(self) -> dict:
        return dict(
            edge_steps=[(int(e), float(s)) for e, s in self.edge_steps],
            node_steps=[(int(v), float(s)) for v, s in self.node_steps],
            s0=float(self.s0),
            s_final=float(self.s_final),
            stage=self.stage,
            succeeded=bool(self.succeeded),
            forbidden_edges_used=list(self.forbidden_edges_used),
        )


def _edges_incident_to(ctx: CFContext, sensor_v: int) -> set[int]:
    """Set of edge_index_sample indices incident to sensor v
    (either as source or as target)."""
    src = ctx.edge_index_sample[0]
    tgt = ctx.edge_index_sample[1]
    mask = (src == sensor_v) | (tgt == sensor_v)
    return set(np.where(mask)[0].tolist())


def staged_greedy(
    ctx: CFContext,
    alarm_t: int,
    U_str_t: np.ndarray,
    U_par_t: np.ndarray,
    forbidden_edges: set[int],
    K_edge_max: int = 15,
    K_node_max: int = 5,
    s0: Optional[float] = None,
    faithful_smoothing: bool = False,
) -> SingleCF:
    """Run one CF: stage A then optional stage B."""
    tau = ctx.m10_tau_star
    if s0 is None:
        s0 = cf_engine(ctx, alarm_t, edge_mask=None, node_mask=None,
                       faithful_smoothing=faithful_smoothing)
    cf = SingleCF(s0=float(s0), s_final=float(s0),
                  forbidden_edges_used=tuple(sorted(forbidden_edges)))

    # ---- Stage A: greedy edge removal, U_str-ranked ----
    edge_order = np.argsort(-U_str_t)
    removed_edges: list[int] = []
    s_current = s0
    edge_tries = 0
    for e in edge_order:
        e = int(e)
        if e in forbidden_edges:
            continue
        if edge_tries >= K_edge_max:
            break
        edge_tries += 1
        trial = removed_edges + [e]
        s_new = cf_engine(ctx, alarm_t, edge_mask=trial, node_mask=None,
                          faithful_smoothing=faithful_smoothing)
        # Reject if the score got worse (monotone non-increase guard).
        if s_new > s_current + 1e-9:
            continue
        removed_edges = trial
        cf.edge_steps.append((e, float(s_new)))
        s_current = s_new
        if s_new <= tau:
            cf.stage = 'A'
            cf.succeeded = True
            cf.s_final = float(s_new)
            return cf

    cf.s_final = float(s_current)

    if cf.succeeded:
        return cf  # unreachable but defensive

    # ---- Stage B: node-level escalation, U_par-ranked ----
    node_order = np.argsort(-U_par_t)
    removed_nodes: list[int] = []
    for v in node_order:
        v = int(v)
        if len(removed_nodes) >= K_node_max:
            break
        trial_nodes = removed_nodes + [v]
        s_new = cf_engine(ctx, alarm_t,
                          edge_mask=removed_edges, node_mask=trial_nodes,
                          faithful_smoothing=faithful_smoothing)
        if s_new > s_current + 1e-9:
            continue
        removed_nodes = trial_nodes
        cf.node_steps.append((v, float(s_new)))
        s_current = s_new
        if s_new <= tau:
            cf.stage = 'B'
            cf.succeeded = True
            cf.s_final = float(s_new)
            return cf

    cf.stage = 'failed'
    cf.s_final = float(s_current)
    return cf


def diverse_cf(
    ctx: CFContext,
    alarm_t: int,
    N: int = 5,
    K_edge_max: int = 15,
    K_node_max: int = 5,
    diversity_mode: str = 'first_edge',
    faithful_smoothing: bool = False,
) -> dict:
    """Generate N diverse CFs at alarm_t by forbidding edges of preceding
    CFs. diversity_mode='first_edge' forbids only each CF's first removed
    edge (light); 'all_edges' forbids every edge any prior CF removed
    (strong).

    Returns dict with keys:
      cfs:            list[SingleCF] of length N
      sensor_votes:   {sensor_v: vote_count in [0, N]} aggregating each CF's
                       implicated sensors (edges' endpoints + nodes removed)
      ranked_sensors: list of (sensor_v, vote_count) sorted desc, ties broken
                       by U_par at alarm_t
    """
    U_str_t = ctx.test_U_str[alarm_t]
    U_par_t = ctx.test_U_par[alarm_t]
    s0 = cf_engine(ctx, alarm_t, edge_mask=None, node_mask=None,
                   faithful_smoothing=faithful_smoothing)

    cfs: list[SingleCF] = []
    forbidden: set[int] = set()
    for i in range(N):
        cf = staged_greedy(
            ctx, alarm_t, U_str_t, U_par_t,
            forbidden_edges=forbidden,
            K_edge_max=K_edge_max, K_node_max=K_node_max,
            s0=s0,
            faithful_smoothing=faithful_smoothing,
        )
        cfs.append(cf)
        if cf.edge_steps:
            if diversity_mode == 'all_edges':
                forbidden.update(cf.removed_edges)
            else:  # 'first_edge'
                forbidden.add(cf.edge_steps[0][0])
        else:
            # CF had no edge removals (Stage A skipped — went straight to B or
            # nothing). Forbid an arbitrary previously-untried U_str-top edge.
            edge_order = np.argsort(-U_str_t)
            for e in edge_order:
                e = int(e)
                if e not in forbidden:
                    forbidden.add(e)
                    break

    # Aggregate sensor votes
    src = ctx.edge_index_sample[0]
    tgt = ctx.edge_index_sample[1]
    sensor_votes: dict[int, int] = {v: 0 for v in range(ctx.V)}
    for cf in cfs:
        sensors_in_this_cf: set[int] = set()
        for e in cf.removed_edges:
            sensors_in_this_cf.add(int(src[e]))
            sensors_in_this_cf.add(int(tgt[e]))
        for v in cf.removed_nodes:
            sensors_in_this_cf.add(int(v))
        for v in sensors_in_this_cf:
            sensor_votes[v] += 1

    # Rank sensors by votes (tie-break by U_par at alarm_t descending)
    ranked = sorted(
        sensor_votes.items(),
        key=lambda kv: (-kv[1], -U_par_t[kv[0]]),
    )
    ranked = [(int(v), int(c)) for v, c in ranked if c > 0]

    return dict(
        cfs=cfs,
        s0=float(s0),
        sensor_votes=sensor_votes,
        ranked_sensors=ranked,
    )


# ---------------------------------------------------------------------- #
# CLI for one-alarm smoke
# ---------------------------------------------------------------------- #

def main():
    import argparse
    import json
    from cf_engine import build_cf_context
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
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--alarm-t', type=int, required=True,
                    help='timestep index into test arrays at which to run the diverse CF')
    ap.add_argument('--N', type=int, default=5)
    ap.add_argument('--K-edge-max', type=int, default=15)
    ap.add_argument('--K-node-max', type=int, default=5)
    args = ap.parse_args()

    ctx = build_cf_context(
        arrays_path=args.arrays, checkpoint_path=args.checkpoint,
        hp_path=args.hp, bundle_dir=args.bundle,
        cal_split_path=args.cal_split, device=args.device,
    )
    res = diverse_cf(ctx, args.alarm_t,
                      N=args.N,
                      K_edge_max=args.K_edge_max,
                      K_node_max=args.K_node_max)

    print(f'\n=== diverse_cf @ alarm_t={args.alarm_t} ===')
    print(f's0 (baseline log-odds) = {res["s0"]:.4f}, tau* = {ctx.m10_tau_star:.4f}')
    feat_map = {i: name for i, name in enumerate(
        # sensor names
        __import__('pandas').read_csv(f'./data/swat/test.csv', sep=',', index_col=0)
        .drop(columns=['attack']).columns.tolist()
    )}
    for i, cf in enumerate(res['cfs']):
        edge_seq = ', '.join(f'e{e}({src_name}->{tgt_name})|{s:+.3f}'
                              for (e, s) in cf.edge_steps
                              for src_name, tgt_name in [
                                  (feat_map[int(ctx.edge_index_sample[0, e])],
                                   feat_map[int(ctx.edge_index_sample[1, e])])])
        node_seq = ', '.join(f'{feat_map[v]}|{s:+.3f}' for (v, s) in cf.node_steps)
        print(f'  CF{i+1}: stage={cf.stage:6s}  succeeded={cf.succeeded}  '
              f'|E_remove|={len(cf.edge_steps)}  |V_remove|={len(cf.node_steps)}  '
              f's_final={cf.s_final:+.4f}')
        if edge_seq:
            print(f'    edges: {edge_seq}')
        if node_seq:
            print(f'    nodes: {node_seq}')

    print('\n  ranked sensors (vote-count, ties broken by U_par):')
    for v, c in res['ranked_sensors'][:10]:
        print(f'    {feat_map[v]:>10s}  votes={c}/{args.N}')


if __name__ == '__main__':
    main()
