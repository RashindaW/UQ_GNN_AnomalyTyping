"""Build a directed ACYCLIC actuator-exogenous causal scaffold for SWaT.

Derived from the domain edge list in scripts/cf_domain_causal.py, but made
into a DAG suitable for the do()/root-cause framing by:

  1. ACTUATOR EXOGENEITY: remove every edge whose DESTINATION is an actuator
     (MV / P<digit> / UV). This drops all sensor->actuator PLC control-feedback
     edges (the class that creates the 2-cycles) and guarantees actuators have
     zero in-degree (they are exogenous intervention targets).
  2. ACYCLICITY: among the remaining actuator->sensor and sensor->sensor edges,
     run Kahn topological sort; if any residual cycle exists (only possible
     among sensors), drop the minimal set of back-edges (by a DFS) and log them.

Convention preserved from cf_domain_causal: A[i, j] = 1 means i -> j
(row = source, column = target). Saved as int8 (51, 51) with a features sidecar
in the exact format models/causal_mask.py::load_causal_mask expects.

Usage:
    python scripts/cf_build_dag_scaffold.py        # writes to data/swat/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from cf_domain_causal import NODES, EDGES  # reuse the audited domain edge list

ACTUATOR_RE = re.compile(r'^(MV|UV|P)\d')   # MV101, UV401, P101 ; PIT501 -> sensor


def is_actuator(name: str) -> bool:
    return bool(ACTUATOR_RE.match(name))


def drop_back_edges(edges: list[tuple[str, str]], nodes: list[str]):
    """Return (dag_edges, dropped) — remove a minimal set of back-edges found by
    DFS so the result is acyclic. Edge order is preserved; the first time an
    edge would close a cycle (points to a node currently on the DFS stack) it is
    dropped."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for s, d in edges:
        adj[s].append(d)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    dropped: set[tuple[str, str]] = set()

    def dfs(u: str):
        color[u] = GRAY
        for v in adj[u]:
            if (u, v) in dropped:
                continue
            if color[v] == GRAY:          # back-edge -> would form a cycle
                dropped.add((u, v))
                continue
            if color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    sys.setrecursionlimit(10000)
    for n in nodes:
        if color[n] == WHITE:
            dfs(n)
    dag = [(s, d) for (s, d) in edges if (s, d) not in dropped]
    return dag, sorted(dropped)


def is_acyclic(edges: list[tuple[str, str]], nodes: list[str]) -> bool:
    indeg = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for s, d in edges:
        indeg[d] += 1
        adj[s].append(d)
    queue = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while queue:
        u = queue.pop()
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return seen == len(nodes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'data' / 'swat'))
    ap.add_argument('--stem', default='causal_scaffold_dag')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # de-dup the raw domain edges
    raw = []
    seen = set()
    for s, d in EDGES:
        if s != d and (s, d) not in seen:
            seen.add((s, d))
            raw.append((s, d))

    actuators = [n for n in NODES if is_actuator(n)]
    sensors = [n for n in NODES if not is_actuator(n)]
    print(f'nodes={len(NODES)}  actuators={len(actuators)}  sensors={len(sensors)}',
          flush=True)

    # 1) actuator exogeneity: drop every edge INTO an actuator
    into_actuator = [(s, d) for (s, d) in raw if is_actuator(d)]
    kept = [(s, d) for (s, d) in raw if not is_actuator(d)]
    print(f'raw edges={len(raw)}  dropped (into actuator)={len(into_actuator)}  '
          f'kept={len(kept)}', flush=True)

    # 2) acyclicity
    if is_acyclic(kept, NODES):
        dag, dropped_back = kept, []
        print('kept graph is already acyclic', flush=True)
    else:
        dag, dropped_back = drop_back_edges(kept, NODES)
        print(f'dropped {len(dropped_back)} residual back-edge(s): {dropped_back}',
              flush=True)
    assert is_acyclic(dag, NODES), 'DAG construction failed: still cyclic'

    # build matrix (src -> tgt)
    idx = {n: i for i, n in enumerate(NODES)}
    A = np.zeros((len(NODES), len(NODES)), dtype=np.int8)
    for s, d in dag:
        A[idx[s], idx[d]] = 1

    # verification + stats
    in_deg = A.sum(axis=0)
    out_deg = A.sum(axis=1)
    act_in = {n: int(in_deg[idx[n]]) for n in actuators if in_deg[idx[n]] > 0}
    assert not act_in, f'actuators with nonzero in-degree: {act_in}'
    sensor_parents = {n: int(in_deg[idx[n]]) for n in sensors}
    n_edges = int(A.sum())
    density = n_edges / (len(NODES) * (len(NODES) - 1))
    print(f'\nDAG edges={n_edges}  density={density:.4f}', flush=True)
    print(f'actuator in-degree: all 0 (exogenous)  [verified]', flush=True)
    pc = np.array(list(sensor_parents.values()))
    print(f'sensor parent-count  min/median/mean/max = '
          f'{pc.min()}/{int(np.median(pc))}/{pc.mean():.1f}/{pc.max()}', flush=True)
    zero_parent_sensors = [n for n, c in sensor_parents.items() if c == 0]
    if zero_parent_sensors:
        print(f'  sensors with 0 causal parents (self-loop only): {zero_parent_sensors}',
              flush=True)
    print('\nper-sensor parent counts:')
    for n in sensors:
        print(f'  {n:8s} parents={sensor_parents[n]}', flush=True)

    # write
    npy_path = out_dir / f'{args.stem}.npy'
    csv_path = out_dir / f'{args.stem}.csv'
    feat_path = out_dir / f'{args.stem}_features.json'
    np.save(npy_path, A)
    feat_path.write_text(json.dumps(NODES, indent=2))
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow([''] + NODES)
        for i, name in enumerate(NODES):
            w.writerow([name] + A[i].tolist())
    print(f'\nwrote {npy_path}\nwrote {feat_path}\nwrote {csv_path}', flush=True)


if __name__ == '__main__':
    main()
