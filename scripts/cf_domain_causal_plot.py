"""Render the SWaT domain-knowledge causal graph as a PNG (and SVG).

Reads data/swat/causal_adjacency.npy (built by scripts/cf_domain_causal.py)
and lays nodes out as 6 stage columns, sensors as circles + actuators as
squares, colored by stage. Edges are drawn as curved arrows; within-stage
edges are tinted lighter than cross-stage flow edges so the process flow
stays readable.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

NODES = [
    'FIT101', 'LIT101', 'MV101', 'P101', 'P102',
    'AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201',
    'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
    'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304',
    'P301', 'P302',
    'AIT401', 'AIT402', 'FIT401', 'LIT401',
    'P401', 'P402', 'P403', 'P404', 'UV401',
    'AIT501', 'AIT502', 'AIT503', 'AIT504',
    'FIT501', 'FIT502', 'FIT503', 'FIT504',
    'P501', 'P502', 'PIT501', 'PIT502', 'PIT503',
    'FIT601', 'P601', 'P602', 'P603',
]

STAGE_COLOR = {
    1: '#4C72B0',  # raw water
    2: '#DD8452',  # chemical dosing
    3: '#55A467',  # ultrafiltration
    4: '#C44E52',  # dechlorination
    5: '#8172B3',  # reverse osmosis
    6: '#937860',  # backwash / recycle
}
STAGE_LABEL = {
    1: 'P1: Raw water',
    2: 'P2: Pre-treatment\n+ chemical dosing',
    3: 'P3: Ultrafiltration',
    4: 'P4: Dechlorination',
    5: 'P5: Reverse osmosis',
    6: 'P6: Backwash + recycle',
}


def stage_of(name: str) -> int:
    return int(re.search(r'(\d)\d\d$', name).group(1))


def node_type(name: str) -> str:
    for p in ('LIT', 'FIT', 'AIT', 'DPIT', 'PIT'):
        if name.startswith(p):
            return 'sensor'
    return 'actuator'


def build_layout() -> dict[str, tuple[float, float]]:
    """Stage = x-column, vertically packed within the column."""
    by_stage: dict[int, list[str]] = {}
    for n in NODES:
        by_stage.setdefault(stage_of(n), []).append(n)
    pos: dict[str, tuple[float, float]] = {}
    col_w = 4.0
    for st in sorted(by_stage):
        names = by_stage[st]
        # actuators on the left half of the column, sensors on the right half
        names = sorted(names, key=lambda n: (node_type(n) != 'actuator', n))
        n = len(names)
        for i, name in enumerate(names):
            y = (n - 1) / 2 - i  # centred around 0
            x = st * col_w + (0.6 if node_type(name) == 'sensor' else -0.6)
            pos[name] = (x, y * 1.4)
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--adj', default=str(REPO_ROOT / 'data' / 'swat' / 'causal_adjacency.npy'))
    ap.add_argument('--out-png', default=str(REPO_ROOT / 'data' / 'swat' / 'causal_adjacency.png'))
    ap.add_argument('--out-svg', default=str(REPO_ROOT / 'data' / 'swat' / 'causal_adjacency.svg'))
    args = ap.parse_args()

    A = np.load(args.adj)
    assert A.shape == (len(NODES), len(NODES))

    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    same_stage_edges = []
    cross_stage_edges = []
    for i, src in enumerate(NODES):
        for j, dst in enumerate(NODES):
            if A[i, j] == 0:
                continue
            G.add_edge(src, dst)
            if stage_of(src) == stage_of(dst):
                same_stage_edges.append((src, dst))
            else:
                cross_stage_edges.append((src, dst))

    pos = build_layout()
    fig, ax = plt.subplots(figsize=(22, 14))

    sensors = [n for n in NODES if node_type(n) == 'sensor']
    actuators = [n for n in NODES if node_type(n) == 'actuator']

    nx.draw_networkx_nodes(
        G, pos, nodelist=sensors, ax=ax,
        node_shape='o', node_size=1300,
        node_color=[STAGE_COLOR[stage_of(n)] for n in sensors],
        edgecolors='black', linewidths=1.0,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=actuators, ax=ax,
        node_shape='s', node_size=1300,
        node_color=[STAGE_COLOR[stage_of(n)] for n in actuators],
        edgecolors='black', linewidths=1.0,
    )

    nx.draw_networkx_edges(
        G, pos, edgelist=same_stage_edges, ax=ax,
        edge_color='#888888', width=0.9, alpha=0.7,
        arrows=True, arrowstyle='-|>', arrowsize=10,
        connectionstyle='arc3,rad=0.15',
        node_size=1300,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=cross_stage_edges, ax=ax,
        edge_color='#1b1b1b', width=1.4, alpha=0.85,
        arrows=True, arrowstyle='-|>', arrowsize=12,
        connectionstyle='arc3,rad=0.10',
        node_size=1300,
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=7.5, font_weight='bold',
    )

    # stage banner across the top
    for st in sorted(STAGE_COLOR):
        x = st * 4.0
        y = max(p[1] for p in pos.values()) + 1.6
        ax.text(x, y, STAGE_LABEL[st],
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=STAGE_COLOR[st],
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white',
                          edgecolor=STAGE_COLOR[st], linewidth=1.5))

    # legends
    stage_handles = [
        mpatches.Patch(color=STAGE_COLOR[st], label=STAGE_LABEL[st].replace('\n', ' '))
        for st in sorted(STAGE_COLOR)
    ]
    shape_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markeredgecolor='black', markersize=12, label='Sensor (LIT/FIT/AIT/DPIT/PIT)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray',
                   markeredgecolor='black', markersize=12, label='Actuator (MV/P/UV)'),
    ]
    edge_handles = [
        plt.Line2D([0], [0], color='#1b1b1b', lw=1.6, label='Cross-stage causal edge'),
        plt.Line2D([0], [0], color='#888888', lw=1.0, label='Within-stage causal edge'),
    ]
    legend1 = ax.legend(handles=stage_handles, loc='lower left', fontsize=9,
                        title='Process stage', frameon=True)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=shape_handles + edge_handles,
                        loc='lower right', fontsize=9, frameon=True)
    ax.add_artist(legend2)

    n_edges = int(A.sum())
    ax.set_title(
        f'SWaT domain-knowledge causal graph  '
        f'({len(NODES)} nodes, {n_edges} directed edges)',
        fontsize=14, pad=18,
    )
    ax.set_axis_off()
    ax.margins(0.05)
    fig.tight_layout()

    fig.savefig(args.out_png, dpi=180, bbox_inches='tight')
    fig.savefig(args.out_svg, bbox_inches='tight')
    print(f'Wrote {args.out_png}')
    print(f'Wrote {args.out_svg}')


if __name__ == '__main__':
    main()
