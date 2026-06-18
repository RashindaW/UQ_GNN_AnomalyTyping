"""Build a 51x51 binary causal adjacency matrix for SWaT from domain knowledge.

Edges encode the physical process and PLC control loops described in
  "A Dataset to Support Research in the Design of Secure Water Treatment
   Systems" (Goh et al., 2016) -- Table 2 (sensor/actuator descriptions)
   and Figure 2 (six-stage process overview).

Convention: A[i, j] = 1 means node i causally influences node j ("i -> j").
            A[i, i] = 0 (no self-loops).

Causal edge taxonomy used here:
  (1) Actuator -> downstream sensor: pump/valve presence drives a flow or
      level reading (mass-flow effect).
  (2) Sensor -> actuator: closed-loop PLC control (level governs pump,
      analyzer governs dosing, dP governs backwash). Different from
      get_typed_causal_scaffold_swat in util/net_struct.py, which treats
      actuators as exogenous; here we include the control feedback.
  (3) Dosing pump -> matching analyzer: chemistry mass-balance.
  (4) Tank inflow actuator -> level; tank outflow actuator -> level.
  (5) Cross-stage chemistry: upstream water-quality sensor -> downstream
      analyzer reading carried by the water itself.

Outputs (in repo root unless overridden by --out-dir):
  data/swat/causal_adjacency.csv  -- header row + index column, 0/1 cells
  data/swat/causal_adjacency.npy  -- raw int8 matrix, shape (51, 51)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

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
assert len(NODES) == 51

# (parent, child) edges. Keep grouped by physical stage for review.
EDGES: list[tuple[str, str]] = []

# -- Stage 1: raw water tank ------------------------------------------------
# MV101 opens -> water flows in (FIT101 reads it) -> raw tank fills (LIT101).
EDGES += [
    ('MV101', 'FIT101'),
    ('MV101', 'LIT101'),
    ('FIT101', 'LIT101'),
    # PLC level control of the inlet valve and outlet pumps.
    ('LIT101', 'MV101'),
    ('LIT101', 'P101'),
    ('LIT101', 'P102'),
    # Outlet pumps drain the raw tank.
    ('P101', 'LIT101'),
    ('P102', 'LIT101'),
]

# -- Stage 1 -> Stage 2: water carried by P101/P102 to the pre-treatment line
EDGES += [
    ('P101', 'FIT201'), ('P102', 'FIT201'),
    ('P101', 'AIT201'), ('P101', 'AIT202'), ('P101', 'AIT203'),
    ('P102', 'AIT201'), ('P102', 'AIT202'), ('P102', 'AIT203'),
]

# -- Stage 2: chemical dosing --------------------------------------------------
# Analyzer -> matching dosing pump (closed-loop chemistry control).
EDGES += [
    ('AIT201', 'P201'), ('AIT201', 'P202'),     # NaCl conductivity -> NaCl dose
    ('AIT202', 'P203'), ('AIT202', 'P204'),     # HCl pH          -> HCl dose
    ('AIT203', 'P205'), ('AIT203', 'P206'),     # NaOCl ORP       -> NaOCl dose
]
# Dosing pump -> matching analyzer (mass-balance: dose changes reading).
EDGES += [
    ('P201', 'AIT201'), ('P202', 'AIT201'),
    ('P203', 'AIT202'), ('P204', 'AIT202'),
    ('P205', 'AIT203'), ('P206', 'AIT203'),
]
# Per Table 2 entry 9: FIT201 "Control dosing pumps" -- flow gates dosing.
EDGES += [('FIT201', f'P20{i}') for i in range(1, 7)]

# -- Stage 2 -> Stage 3: MV201 fills the UF feed tank (LIT301) ----------------
# Strict-causality pruning:
#   MV201 -> FIT301: dropped. The tank LIT301 buffers between MV201's inflow
#     rate and the UF outflow rate; the real path is
#     MV201 -> LIT301 -> P301/P302 -> FIT301.
#   P101/P102/FIT201 -> LIT301: dropped. Upstream flow is fully mediated by
#     MV201 before it reaches the UF feed tank.
EDGES += [
    ('MV201', 'LIT301'),
    ('LIT301', 'MV201'),
]

# -- Stage 3: ultrafiltration --------------------------------------------------
EDGES += [
    # PLC: tank level controls feed pumps.
    ('LIT301', 'P301'), ('LIT301', 'P302'),
    # UF feed pumps drain the UF feed tank.
    ('P301', 'LIT301'), ('P302', 'LIT301'),
    # UF feed pumps drive flow through the UF stage.
    ('P301', 'FIT301'), ('P302', 'FIT301'),
    # Flow generates the membrane differential pressure.
    ('P301', 'DPIT301'), ('P302', 'DPIT301'),
    # dP triggers backwash valves and the backwash pump.
    ('DPIT301', 'MV301'), ('DPIT301', 'MV302'),
    ('DPIT301', 'MV303'), ('DPIT301', 'MV304'),
    ('DPIT301', 'P602'),
    # Backwash valves / pump drive the backwash flow reading.
    ('MV301', 'FIT601'),
    ('MV303', 'FIT601'),
    ('P602', 'FIT601'),
]

# -- Stage 3 -> Stage 4: MV302 routes UF output to the RO feed tank -----------
# Strict-causality pruning:
#   P301/P302 -> LIT401: dropped. Mediated by MV302 (parallel to
#     P101 -> LIT301 above).
#   MV302 -> AIT401: dropped. Mediated by LIT401 -> P401/P402 -> AIT401.
EDGES += [
    ('MV302', 'LIT401'),
]

# -- Stage 4: dechlorination ---------------------------------------------------
EDGES += [
    # PLC: RO feed tank level controls RO feed pumps.
    ('LIT401', 'P401'), ('LIT401', 'P402'),
    # RO feed pumps drain LIT401.
    ('P401', 'LIT401'), ('P402', 'LIT401'),
    # RO feed pumps drive flow through the UV dechlorinator.
    ('P401', 'FIT401'), ('P402', 'FIT401'),
    # Same flow is read by the hardness / ORP analyzers in stage 4.
    ('P401', 'AIT401'), ('P402', 'AIT401'),
    ('P401', 'AIT402'), ('P402', 'AIT402'),
    # Per Table 2 entry 28: FIT401 controls UV401.
    ('FIT401', 'UV401'),
    # AIT402 (ORP) sits upstream of UV401 -- it controls NaHSO3 dosing
    # before the water hits the dechlorinator, so UV401 cannot causally
    # affect AIT402 (edge dropped).
    # AIT402 ORP controls NaHSO3 dosing (P403/P404); per Table 2 it also
    # gates NaOCl dosing (P205) upstream in stage 2.
    ('AIT402', 'P403'), ('AIT402', 'P404'), ('AIT402', 'P205'),
    ('P403', 'AIT402'), ('P404', 'AIT402'),
]

# -- Stage 4 -> Stage 5: chemistry carried through dechlorinator to RO --------
EDGES += [
    ('UV401', 'AIT501'), ('UV401', 'AIT502'), ('UV401', 'AIT503'),
    ('AIT401', 'AIT501'), ('AIT401', 'AIT503'),
    ('AIT402', 'AIT502'),
]

# -- Stage 5: reverse osmosis --------------------------------------------------
EDGES += [
    # RO booster pumps drive RO inlet flow and feed pressure.
    ('P501', 'FIT501'), ('P502', 'FIT501'),
    ('P501', 'PIT501'), ('P502', 'PIT501'),
    # Feed pressure -> RO inlet flow.
    ('PIT501', 'FIT501'),
    # Membrane: inlet flow splits into permeate, reject, recirculation.
    ('FIT501', 'FIT502'), ('FIT501', 'FIT503'), ('FIT501', 'FIT504'),
    # Membrane: feed pressure produces permeate / reject side pressures.
    ('PIT501', 'PIT502'), ('PIT501', 'PIT503'),
    # Higher feed pressure -> better salt rejection -> lower permeate cond.
    ('PIT501', 'AIT504'),
    # Salt rejection: feed conductivity drives permeate conductivity.
    ('AIT503', 'AIT504'),
    # Booster pumps push water past all RO-side analyzers.
    ('P501', 'AIT501'), ('P501', 'AIT502'), ('P501', 'AIT503'), ('P501', 'AIT504'),
    ('P502', 'AIT501'), ('P502', 'AIT502'), ('P502', 'AIT503'), ('P502', 'AIT504'),
]

# -- Stage 6: backwash + recycle ----------------------------------------------
# P601: per Table 2 entry 49, pumps RO permeate back to the raw tank
# ("not used for data collection") -- include for completeness.
EDGES += [
    ('FIT502', 'P601'),
    ('P601', 'LIT101'),
]
# P603 not implemented (Table 2 entry 51) -- no edges.


def build_matrix() -> np.ndarray:
    idx = {n: i for i, n in enumerate(NODES)}
    A = np.zeros((len(NODES), len(NODES)), dtype=np.int8)
    seen: set[tuple[str, str]] = set()
    for src, dst in EDGES:
        if src == dst:
            continue
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        A[idx[src], idx[dst]] = 1
    return A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'data' / 'swat'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A = build_matrix()
    npy_path = out_dir / 'causal_adjacency.npy'
    csv_path = out_dir / 'causal_adjacency.csv'
    feat_path = out_dir / 'causal_adjacency_features.json'

    np.save(npy_path, A)
    import json as _json
    feat_path.write_text(_json.dumps(NODES, indent=2))

    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow([''] + NODES)
        for i, name in enumerate(NODES):
            w.writerow([name] + A[i].tolist())

    n_edges = int(A.sum())
    density = n_edges / (len(NODES) * (len(NODES) - 1))
    in_deg = A.sum(axis=0)
    out_deg = A.sum(axis=1)
    print(f'Wrote {csv_path}')
    print(f'Wrote {npy_path}')
    print(f'Wrote {feat_path}')
    print(f'Nodes: {len(NODES)}   Edges: {n_edges}   Density: {density:.4f}')
    print(f'Out-degree  min/median/max: {out_deg.min()} / '
          f'{int(np.median(out_deg))} / {out_deg.max()}')
    print(f'In-degree   min/median/max: {in_deg.min()} / '
          f'{int(np.median(in_deg))} / {in_deg.max()}')


if __name__ == '__main__':
    main()
