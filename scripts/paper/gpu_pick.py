#!/usr/bin/env python3
"""Pick the N GPUs with the most free memory (good-neighbour scheduling).

Prints a comma-separated list of GPU indices to stdout, e.g. "2,0".
Sorted by free memory desc, tie-broken by utilization asc.
Robust: if nvidia-smi is missing/unparseable, falls back to "0,1,...".

Usage:
    python scripts/paper/gpu_pick.py [N]      # default N=2
"""
import subprocess
import sys


def query():
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True, timeout=30,
    )
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, free, util = int(parts[0]), int(parts[1]), int(parts[2])
        rows.append((idx, free, util))
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    try:
        rows = query()
        if not rows:
            raise RuntimeError("no GPUs parsed")
        # most free memory first; tie-break by lower utilization
        rows.sort(key=lambda r: (-r[1], r[2]))
        picked = [str(r[0]) for r in rows[:n]]
    except Exception as e:  # noqa: BLE001 - must always print something
        sys.stderr.write(f"[gpu_pick] fallback ({e})\n")
        picked = [str(i) for i in range(n)]
    print(",".join(picked))


if __name__ == "__main__":
    main()
