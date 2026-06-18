#!/usr/bin/env python3
"""Re-slice CST-GL's all-normal windows into V1/V2 contiguous dataset dirs.

Concats the existing swat_canon train.npz + val.npz (the all-normal stream, in
time order) and cuts:
  V1: train [0,70%), val [85,100%)   (70-85% unused)
  V2: train [0,85%), val [85,100%)
test.npz is copied unchanged. Writes data/swat_canon_V1/ and _V2/ with
{train,val,test}.npz (x,y keys), drop-in for run.py --data.
"""
import argparse
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CST = os.path.join(os.path.dirname(os.path.dirname(HERE)), "competitors", "CST-GL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", default="swat_canon",
                    help="source canon dir under CST-GL/data (e.g. wadi_canon)")
    ap.add_argument("--variants", default="V1,V2")
    args = ap.parse_args()
    SRC = os.path.join(CST, "data", args.canon)
    tr = np.load(os.path.join(SRC, "train.npz"))
    va = np.load(os.path.join(SRC, "val.npz"))
    te = np.load(os.path.join(SRC, "test.npz"))
    # all-normal stream in time order = train then val
    X = np.concatenate([tr["x"], va["x"]], axis=0)
    Y = np.concatenate([tr["y"], va["y"]], axis=0)
    N = X.shape[0]
    p70, p85 = int(0.70 * N), int(0.85 * N)
    print(f"[cstgl-reslice] all-normal N={N}  70%={p70}  85%={p85}")

    sel = set(args.variants.split(","))
    for variant, tr_end in [("V1", p70), ("V2", p85)]:
        if variant not in sel:
            continue
        d = os.path.join(CST, "data", f"{args.canon}_{variant}")
        os.makedirs(d, exist_ok=True)
        # train = [0,tr_end); val = [85%,100%) contiguous last 15%
        np.savez(os.path.join(d, "train.npz"), x=X[:tr_end], y=Y[:tr_end],
                 x_offsets=tr["x_offsets"], y_offsets=tr["y_offsets"])
        np.savez(os.path.join(d, "val.npz"), x=X[p85:], y=Y[p85:],
                 x_offsets=tr["x_offsets"], y_offsets=tr["y_offsets"])
        # test unchanged
        np.savez(os.path.join(d, "test.npz"), x=te["x"], y=te["y"],
                 x_offsets=te["x_offsets"], y_offsets=te["y_offsets"])
        print(f"  {variant}: train={tr_end} val={N-p85} test={te['x'].shape[0]} -> {d}")


if __name__ == "__main__":
    main()
