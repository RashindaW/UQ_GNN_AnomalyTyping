#!/usr/bin/env python3
"""Build competitors/CST-GL/data/wadi_canon/{train,val,test}.npz + test_label.pkl.

Mirrors the swat_canon contract exactly (verified empirically):
  x[i] = the 60 rows BEFORE target row i (x[i][-1] == y[i-1]); y[i] = row i;
  test window count == test ROW count (the first 60 test windows draw their
  history from the train tail, which is wall-clock contiguous: WADI_14days
  ends 2017-10-09 18:00:00 exactly where attackdata starts);
  x_offsets = [-59..0], y_offsets = [1]; train/val = 90/10 cut of the
  all-normal window stream (the V2 reslicer concatenates and recuts anyway).

Values come from data/wadi/{train,test}.csv which are ALREADY min-max
normalized by prepare_wadi.py, so no further scaling here (load_dataset is
called with scaling_required=False). float32 to halve the ~3.5 GB train x.

Run in rashindaNew-torch-env (CPU, a few minutes).
"""
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CST = os.path.join(ROOT, "competitors", "CST-GL")
OUT = os.path.join(CST, "data", "wadi_canon")
W = 60


def windows(stream: np.ndarray):
    """x[i] = stream[i:i+W], y[i] = stream[i+W]; returns (N-W, W, V, 1), (N-W, 1, V, 1)."""
    sw = np.lib.stride_tricks.sliding_window_view(stream, W, axis=0)  # (N-W+1, V, W)
    x = sw[:-1].transpose(0, 2, 1)[..., None].astype(np.float32)      # (N-W, W, V, 1)
    y = stream[W:][:, None, :, None].astype(np.float32)               # (N-W, 1, V, 1)
    return x, y


def main():
    train = pd.read_csv(os.path.join(ROOT, "data/wadi/train.csv"), index_col=0)
    test = pd.read_csv(os.path.join(ROOT, "data/wadi/test.csv"), index_col=0)
    lab = test["attack"].to_numpy().astype(np.int8)
    test = test.drop(columns=["attack"])
    feats = open(os.path.join(ROOT, "data/wadi/list.txt")).read().split()
    assert list(train.columns) == feats == list(test.columns), "column order mismatch"
    V = len(feats)
    S_tr, S_te = train.to_numpy(np.float32), test.to_numpy(np.float32)
    print(f"[wadi-canon] train {S_tr.shape} test {S_te.shape} V={V}", flush=True)

    xo = np.arange(-(W - 1), 1).reshape(-1, 1)
    yo = np.array([[1]])

    # all-normal stream windows, 90/10 cut (recut later by reslice_cstgl_v1v2)
    x_all, y_all = windows(S_tr)
    n = x_all.shape[0]
    cut = int(round(0.9 * n))
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "train.npz"), x=x_all[:cut], y=y_all[:cut], x_offsets=xo, y_offsets=yo)
    np.savez(os.path.join(OUT, "val.npz"), x=x_all[cut:], y=y_all[cut:], x_offsets=xo, y_offsets=yo)

    # test: pad with the train tail so window count == row count
    padded = np.vstack([S_tr[-W:], S_te])
    x_te, y_te = windows(padded)                 # (n_test, W, V, 1), y[i] = test row i
    assert x_te.shape[0] == len(S_te) == len(lab)
    assert np.allclose(y_te[:-1, 0, :, 0], x_te[1:, -1, :, 0]), "canon alignment broken"
    np.savez(os.path.join(OUT, "test.npz"), x=x_te, y=y_te, x_offsets=xo, y_offsets=yo)
    with open(os.path.join(OUT, "test_label.pkl"), "wb") as f:
        pickle.dump(lab.tolist(), f)

    print(f"[wadi-canon] train {cut} val {n - cut} test {x_te.shape[0]} windows, "
          f"label frac {lab.mean():.4f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
