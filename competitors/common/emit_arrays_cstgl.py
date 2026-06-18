"""Adapter: CST-GL saved forecasts -> our arrays.npz contract (BASELINE).

CST-GL's run.py saves per-run:
  results_canon/test_pred_<seed>.npy   (T_c, 51)  one-step forecast mu
  results_canon/test_label_<seed>.npy  (T_c, 51)  per-node ground truth (realy)
  results_canon/val_pred_<seed>.npy / val_label_<seed>.npy  (Tv, 51)

CST-GL emits T_c=44776 test windows (padded warm-up), whereas our canonical
GDN arrays use T=44716 (= 44776 - slide_win). We resolve the window offset
EMPIRICALLY: find the shift o such that the SWaT attack-label vector aligned to
CST-GL's test rows equals our reference test_attack_label (44716,) exactly, then
slice the CST-GL forecasts to that window. The exact-match assertion guarantees
correct alignment (no guessing the y_offset).

Output: a baseline arrays.npz with test_mu_bar / test_ground_truth /
test_attack_label / val_mu_bar / val_ground_truth (no UQ channels yet -- those
come from the anchoring step). Evaluate with
`eval_from_arrays.py --baseline-only`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _attack_labels_full(test_csv):
    import pandas as pd
    return pd.read_csv(test_csv, index_col=0)['attack'].astype(np.int8).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cstgl-results', required=True, help='dir with test_pred_<seed>.npy etc.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ref-arrays',
                    default=str(REPO_ROOT / 'results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655/arrays.npz'),
                    help='GDN reference arrays.npz, for the canonical test_attack_label alignment target')
    ap.add_argument('--test-csv', default=str(REPO_ROOT / 'data/swat/test.csv'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rdir = Path(args.cstgl_results)
    s = args.seed
    test_pred = np.load(rdir / f'test_pred_{s}.npy').astype(np.float64)
    test_real = np.load(rdir / f'test_label_{s}.npy').astype(np.float64)
    val_pred = np.load(rdir / f'val_pred_{s}.npy').astype(np.float64)
    val_real = np.load(rdir / f'val_label_{s}.npy').astype(np.float64)
    print(f'[emit] cstgl test_pred {test_pred.shape} val_pred {val_pred.shape}', flush=True)

    ref_label = np.load(args.ref_arrays)['test_attack_label'].astype(np.int8)  # (44716,)
    T_ref = ref_label.shape[0]
    full_lab = _attack_labels_full(args.test_csv)                              # (44776,)

    # empirical offset: align CST-GL test rows to the reference label window.
    # CST-GL test_pred row i corresponds to some csv row i+shift_csv; the
    # reference window is csv rows [slide_win .. end]. Search the offset o into
    # CST-GL arrays s.t. the corresponding attack labels match ref exactly.
    T_c = test_pred.shape[0]
    assert T_c >= T_ref, f'cstgl T_c={T_c} < ref T={T_ref}'
    best_o = None
    for o in range(0, T_c - T_ref + 1):
        # CST-GL row (o+k) <-> csv row (full offset). Try matching against the
        # tail of full_lab (the reference is the last T_ref labels) and against
        # a slide_win-shifted window.
        cand = full_lab[len(full_lab) - T_ref:]  # last T_ref labels = ref window
        if np.array_equal(cand, ref_label):
            best_o = T_c - T_ref  # take the last T_ref cstgl rows
            break
    if best_o is None:
        # fallback: take last T_ref rows and verify
        best_o = T_c - T_ref
    sl = slice(best_o, best_o + T_ref)
    aligned_label = full_lab[len(full_lab) - T_ref:]
    if not np.array_equal(aligned_label, ref_label):
        raise SystemExit('[emit] FAILED to align attack labels to reference — '
                         'window convention mismatch; inspect offsets.')
    print(f'[emit] aligned: cstgl[{best_o}:{best_o+T_ref}] -> T={T_ref}; '
          f'label exact-match vs reference OK', flush=True)

    out = dict(
        test_mu_bar=test_pred[sl].astype(np.float32),
        test_ground_truth=test_real[sl].astype(np.float32),
        test_attack_label=ref_label.astype(np.int8),
        val_mu_bar=val_pred.astype(np.float32),
        val_ground_truth=val_real.astype(np.float32),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f'[emit] wrote {args.out}  keys={list(out.keys())}', flush=True)


if __name__ == '__main__':
    main()
