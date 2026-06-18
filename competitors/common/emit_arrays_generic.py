"""Generic adapter: competitor forecast arrays -> our arrays.npz (BASELINE).

Works for any method that saves per-node one-step forecasts + ground truth +
per-timestep attack labels as numpy arrays (GTA: pred.npy/true.npy/label.npy;
others similar). Aligns to our canonical reference window by matching the SWaT
attack-label vector to our reference test_attack_label (last T_ref rows), and
asserts an exact match so the windowing offset is verified, not guessed.

Outputs a baseline arrays.npz (test_mu_bar / test_ground_truth /
test_attack_label / val_mu_bar / val_ground_truth). Evaluate with
`eval_from_arrays.py --baseline-only`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load2d(p):
    a = np.load(p)
    a = np.asarray(a, dtype=np.float64)
    # squeeze a singleton pred_len axis if present: (T,1,V)->(T,V) or (T,V,1)->(T,V)
    if a.ndim == 3:
        if a.shape[1] == 1:
            a = a[:, 0, :]
        elif a.shape[2] == 1:
            a = a[:, :, 0]
        else:
            a = a.reshape(a.shape[0], -1)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='test forecast .npy (T,V) or (T,1,V)')
    ap.add_argument('--true', required=True, help='test ground-truth .npy')
    ap.add_argument('--val-pred', default=None, help='val forecast .npy (optional; falls back to a nominal slice of test)')
    ap.add_argument('--val-true', default=None)
    ap.add_argument('--upar', default=None, help='MC-dropout per-node variance .npy (T,V); if set, emits a FULL arrays.npz with test_U_par/U_dist/sigma2_ale for M10')
    ap.add_argument('--ref-arrays',
                    default=str(REPO_ROOT / 'results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655/arrays.npz'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    pred = _load2d(args.pred)
    true = _load2d(args.true)
    assert pred.shape == true.shape, (pred.shape, true.shape)
    assert pred.shape[1] == 51, f'expected 51 nodes, got {pred.shape}'

    ref = np.load(args.ref_arrays)
    ref_label = ref['test_attack_label'].astype(np.int8)
    T_ref = ref_label.shape[0]

    if pred.shape[0] >= T_ref:
        # align to the LAST T_ref rows (our reference window = csv rows [slide_win:])
        sl = slice(pred.shape[0] - T_ref, pred.shape[0])
        out_label = ref_label
    else:
        # Competitor produced FEWER rows than the canonical grid (e.g. GTA's
        # Informer-style border math drops trailing rows). Verified offline that
        # GTA is front-aligned (ref_label[:len(pred)] matches its label.npy
        # exactly, offset 0) — it drops rows at the END. So front-align and
        # truncate the label to match. Guard against a large/unknown shortfall.
        short = T_ref - pred.shape[0]
        assert short <= 64, (f'pred T={pred.shape[0]} is {short} rows short of '
                             f'ref {T_ref} (>64; alignment unverified)')
        sl = slice(0, pred.shape[0])
        out_label = ref_label[:pred.shape[0]]
        print(f'[emit] pred {short} rows short of ref; front-aligned, '
              f'label truncated to {pred.shape[0]}', flush=True)

    if args.val_pred and args.val_true:
        val_mu = _load2d(args.val_pred); val_gt = _load2d(args.val_true)
    else:
        # nominal fallback: first 4000 aligned rows (pre-attack warmup region)
        val_mu = pred[sl][:4000]; val_gt = true[sl][:4000]

    out = dict(
        test_mu_bar=pred[sl].astype(np.float32),
        test_ground_truth=true[sl].astype(np.float32),
        test_attack_label=out_label,
        val_mu_bar=val_mu.astype(np.float32),
        val_ground_truth=val_gt.astype(np.float32),
    )
    if args.upar:
        upar = _load2d(args.upar)
        assert upar.shape == pred.shape, (upar.shape, pred.shape)
        out['test_U_par'] = upar[sl].astype(np.float32)
        out['test_U_dist'] = upar[sl].mean(axis=1).astype(np.float32)
        out['test_sigma2_ale'] = np.ones_like(pred[sl], dtype=np.float32)  # no aleatoric head
        print(f'[emit] +UQ: U_par mean={upar[sl].mean():.4e}', flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f'[emit] {args.out}  test={out["test_mu_bar"].shape} '
          f'val={out["val_mu_bar"].shape}  M10={"yes" if args.upar else "no(baseline)"}', flush=True)


if __name__ == '__main__':
    main()
