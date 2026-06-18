"""PA%K — Kim et al. 2022 rigorous TAD evaluation protocol.

Reference: "Towards a Rigorous Evaluation of Time-series Anomaly Detection"
(Kim, Choi, Choi, Lee, Yoon. AAAI 2022, arXiv 2109.05257v2).

The paper shows that the standard Point-Adjustment (PA) protocol used by
most TAD evaluations dramatically overestimates F1 — a random anomaly score
can hit F1_PA ≈ 0.99 once segments are long enough. PA%K is the proposed
fix: promote a ground-truth anomaly segment to all-positive **only if at
least K% of its timesteps already exceeded the threshold**. K=0 reduces to
standard PA (any single detection promotes whole segment); K=100 reduces
to standard F1 (no promotion).

PUBLIC API
==========

| Function | Signature | Paper ref |
|---|---|---|
| extract_runs | (v: np.ndarray) -> list[(start, end_incl)] | §3.1 segment def |
| point_adjust_K | (scores, labels, delta, K_pct) -> pred (T,) | Eq.4-modified §4.2 |
| point_adjust | (scores, labels, delta) -> pred (T,) | Eq.4 (= PA%K at K=0) |
| metrics_from_pred | (pred, label) -> dict(F1, P, R, ...) | Eq.3 |
| f1_at_threshold | (scores, labels, delta, K_pct) -> dict | Eq.3 with PA%K |
| best_f1_pa_k | (scores, labels, K_pct, n_thresholds) -> dict | δ-sweep at given K |
| f1_pa_k_auc | (scores, labels, K_grid, n_thresholds) -> dict | §4.2 AUC over K |
| random_baseline_score | (T, seed) -> (T,) array | §4.1 Case 1 |
| input_norm_baseline_score | (test_ground_truth, slide_win) -> (T,) array | §4.1 Case 2, Eq.8 |
| evaluate_method_pa_k | (method_name, scores, labels, K_grid) -> dict | adapter |

PROTOCOL NOTES
==============

PA%K rule (Eq. 4-modified, page 4):
    ŷ_t = 1 if A(w_t) > δ
           OR (t ∈ S_m AND |{t' ∈ S_m : A(w_t') > δ}| / |S_m| > K/100)
        = 0 otherwise

- K ∈ [0, 100] in PERCENT.
- The fraction comparison uses STRICT inequality `>`, not `≥` — matches the
  paper text. Concretely K=30 means "promote only if STRICTLY more than 30%
  of segment timesteps exceeded δ", so a segment with exactly 30% detection
  is NOT promoted at K=30.
- K=0 reduces to standard PA (n_hit/|S| > 0 ⟺ n_hit ≥ 1).
- K=100 reduces to standard F1 (no fraction can strictly exceed 100%).

PA%K-AUC (Section 4.2 last paragraph):
    AUC = (1/100) · ∫_0^{100} F1_{PA%K}(K) dK
        ≈ trapezoidal sum over K-grid {0, 1, ..., 100}, normalized by 100.

For each K in the grid we sweep δ over score quantiles (default 400) and
pick the best F1 at that K. The AUC is then the area under that
best-F1-vs-K curve. AUC ∈ [0, 1].

Baselines (Section 4.1):
- Random uniform: A(w_t) ~ U(0, 1). Paper proves analytically (Eqs. 5-6)
  that under PA the recall approaches 1 for long segments, giving F1_PA
  ≈ 1.0 from pure noise.
- Input L2-norm: A(w_t) = ||w_t||_2 (Eq. 8 page 3). The "extreme" untrained
  case where f_θ(w_t) ≈ 0.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Run extraction
# ----------------------------------------------------------------------------

def extract_runs(v: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a (T,) binary vector to a list of (start, end_inclusive) runs."""
    v = np.asarray(v).astype(bool)
    if v.size == 0 or not v.any():
        return []
    diff = np.diff(np.concatenate([[False], v, [False]]).astype(np.int8))
    starts = np.where(diff == 1)[0].astype(int)
    ends = (np.where(diff == -1)[0] - 1).astype(int)
    return list(zip(starts.tolist(), ends.tolist()))


# ----------------------------------------------------------------------------
# Core: PA%K rule (Eq. 4-modified)
# ----------------------------------------------------------------------------

def point_adjust_K(scores: np.ndarray,
                   labels: np.ndarray,
                   delta: float,
                   K_pct: float = 0.0) -> np.ndarray:
    """Apply PA%K labeling per Kim et al. 2022 Eq. 4-modified.

    Parameters
    ----------
    scores
        (T,) continuous anomaly score.
    labels
        (T,) binary ground-truth attack labels.
    delta
        Threshold to binarize scores: base condition is `scores > delta`.
    K_pct
        K in [0, 100] (percent). K=0 ⇔ standard PA; K=100 ⇔ no promotion.

    Returns
    -------
    pred : (T,) int8
        Adjusted prediction after PA%K rule.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(np.int8)
    base_pred = (scores > delta).astype(np.int8)
    segments = extract_runs(labels)
    if not segments:
        return base_pred
    pred = base_pred.copy()
    K_frac = K_pct / 100.0
    for (s_lo, s_hi) in segments:
        seg_scores = scores[s_lo:s_hi + 1]
        n_hit = int((seg_scores > delta).sum())
        seg_len = s_hi - s_lo + 1
        if seg_len <= 0:
            continue
        frac = n_hit / seg_len
        # Strict inequality per Eq. 4-modified.
        if frac > K_frac:
            pred[s_lo:s_hi + 1] = 1
    return pred


def point_adjust(scores: np.ndarray, labels: np.ndarray,
                 delta: float) -> np.ndarray:
    """Standard Point Adjustment (Eq. 4). Equivalent to PA%K with K=0."""
    return point_adjust_K(scores, labels, delta, K_pct=0.0)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def metrics_from_pred(pred: np.ndarray, label: np.ndarray) -> Dict:
    pred = np.asarray(pred).astype(np.int8)
    label = np.asarray(label).astype(np.int8)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return dict(F1=f1, P=p, R=r, TP=tp, FP=fp, FN=fn, TN=tn)


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray,
                    delta: float, K_pct: float = 0.0) -> Dict:
    """Compute F1/P/R after applying PA%K at (delta, K_pct)."""
    pred = point_adjust_K(scores, labels, delta, K_pct=K_pct)
    return metrics_from_pred(pred, labels)


# ----------------------------------------------------------------------------
# Threshold sweep + AUC
# ----------------------------------------------------------------------------

def best_f1_pa_k(scores: np.ndarray,
                 labels: np.ndarray,
                 K_pct: float = 0.0,
                 n_thresholds: int = 400) -> Dict:
    """Sweep δ over `n_thresholds` quantiles of `scores`; return best F1
    after PA%K at the given K_pct.

    The quantile grid is `np.linspace(0, 0.9999, n_thresholds)` — it covers
    the full range of the score distribution except the absolute max,
    which is often an outlier that thresholds at it would collapse recall.
    For pathological datasets where the optimum δ is above the 99.99th
    percentile, increase `n_thresholds` or pass a custom grid by editing
    this function.
    """
    scores = np.asarray(scores)
    qs = np.linspace(0.0, 0.9999, n_thresholds)
    taus = np.quantile(scores, qs)
    best = {'F1': -1.0, 'P': 0.0, 'R': 0.0, 'tau': float('nan'),
            'q': float('nan'),
            'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    for tau, q in zip(taus, qs):
        m = f1_at_threshold(scores, labels, float(tau), K_pct=K_pct)
        if m['F1'] > best['F1']:
            best = dict(m)
            best['tau'] = float(tau)
            best['q'] = float(q)
    return best


def f1_pa_k_auc(scores: np.ndarray,
                labels: np.ndarray,
                K_grid: Optional[np.ndarray] = None,
                n_thresholds: int = 400) -> Dict:
    """Compute F1_PA%K at each K in K_grid and return the AUC (trapezoidal,
    normalized to [0, 1] over the K range).

    Parameters
    ----------
    K_grid
        Grid of K values to sweep (each in [0, 100]). Default is
        `np.arange(0, 101, 1)` (101 points). Coarser grids (e.g. step=10
        → 11 points) approximate the AUC less precisely via trapezoidal
        integration, especially for methods where F1 changes sharply with
        K. Finer grids are more accurate but slower (compute is roughly
        linear in len(K_grid)).
    """
    if K_grid is None:
        K_grid = np.arange(0, 101, 1)  # 101 points: 0, 1, 2, ..., 100
    K_grid = np.asarray(K_grid, dtype=float)
    rows = []
    for K in K_grid:
        b = best_f1_pa_k(scores, labels, K_pct=float(K),
                         n_thresholds=n_thresholds)
        rec = {'K_pct': float(K), 'F1': b['F1'], 'P': b['P'], 'R': b['R'],
               'tau': b['tau'], 'q': b['q']}
        rows.append(rec)
    curve = pd.DataFrame(rows)
    auc = float(np.trapz(curve['F1'].values, K_grid)
                / (K_grid[-1] - K_grid[0]))
    out = {
        'F1_PA_K0': float(curve.iloc[0]['F1']),      # K=0 = standard PA
        'F1_PA_K100': float(curve.iloc[-1]['F1']),   # K=100 = standard F1
        'PA_K_AUC': auc,
        'curve': curve,
    }
    # Convenience midpoint at K=50 if in grid
    k50 = curve[curve['K_pct'] == 50.0]
    out['F1_PA_K50'] = float(k50.iloc[0]['F1']) if len(k50) > 0 else float('nan')
    return out


# ----------------------------------------------------------------------------
# Baselines from Section 4.1
# ----------------------------------------------------------------------------

def random_baseline_score(T: int, seed: int = 42) -> np.ndarray:
    """Uniform-random anomaly score (Kim et al. 2022 §4.1 Case 1)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=int(T)).astype(np.float64)


def input_norm_baseline_score(test_ground_truth: np.ndarray,
                              slide_win: int = 60) -> np.ndarray:
    """Input L2-norm as anomaly score (Kim et al. 2022 §4.1 Case 2, Eq. 8).

    s(t) = ||w_t||_F where w_t = test_ground_truth[t - slide_win + 1 : t + 1, :].

    For t < slide_win - 1 the window is left-truncated. Result is (T,) float64.
    """
    X = np.asarray(test_ground_truth, dtype=np.float64)
    T, V = X.shape
    s = np.zeros(T, dtype=np.float64)
    sq = X * X
    # Rolling cumulative sum over each sensor, then sum across sensors.
    csum = np.concatenate([np.zeros((1, V)), sq.cumsum(axis=0)], axis=0)
    for t in range(T):
        lo = max(0, t - slide_win + 1)
        hi = t + 1
        win_sq_sum = (csum[hi] - csum[lo]).sum()
        s[t] = math.sqrt(win_sq_sum)
    return s


# ----------------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------------

def evaluate_method_pa_k(method_name: str,
                         scores: np.ndarray,
                         labels: np.ndarray,
                         K_grid: Optional[np.ndarray] = None,
                         n_thresholds: int = 400) -> Dict:
    """Compute the PA%K headline numbers for one (method, scores, labels)
    triple. Returns a flat dict suitable for a CSV row."""
    res = f1_pa_k_auc(scores, labels, K_grid=K_grid,
                       n_thresholds=n_thresholds)
    return {
        'method': method_name,
        'F1_PA': res['F1_PA_K0'],          # K=0: standard PA F1
        'F1_PA_K50': res['F1_PA_K50'],      # K=50: midpoint
        'F1': res['F1_PA_K100'],            # K=100: standard F1 (no PA)
        'PA_K_AUC': res['PA_K_AUC'],        # AUC over K ∈ [0, 100]
        '_curve': res['curve'],
    }


if __name__ == '__main__':
    # Quick smoke: random scores → high F1_PA, low standard F1.
    T = 1000
    labels = np.zeros(T, dtype=np.int8)
    labels[300:800] = 1   # one long segment
    rng = np.random.default_rng(0)
    rand_scores = rng.uniform(0.0, 1.0, T)
    res = evaluate_method_pa_k('random', rand_scores, labels)
    print(f"random: F1_PA={res['F1_PA']:.4f}  F1={res['F1']:.4f}  "
          f"AUC={res['PA_K_AUC']:.4f}")
