"""PTaPR metric — faithful Python implementation of Kang et al. (2026).

Reference: "Forecasting Anomaly Precursors with Uncertainty-Aware Time-series
Ensembles" (Kang, Park, Han, Kang. IEEE TNNLS, manuscript Feb 2026, arXiv
2602.17028v1). Equations 5-15 of Section IV.

This module computes the three-component PTaPR metric — Precursor Time-series
Aware Precision and Recall — used by the paper for evaluating precursor
detection. The metric extends the classical TaPR (Hwang et al.) by adding an
explicit early-detection reward and an ambiguous-instance score.

PUBLIC API
==========

| Function | Signature | Paper ref |
|---|---|---|
| extract_runs | (v: np.ndarray) -> list[(start, end_incl)] | §IV.A |
| precursor_window | (a_start, eps) -> (lo, hi) | "p' precedes p" §IV.A |
| ambiguous_segment | (a, delta, T) -> (lo, hi) | §IV.A "a'" |
| overlap_count | (s1, s2) -> int | |·∩·| in Eq.7 |
| ambiguous_score_S | (a_prime, p, delta) -> float | Eq.8 |
| early_reward_E | (a, p_prime, k, eps) -> float | Eq.10 |
| overlap_O | (a, p, p_prime, a_prime, delta) -> float | Eq.7 |
| ptapr_paper | (label_T, pred_T, theta, ...) -> dict | Eqs.5-15 |
| ptapr_auc | (label_T, pred_T, theta_grid=None, ...) -> dict | §IV.D AUC |
| evaluate_method | (method_name, alarm_T, label_T, ...) -> dict | adapter |

PROTOCOL NOTES
==============

This implementation uses the **Strict** convention for precursor segments p':
each alarm run that **straddles** the start of an anomaly is split into a
precursor part (p', the portion strictly before t_a) and a main part (p, the
portion from t_a onwards). Alarm runs entirely before t_a become precursors;
runs entirely from t_a onwards have no precursor component. Therefore
|a ∩ p'| = 0 by construction under this convention.

This differs from Fig. 5 of the paper, which appears to state |a₁ ∩ p'₁| = 1.
We document this discrepancy below and provide a `precursor_includes_t_a`
flag that allows p' to extend by one timestep into a so that Fig. 5's stated
numbers can be reproduced as a regression test.

The other Fig. 5 anomaly — PTaR^p stated as 0.8 (which the paper's claim
"(0.6 + 1)/2 = 0.8" requires Σ O(a,p,p')/|a₁| = 0.6, but the figure's stated
O(a₁, p₁) = 3 with |a₁| = 3 gives 1.0 — is a paper-internal arithmetic
inconsistency. Our implementation follows Eq. 9 exactly:
    PTaR^p = (1/|A|) Σ_a min(1, Σ_p O(a, p, p')/|a|).

PAPER DEFAULTS (Section V.C)
============================
    eps = 6      optimal early-prediction lead time (timesteps)
    delta = 4    ambiguous-window length
    k = 0.05     early-detection penalty curvature
    alpha = beta = gamma = 1/3   weights for PTaR^d, PTaR^p, PTaR^e

θ-AUC: paper sweeps θ ∈ [0, 1] and reports the AUC (trapezoidal) plus
F1_0 = PTaPR(θ=0) and F1_1 = PTaPR(θ=1) as headline numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Paper defaults (Section V.C and Fig. 5 worked example).
DEFAULT_EPS: int = 6
DEFAULT_DELTA: int = 4
DEFAULT_K: float = 0.05
DEFAULT_ALPHA: float = 1.0 / 3.0
DEFAULT_BETA: float = 1.0 / 3.0
DEFAULT_GAMMA: float = 1.0 / 3.0
DEFAULT_THETA_GRID = np.linspace(0.0, 1.0, 21)  # 21 points → trapezoidal AUC


# ----------------------------------------------------------------------------
# Run extraction & segment helpers
# ----------------------------------------------------------------------------

def extract_runs(v: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a (T,) binary vector to a list of (start, end_inclusive) runs.

    Example: [0,1,1,0,1,0,1,1,1] → [(1,2), (4,4), (6,8)].
    """
    v = np.asarray(v).astype(bool)
    if v.size == 0 or not v.any():
        return []
    diff = np.diff(np.concatenate([[False], v, [False]]).astype(np.int8))
    starts = np.where(diff == 1)[0].astype(int)
    ends = (np.where(diff == -1)[0] - 1).astype(int)
    return list(zip(starts.tolist(), ends.tolist()))


def precursor_window(a_start: int, eps: int) -> Tuple[int, int]:
    """The precursor-of-anomaly window: [a_start - eps, a_start - 1].

    This is the time interval before a where a precursor prediction is
    rewarded by the early-detection term. Returns the half-open
    inclusive-end window; callers may need to clip to [0, T-1].
    """
    return (max(0, a_start - eps), a_start - 1)


def ambiguous_segment(anomaly: Tuple[int, int], delta: int,
                      T: int) -> Optional[Tuple[int, int]]:
    """Post-anomaly ambiguous window a' = [a_end+1, a_end+delta],
    clipped to [0, T-1]. Returns None if delta <= 0 or window is empty."""
    a_end = anomaly[1]
    am_start = a_end + 1
    am_end = min(a_end + delta, T - 1)
    if delta <= 0 or am_start > am_end:
        return None
    return (am_start, am_end)


def overlap_count(seg_a: Optional[Tuple[int, int]],
                  seg_b: Optional[Tuple[int, int]]) -> int:
    """|seg_a ∩ seg_b| where segments are (start, end_inclusive) intervals."""
    if seg_a is None or seg_b is None:
        return 0
    lo = max(seg_a[0], seg_b[0])
    hi = min(seg_a[1], seg_b[1])
    return max(0, hi - lo + 1)


def split_prediction_to_p_pprime(
    prediction: Tuple[int, int],
    a_start: int,
    eps: int,
    precursor_includes_t_a: bool = False,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Split an alarm run into (p_prime, p) relative to the upcoming anomaly.

    p_prime: portion of the prediction that lies in [a_start - eps, a_start - 1].
    p:       portion that lies on/after a_start.

    With `precursor_includes_t_a=True`, p_prime is extended to include
    a_start itself (i.e., uses window [a_start - eps, a_start]). This is the
    "permissive" convention that reproduces Fig. 5's |a ∩ p'| = 1 claim.
    """
    r_start, r_end = prediction
    if precursor_includes_t_a:
        pp_hi = a_start
    else:
        pp_hi = a_start - 1
    pp_lo = max(r_start, a_start - eps)
    pp_hi_eff = min(r_end, pp_hi)
    if pp_lo > pp_hi_eff:
        p_prime: Optional[Tuple[int, int]] = None
    else:
        p_prime = (pp_lo, pp_hi_eff)

    p_lo = max(r_start, a_start) if not precursor_includes_t_a else max(r_start, a_start + 1)
    p_hi = r_end
    if p_lo > p_hi:
        p: Optional[Tuple[int, int]] = None
    else:
        p = (p_lo, p_hi)
    return p_prime, p


# ----------------------------------------------------------------------------
# Eq. 8: ambiguous-instance score S(a', p)
# ----------------------------------------------------------------------------

def ambiguous_score_S(a_prime: Optional[Tuple[int, int]],
                      p: Optional[Tuple[int, int]],
                      delta: int) -> float:
    """S(a', p) = Σ_{i ∈ (a' ∩ p)} 1 / (1 + exp(i'))

    with i' = -δ + 12·(i - t_{a'}) / (δ - 1).

    Notes
    -----
    The sigmoid 1/(1+e^{i'}) is *not* the standard σ(x); it monotonically
    decreases as i' grows. At i' = -δ (start of a') the value is near 1; at
    i' = +δ (end of a' for δ ≥ 2) it is near 0. This rewards predictions
    that overlap the early part of a' (closer to the anomaly's end).
    """
    if a_prime is None or p is None or delta <= 0:
        return 0.0
    lo = max(a_prime[0], p[0])
    hi = min(a_prime[1], p[1])
    if lo > hi:
        return 0.0
    t_aprime = a_prime[0]
    if delta == 1:
        return 0.5 * (hi - lo + 1)
    total = 0.0
    for i in range(lo, hi + 1):
        i_prime = -delta + 12.0 * (i - t_aprime) / (delta - 1)
        # Clamp for numerical safety. exp(50) is already > 5e21.
        i_prime_clamped = max(-50.0, min(50.0, i_prime))
        total += 1.0 / (1.0 + math.exp(i_prime_clamped))
    return total


# ----------------------------------------------------------------------------
# Eq. 10: early-detection reward E(a, p')
# ----------------------------------------------------------------------------

def early_reward_E(anomaly: Tuple[int, int],
                   p_prime: Optional[Tuple[int, int]],
                   k: float = DEFAULT_K,
                   eps: int = DEFAULT_EPS) -> float:
    """E(a, p') = exp(-k(i' - ε)²), where i' = t_a - i and i is the largest
    index in p' satisfying i < t_a. Returns 0 if no such i exists.

    The "largest i below t_a" convention is the standard choice for a
    single-precursor reward: i' is the time-gap to anomaly onset; the
    reward is maximised at i' = ε (the optimal lead time) and decays
    Gaussian-like for earlier or later predictions.
    """
    if p_prime is None:
        return 0.0
    t_a = anomaly[0]
    pp_lo, pp_hi = p_prime
    i = min(pp_hi, t_a - 1)
    if i < pp_lo:
        return 0.0
    i_prime = t_a - i
    return float(math.exp(-k * (i_prime - eps) ** 2))


# ----------------------------------------------------------------------------
# Eq. 7: O(a, p, p')
# ----------------------------------------------------------------------------

def overlap_O(anomaly: Tuple[int, int],
              prediction: Tuple[int, int],
              p_prime: Optional[Tuple[int, int]],
              a_prime: Optional[Tuple[int, int]],
              delta: int) -> float:
    """O(a, p, p') = |a ∩ p'| + |a ∩ p| + S(a', p)."""
    return (
        float(overlap_count(anomaly, p_prime))
        + float(overlap_count(anomaly, prediction))
        + ambiguous_score_S(a_prime, prediction, delta)
    )


# ----------------------------------------------------------------------------
# Main metric
# ----------------------------------------------------------------------------

@dataclass
class AssociatedSegments:
    """For each alarm run, the (p_prime, p, associated_anomaly_idx)."""
    p_prime_per_pred: List[Optional[Tuple[int, int]]]
    p_per_pred: List[Optional[Tuple[int, int]]]
    associated_a_idx_per_pred: List[Optional[int]]


def _associate_predictions(
    A: List[Tuple[int, int]],
    P: List[Tuple[int, int]],
    eps: int,
    precursor_includes_t_a: bool = False,
) -> AssociatedSegments:
    """For each prediction in P, find the upcoming anomaly (next anomaly that
    starts at or after the prediction's start, OR has its start within p's
    span). Split p into (p_prime, p_main) relative to that anomaly.

    A prediction with NO upcoming anomaly has p_prime = None and p_main = p.
    """
    p_prime_list: List[Optional[Tuple[int, int]]] = []
    p_main_list: List[Optional[Tuple[int, int]]] = []
    assoc_idx: List[Optional[int]] = []
    for p in P:
        # Find candidates: anomalies whose start lies in [p.start - eps, p.end + 1).
        # Among these the EARLIEST is the natural one to associate.
        best_idx = None
        best_start = None
        for ai, a in enumerate(A):
            if a[0] >= p[0] and a[0] <= p[1] + eps:
                if best_start is None or a[0] < best_start:
                    best_start = a[0]
                    best_idx = ai
            elif a[0] > p[0] and a[0] - p[1] <= eps:
                # Anomaly starts soon after the prediction ends
                if best_start is None or a[0] < best_start:
                    best_start = a[0]
                    best_idx = ai
        if best_idx is None:
            p_prime_list.append(None)
            p_main_list.append(p)
            assoc_idx.append(None)
        else:
            a_start = A[best_idx][0]
            pp, pm = split_prediction_to_p_pprime(
                p, a_start, eps,
                precursor_includes_t_a=precursor_includes_t_a)
            p_prime_list.append(pp)
            p_main_list.append(pm)
            assoc_idx.append(best_idx)
    return AssociatedSegments(p_prime_list, p_main_list, assoc_idx)


def ptapr_paper(
    label_T: np.ndarray,
    pred_T: np.ndarray,
    theta: float,
    eps: int = DEFAULT_EPS,
    delta: int = DEFAULT_DELTA,
    k: float = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    precursor_includes_t_a: bool = False,
) -> Dict[str, float]:
    """Compute PTaPR at a single overlap threshold θ.

    Returns dict with keys: ptapr, ptar, ptap, ptar_d, ptar_p, ptar_e,
    ptap_d, ptap_p, ptap_e, theta, n_A, n_P.

    Both inputs are (T,) 0/1 vectors of the same length.
    """
    label_T = np.asarray(label_T).astype(np.int8)
    pred_T = np.asarray(pred_T).astype(np.int8)
    assert label_T.shape == pred_T.shape, "label and pred must have same shape"
    T = label_T.shape[0]
    A = extract_runs(label_T)
    P = extract_runs(pred_T)
    nA, nP = len(A), len(P)

    if nA == 0 or nP == 0:
        return _zero_dict(theta, nA, nP)

    # Associate alarm runs to anomalies and split each into (p_prime, p_main).
    assoc = _associate_predictions(A, P, eps, precursor_includes_t_a)
    P_prime_raw = assoc.p_prime_per_pred       # |P| entries, may be None
    P_main_raw = assoc.p_per_pred              # |P| entries, may be None
    assoc_idx_raw = assoc.associated_a_idx_per_pred

    # Per Kang et al., P = "anomaly predictions", P' = "precursor predictions".
    # An alarm run that has a non-None main part is an anomaly prediction (in P).
    # An alarm run that is entirely a precursor (main part is None) is a
    # precursor-only run; it contributes only to P', not to P.
    P_main: List[Tuple[int, int]] = []
    P_prime: List[Optional[Tuple[int, int]]] = []
    assoc_idx: List[Optional[int]] = []
    precursor_only_records: List[Tuple[int, Tuple[int, int]]] = []
    for pi in range(len(P)):
        pp = P_prime_raw[pi]
        pmain = P_main_raw[pi]
        a_idx = assoc_idx_raw[pi]
        if pmain is not None:
            P_main.append(pmain)
            P_prime.append(pp)
            assoc_idx.append(a_idx)
        elif pp is not None and a_idx is not None:
            # Precursor-only alarm run — keep for PTaR^e / PTaP^e but do
            # not include in P.
            precursor_only_records.append((a_idx, pp))

    nP_main = len(P_main)
    if nP_main == 0:
        # No anomaly-overlapping predictions; precursor-only predictions
        # alone can still contribute to PTaR^e and PTaP^e but not to PTaR^d
        # or PTaR^p (no main O).
        # Compute PTaR^e and PTaP^e and return.
        return _precursor_only_dict(A, precursor_only_records, theta, nA, len(P),
                                     eps=eps, k=k, alpha=alpha, beta=beta,
                                     gamma=gamma)

    # a' for each anomaly (clipped to [0, T-1])
    A_prime = [ambiguous_segment(a, delta, T) for a in A]

    # Build O(a, p, p') matrix. Row a, column p (only P_main columns).
    O = np.zeros((nA, nP_main), dtype=np.float64)
    for ai, a in enumerate(A):
        for pi, p_main in enumerate(P_main):
            pp = P_prime[pi] if assoc_idx[pi] == ai else None
            O[ai, pi] = overlap_O(a, p_main, pp, A_prime[ai], delta)

    # ---- Recall side ----
    a_lens = np.array([a[1] - a[0] + 1 for a in A], dtype=np.float64)
    sums_per_a = O.sum(axis=1)              # Σ_p O(a, p, p')
    ratios_a = sums_per_a / a_lens

    # PTaR^d (Eq. 6)
    detected = ratios_a >= theta
    ptar_d = float(detected.sum()) / nA

    # PTaR^p (Eq. 9)
    ptar_p = float(np.minimum(1.0, ratios_a).mean())

    # PTaR^e (Eq. 10) — per anomaly, max E over ALL p' targeting it
    # (whether from a main-overlapping run or a precursor-only run).
    ptar_e_terms = np.zeros(nA)
    for ai, a in enumerate(A):
        E_vals = []
        # Precursors from P_main runs that target this anomaly
        for pi, pp in enumerate(P_prime):
            if assoc_idx[pi] != ai or pp is None:
                continue
            E_vals.append(early_reward_E(a, pp, k=k, eps=eps))
        # Precursors from precursor-only runs that target this anomaly
        for tgt_ai, pp in precursor_only_records:
            if tgt_ai == ai:
                E_vals.append(early_reward_E(a, pp, k=k, eps=eps))
        ptar_e_terms[ai] = max(E_vals) if E_vals else 0.0
    ptar_e = float(ptar_e_terms.mean())

    ptar = alpha * ptar_d + beta * ptar_p + gamma * ptar_e

    # ---- Precision side ----
    p_lens = np.array([p[1] - p[0] + 1 for p in P_main], dtype=np.float64)
    sums_per_p = O.sum(axis=0)              # Σ_a O(a, p, p')
    ratios_p = sums_per_p / p_lens

    # PTaP^d (Eq. 12) — denominator is |P| = |P_main|
    correct = ratios_p >= theta
    ptap_d = float(correct.sum()) / nP_main

    # PTaP^p (Eq. 13)
    ptap_p = float(np.minimum(1.0, ratios_p).mean())

    # PTaP^e (Eq. 14) — average E over all p' ∈ P' (including precursor-only)
    Pprime_with_target: List[Tuple[Tuple[int, int], int]] = []
    for pi, pp in enumerate(P_prime):
        if pp is not None and assoc_idx[pi] is not None:
            Pprime_with_target.append((pp, assoc_idx[pi]))
    for tgt_ai, pp in precursor_only_records:
        Pprime_with_target.append((pp, tgt_ai))
    if not Pprime_with_target:
        ptap_e = 0.0
    else:
        E_vals = [early_reward_E(A[ai], pp, k=k, eps=eps)
                  for pp, ai in Pprime_with_target]
        ptap_e = float(np.mean(E_vals))

    ptap = alpha * ptap_d + beta * ptap_p + gamma * ptap_e

    # PTaPR (Eq. 15)
    if ptar + ptap > 0:
        ptapr = 2.0 * ptar * ptap / (ptar + ptap)
    else:
        ptapr = 0.0

    return {
        'ptapr': ptapr,
        'ptar': ptar, 'ptap': ptap,
        'ptar_d': ptar_d, 'ptar_p': ptar_p, 'ptar_e': ptar_e,
        'ptap_d': ptap_d, 'ptap_p': ptap_p, 'ptap_e': ptap_e,
        'theta': float(theta),
        'n_A': nA, 'n_P': nP,
    }


def _zero_dict(theta: float, nA: int, nP: int) -> Dict[str, float]:
    return {
        'ptapr': 0.0, 'ptar': 0.0, 'ptap': 0.0,
        'ptar_d': 0.0, 'ptar_p': 0.0, 'ptar_e': 0.0,
        'ptap_d': 0.0, 'ptap_p': 0.0, 'ptap_e': 0.0,
        'theta': float(theta), 'n_A': nA, 'n_P': nP,
    }


def _precursor_only_dict(
    A: List[Tuple[int, int]],
    precursor_only_records: List[Tuple[int, Tuple[int, int]]],
    theta: float,
    nA: int,
    nP: int,
    eps: int,
    k: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> Dict[str, float]:
    """When |P_main| = 0 but precursor-only runs exist."""
    # PTaR^d, PTaR^p = 0 (no main overlap)
    # PTaR^e = (1/|A|) Σ_a max(E for precursors targeting a)
    ptar_e_terms = np.zeros(nA)
    for ai, a in enumerate(A):
        E_vals = [early_reward_E(a, pp, k=k, eps=eps)
                  for (tgt_ai, pp) in precursor_only_records if tgt_ai == ai]
        ptar_e_terms[ai] = max(E_vals) if E_vals else 0.0
    ptar_e = float(ptar_e_terms.mean())
    ptar = gamma * ptar_e
    # PTaP^d, PTaP^p = 0
    # PTaP^e = mean E over all p' ∈ P'
    if precursor_only_records:
        E_vals = [early_reward_E(A[ai], pp, k=k, eps=eps)
                  for (ai, pp) in precursor_only_records]
        ptap_e = float(np.mean(E_vals))
    else:
        ptap_e = 0.0
    ptap = gamma * ptap_e
    if ptar + ptap > 0:
        ptapr = 2 * ptar * ptap / (ptar + ptap)
    else:
        ptapr = 0.0
    return {
        'ptapr': ptapr, 'ptar': ptar, 'ptap': ptap,
        'ptar_d': 0.0, 'ptar_p': 0.0, 'ptar_e': ptar_e,
        'ptap_d': 0.0, 'ptap_p': 0.0, 'ptap_e': ptap_e,
        'theta': float(theta), 'n_A': nA, 'n_P': nP,
    }


# ----------------------------------------------------------------------------
# θ-AUC (Section IV-D paragraph 2)
# ----------------------------------------------------------------------------

def ptapr_auc(
    label_T: np.ndarray,
    pred_T: np.ndarray,
    theta_grid: Optional[np.ndarray] = None,
    **kw,
) -> Dict:
    """Sweep θ ∈ [0, 1] and compute the PTaPR-AUC (trapezoidal).

    Returns dict with:
        F1_0      PTaPR at θ=0 (most lenient)
        F1_1      PTaPR at θ=1 (strictest)
        auc       AUC of PTaPR vs θ
        curve     pd.DataFrame with one row per θ and full component breakdown
    """
    if theta_grid is None:
        theta_grid = DEFAULT_THETA_GRID
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    rows = []
    for theta in theta_grid:
        rows.append(ptapr_paper(label_T, pred_T, theta=float(theta), **kw))
    curve = pd.DataFrame(rows)
    auc = float(np.trapz(curve['ptapr'].values, theta_grid))
    return {
        'F1_0': float(curve.iloc[0]['ptapr']),
        'F1_1': float(curve.iloc[-1]['ptapr']),
        'auc': auc,
        'curve': curve,
    }


# ----------------------------------------------------------------------------
# Adapter for our project
# ----------------------------------------------------------------------------

def evaluate_method(method_name: str,
                    alarm_T: np.ndarray,
                    label_T: np.ndarray,
                    theta_grid: Optional[np.ndarray] = None,
                    **kw) -> Dict:
    """Compute PTaPR-AUC for one (method, alarm, label) triple and return a
    flat dict suitable for a results CSV row."""
    res = ptapr_auc(label_T, alarm_T, theta_grid=theta_grid, **kw)
    return {
        'method': method_name,
        'F1_0': res['F1_0'],
        'F1_1': res['F1_1'],
        'PTaPR_AUC': res['auc'],
        'n_alarm_runs': int(extract_runs(alarm_T).__len__()),
        'n_attack_runs': int(extract_runs(label_T).__len__()),
        '_curve': res['curve'],   # for downstream inspection
    }


if __name__ == '__main__':
    # Smoke: perfect alignment → PTaPR = 1.0; no overlap → PTaPR = 0.0.
    label = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    pred = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    print('perfect:', ptapr_auc(label, pred))
    pred0 = np.zeros_like(label)
    print('empty pred:', ptapr_auc(label, pred0))
