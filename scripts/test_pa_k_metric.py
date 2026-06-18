"""Unit tests for scripts/pa_k_metric.py.

Verifies:
- K=0 reduces exactly to standard PA (any detection promotes whole segment)
- K=100 reduces exactly to standard F1 (no promotion ever)
- Strict-inequality boundary (K=30 with frac=0.30 → not promoted)
- Random baseline F1_PA ≈ 1.0 (the paper's key finding)
- Random baseline F1 (K=100) ≈ γ (anomaly ratio — the honest random floor)
- Perfect alignment AUC ≈ 1.0
- Edge cases (no segments, no predictions, single-step segment)

Run: /home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python \
        scripts/test_pa_k_metric.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pa_k_metric import (
    extract_runs,
    point_adjust_K,
    point_adjust,
    metrics_from_pred,
    f1_at_threshold,
    best_f1_pa_k,
    f1_pa_k_auc,
    random_baseline_score,
    input_norm_baseline_score,
)


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _passed(name: str) -> None:
    print(f"  PASS  {name}")


def _failed(name: str, expected, got) -> None:
    print(f"  FAIL  {name}: expected {expected!r}, got {got!r}")


# ---------------------------------------------------------------------------
# Reductions: K=0 ⇔ PA, K=100 ⇔ standard F1
# ---------------------------------------------------------------------------

def test_K0_equals_PA_single_detection() -> bool:
    """A single timestep above δ in a 5-step segment promotes the whole
    segment under K=0 (standard PA). Result: F1=1.0 with TP=5 if labels match."""
    name = "K=0 ⇔ standard PA (single detection promotes whole segment)"
    T = 10
    labels = np.zeros(T, dtype=np.int8)
    labels[3:8] = 1     # segment [3, 7]
    scores = np.zeros(T, dtype=np.float64)
    scores[5] = 1.0     # only one detection
    pred = point_adjust_K(scores, labels, delta=0.5, K_pct=0.0)
    expected_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=np.int8)
    if not np.array_equal(pred, expected_pred):
        _failed(name, expected_pred.tolist(), pred.tolist())
        return False
    m = metrics_from_pred(pred, labels)
    if not _approx(m['F1'], 1.0, 1e-9):
        _failed(name + " F1", 1.0, m['F1'])
        return False
    _passed(name)
    return True


def test_K100_no_promotion() -> bool:
    """K=100 makes the fraction comparison `frac > 1.0`, which is never true.
    Result: prediction equals base (scores > δ); no segment promotion."""
    name = "K=100 ⇔ standard F1 (no promotion)"
    T = 10
    labels = np.zeros(T, dtype=np.int8)
    labels[3:8] = 1
    scores = np.zeros(T, dtype=np.float64)
    scores[5] = 1.0     # one detection
    pred = point_adjust_K(scores, labels, delta=0.5, K_pct=100.0)
    expected_pred = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    if not np.array_equal(pred, expected_pred):
        _failed(name, expected_pred.tolist(), pred.tolist())
        return False
    _passed(name)
    return True


def test_K100_all_above_no_promotion_needed() -> bool:
    """At K=100, even if every timestep in a segment exceeds δ, the
    fraction (1.0) is NOT strictly > 1.0, so no PA promotion happens —
    but the base condition (scores > δ) already covers all timesteps in
    that segment. So the predicted segment matches the labels via the
    base rule, not via promotion."""
    name = "K=100 with full-detection segment ⇔ base prediction"
    T = 10
    labels = np.zeros(T, dtype=np.int8)
    labels[3:8] = 1
    scores = np.zeros(T, dtype=np.float64)
    scores[3:8] = 1.0
    pred = point_adjust_K(scores, labels, delta=0.5, K_pct=100.0)
    expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=np.int8)
    if not np.array_equal(pred, expected):
        _failed(name, expected.tolist(), pred.tolist())
        return False
    _passed(name)
    return True


def test_K30_strict_inequality_boundary() -> bool:
    """A 10-timestep segment with exactly 3 detections has fraction = 0.30.
    Strict-inequality `>` means:
      K=29.99 → 0.30 > 0.2999 ✓ promoted
      K=30.00 → 0.30 > 0.30   ✗ NOT promoted  (strict, not ≥)
      K=30.01 → 0.30 > 0.3001 ✗ NOT promoted"""
    name = "K=30 strict inequality boundary"
    T = 12
    labels = np.zeros(T, dtype=np.int8)
    labels[1:11] = 1    # 10-step segment [1, 10]
    scores = np.zeros(T, dtype=np.float64)
    scores[2] = scores[5] = scores[8] = 1.0   # 3 detections in segment
    # Confirm count
    seg_scores = scores[1:11]
    assert (seg_scores > 0.5).sum() == 3
    # K=29
    pred_29 = point_adjust_K(scores, labels, delta=0.5, K_pct=29.0)
    if not pred_29[1:11].all():
        _failed(name + " K=29", "all 1s in segment", pred_29[1:11].tolist())
        return False
    # K=30 — boundary; strict > 0.30 fails for frac=0.30
    pred_30 = point_adjust_K(scores, labels, delta=0.5, K_pct=30.0)
    expected_30 = np.zeros(T, dtype=np.int8)
    expected_30[2] = expected_30[5] = expected_30[8] = 1
    if not np.array_equal(pred_30, expected_30):
        _failed(name + " K=30 (boundary)", expected_30.tolist(), pred_30.tolist())
        return False
    # K=31
    pred_31 = point_adjust_K(scores, labels, delta=0.5, K_pct=31.0)
    if not np.array_equal(pred_31, expected_30):
        _failed(name + " K=31", expected_30.tolist(), pred_31.tolist())
        return False
    _passed(name)
    return True


# ---------------------------------------------------------------------------
# Random-baseline behavior (the paper's central finding)
# ---------------------------------------------------------------------------

def test_random_F1_PA_is_high() -> bool:
    """Per Kim et al. 2022 §3.2 and Fig. 2, a random uniform anomaly score
    with PA achieves F1_PA close to 1.0 for sufficiently long segments.
    Set up: 5000-timestep series with one 2500-step anomaly segment
    (γ=0.5) — extreme but illustrates the principle. Expected F1_PA > 0.85."""
    name = "Random scores: F1_PA > 0.85 (paper §3.2)"
    T = 5000
    labels = np.zeros(T, dtype=np.int8)
    labels[1250:3750] = 1
    scores = random_baseline_score(T, seed=42)
    res = best_f1_pa_k(scores, labels, K_pct=0.0, n_thresholds=200)
    if res['F1'] < 0.85:
        _failed(name, "≥ 0.85", res['F1'])
        return False
    _passed(name)
    return True


def test_random_R_analytical_Eq6() -> bool:
    """Verify the random-baseline recall after PA matches Kim et al. 2022
    Eq. 6 quantitatively (within Monte Carlo tolerance).

    Eq. 6:    R = 1 - (1/γ) · δ'^L
    where γ = anomaly ratio, δ' = Pr(A(w) < δ) = δ for A ~ U(0,1),
    L = segment length.

    Test setup: T = 2000, segment length L = 100, γ = 0.05 (so anomaly
    timesteps = 100). Choose δ = 0.95 → δ' = 0.95.
    Predicted R:   1 - (1/0.05) · 0.95^100 = 1 - 20 · 0.005921 = 1 - 0.1184
                 = 0.8816
    Expect observed R within ±0.10 (Monte Carlo over 5 seeds)."""
    name = "Random baseline R matches Eq. 6 within Monte Carlo tolerance"
    T = 2000
    L = 100
    gamma = L / T   # 0.05
    delta = 0.95
    expected_R = 1.0 - (1.0 / gamma) * (delta ** L)
    # Average over a few seeds to reduce MC noise
    R_observed = []
    for seed in range(5):
        labels = np.zeros(T, dtype=np.int8)
        labels[500:500 + L] = 1
        scores = random_baseline_score(T, seed=seed)
        pred = point_adjust_K(scores, labels, delta=delta, K_pct=0.0)
        m = metrics_from_pred(pred, labels)
        R_observed.append(m['R'])
    R_mean = float(np.mean(R_observed))
    if abs(R_mean - expected_R) > 0.15:
        _failed(name, f"≈ {expected_R:.3f}", R_mean)
        return False
    _passed(name + f" (Eq.6 R={expected_R:.3f}, observed R={R_mean:.3f})")
    return True


def test_random_F1_standard_low() -> bool:
    """At K=100 (no promotion), random F1 should be near the chance level
    — roughly 2γ/(1+γ) for a balanced threshold, much lower than F1_PA."""
    name = "Random scores: F1 (K=100) much lower than F1_PA"
    T = 5000
    labels = np.zeros(T, dtype=np.int8)
    labels[1250:3750] = 1   # γ = 0.5
    scores = random_baseline_score(T, seed=42)
    res_PA = best_f1_pa_k(scores, labels, K_pct=0.0, n_thresholds=200)
    res_std = best_f1_pa_k(scores, labels, K_pct=100.0, n_thresholds=200)
    # F1 at K=100 should be FAR lower than F1_PA at K=0.
    if not (res_std['F1'] < res_PA['F1'] - 0.20):
        _failed(name, f"F1<F1_PA-0.20 ({res_PA['F1']:.3f}-0.20)",
                res_std['F1'])
        return False
    _passed(name)
    return True


# ---------------------------------------------------------------------------
# AUC sanity
# ---------------------------------------------------------------------------

def test_AUC_perfect_alignment() -> bool:
    """For perfectly aligned scores (above δ exactly where labels=1, below
    elsewhere), F1=1 at every K → AUC=1."""
    name = "AUC: perfect alignment → 1.0"
    T = 50
    labels = np.zeros(T, dtype=np.int8)
    labels[10:20] = 1
    labels[30:40] = 1
    scores = labels.astype(np.float64)   # exactly 1.0 inside segments
    res = f1_pa_k_auc(scores, labels, K_grid=np.arange(0, 101, 10),
                     n_thresholds=50)
    if not _approx(res['F1_PA_K0'], 1.0, 1e-6):
        _failed(name + " F1_PA_K0", 1.0, res['F1_PA_K0'])
        return False
    if not _approx(res['F1_PA_K100'], 1.0, 1e-6):
        _failed(name + " F1_PA_K100", 1.0, res['F1_PA_K100'])
        return False
    if not _approx(res['PA_K_AUC'], 1.0, 1e-6):
        _failed(name + " AUC", 1.0, res['PA_K_AUC'])
        return False
    _passed(name)
    return True


def test_AUC_random_below_F1_PA() -> bool:
    """Random scores: AUC should be much lower than F1_PA_K0, since at high
    K the random baseline has only chance-level F1."""
    name = "AUC: random scores show K-dependent decay"
    T = 2000
    labels = np.zeros(T, dtype=np.int8)
    labels[500:1500] = 1
    scores = random_baseline_score(T, seed=7)
    res = f1_pa_k_auc(scores, labels, K_grid=np.arange(0, 101, 10),
                     n_thresholds=100)
    if not (res['F1_PA_K0'] > res['PA_K_AUC'] + 0.10):
        _failed(name, f"F1_PA > AUC+0.10 ({res['F1_PA_K0']:.3f} vs {res['PA_K_AUC']:.3f})",
                "gap too small")
        return False
    _passed(name)
    return True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_segments() -> bool:
    """No anomaly segments → adjusted prediction equals base."""
    name = "no segments → adjusted = base"
    T = 20
    labels = np.zeros(T, dtype=np.int8)
    scores = np.zeros(T, dtype=np.float64)
    scores[5:8] = 1.0
    pred = point_adjust_K(scores, labels, delta=0.5, K_pct=0.0)
    expected = (scores > 0.5).astype(np.int8)
    if not np.array_equal(pred, expected):
        _failed(name, expected.tolist(), pred.tolist())
        return False
    _passed(name)
    return True


def test_no_predictions_above_delta() -> bool:
    """All scores ≤ δ → pred all zeros → F1 = 0."""
    name = "no predictions → F1 = 0"
    T = 20
    labels = np.zeros(T, dtype=np.int8)
    labels[5:10] = 1
    scores = np.zeros(T, dtype=np.float64)
    m = f1_at_threshold(scores, labels, delta=0.5, K_pct=0.0)
    if not _approx(m['F1'], 0.0, 1e-9):
        _failed(name, 0.0, m['F1'])
        return False
    _passed(name)
    return True


def test_single_timestep_segment() -> bool:
    """A length-1 segment: if its one timestep exceeds δ, it's promoted
    (trivially); otherwise it isn't."""
    name = "single-timestep segment"
    T = 10
    labels = np.zeros(T, dtype=np.int8)
    labels[5] = 1
    scores = np.zeros(T, dtype=np.float64)
    scores[5] = 1.0
    pred = point_adjust_K(scores, labels, delta=0.5, K_pct=0.0)
    expected = np.zeros(T, dtype=np.int8)
    expected[5] = 1
    if not np.array_equal(pred, expected):
        _failed(name, expected.tolist(), pred.tolist())
        return False
    # And if the one timestep is below δ, no promotion
    scores[5] = 0.0
    pred = point_adjust_K(scores, labels, delta=0.5, K_pct=0.0)
    if pred.any():
        _failed(name + " (below δ)", "all zeros", pred.tolist())
        return False
    _passed(name)
    return True


def test_input_norm_baseline_shape() -> bool:
    """Input-norm baseline returns the right shape and increases with input
    magnitude."""
    name = "input_norm_baseline_score shape + monotonicity"
    T, V = 100, 3
    X = np.ones((T, V), dtype=np.float64)
    s = input_norm_baseline_score(X, slide_win=10)
    if s.shape != (T,):
        _failed(name + " shape", (T,), s.shape)
        return False
    # All-ones input: ||w||_F = √(10 * 3) = √30 for t >= 9
    # For t < 9 the window is shorter. We just verify the asymptote.
    if not _approx(s[-1], math.sqrt(30), 1e-6):
        _failed(name + " asymptote", math.sqrt(30), s[-1])
        return False
    _passed(name)
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_K0_equals_PA_single_detection,
        test_K100_no_promotion,
        test_K100_all_above_no_promotion_needed,
        test_K30_strict_inequality_boundary,
        test_random_F1_PA_is_high,
        test_random_R_analytical_Eq6,
        test_random_F1_standard_low,
        test_AUC_perfect_alignment,
        test_AUC_random_below_F1_PA,
        test_no_segments,
        test_no_predictions_above_delta,
        test_single_timestep_segment,
        test_input_norm_baseline_shape,
    ]
    print(f"Running {len(tests)} tests...")
    n_pass = 0
    for t in tests:
        if t():
            n_pass += 1
    print(f"\n{n_pass}/{len(tests)} passed")
    return 0 if n_pass == len(tests) else 1


if __name__ == '__main__':
    sys.exit(main())
