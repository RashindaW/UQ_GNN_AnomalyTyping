"""Unit tests for scripts/ptapr_metric.py.

Verifies the implementation against:
    * Hand-computed expected outputs for run-extraction and segment helpers.
    * The Fig. 5 worked example from Kang et al. 2026 (with documented gaps).
    * Degenerate cases (empty A, empty P, perfect alignment).
    * Monotonicity properties (PTaR^d non-increasing in θ).

Run with:
    /home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python -m pytest \
        scripts/test_ptapr_metric.py -v
or:
    /home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python \
        scripts/test_ptapr_metric.py
to use the built-in self-test driver at the bottom of the file.
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

from ptapr_metric import (
    extract_runs,
    precursor_window,
    ambiguous_segment,
    overlap_count,
    ambiguous_score_S,
    early_reward_E,
    overlap_O,
    split_prediction_to_p_pprime,
    ptapr_paper,
    ptapr_auc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _passed(name: str) -> None:
    print(f"  PASS  {name}")


def _failed(name: str, expected, got) -> None:
    print(f"  FAIL  {name}: expected {expected!r}, got {got!r}")


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def test_extract_runs() -> bool:
    name = "extract_runs"
    cases = [
        ([0, 0, 0], []),
        ([1, 0, 0], [(0, 0)]),
        ([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], [(1, 2), (4, 4), (7, 9)]),
        ([1, 1, 1], [(0, 2)]),
    ]
    for inp, expected in cases:
        got = extract_runs(np.array(inp))
        if got != expected:
            _failed(name, expected, got)
            return False
    _passed(name)
    return True


def test_precursor_window() -> bool:
    name = "precursor_window"
    assert precursor_window(14, 6) == (8, 13)
    assert precursor_window(3, 6) == (0, 2)
    assert precursor_window(0, 6) == (0, -1)  # empty (lo > hi)
    _passed(name)
    return True


def test_ambiguous_segment() -> bool:
    name = "ambiguous_segment"
    assert ambiguous_segment((14, 16), 4, 100) == (17, 20)
    assert ambiguous_segment((97, 99), 4, 100) == (None)
    assert ambiguous_segment((90, 95), 4, 100) == (96, 99)
    assert ambiguous_segment((10, 10), 0, 100) is None
    _passed(name)
    return True


def test_overlap_count() -> bool:
    name = "overlap_count"
    assert overlap_count((10, 15), (13, 20)) == 3   # {13,14,15}
    assert overlap_count((0, 4), (5, 9)) == 0
    assert overlap_count((10, 12), (10, 12)) == 3
    assert overlap_count(None, (1, 2)) == 0
    assert overlap_count((1, 2), None) == 0
    _passed(name)
    return True


def test_ambiguous_score_S_endpoints() -> bool:
    name = "ambiguous_score_S endpoints"
    # delta=4 → i' values for indices [t_a' + 0..3] are -4, 0, 4, 8.
    # 1/(1+e^{-4}) ≈ 0.9820, 1/(1+e^{0}) = 0.5, 1/(1+e^{4}) ≈ 0.0180,
    # 1/(1+e^{8}) ≈ 0.000335.
    a_prime = (17, 20)   # length 4 starting at 17
    p_full = (17, 20)
    s_full = ambiguous_score_S(a_prime, p_full, delta=4)
    expected = (1/(1+math.exp(-4)) + 1/(1+math.exp(0))
                + 1/(1+math.exp(4)) + 1/(1+math.exp(8)))
    if not _approx(s_full, expected, 1e-8):
        _failed(name + " (full overlap)", expected, s_full)
        return False
    # Only first two indices of a'
    s_first_two = ambiguous_score_S(a_prime, (17, 18), delta=4)
    expected2 = 1/(1+math.exp(-4)) + 1/(1+math.exp(0))
    if not _approx(s_first_two, expected2, 1e-8):
        _failed(name + " (first two)", expected2, s_first_two)
        return False
    # Empty / None
    assert ambiguous_score_S(None, (10, 20), 4) == 0.0
    assert ambiguous_score_S((10, 14), None, 4) == 0.0
    _passed(name)
    return True


def test_early_reward_E_fig5_a1() -> bool:
    """Fig. 5 stated: E(a_1, p'_1) = exp(-0.05·(1-6)²) ≈ 0.287."""
    name = "early_reward_E Fig5 a1"
    # If p_prime ends at t_a - 1 (gap of 1), then i' = 1, and
    # E = exp(-0.05·(1-6)²) = exp(-1.25) ≈ 0.286505.
    E = early_reward_E((14, 16), (8, 13), k=0.05, eps=6)
    expected = math.exp(-0.05 * 25)
    if not _approx(E, expected, 1e-8):
        _failed(name, expected, E)
        return False
    # If p_prime has no element < t_a, E = 0.
    E_no = early_reward_E((14, 16), (15, 17), k=0.05, eps=6)
    if E_no != 0.0:
        _failed(name + " (no prior)", 0.0, E_no)
        return False
    # If p_prime is None
    if early_reward_E((14, 16), None, k=0.05, eps=6) != 0.0:
        _failed(name + " (None)", 0.0, "non-zero")
        return False
    _passed(name)
    return True


def test_early_reward_E_optimal_lead() -> bool:
    """E peaks at i' = eps and decays for shorter/longer leads."""
    name = "early_reward_E peak at i'=eps"
    a = (10, 12)
    # At i = 10 - eps = 4, i' = 6 = eps. Need p_prime ending at 4.
    E_opt = early_reward_E(a, (3, 4), k=0.05, eps=6)
    # E_opt = exp(-0.05·(6-6)²) = 1.0
    if not _approx(E_opt, 1.0, 1e-8):
        _failed(name, 1.0, E_opt)
        return False
    # At i = 9 (immediately before), i' = 1, E = exp(-1.25)
    E_close = early_reward_E(a, (8, 9), k=0.05, eps=6)
    if not _approx(E_close, math.exp(-1.25), 1e-8):
        _failed(name + " (close)", math.exp(-1.25), E_close)
        return False
    # Far precursor: i = 0, i' = 10, E = exp(-0.05·16) = exp(-0.8)
    E_far = early_reward_E(a, (0, 0), k=0.05, eps=6)
    if not _approx(E_far, math.exp(-0.8), 1e-8):
        _failed(name + " (far)", math.exp(-0.8), E_far)
        return False
    _passed(name)
    return True


# ---------------------------------------------------------------------------
# PTaPR — degenerate cases (must hold)
# ---------------------------------------------------------------------------

def test_perfect_alignment_gives_two_thirds() -> bool:
    """label == pred → PTaPR = 2/3, not 1.0.

    Per Eq. 5, PTaR = α·PTaR^d + β·PTaR^p + γ·PTaR^e with α=β=γ=1/3.
    When the prediction exactly covers each anomaly with no precursor
    portion, PTaR^d = PTaR^p = 1 but PTaR^e = 0 (no early predictions).
    So PTaR = 2/3, PTaP = 2/3, and PTaPR = 2·(2/3)²/(4/3) = 2/3.

    This is by design: PTaPR rewards EARLY detection as a third component,
    and a detector that fires exactly at anomaly onset gets only 2/3 credit.
    """
    name = "perfect_alignment → 2/3 (early reward missing)"
    label = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    pred = label.copy()
    res = ptapr_paper(label, pred, theta=0.0)
    expected = 2.0 / 3.0
    if not _approx(res['ptapr'], expected, 1e-9):
        _failed(name, expected, res['ptapr'])
        return False
    # PTaR^d, PTaR^p should be 1 (perfect detection + coverage)
    if not _approx(res['ptar_d'], 1.0, 1e-9):
        _failed(name + " (ptar_d)", 1.0, res['ptar_d'])
        return False
    if not _approx(res['ptar_p'], 1.0, 1e-9):
        _failed(name + " (ptar_p)", 1.0, res['ptar_p'])
        return False
    # PTaR^e should be 0 (no precursor portion in any prediction)
    if not _approx(res['ptar_e'], 0.0, 1e-9):
        _failed(name + " (ptar_e)", 0.0, res['ptar_e'])
        return False
    _passed(name)
    return True


def test_optimally_early_prediction() -> bool:
    """Predictions emitting a single precursor at t_a - eps AND the anomaly.

    Under the rightmost-index convention for E (matching Fig. 5):
    i' = t_a - rightmost(p_prime). For i' = eps, the precursor must END
    at exactly t_a - eps. So we use a single-timestep precursor at
    t_a - eps, then the main prediction during the anomaly.
    """
    name = "optimally_early_prediction → ~1.0"
    T = 60
    label = np.zeros(T, dtype=np.int8)
    label[20:25] = 1
    label[40:45] = 1
    eps = 6
    pred = np.zeros(T, dtype=np.int8)
    # Single-step precursor at t_a - eps, then gap, then main prediction.
    pred[20 - eps] = 1     # precursor at t=14 (one step), then gap [15,19]
    pred[20:25] = 1        # main prediction during anomaly 1
    pred[40 - eps] = 1     # precursor at t=34
    pred[40:45] = 1        # main prediction during anomaly 2
    res = ptapr_paper(label, pred, theta=0.0,
                      eps=eps, delta=4, k=0.05)
    # E for each precursor: i'=eps → exp(0) = 1.0
    if not _approx(res['ptar_e'], 1.0, 1e-6):
        _failed(name + " (ptar_e)", 1.0, res['ptar_e'])
        return False
    if not _approx(res['ptar'], 1.0, 1e-6):
        _failed(name + " (ptar)", 1.0, res['ptar'])
        return False
    if res['ptapr'] < 0.9:
        _failed(name + " (ptapr ≥ 0.9)", "≥ 0.9", res['ptapr'])
        return False
    _passed(name)
    return True


def test_empty_pred() -> bool:
    name = "empty_pred"
    label = np.array([0, 1, 1, 0])
    pred = np.zeros_like(label)
    res = ptapr_paper(label, pred, theta=0.0)
    if not _approx(res['ptapr'], 0.0, 1e-9):
        _failed(name, 0.0, res['ptapr'])
        return False
    _passed(name)
    return True


def test_empty_label() -> bool:
    name = "empty_label"
    label = np.zeros(10, dtype=np.int8)
    pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    res = ptapr_paper(label, pred, theta=0.0)
    if not _approx(res['ptapr'], 0.0, 1e-9):
        _failed(name, 0.0, res['ptapr'])
        return False
    _passed(name)
    return True


def test_theta_monotone_d() -> bool:
    """PTaR^d is non-increasing in θ for a fixed alarm vector."""
    name = "PTaR^d monotone in θ"
    rng = np.random.default_rng(0)
    T = 200
    label = np.zeros(T, dtype=np.int8)
    label[40:55] = 1
    label[110:130] = 1
    label[170:180] = 1
    pred = (rng.random(T) > 0.85).astype(np.int8)
    last = 1.0
    for theta in np.linspace(0.0, 1.0, 11):
        res = ptapr_paper(label, pred, theta=theta)
        if res['ptar_d'] > last + 1e-9:
            _failed(name, f"≤ {last}", res['ptar_d'])
            return False
        last = res['ptar_d']
    _passed(name)
    return True


# ---------------------------------------------------------------------------
# Fig. 5 worked example (Kang et al. 2026 page 6).
#
# Reconstruction (best interpretation faithful to the equations):
#   a_1 = [14, 16]  (length 3, paper's "anomaly 1" centered around index ~15)
#   a_2 = [23, 26]  (length 4)
#   p_1 = [15, 16]  (length 2, predicts during anomaly 1)
#   p'_1 = [8, 13]  (precursor window for a_1, eps=6 → [8, 13])
#         But |a_1 ∩ p'_1| = 0 under strict convention.
#   p_2 = [23, 28]  (length 6, extends past a_2 by 2 timesteps)
#   p'_2 = None     (p_2 starts AT a_2, no precursor portion)
#
# The paper's Fig. 5 captions claim |a_1 ∩ p'_1| = 1 (under "permissive"
# convention where p' can extend into a). Our default is "strict"; turn on
# precursor_includes_t_a=True to match the figure's |a∩p'|=1 claim.
# ---------------------------------------------------------------------------

def test_fig5_S_a2() -> bool:
    """S(a_2', p_2) for the Fig. 5 a_2 with p_2 extending past a_2 by 2.

    a_2 = [23, 26], a_2' = [27, 30] (delta=4), p_2 = [23, 28].
    a_2' ∩ p_2 = [27, 28]. With delta=4:
        i'(27) = -4 + 12·0/3 = -4
        i'(28) = -4 + 12·1/3 = 0
    S = 1/(1+e^-4) + 1/(1+e^0) ≈ 0.9820 + 0.5 = 1.4820.

    Paper claims: O(a_2, p_2) = 3 + 1 + (1/(1+e^1) + 1/(1+e^0)) ≈ 5.88,
    implying S ≈ 0.769 with sigmoid args 1, 0. This is INCONSISTENT with
    Eq. 8's formula. We document this and assert our value (1.4820).
    """
    name = "Fig5 S(a_2', p_2) per Eq. 8"
    s = ambiguous_score_S((27, 30), (23, 28), delta=4)
    expected = 1/(1+math.exp(-4)) + 1/(1+math.exp(0))
    if not _approx(s, expected, 1e-6):
        _failed(name, expected, s)
        return False
    _passed(name)
    return True


def test_fig5_E_a1() -> bool:
    """E(a_1, p'_1) where p'_1 has its right endpoint at t_a - 1.
    Paper claim: E ≈ 0.287 = exp(-1.25)."""
    name = "Fig5 E(a_1, p'_1)"
    # Use precursor segment ending at 13 (= t_a - 1 where t_a = 14).
    E = early_reward_E((14, 16), (8, 13), k=0.05, eps=6)
    if not _approx(E, math.exp(-1.25), 1e-6):
        _failed(name, math.exp(-1.25), E)
        return False
    _passed(name)
    return True


def test_fig5_full_strict() -> bool:
    """Full Fig. 5 PTaPR computation under the STRICT convention.

    Constructs the label and prediction vectors consistent with Fig. 5
    intervals, then computes PTaPR. Under strict convention |a∩p'| = 0,
    so the numbers will DIFFER from the paper's stated values in Fig. 5.
    We assert against equation-faithful expected values, not the figure's
    stated numbers.
    """
    name = "Fig5 PTaPR (strict)"
    T = 40
    label = np.zeros(T, dtype=np.int8)
    label[14:17] = 1   # a_1 = [14, 16]
    label[23:27] = 1   # a_2 = [23, 26]
    pred = np.zeros(T, dtype=np.int8)
    # p_1: extends from precursor through anomaly. Make it cover [10, 16]
    # so the precursor part p'_1 = [10, 13] and main part p_1 = [14, 16].
    pred[10:17] = 1
    # p_2: from a_2 onwards plus 2 ambiguous timesteps
    pred[23:29] = 1

    res = ptapr_paper(label, pred, theta=0.5,
                      eps=6, delta=4, k=0.05,
                      precursor_includes_t_a=False)

    # Under strict convention:
    #   For a_1: O(a_1, p_1) = |a_1 ∩ p'_1| + |a_1 ∩ p_main| + S(a_1', p_main)
    #          = 0 + 3 + S([17,20], [14,16], delta=4) = 0 + 3 + 0 = 3
    #   For a_2: O(a_2, p_2) = 0 + 4 + S([27,30], [23,28], 4)
    #          = 0 + 4 + (1/(1+e^-4) + 1/(1+e^0)) ≈ 5.482
    # ratios_a = [3/3, 5.482/4] = [1.0, 1.371]
    # PTaR^d at θ=0.5 = both ≥ 0.5 → 1.0
    # PTaR^p = (min(1, 1.0) + min(1, 1.371)) / 2 = (1+1)/2 = 1.0
    # PTaR^e: For a_1 the associated p has p'=[10,13], E(a_1, (10,13))
    #         i=13, i'=1, E=exp(-1.25)≈0.2865.
    #         For a_2: p_2 starts AT 23 = t_a, so no precursor; E=0.
    #         PTaR^e = (0.2865 + 0) / 2 ≈ 0.143
    # PTaR ≈ (1/3)·(1.0 + 1.0 + 0.143) ≈ 0.714
    expected_ptar_d = 1.0
    expected_ptar_p = 1.0
    expected_ptar_e = math.exp(-1.25) / 2.0
    expected_ptar = (expected_ptar_d + expected_ptar_p + expected_ptar_e) / 3.0
    if not (_approx(res['ptar_d'], expected_ptar_d, 1e-6) and
            _approx(res['ptar_p'], expected_ptar_p, 1e-6) and
            _approx(res['ptar_e'], expected_ptar_e, 1e-6)):
        _failed(name + " PTaR components",
                f"({expected_ptar_d}, {expected_ptar_p}, {expected_ptar_e:.4f})",
                f"({res['ptar_d']:.4f}, {res['ptar_p']:.4f}, {res['ptar_e']:.4f})")
        return False
    if not _approx(res['ptar'], expected_ptar, 1e-6):
        _failed(name + " PTaR", expected_ptar, res['ptar'])
        return False
    _passed(name)
    return True


# ---------------------------------------------------------------------------
# AUC sanity
# ---------------------------------------------------------------------------

def test_auc_keys() -> bool:
    name = "ptapr_auc keys + AUC shape"
    label = np.zeros(60, dtype=np.int8)
    label[20:30] = 1
    pred = np.zeros(60, dtype=np.int8)
    pred[19:31] = 1
    res = ptapr_auc(label, pred)
    for k in ('F1_0', 'F1_1', 'auc', 'curve'):
        if k not in res:
            _failed(name + f" (missing {k})", "present", "missing")
            return False
    if not (0.0 <= res['auc'] <= 1.0):
        _failed(name, "0 ≤ AUC ≤ 1", res['auc'])
        return False
    _passed(name)
    return True


def test_auc_perfect_alignment_is_two_thirds() -> bool:
    """Perfect alignment (label==pred) → F1_0 = 2/3, F1_1 = 2/3, AUC = 2/3.
    See test_perfect_alignment_gives_two_thirds for the explanation."""
    name = "AUC perfect alignment → 2/3"
    label = np.zeros(60, dtype=np.int8)
    label[20:30] = 1
    label[40:50] = 1
    pred = label.copy()
    res = ptapr_auc(label, pred)
    expected = 2.0 / 3.0
    if not _approx(res['F1_0'], expected, 1e-6):
        _failed(name + " F1_0", expected, res['F1_0'])
        return False
    if not _approx(res['auc'], expected, 1e-6):
        _failed(name + " AUC", expected, res['auc'])
        return False
    _passed(name)
    return True


def test_auc_optimally_early() -> bool:
    """Optimally-early-prediction case → AUC near 1.0.

    Uses the single-step-precursor pattern: an isolated 1 at t_a - eps,
    plus the main prediction during the anomaly."""
    name = "AUC optimally-early prediction → ~1.0"
    T = 60
    label = np.zeros(T, dtype=np.int8)
    label[20:25] = 1
    label[40:45] = 1
    eps = 6
    pred = np.zeros(T, dtype=np.int8)
    pred[20 - eps] = 1     # precursor at t=14
    pred[20:25] = 1        # main
    pred[40 - eps] = 1     # precursor at t=34
    pred[40:45] = 1        # main
    res = ptapr_auc(label, pred, eps=eps)
    if res['auc'] < 0.9:
        _failed(name, "≥ 0.9", res['auc'])
        return False
    _passed(name)
    return True


# ---------------------------------------------------------------------------
# Self-test driver
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_extract_runs,
        test_precursor_window,
        test_ambiguous_segment,
        test_overlap_count,
        test_ambiguous_score_S_endpoints,
        test_early_reward_E_fig5_a1,
        test_early_reward_E_optimal_lead,
        test_perfect_alignment_gives_two_thirds,
        test_optimally_early_prediction,
        test_empty_pred,
        test_empty_label,
        test_theta_monotone_d,
        test_fig5_S_a2,
        test_fig5_E_a1,
        test_fig5_full_strict,
        test_auc_keys,
        test_auc_perfect_alignment_is_two_thirds,
        test_auc_optimally_early,
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
