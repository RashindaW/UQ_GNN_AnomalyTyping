"""Unit tests for competitors/common/extra_metrics.py (Track-B extended metrics).

Run:
    $PY competitors/common/test_extra_metrics.py

Prints a per-test PASS/FAIL line and a final summary. Exit code 0 iff all pass.
No pytest dependency -- pure asserts so it runs under the cached CPU env.

Note on VUS over a finite buffer: even a perfectly separable score gives a VUS
strictly below 1, because event dilation injects nominal-scored timesteps into
the positive class (they then rank like negatives). This is a documented
property of the Paparrizos VUS, NOT a bug. The tests therefore assert "clearly
high" for VUS at a wide buffer and verify the exact reductions separately:
VUS(window=0) == plain AUC, and VUS-ROC at the smallest buffer ~ 1.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import extra_metrics as em  # noqa: E402

_PASS = 0
_FAIL = 0
_FAILED = []


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print("PASS  %s" % name)
    else:
        _FAIL += 1
        _FAILED.append(name)
        print("FAIL  %s  %s" % (name, detail))


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_separable(n=400, seed=0):
    """A score that is perfectly separable from a block-structured label."""
    rng = np.random.default_rng(seed)
    label = np.zeros(n, dtype=int)
    label[100:130] = 1   # event 1
    label[300:310] = 1   # event 2
    score = rng.uniform(0.0, 0.4, size=n)
    score[label == 1] = rng.uniform(0.6, 1.0, size=int(label.sum()))
    return score, label


# ---------------------------------------------------------------------------
# 1. Perfect score -> AUPRC/AUROC == 1 ; VUS clearly high (+ exact reductions)
# ---------------------------------------------------------------------------
def test_perfect():
    score, label = make_separable()
    check("perfect_AUPRC~1", approx(em.auprc(score, label), 1.0, 1e-9),
          "got %r" % em.auprc(score, label))
    check("perfect_AUROC~1", approx(em.auroc(score, label), 1.0, 1e-9),
          "got %r" % em.auroc(score, label))
    vr = em.vus_roc(score, label, window=10)
    vp = em.vus_pr(score, label, window=10)
    check("perfect_VUS_ROC_high", vr > 0.80, "got %r" % vr)
    check("perfect_VUS_PR_high", vp > 0.50, "got %r" % vp)
    # smallest non-trivial buffer must still be ~1 for a clean score
    vr1 = em.vus_roc(score, label, window=1)
    check("perfect_VUS_ROC_w1_near1", vr1 > 0.95, "got %r" % vr1)


# ---------------------------------------------------------------------------
# 2. Random / constant -> AUPRC ~ base rate, AUROC ~ 0.5
# ---------------------------------------------------------------------------
def test_random_baseline():
    rng = np.random.default_rng(7)
    n = 20000
    rate = 0.12
    label = (rng.uniform(size=n) < rate).astype(int)
    score = rng.uniform(size=n)  # independent of label
    ap = em.auprc(score, label)
    br = float(label.mean())
    check("random_AUPRC~base_rate", approx(ap, br, 0.02),
          "ap=%.4f base=%.4f" % (ap, br))
    roc = em.auroc(score, label)
    check("random_AUROC~0.5", approx(roc, 0.5, 0.03), "got %.4f" % roc)
    const = np.zeros(n)
    apc = em.auprc(const, label)
    check("constant_AUPRC==base_rate", approx(apc, br, 1e-6),
          "apc=%.6f base=%.6f" % (apc, br))


# ---------------------------------------------------------------------------
# 3. Hand-built 2-event affiliation example with a known answer
# ---------------------------------------------------------------------------
def test_affiliation_handbuilt():
    n = 20
    label = np.zeros(n, dtype=int)
    label[4:7] = 1     # event 1: [4,6]
    label[14:17] = 1   # event 2: [14,16]

    # (a) prediction == labels -> precision == recall == 1
    r = em.affiliation_precision_recall(label.copy(), label)
    check("aff_perfect_precision==1", approx(r["aff_precision"], 1.0, 1e-9),
          "got %r" % r["aff_precision"])
    check("aff_perfect_recall==1", approx(r["aff_recall"], 1.0, 1e-9),
          "got %r" % r["aff_recall"])

    # (b) no prediction -> recall 0, no precision evidence
    r0 = em.affiliation_precision_recall(np.zeros(n, dtype=int), label)
    check("aff_none_recall==0", approx(r0["aff_recall"], 0.0, 1e-9),
          "got %r" % r0["aff_recall"])
    check("aff_none_zones_with_pred==0", r0["n_zones_with_pred"] == 0,
          "got %r" % r0["n_zones_with_pred"])

    # (c) one on-target prediction inside event 1 (t=5, the centre).
    # zone 1 = [0,10] (boundary at (6+14)//2=10); gt pts {4,5,6}; nearest pred=5.
    # zspan = max(4-0, 10-6) = 4 ; recall1 = mean(1 - [1,0,1]/4) = 0.83333
    # overall recall = mean(0.83333, 0) = 0.41667 ; precision (on-target) = 1.0
    pred_one = np.zeros(n, dtype=int); pred_one[5] = 1
    r1 = em.affiliation_precision_recall(pred_one, label)
    check("aff_one_recall_value", approx(r1["aff_recall"], 0.4166667, 1e-4),
          "got %r" % r1["aff_recall"])
    check("aff_one_precision==1", approx(r1["aff_precision"], 1.0, 1e-9),
          "got %r" % r1["aff_precision"])
    check("aff_one_zones_with_pred==1", r1["n_zones_with_pred"] == 1,
          "got %r" % r1["n_zones_with_pred"])

    # (d) add a far false alarm in zone 1 -> precision drops below the on-target case
    pred_far = np.zeros(n, dtype=int); pred_far[5] = 1; pred_far[10] = 1
    r2 = em.affiliation_precision_recall(pred_far, label)
    check("aff_farFA_lowers_precision", r2["aff_precision"] < r1["aff_precision"] - 1e-6,
          "far=%r on=%r" % (r2["aff_precision"], r1["aff_precision"]))


# ---------------------------------------------------------------------------
# 4. raw_f1 of perfect alignment == 1 (oracle and val-fit)
# ---------------------------------------------------------------------------
def test_raw_f1_perfect():
    score, label = make_separable()
    rf = em.raw_f1_best(score, label)
    check("raw_f1_oracle_perfect==1", approx(rf["oracle"]["F1"], 1.0, 1e-9),
          "got %r (tau=%r)" % (rf["oracle"]["F1"], rf["oracle"]["tau"]))

    # val nominal band sits in a GAP strictly between test-nominal [0,0.4] and
    # anomaly [0.6,1.0] -> val-max threshold separates the test perfectly.
    rng = np.random.default_rng(3)
    val_score = rng.uniform(0.45, 0.5, size=500)
    rf2 = em.raw_f1_best(score, label, val_score=val_score)
    check("raw_f1_valfit_perfect==1", approx(rf2["val_fit"]["F1"], 1.0, 1e-9),
          "got %r (tau=%r)" % (rf2["val_fit"]["F1"], rf2["val_fit"]["tau"]))


# ---------------------------------------------------------------------------
# 5. event_level_counts on a known layout
# ---------------------------------------------------------------------------
def test_event_counts():
    n = 30
    label = np.zeros(n, dtype=int)
    label[5:8] = 1     # event A
    label[20:23] = 1   # event B
    pred = np.zeros(n, dtype=int)
    pred[6] = 1        # hits A
    pred[12] = 1       # FA run (contiguous with 13)
    pred[13] = 1
    ev = em.event_level_counts(pred, label)
    check("ev_n_true==2", ev["n_true_events"] == 2, "got %r" % ev["n_true_events"])
    check("ev_detected==1", ev["n_detected_events"] == 1, "got %r" % ev["n_detected_events"])
    check("ev_missed==1", ev["n_missed_events"] == 1, "got %r" % ev["n_missed_events"])
    check("ev_false_alarm_runs==1", ev["n_false_alarm_runs"] == 1,
          "got %r" % ev["n_false_alarm_runs"])
    check("ev_pred_runs==2", ev["n_pred_runs"] == 2, "got %r" % ev["n_pred_runs"])


# ---------------------------------------------------------------------------
# 6. VUS reductions at window=0
# ---------------------------------------------------------------------------
def test_vus_window0():
    score, label = make_separable(seed=11)
    check("vus_roc_w0==auroc", approx(em.vus_roc(score, label, window=0),
                                      em.auroc(score, label), 1e-9))
    check("vus_pr_w0==auprc", approx(em.vus_pr(score, label, window=0),
                                     em.auprc(score, label), 1e-9))


# ---------------------------------------------------------------------------
# 7. full_panel smoke + internal consistency
# ---------------------------------------------------------------------------
def test_full_panel():
    score, label = make_separable(seed=5)
    rng = np.random.default_rng(9)
    val_score = rng.uniform(0.45, 0.5, size=300)
    panel = em.full_panel(score, label, val_score=val_score, vus_window=5)
    for key in ("AUPRC", "VUS_ROC", "VUS_PR", "raw_F1_oracle", "aff_precision",
                "aff_recall", "n_true_events", "n_detected_events"):
        check("panel_has_%s" % key, key in panel and panel[key] is not None,
              "missing %s" % key)
    check("panel_n_true==2", panel["n_true_events"] == 2, "got %r" % panel["n_true_events"])
    check("panel_detected_le_true",
          panel["n_detected_events"] <= panel["n_true_events"])
    check("panel_raw_oracle_perfect", approx(panel["raw_F1_oracle"], 1.0, 1e-9),
          "got %r" % panel["raw_F1_oracle"])


def main():
    tests = [
        test_perfect,
        test_random_baseline,
        test_affiliation_handbuilt,
        test_raw_f1_perfect,
        test_event_counts,
        test_vus_window0,
        test_full_panel,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:
            global _FAIL
            _FAIL += 1
            _FAILED.append(t.__name__ + " (exception)")
            print("FAIL  %s  EXCEPTION: %r" % (t.__name__, exc))
    print("\n==== SUMMARY: %d passed, %d failed ====" % (_PASS, _FAIL))
    if _FAILED:
        print("failed:", ", ".join(_FAILED))
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
