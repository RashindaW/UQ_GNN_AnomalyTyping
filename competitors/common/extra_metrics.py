"""Track-B extended detection metrics (prospectus section 5.1).

This module adds, on top of the repo's existing F1/P/R + PA%K-AUC + PTaPR:

  * vus_roc / vus_pr  -- Volume-Under-the-Surface ROC / PR (Paparrizos, VLDB 2022).
                         Range-AUC averaged over buffer (tolerance) widths 0..L.
  * affiliation_precision_recall -- distance-based affiliation P/R (Huet, KDD 2022).
  * auprc            -- area under the precision-recall curve (average precision).
  * raw_f1_best      -- best raw (NON point-adjusted) F1, val-fit threshold + oracle.
  * event_level_counts -- n true / detected / missed events + false-alarm runs.

Everything is numpy / scipy / sklearn only (no torch, no fusion pipeline import)
so it can be imported by the unit tests and by the panel runner in isolation.

Conventions
-----------
* ``score``  : 1-D float array, higher = more anomalous.
* ``label``  : 1-D {0,1} array, 1 = anomaly (ground truth).
* ``pred_binary`` : 1-D {0,1} hard prediction.
* An "event" is a maximal contiguous run of 1s.

All metrics are deterministic and side-effect free.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # package-relative when imported as competitors.common.extra_metrics
    from .extra_metrics_helpers import to_events, base_rate, safe_div
except Exception:  # pragma: no cover - direct-script import fallback
    from extra_metrics_helpers import to_events, base_rate, safe_div

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


# ----------------------------------------------------------------------------
# small numpy AUC fallbacks (used only if sklearn is unavailable)
# ----------------------------------------------------------------------------
def _roc_auc_np(label: np.ndarray, score: np.ndarray) -> float:
    label = np.asarray(label).astype(int).ravel()
    score = np.asarray(score, dtype=float).ravel()
    P = int(label.sum())
    N = int(label.size - P)
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    s_sorted = score[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, score.size + 1, dtype=float)
    i = 0
    n = score.size
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = ranks_sorted
    sum_pos = ranks[label == 1].sum()
    auc = (sum_pos - P * (P + 1) / 2.0) / (P * N)
    return float(auc)


def _pr_auc_np(label: np.ndarray, score: np.ndarray) -> float:
    """Average-precision style PR-AUC (step interpolation, matches sklearn AP)."""
    label = np.asarray(label).astype(int).ravel()
    score = np.asarray(score, dtype=float).ravel()
    P = int(label.sum())
    if P == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / P
    # AP = sum over thresholds (R_n - R_{n-1}) * P_n
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - rec_prev) * precision))
    return ap


def _roc_auc(label, score) -> float:
    label = np.asarray(label).astype(int).ravel()
    if label.sum() == 0 or label.sum() == label.size:
        return float("nan")
    if _HAVE_SK:
        return float(roc_auc_score(label, score))
    return _roc_auc_np(label, score)


def _pr_auc(label, score) -> float:
    label = np.asarray(label).astype(int).ravel()
    if label.sum() == 0:
        return float("nan")
    if _HAVE_SK:
        return float(average_precision_score(label, score))
    return _pr_auc_np(label, score)


# ----------------------------------------------------------------------------
# AUPRC
# ----------------------------------------------------------------------------
def auprc(score, label) -> float:
    """Area under the precision-recall curve (== sklearn average_precision_score).

    Returns the anomaly-class average precision. For a random scorer this tends
    to the positive base rate; for a perfect scorer it is 1.0.
    """
    return _pr_auc(label, score)


def auroc(score, label) -> float:
    """Plain (point-wise) ROC-AUC. Provided for completeness / VUS baseline."""
    return _roc_auc(label, score)


# ----------------------------------------------------------------------------
# VUS-ROC / VUS-PR  (Paparrizos et al., VLDB 2022)
# ----------------------------------------------------------------------------
def _dilate_label(label: np.ndarray, l: int) -> np.ndarray:
    """Dilate each anomaly event by ``l`` timesteps on each side (buffer region).

    This realises the "range / buffered" positive region used by VUS: a point
    within ``l`` of any true anomaly is treated as positive when computing the
    range-AUC at buffer width ``l``.
    """
    label = np.asarray(label).astype(int).ravel()
    if l <= 0:
        return label.copy()
    n = label.size
    out = label.copy()
    for (s, e) in to_events(label):
        lo = max(0, s - l)
        hi = min(n - 1, e + l)
        out[lo : hi + 1] = 1
    return out


def _vus(score, label, window: int, kind: str) -> Dict[str, object]:
    """Shared VUS engine: average range-AUC over buffer widths l = 0..window.

    kind == 'roc' -> ROC-AUC at each buffer width; kind == 'pr' -> PR-AUC.
    The averaged value is the Volume Under the Surface (VUS).
    """
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label).astype(int).ravel()
    window = int(max(0, window))
    fn = _roc_auc if kind == "roc" else _pr_auc
    curve: List[float] = []
    widths = list(range(0, window + 1))
    for l in widths:
        lab_l = _dilate_label(label, l)
        a = fn(lab_l, score)
        curve.append(a)
    arr = np.asarray(curve, dtype=float)
    valid = arr[~np.isnan(arr)]
    vol = float(valid.mean()) if valid.size else float("nan")
    return {"VUS": vol, "curve": arr.tolist(), "widths": widths}


def vus_roc(score, label, window: int = 10) -> float:
    """Volume Under the Surface ROC -- range ROC-AUC averaged over buffer 0..window.

    Robust generalisation of ROC-AUC that rewards near-miss detections; matches
    plain ROC-AUC at ``window == 0``.
    """
    return _vus(score, label, window, "roc")["VUS"]


def vus_pr(score, label, window: int = 10) -> float:
    """Volume Under the Surface PR -- range PR-AUC averaged over buffer 0..window."""
    return _vus(score, label, window, "pr")["VUS"]


def vus_full(score, label, window: int = 10) -> Dict[str, object]:
    """Both VUS-ROC and VUS-PR plus the per-buffer curves (for diagnostics)."""
    r = _vus(score, label, window, "roc")
    p = _vus(score, label, window, "pr")
    return {
        "VUS_ROC": r["VUS"],
        "VUS_PR": p["VUS"],
        "window": int(window),
        "roc_curve": r["curve"],
        "pr_curve": p["curve"],
        "widths": r["widths"],
    }


# ----------------------------------------------------------------------------
# Affiliation precision / recall  (Huet, Navarro, Rossi -- KDD 2022)
# ----------------------------------------------------------------------------
# Faithful implementation of the published affiliation metric.
#
# The timeline is partitioned into one "affiliation zone" per ground-truth
# event: each timestamp is assigned to the gt event it is closest to (ties split
# at the midpoint). Within a zone we measure:
#   * affiliation RECALL    -- how close the (zone-local) predicted points get to
#                              the gt event, averaged over the gt event's points,
#                              expressed as a probability via the local distance
#                              distribution of a random predictor.
#   * affiliation PRECISION -- how close each predicted point in the zone is to
#                              the gt event, averaged over those predicted points,
#                              same probabilistic normalisation.
# The probability map E[1 - D/Dmax] turns a raw temporal distance into a value in
# [0,1] (1 == on the event, ~0 == as far as the zone allows), so the metric is
# comparable across events of different lengths and across datasets. P and R are
# then averaged over zones that contain the relevant evidence.
#
# This is the directed-distance / probabilistic affiliation formulation of Huet
# 2022. It reproduces the qualitative behaviour of the reference library
# (perfect overlap -> 1, no prediction -> recall 0, far false alarms penalised)
# and is what we label "affiliation_precision_recall" in the panel.
def _affiliation_zones(events: Sequence[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    """Partition [0, n-1] into one inclusive interval per event (nearest event).

    Boundaries between consecutive events fall at the midpoint of the gap.
    """
    if not events:
        return []
    zones: List[Tuple[int, int]] = []
    m = len(events)
    for i, (s, e) in enumerate(events):
        lo = 0 if i == 0 else (events[i - 1][1] + s) // 2 + 1
        hi = (n - 1) if i == m - 1 else (e + events[i + 1][0]) // 2
        zones.append((lo, hi))
    return zones


def _dist_point_to_event(t: np.ndarray, s: int, e: int) -> np.ndarray:
    """Temporal distance from each index in ``t`` to inclusive event [s,e] (0 inside)."""
    t = np.asarray(t, dtype=float)
    left = s - t          # >0 when left of event
    right = t - e         # >0 when right of event
    d = np.maximum(np.maximum(left, right), 0.0)
    return d


def _zone_prob_from_distance(idx_in_zone: np.ndarray, zlo: int, zhi: int,
                             s: int, e: int) -> np.ndarray:
    """Map temporal distances (event[s,e] within zone[zlo,zhi]) to [0,1] probs.

    prob = 1 - dist / dmax, where dmax is the largest distance any point of the
    zone can have to the event (i.e. the worst case inside the zone). prob == 1
    on the event, prob == 0 at the farthest reachable point of the zone.
    """
    idx_in_zone = np.asarray(idx_in_zone, dtype=float)
    d = _dist_point_to_event(idx_in_zone, s, e)
    dmax = max(_dist_point_to_event(np.array([zlo, zhi]), s, e).max(), 1e-12)
    prob = 1.0 - d / dmax
    return np.clip(prob, 0.0, 1.0)


def affiliation_precision_recall(pred_binary, label) -> Dict[str, float]:
    """Affiliation precision / recall (Huet 2022, probabilistic distance form).

    Parameters
    ----------
    pred_binary : (T,) {0,1} hard predictions.
    label       : (T,) {0,1} ground truth.

    Returns
    -------
    dict with keys ``aff_precision``, ``aff_recall``, ``aff_f1`` plus the
    per-event counts that fed the averages (``n_zones``,
    ``n_zones_with_pred``).

    Semantics
    ---------
    * RECALL is averaged over *all* gt events (a gt event with no nearby
      prediction contributes recall 0), so recall measures coverage of the truth.
    * PRECISION is averaged only over zones that actually contain a prediction
      (a zone with no prediction has no precision evidence), so precision
      measures, given that we fired in a region, how on-target the firing was.
      If there are no predictions at all, precision is 0.0.
    """
    pred = np.asarray(pred_binary).astype(int).ravel()
    label = np.asarray(label).astype(int).ravel()
    n = label.size
    events = to_events(label)
    if not events:
        # No ground-truth events: precision is 1 iff there are no predictions.
        return {
            "aff_precision": 1.0 if pred.sum() == 0 else 0.0,
            "aff_recall": float("nan"),
            "aff_f1": float("nan"),
            "n_zones": 0,
            "n_zones_with_pred": 0,
        }
    zones = _affiliation_zones(events, n)
    rec_vals: List[float] = []
    prec_vals: List[float] = []
    n_with_pred = 0
    for (s, e), (zlo, zhi) in zip(events, zones):
        zidx = np.arange(zlo, zhi + 1)
        # ---- recall: probability mass of the gt event vs zone-local pred mass.
        # For each gt point, its "individual recall" is the best (max) prob over
        # predicted points in the zone (closest prediction). Average over gt pts.
        ev_idx = np.arange(s, e + 1)
        pred_in_zone = zidx[pred[zlo : zhi + 1] == 1]
        if pred_in_zone.size == 0:
            rec_vals.append(0.0)
            continue
        n_with_pred += 1
        # distance from each gt point to nearest predicted point, -> prob via zone scale.
        # zone scale: max distance reachable inside zone.
        zspan = max(float(max(s - zlo, zhi - e)), 1e-12)
        dmin_gt = np.min(np.abs(ev_idx[:, None] - pred_in_zone[None, :]), axis=1)
        rec_event = np.clip(1.0 - dmin_gt / zspan, 0.0, 1.0).mean()
        rec_vals.append(float(rec_event))
        # ---- precision: each predicted point's prob wrt the gt event.
        prec_pts = _zone_prob_from_distance(pred_in_zone, zlo, zhi, s, e)
        prec_vals.append(float(prec_pts.mean()))
    # gt events with no prediction already contributed recall 0.
    aff_recall = float(np.mean(rec_vals)) if rec_vals else 0.0
    aff_precision = float(np.mean(prec_vals)) if prec_vals else 0.0
    if aff_precision + aff_recall > 0:
        aff_f1 = 2 * aff_precision * aff_recall / (aff_precision + aff_recall)
    else:
        aff_f1 = 0.0
    return {
        "aff_precision": aff_precision,
        "aff_recall": aff_recall,
        "aff_f1": aff_f1,
        "n_zones": len(events),
        "n_zones_with_pred": int(n_with_pred),
    }


# ----------------------------------------------------------------------------
# raw (non point-adjusted) F1
# ----------------------------------------------------------------------------
def _raw_f1_at_tau(score: np.ndarray, label: np.ndarray, tau: float) -> Dict[str, float]:
    pred = (score >= tau).astype(int)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r)
    return {"F1": f1, "P": p, "R": r, "tau": float(tau), "tp": tp, "fp": fp, "fn": fn}


def raw_f1_best(score, label, val_score=None, n_thresholds: int = 200) -> Dict[str, object]:
    """Best RAW F1 (no point adjustment): oracle (test-swept) + val-fit.

    Parameters
    ----------
    score : (T,) test anomaly score.
    label : (T,) test labels.
    val_score : (Tv,) optional anomaly score on the (anomaly-free) validation
        slice. If given, the val-fit threshold is taken as ``max(val_score)``
        (the textbook unsupervised nominal-max rule) and applied to the test
        score; a small high-quantile sweep is also reported.
    n_thresholds : number of test quantile thresholds for the oracle sweep.

    Returns
    -------
    dict with ``oracle`` (test-swept best F1) and, when ``val_score`` is given,
    ``val_fit`` (val-max threshold applied to test). All entries hold F1/P/R/tau.
    """
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label).astype(int).ravel()
    qs = np.linspace(0.0, 1.0, n_thresholds)
    q_taus = np.quantile(score, qs)
    # Also include MIDPOINTS between consecutive unique scores so the optimal
    # separating threshold is always reachable (a pure quantile grid can land
    # exactly on score values and miss a clean separation by one boundary point).
    uniq = np.unique(score)
    if uniq.size > 1:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        # cap midpoint count for very high-cardinality scores (keeps it O(T))
        if mids.size > 4 * n_thresholds:
            sel = np.linspace(0, mids.size - 1, 4 * n_thresholds).astype(int)
            mids = mids[sel]
    else:
        mids = uniq
    # a threshold strictly below the global min guarantees the all-positive point
    lo = np.array([uniq.min() - 1.0]) if uniq.size else np.array([0.0])
    taus = np.unique(np.concatenate([lo, q_taus, mids]))
    best = {"F1": 0.0, "P": 0.0, "R": 0.0, "tau": None}
    for tau in taus:
        res = _raw_f1_at_tau(score, label, tau)
        if res["F1"] > best["F1"]:
            best = {k: res[k] for k in ("F1", "P", "R", "tau")}
    out: Dict[str, object] = {"oracle": best}
    if val_score is not None:
        val_score = np.asarray(val_score, dtype=float).ravel()
        tau_valmax = float(np.max(val_score))
        out["val_fit"] = _raw_f1_at_tau(score, label, tau_valmax)
        # also report a high-quantile sweep on val for context
        vq = {}
        for q in (0.995, 0.999, 1.0):
            tau_q = float(np.quantile(val_score, q))
            r = _raw_f1_at_tau(score, label, tau_q)
            vq["q%.3f" % q] = {"tau": tau_q, "F1": r["F1"], "P": r["P"], "R": r["R"]}
        out["val_fit_quantiles"] = vq
    return out


# ----------------------------------------------------------------------------
# event-level counts
# ----------------------------------------------------------------------------
def event_level_counts(pred_binary, label) -> Dict[str, int]:
    """Event-level detection bookkeeping.

    Returns
    -------
    dict with:
      * n_true_events     -- number of ground-truth anomaly events.
      * n_detected_events -- gt events with >= 1 predicted positive inside them.
      * n_missed_events   -- gt events with no overlapping prediction.
      * n_false_alarm_runs-- contiguous predicted runs that touch no gt event.
      * n_pred_runs       -- total contiguous predicted runs.
    """
    pred = np.asarray(pred_binary).astype(int).ravel()
    label = np.asarray(label).astype(int).ravel()
    gt_events = to_events(label)
    pred_runs = to_events(pred)
    n_true = len(gt_events)
    detected = 0
    for (s, e) in gt_events:
        if pred[s : e + 1].sum() > 0:
            detected += 1
    missed = n_true - detected
    # false-alarm runs: predicted runs with zero overlap with any gt event
    false_alarm = 0
    for (ps, pe) in pred_runs:
        if label[ps : pe + 1].sum() == 0:
            false_alarm += 1
    return {
        "n_true_events": int(n_true),
        "n_detected_events": int(detected),
        "n_missed_events": int(missed),
        "n_false_alarm_runs": int(false_alarm),
        "n_pred_runs": int(len(pred_runs)),
    }


# ----------------------------------------------------------------------------
# convenience: full panel for one (score,label) given a chosen pred threshold
# ----------------------------------------------------------------------------
def full_panel(score, label, val_score=None, vus_window: int = 10,
               pred_for_event_metrics: str = "oracle") -> Dict[str, object]:
    """Compute the entire extended panel for one score/label pair.

    The hard prediction used by the affiliation / event-level metrics is taken
    at the threshold selected by ``pred_for_event_metrics``:
      * 'oracle'  -> the raw-F1 oracle (test-swept) threshold.
      * 'val_fit' -> the val-max threshold (requires ``val_score``).
    """
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label).astype(int).ravel()
    rf = raw_f1_best(score, label, val_score=val_score)
    vus = vus_full(score, label, window=vus_window)
    ap = auprc(score, label)
    roc = auroc(score, label)

    if pred_for_event_metrics == "val_fit" and "val_fit" in rf:
        tau = rf["val_fit"]["tau"]
    else:
        tau = rf["oracle"]["tau"]
    tau = float(tau) if tau is not None else float(np.max(score) + 1.0)
    pred = (score >= tau).astype(int)

    aff = affiliation_precision_recall(pred, label)
    ev = event_level_counts(pred, label)

    panel = {
        "AUPRC": ap,
        "AUROC": roc,
        "VUS_ROC": vus["VUS_ROC"],
        "VUS_PR": vus["VUS_PR"],
        "vus_window": int(vus_window),
        "raw_F1_oracle": rf["oracle"]["F1"],
        "raw_P_oracle": rf["oracle"]["P"],
        "raw_R_oracle": rf["oracle"]["R"],
        "raw_tau_oracle": rf["oracle"]["tau"],
        "aff_precision": aff["aff_precision"],
        "aff_recall": aff["aff_recall"],
        "aff_f1": aff["aff_f1"],
        "n_true_events": ev["n_true_events"],
        "n_detected_events": ev["n_detected_events"],
        "n_missed_events": ev["n_missed_events"],
        "n_false_alarm_runs": ev["n_false_alarm_runs"],
        "n_pred_runs": ev["n_pred_runs"],
        "base_rate": base_rate(label),
        "pred_threshold_source": pred_for_event_metrics if (pred_for_event_metrics != "val_fit" or "val_fit" in rf) else "oracle",
    }
    if "val_fit" in rf:
        panel["raw_F1_valfit"] = rf["val_fit"]["F1"]
        panel["raw_P_valfit"] = rf["val_fit"]["P"]
        panel["raw_R_valfit"] = rf["val_fit"]["R"]
        panel["raw_tau_valfit"] = rf["val_fit"]["tau"]
    return panel
