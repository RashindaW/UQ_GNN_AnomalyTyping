"""Small shared helpers for competitors/common/extra_metrics.py.

Kept in a separate module so both the metric implementation and the unit tests
import the exact same primitives (run extraction, base rate, safe division).
Pure numpy, no external deps.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def to_events(binary) -> List[Tuple[int, int]]:
    """Convert a (T,) {0,1} vector into a list of (start, end_inclusive) runs.

    Identical semantics to scripts/pa_k_metric.extract_runs, re-implemented here
    to keep this module dependency-free.
    """
    v = np.asarray(binary).astype(bool)
    if v.size == 0 or not v.any():
        return []
    diff = np.diff(np.concatenate([[False], v, [False]]).astype(np.int8))
    starts = np.where(diff == 1)[0].astype(int)
    ends = (np.where(diff == -1)[0] - 1).astype(int)
    return list(zip(starts.tolist(), ends.tolist()))


def base_rate(label) -> float:
    """Positive-class fraction (anomaly base rate)."""
    label = np.asarray(label).astype(int).ravel()
    if label.size == 0:
        return float("nan")
    return float(label.mean())


def safe_div(num: float, den: float) -> float:
    """num/den with a 0.0 result when den == 0 (avoids div-by-zero warnings)."""
    den = float(den)
    if den == 0.0:
        return 0.0
    return float(num) / den
