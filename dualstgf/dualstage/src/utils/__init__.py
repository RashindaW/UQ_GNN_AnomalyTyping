"""Utility modules for DualSTAGE."""

import numpy as np
from scipy.stats import genpareto

from .checkpoint import EpochCheckpointManager
from .init import init_weights


def threshold_evt_pot(val_scores, false_alarm_rate=0.05, initial_percentile=90):
    """EVT Peaks-Over-Threshold with GPD tail fit.

    Uses only validation baseline scores — no test data leakage.
    Ref: Siffer et al. (2017) "Anomaly Detection in Streams with EVT" KDD.
    """
    u = np.percentile(val_scores, initial_percentile)
    exceedances = val_scores[val_scores > u] - u
    n_total, n_exceed = len(val_scores), len(exceedances)
    if n_exceed < 10:
        return np.percentile(val_scores, 100 * (1 - false_alarm_rate))
    shape, _, scale = genpareto.fit(exceedances, floc=0)
    if abs(shape) < 1e-8:
        return u + scale * np.log(n_exceed / (n_total * false_alarm_rate))
    return u + (scale / shape) * ((n_exceed / (n_total * false_alarm_rate)) ** shape - 1)


__all__ = ['init_weights', 'EpochCheckpointManager', 'threshold_evt_pot']
