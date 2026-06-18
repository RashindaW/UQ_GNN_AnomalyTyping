"""
Early Detection Metrics for Incipient Fault Detection.

This module provides metrics specifically designed to evaluate how early
a fault detection system can identify developing faults, which is critical
for predictive maintenance applications.

Metrics:
--------
1. Detection Delay: Number of samples from fault onset to first detection
2. Persistent Detection Delay: Delay requiring N consecutive detections
3. Severity at Detection: Minimum fault severity level at first detection
4. Normalized Detection Time: Detection delay as percentage of fault duration
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_detection_delay(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    fault_onset_idx: int = 0,
) -> int:
    """
    Compute number of samples from fault onset to first detection.

    This metric measures how quickly the model detects a fault after it begins.
    Lower values indicate earlier detection capability.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores for the fault period, shape (T,).
    labels : np.ndarray
        Ground truth labels (1 = fault), shape (T,).
    threshold : float
        Detection threshold (scores >= threshold are flagged as anomalies).
    fault_onset_idx : int, optional
        Index where fault begins. Default 0 (assumes all samples are fault).

    Returns
    -------
    int
        Number of samples until first detection.
        Returns -1 if fault was never detected.

    Examples
    --------
    >>> scores = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
    >>> labels = np.array([1, 1, 1, 1, 1])
    >>> compute_detection_delay(scores, labels, threshold=0.5)
    2  # Detected at index 2 (third sample)
    """
    if len(scores) == 0 or len(labels) == 0:
        return -1

    predictions = (scores >= threshold).astype(int)

    # Find first detection after fault onset
    for i in range(fault_onset_idx, len(predictions)):
        if predictions[i] == 1 and labels[i] == 1:
            return i - fault_onset_idx

    return -1  # Never detected


def compute_persistent_detection_delay(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    persistence: int = 3,
    fault_onset_idx: int = 0,
) -> int:
    """
    Compute delay until N consecutive detections (more robust than single detection).

    This metric requires multiple consecutive positive detections to count as
    a true detection, reducing sensitivity to transient false positives.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores for the fault period, shape (T,).
    labels : np.ndarray
        Ground truth labels (1 = fault), shape (T,).
    threshold : float
        Detection threshold (scores >= threshold are flagged as anomalies).
    persistence : int, optional
        Number of consecutive detections required. Default: 3.
    fault_onset_idx : int, optional
        Index where fault begins. Default 0.

    Returns
    -------
    int
        Number of samples until persistent detection achieved.
        Returns -1 if persistent detection never achieved.

    Examples
    --------
    >>> scores = np.array([0.6, 0.4, 0.7, 0.8, 0.9, 0.95])
    >>> labels = np.ones(6)
    >>> compute_persistent_detection_delay(scores, labels, threshold=0.5, persistence=3)
    2  # Consecutive detections start at index 2 (0.7, 0.8, 0.9)
    """
    if len(scores) == 0 or len(labels) == 0:
        return -1

    predictions = (scores >= threshold).astype(int)
    consecutive = 0
    first_consecutive_idx = None

    for i in range(fault_onset_idx, len(predictions)):
        if predictions[i] == 1 and labels[i] == 1:
            if first_consecutive_idx is None:
                first_consecutive_idx = i
            consecutive += 1
            if consecutive >= persistence:
                return first_consecutive_idx - fault_onset_idx
        else:
            consecutive = 0
            first_consecutive_idx = None

    return -1


def compute_severity_at_detection(
    scores: np.ndarray,
    severities: np.ndarray,
    threshold: float,
) -> float:
    """
    Find the minimum fault severity level at which detection first occurs.

    For progressive faults (e.g., PRONTO with increasing valve angles, ASHRAE
    with increasing fouling percentages), this metric identifies at what
    severity level the model first detects the fault.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores, shape (T,).
    severities : np.ndarray
        Severity level for each sample (e.g., [10, 10, 20, 20, 30, ...]).
        Shape (T,).
    threshold : float
        Detection threshold.

    Returns
    -------
    float
        Minimum severity at which detection occurs.
        Returns float('inf') if never detected.

    Examples
    --------
    >>> scores = np.array([0.3, 0.4, 0.6, 0.8])
    >>> severities = np.array([10, 10, 20, 20])
    >>> compute_severity_at_detection(scores, severities, threshold=0.5)
    20.0  # First detected at severity level 20
    """
    if len(scores) == 0:
        return float('inf')

    predictions = scores >= threshold
    detected_severities = severities[predictions]

    if len(detected_severities) == 0:
        return float('inf')  # Never detected

    return float(np.min(detected_severities))


def compute_normalized_detection_time(
    delay_samples: int,
    total_fault_samples: int,
) -> float:
    """
    Express detection delay as percentage of total fault duration.

    This metric normalizes the detection delay to allow comparison across
    faults of different durations.

    Parameters
    ----------
    delay_samples : int
        Number of samples from fault onset to detection.
        -1 indicates no detection.
    total_fault_samples : int
        Total number of samples in the fault period.

    Returns
    -------
    float
        Detection delay as percentage (0% = immediate, 100% = at end or never).

    Examples
    --------
    >>> compute_normalized_detection_time(50, 1000)
    5.0  # Detected after 5% of the fault period
    >>> compute_normalized_detection_time(-1, 1000)
    100.0  # Never detected
    """
    if delay_samples < 0:
        return 100.0  # Never detected

    if total_fault_samples <= 0:
        return 0.0

    return (delay_samples / total_fault_samples) * 100


def compute_time_to_detection(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    sample_rate_hz: float = 1.0,
) -> float:
    """
    Compute time (in seconds) from fault onset to first detection.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores for the fault period.
    labels : np.ndarray
        Ground truth labels (1 = fault).
    threshold : float
        Detection threshold.
    sample_rate_hz : float, optional
        Sampling rate in Hz. Default: 1.0 (returns samples as time).

    Returns
    -------
    float
        Time in seconds until detection. -1.0 if never detected.
    """
    delay_samples = compute_detection_delay(scores, labels, threshold)
    if delay_samples < 0:
        return -1.0
    return delay_samples / sample_rate_hz


def compute_early_detection_metrics(
    baseline_scores: np.ndarray,
    fault_scores: np.ndarray,
    fault_labels: np.ndarray,
    threshold_percentile: float = 95.0,
    severities: Optional[np.ndarray] = None,
    persistence: int = 3,
    total_fault_samples: Optional[int] = None,
) -> Dict[str, Union[int, float]]:
    """
    Compute comprehensive early detection metrics for a fault condition.

    This is the main function for evaluating early detection performance.
    It computes the detection threshold from baseline scores and evaluates
    multiple early detection metrics.

    Parameters
    ----------
    baseline_scores : np.ndarray
        Anomaly scores from normal operation (for threshold computation).
    fault_scores : np.ndarray
        Anomaly scores from fault condition.
    fault_labels : np.ndarray
        Ground truth labels for fault samples.
    threshold_percentile : float, optional
        Percentile of baseline scores to use as threshold. Default: 95.0.
    severities : np.ndarray, optional
        Severity levels for each fault sample (for progressive faults).
    persistence : int, optional
        Number of consecutive detections for persistent delay metric.
    total_fault_samples : int, optional
        Total fault duration for normalized time. Defaults to len(fault_scores).

    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary containing:
        - 'threshold': Detection threshold used
        - 'delay_samples': Samples to first detection
        - 'delay_persistent': Samples to persistent detection
        - 'normalized_delay_pct': Delay as percentage of fault duration
        - 'severity_at_detection': Minimum severity at detection (if severities provided)
        - 'detection_rate': Fraction of fault samples detected

    Examples
    --------
    >>> baseline_scores = np.random.randn(1000) * 0.1
    >>> fault_scores = np.random.randn(500) * 0.1 + 0.5
    >>> fault_labels = np.ones(500)
    >>> metrics = compute_early_detection_metrics(baseline_scores, fault_scores, fault_labels)
    >>> print(metrics['delay_samples'])
    """
    # Compute threshold from baseline
    threshold = np.percentile(baseline_scores, threshold_percentile)

    # Total fault duration
    if total_fault_samples is None:
        total_fault_samples = len(fault_scores)

    # Compute detection metrics
    delay_samples = compute_detection_delay(
        fault_scores, fault_labels, threshold
    )
    delay_persistent = compute_persistent_detection_delay(
        fault_scores, fault_labels, threshold, persistence=persistence
    )
    normalized_delay = compute_normalized_detection_time(
        delay_samples, total_fault_samples
    )

    # Detection rate
    predictions = fault_scores >= threshold
    detection_rate = float(np.mean(predictions[fault_labels == 1])) if np.sum(fault_labels == 1) > 0 else 0.0

    results = {
        'threshold': float(threshold),
        'delay_samples': delay_samples,
        'delay_persistent': delay_persistent,
        'normalized_delay_pct': normalized_delay,
        'detection_rate': detection_rate,
    }

    # Severity at detection (if provided)
    if severities is not None:
        results['severity_at_detection'] = compute_severity_at_detection(
            fault_scores, severities, threshold
        )

    return results


def batch_evaluate_early_detection(
    baseline_scores: np.ndarray,
    fault_data: Dict[str, Dict[str, np.ndarray]],
    threshold_percentile: float = 95.0,
    persistence: int = 3,
) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Evaluate early detection metrics across multiple fault types.

    Parameters
    ----------
    baseline_scores : np.ndarray
        Anomaly scores from normal operation.
    fault_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping fault names to their data:
        {
            'fault_name': {
                'scores': np.ndarray,  # Required
                'labels': np.ndarray,  # Required
                'severities': np.ndarray,  # Optional
            }
        }
    threshold_percentile : float, optional
        Percentile of baseline scores to use as threshold.
    persistence : int, optional
        Number of consecutive detections for persistent delay metric.

    Returns
    -------
    Dict[str, Dict[str, Union[int, float]]]
        Dictionary mapping fault names to their metrics.

    Examples
    --------
    >>> baseline_scores = np.random.randn(1000) * 0.1
    >>> fault_data = {
    ...     'blockage': {
    ...         'scores': np.random.randn(500) * 0.1 + 0.5,
    ...         'labels': np.ones(500),
    ...         'severities': np.repeat([10, 20, 30, 40, 50], 100),
    ...     },
    ...     'leakage': {
    ...         'scores': np.random.randn(300) * 0.1 + 0.3,
    ...         'labels': np.ones(300),
    ...     },
    ... }
    >>> results = batch_evaluate_early_detection(baseline_scores, fault_data)
    """
    results = {}

    for fault_name, data in fault_data.items():
        scores = data.get('scores')
        labels = data.get('labels')
        severities = data.get('severities')

        if scores is None or labels is None:
            print(f"Warning: Skipping {fault_name} - missing scores or labels")
            continue

        metrics = compute_early_detection_metrics(
            baseline_scores=baseline_scores,
            fault_scores=scores,
            fault_labels=labels,
            threshold_percentile=threshold_percentile,
            severities=severities,
            persistence=persistence,
        )

        results[fault_name] = metrics

    return results


def format_early_detection_results(
    results: Dict[str, Dict[str, Union[int, float]]],
    include_header: bool = True,
) -> str:
    """
    Format early detection results as a readable table.

    Parameters
    ----------
    results : Dict[str, Dict[str, Union[int, float]]]
        Results from batch_evaluate_early_detection.
    include_header : bool, optional
        Whether to include column headers.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []

    if include_header:
        header = (
            f"{'Fault Type':<20} | {'Delay':>8} | {'Persist':>8} | "
            f"{'Norm %':>8} | {'Det Rate':>8} | {'Severity':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))

    for fault_name, metrics in results.items():
        delay = metrics.get('delay_samples', -1)
        delay_str = str(delay) if delay >= 0 else "N/A"

        persist = metrics.get('delay_persistent', -1)
        persist_str = str(persist) if persist >= 0 else "N/A"

        norm_delay = metrics.get('normalized_delay_pct', 100.0)
        det_rate = metrics.get('detection_rate', 0.0)

        severity = metrics.get('severity_at_detection', float('inf'))
        sev_str = f"{severity:.1f}" if severity < float('inf') else "N/A"

        line = (
            f"{fault_name:<20} | {delay_str:>8} | {persist_str:>8} | "
            f"{norm_delay:>7.1f}% | {det_rate:>7.1%} | {sev_str:>10}"
        )
        lines.append(line)

    return "\n".join(lines)
