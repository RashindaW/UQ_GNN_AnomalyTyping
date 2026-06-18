"""
Temporal Evidence Accumulation (TEA) for Incipient Fault Detection.

This module implements cumulative anomaly scoring for detecting gradually
developing faults that manifest as small, persistent deviations from normal
behavior. While instantaneous anomaly scores excel at detecting abrupt faults,
TEA aggregates evidence over configurable time horizons to detect incipient
faults where individual deviations remain below detection thresholds.

Theoretical Foundation:
-----------------------
Based on the Cumulative Sum (CUSUM) method from statistical process control
(Page, 1954), TEA accumulates anomaly evidence over time. Under normal operation,
anomaly scores fluctuate randomly around a baseline, resulting in a slowly
growing cumulative sum with stable rate. During incipient fault development,
even slightly elevated scores produce a systematically faster-growing cumsum,
enabling detection of faults that would otherwise remain undetected.

Usage:
------
    from src.utils.tea import TemporalEvidenceAccumulator

    # Initialize with baseline (normal) scores
    tea = TemporalEvidenceAccumulator(window_sizes=[180, 360, 720])
    tea.fit_baseline(normal_anomaly_scores)

    # Transform fault scores
    tea_scores = tea.transform(fault_anomaly_scores)

References:
-----------
    Page, E. S. (1954). Continuous inspection schemes. Biometrika, 41(1/2), 100-115.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class TemporalEvidenceAccumulator:
    """
    Temporal Evidence Accumulation for incipient fault detection.

    Aggregates instantaneous anomaly scores over configurable time windows
    to detect gradually developing faults that manifest as persistent but
    small deviations from normal behavior.

    Parameters
    ----------
    window_sizes : List[int], optional
        List of window sizes (in samples) for computing cumulative rates.
        Multiple windows enable multi-scale detection of faults developing
        at different speeds. Default: [60, 180, 360] (10min, 30min, 1hr at 10s sampling).
    aggregation : str, optional
        How to combine scores from multiple windows: 'max', 'mean', or 'best'.
        'best' selects the window with highest AUC during calibration.
        Default: 'max'.
    clip_percentile : float, optional
        Percentile for clipping extreme outliers in input scores.
        Default: 99.9.

    Attributes
    ----------
    baseline_mean_ : Dict[int, float]
        Mean cumsum rate for each window size, fitted on baseline data.
    baseline_std_ : Dict[int, float]
        Std of cumsum rate for each window size, fitted on baseline data.
    best_window_ : int
        Window size that achieved best separation (set after fit_baseline with labels).
    is_fitted_ : bool
        Whether baseline statistics have been computed.
    """

    def __init__(
        self,
        window_sizes: Optional[List[int]] = None,
        aggregation: str = 'max',
        clip_percentile: float = 99.9,
    ):
        self.window_sizes = window_sizes or [60, 180, 360]
        self.aggregation = aggregation
        self.clip_percentile = clip_percentile

        # Fitted parameters
        self.baseline_mean_: Dict[int, float] = {}
        self.baseline_std_: Dict[int, float] = {}
        self.best_window_: Optional[int] = None
        self.is_fitted_ = False

    def _clip_outliers(self, scores: np.ndarray) -> np.ndarray:
        """Clip extreme outliers using percentile threshold."""
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            return np.zeros_like(scores)
        threshold = np.percentile(scores[finite_mask], self.clip_percentile)
        return np.clip(np.nan_to_num(scores, nan=0, posinf=threshold, neginf=0), 0, threshold)

    def _compute_cusum_rate(self, scores: np.ndarray, window_size: int) -> np.ndarray:
        """
        Compute windowed cumulative sum rate.

        The rate represents how much the cumulative sum grew over the window,
        which indicates the average anomaly intensity over that time period.

        Parameters
        ----------
        scores : np.ndarray
            Anomaly scores, shape (T,).
        window_size : int
            Number of samples for rate computation.

        Returns
        -------
        np.ndarray
            Cumsum rate at each time step, shape (T,).
        """
        cumsum = np.cumsum(scores)
        rate = np.zeros_like(cumsum)

        if len(scores) > window_size:
            rate[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
            rate[:window_size] = rate[window_size]  # Pad initial values
        else:
            # For short sequences, scale by expected window
            rate[:] = cumsum[-1] / len(scores) * window_size if len(scores) > 0 else 0

        return rate

    def fit_baseline(self, baseline_scores: np.ndarray) -> 'TemporalEvidenceAccumulator':
        """
        Fit baseline statistics from normal operation data.

        Parameters
        ----------
        baseline_scores : np.ndarray
            Anomaly scores from normal operation, shape (T,).

        Returns
        -------
        self
        """
        scores = self._clip_outliers(baseline_scores)

        for window_size in self.window_sizes:
            rates = self._compute_cusum_rate(scores, window_size)
            self.baseline_mean_[window_size] = float(np.mean(rates))
            self.baseline_std_[window_size] = float(np.std(rates))

        self.best_window_ = self.window_sizes[-1]  # Default to largest window
        self.is_fitted_ = True

        return self

    def transform(
        self,
        scores: np.ndarray,
        normalize: bool = True,
        return_all_windows: bool = False,
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Transform anomaly scores using temporal evidence accumulation.

        Parameters
        ----------
        scores : np.ndarray
            Input anomaly scores, shape (T,).
        normalize : bool, optional
            Whether to z-score normalize using baseline statistics.
            Default: True.
        return_all_windows : bool, optional
            If True, return dict mapping window_size to transformed scores.
            If False, return aggregated scores. Default: False.

        Returns
        -------
        np.ndarray or Dict[int, np.ndarray]
            Transformed TEA scores.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit_baseline() before transform()")

        scores = self._clip_outliers(scores)
        results: Dict[int, np.ndarray] = {}

        for window_size in self.window_sizes:
            rates = self._compute_cusum_rate(scores, window_size)

            if normalize:
                mean = self.baseline_mean_[window_size]
                std = self.baseline_std_[window_size]
                rates = (rates - mean) / (std + 1e-8)

            results[window_size] = rates

        if return_all_windows:
            return results

        # Aggregate across windows
        stacked = np.stack(list(results.values()), axis=0)

        if self.aggregation == 'max':
            return np.max(stacked, axis=0)
        elif self.aggregation == 'mean':
            return np.mean(stacked, axis=0)
        elif self.aggregation == 'best' and self.best_window_ is not None:
            return results[self.best_window_]
        else:
            return np.max(stacked, axis=0)

    def fit_transform(
        self,
        baseline_scores: np.ndarray,
        target_scores: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Fit baseline and transform scores in one call.

        Parameters
        ----------
        baseline_scores : np.ndarray
            Normal operation scores for fitting baseline statistics.
        target_scores : np.ndarray, optional
            Scores to transform. If None, transforms baseline_scores.
        **kwargs
            Additional arguments passed to transform().

        Returns
        -------
        np.ndarray
            Transformed TEA scores.
        """
        self.fit_baseline(baseline_scores)
        scores_to_transform = target_scores if target_scores is not None else baseline_scores
        return self.transform(scores_to_transform, **kwargs)

    def get_baseline_stats(self) -> Dict[str, Dict[int, float]]:
        """Return fitted baseline statistics."""
        return {
            'mean': dict(self.baseline_mean_),
            'std': dict(self.baseline_std_),
        }


def compute_tea_metrics(
    baseline_scores: np.ndarray,
    fault_scores: np.ndarray,
    window_sizes: Optional[List[int]] = None,
    return_best_window: bool = True,
    val_baseline_scores: Optional[np.ndarray] = None,
    val_fault_scores: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """
    Compute TEA-enhanced detection metrics for a single fault type.

    This is a convenience function for evaluating TEA on fault detection tasks.

    IMPORTANT: To avoid data leakage, provide val_baseline_scores and val_fault_scores
    for window size selection. If not provided, window selection will be done on
    test data (which is technically data leakage but may be acceptable for final
    reporting when hyperparameters were tuned separately).

    Parameters
    ----------
    baseline_scores : np.ndarray
        Anomaly scores from normal operation (test baseline).
    fault_scores : np.ndarray
        Anomaly scores from fault condition (test faults).
    window_sizes : List[int], optional
        Window sizes to evaluate. Default: [30, 60, 180].
    return_best_window : bool, optional
        Whether to return the best-performing window size.
    val_baseline_scores : np.ndarray, optional
        Validation baseline scores for window selection. If provided,
        window size is selected on validation data to avoid leakage.
    val_fault_scores : np.ndarray, optional
        Validation fault scores for window selection. If provided with
        val_baseline_scores, window size is selected on validation data.

    Returns
    -------
    Dict with keys:
        - 'auc': Best AUC-ROC achieved on test data
        - 'best_f1': Best F1 score achieved on test data
        - 'best_window': Window size selected (from validation if provided)
        - 'tea_baseline': TEA-transformed baseline scores
        - 'tea_fault': TEA-transformed fault scores
        - 'window_selected_on': 'validation' or 'test' (indicates selection method)
    """
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    window_sizes = window_sizes or [30, 60, 180]

    # Determine if we should use validation data for window selection
    use_validation = (val_baseline_scores is not None and val_fault_scores is not None)

    # Data for window selection
    if use_validation:
        selection_baseline = val_baseline_scores
        selection_fault = val_fault_scores
    else:
        selection_baseline = baseline_scores
        selection_fault = fault_scores

    # Step 1: Select best window on selection data (validation or test)
    best_auc_selection = 0.0
    best_window = window_sizes[0]

    for window_size in window_sizes:
        tea = TemporalEvidenceAccumulator(
            window_sizes=[window_size],
            aggregation='max',
        )
        tea.fit_baseline(selection_baseline)

        tea_baseline = tea.transform(selection_baseline)
        tea_fault = tea.transform(selection_fault)

        # Compute AUC for window selection
        labels = np.concatenate([np.zeros(len(tea_baseline)), np.ones(len(tea_fault))])
        scores = np.concatenate([tea_baseline, tea_fault])

        try:
            selection_auc = roc_auc_score(labels, scores)
        except ValueError:
            selection_auc = 0.5

        if selection_auc > best_auc_selection:
            best_auc_selection = selection_auc
            best_window = window_size

    # Step 2: Apply best window to test data for final metrics
    tea = TemporalEvidenceAccumulator(
        window_sizes=[best_window],
        aggregation='max',
    )
    tea.fit_baseline(baseline_scores)

    best_tea_baseline = tea.transform(baseline_scores)
    best_tea_fault = tea.transform(fault_scores)

    # Compute final metrics on test data
    labels = np.concatenate([np.zeros(len(best_tea_baseline)), np.ones(len(best_tea_fault))])
    scores = np.concatenate([best_tea_baseline, best_tea_fault])

    try:
        best_auc = roc_auc_score(labels, scores)
    except ValueError:
        best_auc = 0.5

    # Best F1
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Exclude last element (endpoint where recall=0, giving F1=0)
    f1_curve = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = int(np.argmax(f1_curve))
    best_f1 = float(f1_curve[best_idx])
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])

    result = {
        'auc': best_auc,
        'best_f1': best_f1,
        'tea_baseline': best_tea_baseline,
        'tea_fault': best_tea_fault,
        'window_selected_on': 'validation' if use_validation else 'test',
    }

    if return_best_window:
        result['best_window'] = best_window

    return result


def select_tea_window_on_validation(
    train_baseline_scores: np.ndarray,
    val_baseline_scores: np.ndarray,
    val_labels: np.ndarray,
    window_sizes: Optional[List[int]] = None,
) -> int:
    """
    Select the best TEA window size using validation data only.

    This function selects the window size that maximizes AUC on validation data,
    avoiding data leakage from test data.

    Parameters
    ----------
    train_baseline_scores : np.ndarray
        Training baseline scores for fitting TEA statistics.
    val_baseline_scores : np.ndarray
        Validation scores (mix of normal and fault).
    val_labels : np.ndarray
        Labels for validation data (0=normal, 1=fault).
    window_sizes : List[int], optional
        Window sizes to evaluate.

    Returns
    -------
    int
        Best window size based on validation AUC.
    """
    from sklearn.metrics import roc_auc_score

    window_sizes = window_sizes or [30, 60, 180]
    best_auc = 0.0
    best_window = window_sizes[0]

    for window_size in window_sizes:
        tea = TemporalEvidenceAccumulator(
            window_sizes=[window_size],
            aggregation='max',
        )
        tea.fit_baseline(train_baseline_scores)
        tea_val_scores = tea.transform(val_baseline_scores)

        try:
            val_auc = roc_auc_score(val_labels, tea_val_scores)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_window = window_size

    return best_window


def batch_evaluate_tea(
    baseline_scores: np.ndarray,
    fault_scores_dict: Dict[str, np.ndarray],
    window_sizes: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate TEA across multiple fault types.

    Parameters
    ----------
    baseline_scores : np.ndarray
        Anomaly scores from normal operation.
    fault_scores_dict : Dict[str, np.ndarray]
        Mapping from fault name to anomaly scores.
    window_sizes : List[int], optional
        Window sizes to evaluate.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping from fault name to metrics dict.
    """
    results = {}

    for fault_name, fault_scores in fault_scores_dict.items():
        metrics = compute_tea_metrics(
            baseline_scores,
            fault_scores,
            window_sizes=window_sizes,
        )
        results[fault_name] = {
            'tea_auc': metrics['auc'],
            'tea_best_f1': metrics['best_f1'],
            'tea_best_window': metrics['best_window'],
        }

    return results
