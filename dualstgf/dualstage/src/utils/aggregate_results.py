"""
Result Aggregation Utilities for Multi-Seed Experiments

Provides functions for loading, aggregating, and formatting experimental results
across multiple random seeds for reproducible research paper reporting.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def load_run_results(results_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load results from multiple seed runs in a directory.

    Args:
        results_dir: Path to the multi-seed results directory

    Returns:
        List of result dictionaries, one per seed
    """
    results_dir = Path(results_dir)
    results = []

    # Find all seed directories
    seed_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[1])
        result = {"seed": seed, "dir": str(seed_dir)}

        # Try to load various result formats
        # 1. Check for detailed_test_metrics.csv in plots/
        for checkpoint_run in sorted(seed_dir.glob("checkpoints/*"), reverse=True):
            plots_dir = checkpoint_run / "plots"
            metrics_csv = plots_dir / "detailed_test_metrics.csv"
            if metrics_csv.exists():
                result["metrics_path"] = str(metrics_csv)
                result["test_metrics"] = _load_csv_metrics(metrics_csv)
                break

        # 2. Check for epoch metrics
        for checkpoint_run in sorted(seed_dir.glob("checkpoints/*"), reverse=True):
            epoch_csv = checkpoint_run / "metrics.csv"
            if epoch_csv.exists():
                result["epoch_metrics_path"] = str(epoch_csv)
                result["epoch_metrics"] = _load_epoch_metrics(epoch_csv)
                break

        # 3. Check for aggregate JSON
        agg_json = seed_dir / "results.json"
        if agg_json.exists():
            with open(agg_json) as f:
                result["json_results"] = json.load(f)

        if "test_metrics" in result or "epoch_metrics" in result or "json_results" in result:
            results.append(result)

    return results


def _load_csv_metrics(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load metrics from a CSV file with test_set column."""
    metrics = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_name = row.get("test_set", "unknown")
            metrics[test_name] = {}
            for k, v in row.items():
                if k != "test_set" and v:
                    try:
                        metrics[test_name][k] = float(v)
                    except ValueError:
                        metrics[test_name][k] = v
    return metrics


def _load_epoch_metrics(csv_path: Path) -> Dict[str, Any]:
    """Load epoch-level training metrics."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    # Get final and best metrics
    final = rows[-1]
    best_val_loss = float("inf")
    best_epoch = 0

    for row in rows:
        val_loss = row.get("val_loss")
        if val_loss:
            val_loss = float(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = int(row.get("epoch", 0))

    return {
        "final_epoch": int(final.get("epoch", len(rows))),
        "final_train_loss": float(final.get("train_loss", 0)),
        "final_val_loss": float(final.get("val_loss", 0)),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def compute_statistics(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute mean, std, min, max statistics for metrics across seeds.

    Args:
        results: List of result dictionaries from load_run_results
        metrics: Optional list of metric names to compute stats for.
                 If None, computes for all found metrics.

    Returns:
        Nested dict: {test_set: {metric_name: {mean, std, min, max, values}}}
    """
    # Collect all test set names and metric names
    test_sets: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        test_metrics = result.get("test_metrics", {})
        for test_name, test_data in test_metrics.items():
            if test_name not in test_sets:
                test_sets[test_name] = {}

            for metric_name, value in test_data.items():
                if metrics and metric_name not in metrics:
                    continue
                if isinstance(value, (int, float)):
                    if metric_name not in test_sets[test_name]:
                        test_sets[test_name][metric_name] = []
                    test_sets[test_name][metric_name].append(float(value))

    # Compute statistics
    statistics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for test_name, metric_data in test_sets.items():
        statistics[test_name] = {}
        for metric_name, values in metric_data.items():
            if values:
                arr = np.array(values)
                statistics[test_name][metric_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": len(values),
                    "values": values,
                }

    return statistics


def format_mean_std(mean: float, std: float, precision: int = 4) -> str:
    """Format mean ± std string."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def format_markdown_table(
    aggregate: Dict[str, Any],
    metrics_to_show: Optional[List[str]] = None,
) -> str:
    """
    Format aggregate results as a markdown table.

    Args:
        aggregate: Aggregate results dictionary
        metrics_to_show: List of metric names to include (default: auc_roc, f1_score, best_f1)

    Returns:
        Markdown-formatted table string
    """
    if metrics_to_show is None:
        metrics_to_show = ["auc_roc", "f1_score", "best_f1", "tea_auc"]

    metrics_data = aggregate.get("metrics", {})
    if not metrics_data:
        return "No metrics data available."

    # Build table header
    lines = []
    header = "| Test Set |"
    separator = "|:---------|"
    for metric in metrics_to_show:
        header += f" {metric} |"
        separator += ":--------:|"
    lines.append(header)
    lines.append(separator)

    # Build table rows
    for test_name in sorted(metrics_data.keys()):
        test_metrics = metrics_data[test_name]
        row = f"| {test_name} |"
        for metric in metrics_to_show:
            if metric in test_metrics:
                m = test_metrics[metric]
                row += f" {format_mean_std(m['mean'], m['std'])} |"
            else:
                row += " - |"
        lines.append(row)

    return "\n".join(lines)


def format_latex_table(
    aggregate: Dict[str, Any],
    metrics_to_show: Optional[List[str]] = None,
    caption: str = "Multi-seed experimental results",
    label: str = "tab:multi_seed_results",
) -> str:
    """
    Format aggregate results as a LaTeX table.

    Args:
        aggregate: Aggregate results dictionary
        metrics_to_show: List of metric names to include
        caption: Table caption
        label: Table label for referencing

    Returns:
        LaTeX-formatted table string
    """
    if metrics_to_show is None:
        metrics_to_show = ["auc_roc", "f1_score", "best_f1", "tea_auc"]

    metrics_data = aggregate.get("metrics", {})
    n_seeds = aggregate.get("n_seeds", "?")

    # Build column specification
    n_cols = 1 + len(metrics_to_show)
    col_spec = "l" + "c" * len(metrics_to_show)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption} (n={n_seeds} seeds)}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    header = "Test Set"
    for metric in metrics_to_show:
        # Convert snake_case to Title Case
        display_name = metric.replace("_", " ").title()
        header += f" & {display_name}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for test_name in sorted(metrics_data.keys()):
        test_metrics = metrics_data[test_name]
        # Escape underscores in test names for LaTeX
        escaped_name = test_name.replace("_", r"\_")
        row = escaped_name
        for metric in metrics_to_show:
            if metric in test_metrics:
                m = test_metrics[metric]
                row += f" & ${m['mean']:.4f} \\pm {m['std']:.4f}$"
            else:
                row += " & -"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_csv_summary(
    aggregate: Dict[str, Any],
    metrics_to_show: Optional[List[str]] = None,
) -> str:
    """
    Format aggregate results as CSV.

    Args:
        aggregate: Aggregate results dictionary
        metrics_to_show: List of metric names to include

    Returns:
        CSV-formatted string
    """
    if metrics_to_show is None:
        metrics_to_show = ["auc_roc", "f1_score", "best_f1", "tea_auc"]

    metrics_data = aggregate.get("metrics", {})

    lines = []
    # Header
    header_parts = ["test_set"]
    for metric in metrics_to_show:
        header_parts.extend([f"{metric}_mean", f"{metric}_std"])
    lines.append(",".join(header_parts))

    # Data rows
    for test_name in sorted(metrics_data.keys()):
        test_metrics = metrics_data[test_name]
        row_parts = [test_name]
        for metric in metrics_to_show:
            if metric in test_metrics:
                m = test_metrics[metric]
                row_parts.extend([f"{m['mean']:.6f}", f"{m['std']:.6f}"])
            else:
                row_parts.extend(["", ""])
        lines.append(",".join(row_parts))

    return "\n".join(lines)


def save_aggregate_results(
    aggregate: Dict[str, Any],
    output_dir: Union[str, Path],
    prefix: str = "aggregate",
) -> Dict[str, Path]:
    """
    Save aggregate results in multiple formats.

    Args:
        aggregate: Aggregate results dictionary
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Dictionary mapping format to saved file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    # JSON (full results)
    json_path = output_dir / f"{prefix}_results.json"
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    saved["json"] = json_path

    # Markdown summary
    md_path = output_dir / f"{prefix}_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Aggregate Results\n\n")
        f.write(f"Number of seeds: {aggregate.get('n_seeds', '?')}\n")
        f.write(f"Seeds: {aggregate.get('seeds', [])}\n\n")
        f.write(format_markdown_table(aggregate))
    saved["markdown"] = md_path

    # LaTeX table
    tex_path = output_dir / f"{prefix}_summary.tex"
    with open(tex_path, "w") as f:
        f.write(format_latex_table(aggregate))
    saved["latex"] = tex_path

    # CSV summary
    csv_path = output_dir / f"{prefix}_summary.csv"
    with open(csv_path, "w") as f:
        f.write(format_csv_summary(aggregate))
    saved["csv"] = csv_path

    return saved


def compare_methods(
    method_results: Dict[str, Dict[str, Any]],
    metrics_to_compare: Optional[List[str]] = None,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare results across different methods/configurations.

    Args:
        method_results: Dict mapping method name to aggregate results
        metrics_to_compare: Metrics to compare
        significance_level: P-value threshold for significance

    Returns:
        Comparison results with deltas and significance tests
    """
    if metrics_to_compare is None:
        metrics_to_compare = ["auc_roc", "f1_score", "best_f1"]

    comparison = {"methods": list(method_results.keys()), "comparisons": {}}

    # Get all test sets across methods
    all_test_sets = set()
    for method_data in method_results.values():
        metrics = method_data.get("metrics", {})
        all_test_sets.update(metrics.keys())

    for test_set in sorted(all_test_sets):
        comparison["comparisons"][test_set] = {}

        for metric in metrics_to_compare:
            metric_comparison = {}

            for method_name, method_data in method_results.items():
                metrics = method_data.get("metrics", {}).get(test_set, {})
                if metric in metrics:
                    metric_comparison[method_name] = metrics[metric]

            if len(metric_comparison) >= 2:
                comparison["comparisons"][test_set][metric] = metric_comparison

                # Perform pairwise comparisons
                methods = list(metric_comparison.keys())
                if len(methods) == 2:
                    try:
                        from scipy import stats

                        m1_vals = metric_comparison[methods[0]].get("values", [])
                        m2_vals = metric_comparison[methods[1]].get("values", [])

                        if len(m1_vals) >= 2 and len(m2_vals) >= 2:
                            t_stat, p_val = stats.ttest_ind(m1_vals, m2_vals)
                            comparison["comparisons"][test_set][metric]["significance"] = {
                                "t_statistic": float(t_stat),
                                "p_value": float(p_val),
                                "significant": p_val < significance_level,
                            }
                    except ImportError:
                        pass

    return comparison
