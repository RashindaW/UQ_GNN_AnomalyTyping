from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional, Dict
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dualstage.src.data.ashrae_column_config import (
    FAULT_FILES,
    MEASUREMENT_VARS,
    BENCHMARK_DIR,
    BASELINE_FAULT_CODE_WHITELIST,
    BASELINE_UNIT_STATUS_WHITELIST,
    get_measurement_vars,
)
from dualstage.src.data.ashrae_dataset import (
    ASHRAEDataset, 
    ASHRAEFaultDataset,
    get_ashrae_control_variable_names,
)

from .registry import DatasetAdapter, register_adapter


def _resolve_split_files(split_key: str) -> List[str]:
    """Resolve dataset split key to list of file paths."""
    key = split_key.lower()

    if key in ("train", "val", "test", "baseline"):
        benchmark_files = _list_benchmark_files(ASHRAE_DEFAULT_DIR)
        train_files, val_files, test_files = _split_benchmark_files(benchmark_files)
        if key == "train":
            return train_files
        if key == "val":
            return val_files
        return test_files

    # NOTE: Benchmarks are split dynamically in _create_dataloaders. This resolver
    # remains for tooling that expects fault keys.
    if split_key in FAULT_FILES:
        subdir, fault_file = FAULT_FILES[split_key]
        return [os.path.join(subdir, fault_file)]

    # Try case-insensitive match for fault names
    for fault_name, (subdir, fault_file) in FAULT_FILES.items():
        if fault_name.lower() == key:
            return [os.path.join(subdir, fault_file)]

    raise ValueError(
        f"Unknown dataset split '{split_key}'. Valid options: "
        f"one of {list(FAULT_FILES.keys())}"
    )


def _list_benchmark_files(data_dir: str) -> List[str]:
    """List benchmark CSV files (relative paths) under the data directory."""
    benchmark_root = os.path.join(data_dir, BENCHMARK_DIR)
    files: List[str] = []
    for root, _, filenames in os.walk(benchmark_root):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
            full_path = os.path.join(root, filename)
            files.append(os.path.relpath(full_path, data_dir))
    return sorted(files)


def _split_benchmark_files(files: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Split benchmark files into train/val/test by file count (70/20/10)."""
    total = len(files)
    if total == 0:
        return [], [], []
    train_count = int(total * 0.7)
    val_count = int(total * 0.2)
    test_count = total - train_count - val_count
    if train_count == 0:
        train_count = 1
        test_count = max(0, total - train_count - val_count)
    if test_count == 0 and total >= 2:
        test_count = 1
        if val_count > 0:
            val_count -= 1
        else:
            train_count -= 1
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]
    return train_files, val_files, test_files


def _severity_from_filename(filename: str) -> Optional[int]:
    numbers = re.findall(r"\d+", filename)
    if not numbers:
        return None
    return min(int(value) for value in numbers)


def _filter_faults_by_severity(
    fault_items: List[Tuple[str, Tuple[str, str]]],
    severity_range: Tuple[int, int],
) -> List[Tuple[str, Tuple[str, str]]]:
    min_sev, max_sev = severity_range
    filtered = []
    for name, (subdir, filename) in fault_items:
        severity = _severity_from_filename(filename)
        if severity is None:
            continue
        if min_sev <= severity <= max_sev:
            filtered.append((name, (subdir, filename)))
    return filtered


def _list_faults() -> List[str]:
    """List all available fault keys."""
    return sorted(FAULT_FILES.keys())


def _create_dataloaders(
    window_size: int,
    batch_size: int,
    train_stride: int,
    val_stride: int,
    test_stride: int | None,
    data_dir: str,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
    baseline_from: str = "val",
    severity_range: Tuple[int, int] | None = None,
    feature_option: str | None = None,
    fault_keys: List[str] | None = None,
    pred_horizon: int | None = None,
    max_time_gap: float = 12.0,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    Create train, validation, and test dataloaders for ASHRAE dataset.
    
    Args:
        window_size: Length of sliding window
        batch_size: Number of samples per batch
        train_stride: Stride for training data sliding window
        val_stride: Stride for validation data (larger=faster, fewer samples)
        test_stride: Stride for test data (defaults to val_stride when None)
        data_dir: Directory containing ASHRAE CSV files
        num_workers: Number of worker processes for data loading
        distributed: Enable DistributedSampler for multi-process training
        rank: Rank of the current process (used when distributed=True)
        world_size: Total number of processes participating (used when distributed=True)
    
    Returns:
        train_loader: DataLoader for training (benchmark tests)
        val_loader: DataLoader for validation (near normal tests)
        test_loaders: Dict of DataLoaders for testing (baseline + refrigerant leak faults)
    """

    if test_stride is None:
        test_stride = val_stride
    
    print("=" * 70)
    print("CREATING ASHRAE 1043-RP DATALOADERS")
    print("=" * 70)
    
    # ========== Create Training Dataset ==========
    print("\n[1/3] Creating TRAINING dataset (Benchmark Tests)...")
    benchmark_files = _list_benchmark_files(data_dir)
    train_files, val_files, test_files = _split_benchmark_files(benchmark_files)
    print(
        f"  Benchmark files: {len(benchmark_files)} | "
        f"Train/Val/Test: {len(train_files)}/{len(val_files)}/{len(test_files)}"
    )
    filter_kwargs = dict(
        fault_code_whitelist=BASELINE_FAULT_CODE_WHITELIST,
        unit_status_whitelist=BASELINE_UNIT_STATUS_WHITELIST,
    )

    train_dataset = ASHRAEDataset(
        data_files=train_files,
        window_size=window_size,
        stride=train_stride,
        data_dir=data_dir,
        normalize=True,
        feature_option=feature_option,
        pred_horizon=pred_horizon or 0,
        max_time_gap=max_time_gap,
        **filter_kwargs,
    )
    
    # Get normalization statistics from training data
    norm_stats = train_dataset.get_normalization_stats()
    
    # ========== Create Validation Dataset ==========
    print("\n[2/3] Creating VALIDATION dataset (Near Normal Tests)...")
    val_dataset = ASHRAEDataset(
        data_files=val_files,
        window_size=window_size,
        stride=val_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,  # Use training stats
        feature_option=feature_option,
        pred_horizon=pred_horizon or 0,
        max_time_gap=max_time_gap,
        **filter_kwargs,
    )
    
    # ========== Create Test Datasets ==========
    print("\n[3/3] Creating TEST datasets...")
    test_datasets = {}

    selected_faults = list(FAULT_FILES.items())
    if fault_keys is not None:
        requested = {k.lower() for k in fault_keys}
        selected = []
        unknown = []
        for name, file in FAULT_FILES.items():
            if name.lower() in requested:
                selected.append((name, file))
        unknown = [k for k in fault_keys if k.lower() not in {n.lower() for n, _ in selected}]
        if unknown:
            raise ValueError(f"Unknown ASHRAE fault keys: {unknown}. Valid: {list(FAULT_FILES.keys())}")
        selected_faults = selected
    elif severity_range is not None:
        selected_faults = _filter_faults_by_severity(selected_faults, severity_range)
    
    # Baseline test (normal operation)
    print("  - Baseline (near normal operation)")
    if baseline_from not in ("val", "train", "test"):
        raise ValueError(f"baseline_from must be 'train', 'val', or 'test', got {baseline_from}")
    if baseline_from == "val":
        baseline_source_files = val_files
    elif baseline_from == "test":
        baseline_source_files = test_files
    else:
        baseline_source_files = train_files
    baseline_test_dataset = ASHRAEDataset(
        data_files=baseline_source_files,
        window_size=window_size,
        stride=test_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        feature_option=feature_option,
        pred_horizon=pred_horizon or 0,
        max_time_gap=max_time_gap,
        **filter_kwargs,
    )
    test_datasets['baseline'] = baseline_test_dataset
    
    # Fault datasets - Refrigerant leak
    for fault_idx, (fault_name, fault_entry) in enumerate(selected_faults, start=1):
        print(f"  - {fault_name}")
        fault_subdir, fault_file = fault_entry
        fault_file_path = os.path.join(fault_subdir, fault_file)
        fault_dataset = ASHRAEFaultDataset(
            data_files=[fault_file_path],
            fault_label=fault_idx,  # Assign sequential fault labels
            window_size=window_size,
            stride=test_stride,
            data_dir=data_dir,
            normalize=True,
            normalization_stats=norm_stats,  # Use training stats
            feature_option=feature_option,
            unit_status_whitelist=BASELINE_UNIT_STATUS_WHITELIST,
            pred_horizon=pred_horizon or 0,
            max_time_gap=max_time_gap,
        )
        test_datasets[fault_name] = fault_dataset
    
    # ========== Create DataLoaders ==========
    print(f"\nCreating DataLoaders (batch_size={batch_size})...")

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        test_samplers = {
            name: DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            for name, dataset in test_datasets.items()
        }
    else:
        train_sampler = val_sampler = None
        test_samplers = {name: None for name in test_datasets}

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loaders = {}
    for name, dataset in test_datasets.items():
        test_loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_samplers[name],
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("ASHRAE DATALOADER SUMMARY")
    print("=" * 70)
    print(f"Training (Benchmark Tests):")
    print(f"  Files: {len(train_files)}")
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Batches: {len(train_loader)}")
    print()
    print(f"Validation (Near Normal Tests):")
    print(f"  Files: {len(val_files)}")
    print(f"  Samples: {len(val_dataset)}")
    print(f"  Batches: {len(val_loader)}")
    print()
    print(f"Benchmark Test Holdout:")
    print(f"  Files: {len(test_files)}")
    print(f"  Samples: {len(baseline_test_dataset)}")
    print()
    print(f"Testing:")
    for name, loader in test_loaders.items():
        print(f"  {name:30s}: {len(loader.dataset):6d} samples, {len(loader):4d} batches")
    print()
    print(f"Baseline test source: {baseline_from} split")
    print(f"Data dimensions:")
    print(f"  Measurement variables: {train_dataset.n_measurement_vars}")
    print(f"  Control variables: {train_dataset.n_control_vars}")
    print(f"  Window size: {window_size}")
    if distributed:
        print(f"Distributed samplers enabled (rank {rank}/{world_size}).")
    print("=" * 70)

    return train_loader, val_loader, test_loaders


# Default data directory for ASHRAE dataset
ASHRAE_DEFAULT_DIR = os.path.join("data", "ASHRAE_csv")

# Register the ASHRAE adapter
register_adapter(
    DatasetAdapter(
        key="ashrae",
        description=(
            "ASHRAE 1043-RP water-cooled chiller dataset (CSV). "
            "Benchmark tests split by file for train/val/test."
        ),
        default_data_dir=ASHRAE_DEFAULT_DIR,
        measurement_vars=MEASUREMENT_VARS,
        measurement_vars_resolver=get_measurement_vars,
        dataset_cls=ASHRAEDataset,
        control_names_fn=get_ashrae_control_variable_names,
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=_list_faults,
    )
)
