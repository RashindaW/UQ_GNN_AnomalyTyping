"""SWaT (A1 & A2 Dec 2015) DatasetAdapter for DualSTGF.

Reuses `data/swat/{train,test}.csv` produced by `scripts/prepare_swat.py`.
The adapter does not split test.csv into per-fault buckets — that's handled
downstream by `scripts/build_test_split.py` and `scripts/calibrate.py` (which
compute the same labeled-val / final-test ranges as the GDN pipeline).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.loader import DataLoader

from dualstage.src.data.swat_column_config import (
    CONTROL_VARS,
    MEASUREMENT_VARS,
)
from dualstage.src.data.swat_dataset import SWaTDataset

from .registry import DatasetAdapter, register_adapter


# Resolve the project repo root (2 levels up from this file: dualstgf/datasets/swat.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = str(_REPO_ROOT / 'data' / 'swat')


def _resolve_split_files(split_key: str) -> List[str]:
    if split_key in ('train', 'val'):
        return ['train']
    if split_key in ('test', 'attack'):
        return ['test']
    return []


def _control_names(_data_dir: str, _feature_option=None) -> List[str]:
    return list(CONTROL_VARS)


def _create_dataloaders(
    window_size: int,
    batch_size: int,
    train_stride: int,
    val_stride: int,
    test_stride: Optional[int],
    data_dir: str,
    num_workers: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    baseline_from: str = 'val',
    severity_range: Optional[Tuple[int, int]] = None,
    feature_option: Optional[str] = None,
    fault_keys: Optional[List[str]] = None,
    pred_horizon: Optional[int] = None,
    val_ratio: float = 0.2,
    **_kwargs,
):
    """Create (train_loader, val_loader, {"swat_test": test_loader}).

    Train / val are a contiguous 80/20 carve of `data/swat/train.csv` (last 20% as
    val to avoid leaking the σ-floor calibration target into training). Test is
    the full `data/swat/test.csv`; the labeled-val / final-test split is handled
    by `scripts/build_test_split.py` at calibration time.
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f'expected {train_csv}; run scripts/prepare_swat.py first')
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f'expected {test_csv}; run scripts/prepare_swat.py first')

    if test_stride is None:
        test_stride = max(1, val_stride)

    # Train CSV is the source of normalization stats.
    full_train = SWaTDataset(
        train_csv,
        window_size=window_size,
        stride=train_stride,
        normalize=True,
        normalization_stats=None,
    )
    norm_stats = full_train.get_normalization_stats()

    # Train / val carve: last `val_ratio` of windows go to val.
    n_windows = full_train.len()
    val_count = max(1, int(n_windows * val_ratio))
    train_count = n_windows - val_count

    # We rebuild val with a different stride (val_stride) over the same source CSV
    # but only over the LAST val_ratio of the underlying time series. To keep this
    # adapter simple we use index-based subsetting on a fresh dataset built with
    # val_stride: take its tail of size proportional to val_ratio.
    val_full = SWaTDataset(
        train_csv,
        window_size=window_size,
        stride=val_stride,
        normalize=True,
        normalization_stats=norm_stats,
    )
    val_n = val_full.len()
    val_keep = max(1, int(val_n * val_ratio))
    val_dataset = torch.utils.data.Subset(val_full, list(range(val_n - val_keep, val_n)))

    # Train: original train dataset, take everything EXCEPT the tail rows
    # belonging to val. Simpler: take the leading windows.
    train_dataset = torch.utils.data.Subset(full_train, list(range(train_count)))

    test_dataset = SWaTDataset(
        test_csv,
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=not distributed,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, {'swat_test': test_loader}


register_adapter(
    DatasetAdapter(
        key='swat',
        description=(
            'SWaT industrial water-treatment benchmark, A1 & A2 (Dec 2015) release. '
            '51 sensors / actuators, 1 Hz historian (already 5x downsampled to 0.2 Hz '
            'in data/swat/{train,test}.csv).'
        ),
        default_data_dir=_DEFAULT_DATA_DIR,
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=SWaTDataset,
        control_names_fn=_control_names,
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=lambda: ['attack'],
        supports_training=True,
        supports_testing=True,
        supports_plotting=False,
    )
)
