"""Ensemble inference for the trained heteroscedastic GDN_UQ ensemble.

Implements Steps 1, 2, 3 of inference_pipeline_outline.md:
- Load M = 5 trained members from a manifest.
- Run all M forward passes per input window (per-batch, batched on GPU).
- Aggregate to (mu_bar, sigma2_aleatoric, sigma2_epistemic, sigma2_total).

Returns numpy arrays sized (T, V) where T = number of windowed timesteps in the
input dataset and V = number of sensors. Per-member arrays are kept (M, T, V)
for downstream Variant B and OOD computations.

Used by scripts/calibrate.py and scripts/detect.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.TimeDataset import TimeDataset
from models.GDN_UQ import GDN_UQ
from util.env import set_device, get_device
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_repo_path(p: str, repo_root: Path) -> Path:
    if os.path.isabs(p):
        return Path(p)
    return (repo_root / p).resolve()


@dataclass
class EnsembleConfig:
    """Architecture + training-config snapshot derived from the manifest."""
    M: int
    node_num: int
    dim: int
    input_dim: int
    out_layer_num: int
    out_layer_inter_dim: int
    topk: int
    slide_win: int
    slide_stride: int
    batch: int
    val_ratio: float
    seeds: list[int]
    # Tracked through to inference so the loaded model uses the exact clamp the
    # checkpoint was trained with. Default matches the historical hardcoded
    # constructor default for back-compat with manifests that predate the fix.
    logvar_clamp: tuple[float, float] = (-10.0, 10.0)


@dataclass
class LoadedEnsemble:
    """All trained ensemble members loaded onto the device."""
    members: list[GDN_UQ]
    seeds: list[int]
    feature_map: list[str]
    fc_edge_index: torch.Tensor
    cfg: EnsembleConfig
    device: torch.device


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_ensemble(
    manifest_path: str | Path,
    device: str | torch.device,
    repo_root: str | Path | None = None,
) -> LoadedEnsemble:
    """Load all M members + edge_index + feature_map from a manifest."""
    manifest_path = Path(manifest_path)
    if repo_root is None:
        repo_root = manifest_path.parent.parent.parent  # pretrained/swat_ensemble/manifest.json -> repo
    repo_root = Path(repo_root)

    with manifest_path.open() as f:
        manifest = json.load(f)

    hp = manifest['hyperparameters']
    dataset = manifest['dataset']

    train_csv = repo_root / 'data' / dataset / 'train.csv'
    train_df = pd.read_csv(train_csv, sep=',', index_col=0)
    if 'attack' in train_df.columns:
        train_df = train_df.drop(columns=['attack'])

    feature_map = get_feature_map(dataset)
    fc_struc = get_fc_graph_struc(dataset)
    fc_edge_index_list = build_loc_net(fc_struc, list(train_df.columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index_list, dtype=torch.long)

    set_device(device)
    dev = torch.device(device)

    cfg = EnsembleConfig(
        M=manifest['M'],
        node_num=len(feature_map),
        dim=hp['dim'],
        input_dim=hp['slide_win'],
        out_layer_num=hp['out_layer_num'],
        out_layer_inter_dim=hp['out_layer_inter_dim'],
        topk=hp['topk'],
        slide_win=hp['slide_win'],
        slide_stride=hp['slide_stride'],
        batch=hp['batch'],
        val_ratio=hp.get('val_ratio', 0.2),
        seeds=[m['seed'] for m in manifest['members']],
        logvar_clamp=tuple(hp.get('logvar_clamp', [-10.0, 10.0])),
    )

    members: list[GDN_UQ] = []
    for m in manifest['members']:
        ckpt_path = _resolve_repo_path(m['checkpoint'], repo_root)
        model = GDN_UQ(
            [fc_edge_index],
            cfg.node_num,
            dim=cfg.dim,
            input_dim=cfg.input_dim,
            out_layer_num=cfg.out_layer_num,
            out_layer_inter_dim=cfg.out_layer_inter_dim,
            topk=cfg.topk,
            logvar_clamp=cfg.logvar_clamp,    # honour the trained clamp
        ).to(dev)
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state)
        model.eval()
        members.append(model)

    return LoadedEnsemble(
        members=members,
        seeds=cfg.seeds,
        feature_map=feature_map,
        fc_edge_index=fc_edge_index.to(dev),
        cfg=cfg,
        device=dev,
    )


# ----------------------------------------------------------------------
# Dataset construction
# ----------------------------------------------------------------------


def build_dataset_from_csv(
    csv_path: str | Path,
    feature_map: list[str],
    fc_edge_index: torch.Tensor,
    slide_win: int,
    slide_stride: int,
    mode: str = 'test',
    label_override: list[int] | None = None,
) -> TimeDataset:
    """Wrap a CSV (sensors + optional 'attack' col) into TimeDataset.

    mode='test' => stride=1 (every consecutive timestep windowed).
    mode='train' => stride=slide_stride.
    """
    df = pd.read_csv(csv_path, sep=',', index_col=0)
    if 'attack' in df.columns:
        labels = df['attack'].tolist()
        df = df.drop(columns=['attack'])
    else:
        labels = 0

    if label_override is not None:
        labels = label_override

    indata = construct_data(df, feature_map, labels=labels)
    cfg = {'slide_win': slide_win, 'slide_stride': slide_stride}
    return TimeDataset(indata, fc_edge_index.cpu(), mode=mode, config=cfg)


# ----------------------------------------------------------------------
# Forward + aggregate
# ----------------------------------------------------------------------


@dataclass
class InferenceOutputs:
    """Per-window arrays from a full inference pass.

    All arrays are float32 with shape (T, V) unless noted. T is the number of
    windows in the dataset; V is len(feature_map).
    """
    mu_bar:           np.ndarray      # (T, V)  ensemble mean of mu
    sigma2_aleatoric: np.ndarray      # (T, V)  (1/M) sum_m exp(log_var_m)
    sigma2_epistemic: np.ndarray      # (T, V)  variance across members of mu_m
    sigma2_total:     np.ndarray      # (T, V)  sigma2_a + sigma2_e
    mu_per_member:    np.ndarray      # (M, T, V) per-member mu predictions
    logvar_per_member: np.ndarray     # (M, T, V) per-member log_var predictions
    ground_truth:     np.ndarray      # (T, V)  observed sensor values at the predicted timestep
    attack_label:     np.ndarray      # (T,)    binary attack flag (0 if dataset has no labels)


def run_inference(
    ensemble: LoadedEnsemble,
    dataset: TimeDataset,
    batch_size: int | None = None,
) -> InferenceOutputs:
    """Run all M members over `dataset`, return per-window aggregated outputs."""
    if batch_size is None:
        batch_size = ensemble.cfg.batch

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = ensemble.device
    M = ensemble.cfg.M
    V = ensemble.cfg.node_num
    T = len(dataset)

    mu_buf = np.empty((M, T, V), dtype=np.float32)
    lv_buf = np.empty((M, T, V), dtype=np.float32)
    y_buf = np.empty((T, V), dtype=np.float32)
    label_buf = np.empty((T,), dtype=np.int8)

    pos = 0
    for batch in loader:
        x, y, label, edge_index = batch
        x = x.to(device).float()
        edge_index = edge_index.to(device).long()
        b = x.shape[0]

        with torch.no_grad():
            for m_idx, member in enumerate(ensemble.members):
                mu, log_var = member(x, edge_index)
                mu_buf[m_idx, pos:pos + b, :] = mu.detach().cpu().numpy()
                lv_buf[m_idx, pos:pos + b, :] = log_var.detach().cpu().numpy()

        y_buf[pos:pos + b, :] = y.cpu().numpy()
        label_buf[pos:pos + b] = label.cpu().numpy().astype(np.int8)
        pos += b

    assert pos == T, f"buffer underflow: filled {pos} / {T}"

    sigma2_per_member = np.exp(lv_buf)
    mu_bar = mu_buf.mean(axis=0)                                  # (T, V)
    sigma2_a = sigma2_per_member.mean(axis=0)                     # (T, V)
    sigma2_e = mu_buf.var(axis=0)                                 # (T, V) — uses ddof=0
    sigma2_tot = sigma2_a + sigma2_e

    return InferenceOutputs(
        mu_bar=mu_bar.astype(np.float32),
        sigma2_aleatoric=sigma2_a.astype(np.float32),
        sigma2_epistemic=sigma2_e.astype(np.float32),
        sigma2_total=sigma2_tot.astype(np.float32),
        mu_per_member=mu_buf,
        logvar_per_member=lv_buf,
        ground_truth=y_buf,
        attack_label=label_buf,
    )


# ----------------------------------------------------------------------
# Step 6 / Step 8 helpers
# ----------------------------------------------------------------------


def apply_lambda(sigma2_total: np.ndarray, lambda_cal) -> np.ndarray:
    """Step 6 — multiplicative correction on sigma_tot (NOT sigma2_tot).

    Accepts either a scalar λ (global correction) OR a per-sensor vector
    `lambda_v` of shape `(V,)`. The per-sensor form broadcasts to `(T, V)`.
    """
    lam = np.asarray(lambda_cal, dtype=np.float32)
    if lam.ndim == 0:
        return (float(lam) ** 2) * sigma2_total
    if lam.ndim == 1:
        # Broadcast (V,) -> (1, V) so it multiplies correctly with (T, V).
        return (lam ** 2)[None, :] * sigma2_total
    raise ValueError(f'apply_lambda expected scalar or 1-D lambda, got shape {lam.shape}')


def apply_sigma_floor(sigma2_total: np.ndarray, sigma_floor_v: np.ndarray) -> np.ndarray:
    """Floor sigma_total per sensor: σ̂_total[t, v] := max(σ̂_total[t, v], σ_floor[v]).

    Mitigates the σ-collapse failure mode: some sensors (typically binary
    actuators in SWaT) have their log_var head pinned at the lower clamp
    boundary during training, giving σ̂ ≈ 0.0067. Standardised residuals on
    those sensors then explode at inference time. Flooring σ̂ at the empirical
    residual std observed on the calibration set keeps the model from being
    *more* confident than the residuals it actually produces on clean data.

    sigma_floor_v: shape (V,), per-sensor σ floor (NOT σ²).
    """
    return np.maximum(sigma2_total, (sigma_floor_v ** 2)[None, :])


def standardised_residual(
    y: np.ndarray, mu_bar: np.ndarray, sigma2_total_cal: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Step 8 — r_tilde_v(t) = |y - mu_bar| / sqrt(sigma2_total_cal)."""
    return np.abs(y - mu_bar) / np.sqrt(np.maximum(sigma2_total_cal, eps))


def sma_smooth(x: np.ndarray, window: int = 4) -> np.ndarray:
    """Simple 'before-K' moving average matching evaluate.py:53-56.

    For each t, output[t] = mean(x[t-window+1 : t+1]).
    The first window-1 entries are kept as-is (no padding policy in upstream).
    """
    out = x.copy().astype(np.float32)
    if window <= 1 or len(x) <= window:
        return out
    cumsum = np.cumsum(x, axis=0, dtype=np.float64)
    out[window:] = (cumsum[window:] - cumsum[:-window]) / window
    out[:window] = x[:window]
    return out.astype(np.float32)
