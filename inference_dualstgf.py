"""Ensemble inference for the trained DualSTGF_UQ ensemble.

Sibling of `inference.py` (which is GDN-specific). Same public API and
`InferenceOutputs` dataclass shape `(T, V)` so that scripts/calibrate.py and
scripts/detect.py can dispatch on `manifest['model']` without further changes.

Implementation differences vs `inference.py`:
- Loads `DualSTAGE_UQ` (vendored from dualstgf/) instead of `GDN_UQ`.
- Builds SWaT batches via `dualstgf/datasets/swat.py`'s `SWaTDataset` instead of
  `datasets/TimeDataset.py`.
- DualSTAGE_UQ.forward returns reconstruction tensors of shape `[B*N, W]`. We
  post-process them into the `(T, V)` shape calibrate.py expects by selecting
  the LAST timestep of each window (the "current" timestep, matching the
  forecasting-style residual semantics GDN uses).

Reused from `inference.py`: `apply_lambda`, `apply_sigma_floor`, `sma_smooth`,
`standardised_residual`, `InferenceOutputs`.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent

# IMPORTANT: import from `inference` BEFORE we add the dualstgf subtree to
# sys.path. The dualstgf package has its own `datasets/` which would shadow
# the top-level `datasets/TimeDataset.py` that `inference.py` depends on.
from inference import (  # noqa: E402  -- reuse the post-process helpers
    InferenceOutputs,
    apply_lambda,
    apply_sigma_floor,
    sma_smooth,
    standardised_residual,
)

# After the inference imports have resolved, expose the vendored DualSTGF
# subtree on sys.path. From here on, `datasets` refers to the dualstgf adapter
# registry (where `get_adapter("swat")` lives), not our top-level `datasets/`.
sys.path.insert(0, str(REPO_ROOT / 'dualstgf'))
sys.path.insert(0, str(REPO_ROOT / 'dualstgf' / 'dualstage'))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_repo_path(p: str, repo_root: Path) -> Path:
    if os.path.isabs(p):
        return Path(p)
    return (repo_root / p).resolve()


@dataclass
class DualSTGFEnsembleConfig:
    M: int
    node_num: int
    window_size: int
    train_stride: int
    val_stride: int
    batch: int
    gnn_embed_dim: int
    temp_node_embed_dim: int
    recon_hidden_dim: int
    topk: int
    num_gnn_layers: int
    use_spectral_view: bool
    aug_control: bool
    seeds: list[int]

    # Aliases matching `inference.EnsembleConfig` so calibrate.py / detect.py
    # can dispatch on `manifest['model']` without further code changes.
    @property
    def slide_win(self) -> int:
        return self.window_size

    @property
    def slide_stride(self) -> int:
        return self.train_stride


@dataclass
class LoadedDualSTGFEnsemble:
    members: list[Any]                    # list of DualSTAGE_UQ
    seeds: list[int]
    feature_map: list[str]
    cfg: DualSTGFEnsembleConfig
    device: torch.device

    # `inference.LoadedEnsemble` exposes `fc_edge_index` (used by the GDN-side
    # `build_dataset_from_csv`). DualSTGF builds its FC graph internally, so
    # we expose a placeholder None — the dispatch always passes
    # `ensemble.fc_edge_index` to `build_dataset_from_csv`, which our
    # implementation ignores.
    fc_edge_index: Any = None


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_ensemble(
    manifest_path: str | Path,
    device: str | torch.device,
    repo_root: Optional[str | Path] = None,
) -> LoadedDualSTGFEnsemble:
    manifest_path = Path(manifest_path)
    if repo_root is None:
        repo_root = manifest_path.parent.parent.parent
    repo_root = Path(repo_root)

    with manifest_path.open() as f:
        manifest = json.load(f)
    if manifest.get('model') != 'dualstgf_uq':
        raise ValueError(
            f"manifest['model'] = {manifest.get('model')!r}, expected 'dualstgf_uq'"
        )
    hp = manifest['hyperparameters']

    # The vendored DualSTGF cfg singleton must be configured before model __init__.
    from src.config import cfg
    cfg.set_dataset_params(
        n_nodes=51,                      # SWaT
        window_size=hp['window_size'],
        ocvar_dim=0,
    )
    cfg.device = str(device)

    from src.model.dualstage_uq import DualSTAGE_UQ
    from dualstage.src.data.swat_column_config import MEASUREMENT_VARS

    dev = torch.device(device)
    members: list[DualSTAGE_UQ] = []
    for m in manifest['members']:
        ckpt_path = _resolve_repo_path(m['checkpoint'], repo_root)
        model = DualSTAGE_UQ(
            feat_input_node=1,
            feat_target_node=1,
            feat_input_edge=1,
            aug_control=bool(hp.get('aug_control', False)),
            use_spectral_view=bool(hp.get('use_spectral_view', False)),
            gnn_embed_dim=hp['gnn_embed_dim'],
            temp_node_embed_dim=hp['temp_node_embed_dim'],
            recon_hidden_dim=hp['recon_hidden_dim'],
            topk=hp['topk'],
            num_gnn_layers=hp['num_gnn_layers'],
            with_variance_head=True,
            logvar_clamp=tuple(hp.get('logvar_clamp', [-10.0, 10.0])),
        ).to(dev)
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state)
        model.eval()
        members.append(model)

    cfg_dataclass = DualSTGFEnsembleConfig(
        M=manifest['M'],
        node_num=len(MEASUREMENT_VARS),
        window_size=hp['window_size'],
        train_stride=hp.get('train_stride', 1),
        val_stride=hp.get('val_stride', 5),
        batch=hp['batch'],
        gnn_embed_dim=hp['gnn_embed_dim'],
        temp_node_embed_dim=hp['temp_node_embed_dim'],
        recon_hidden_dim=hp['recon_hidden_dim'],
        topk=hp['topk'],
        num_gnn_layers=hp['num_gnn_layers'],
        use_spectral_view=bool(hp.get('use_spectral_view', False)),
        aug_control=bool(hp.get('aug_control', False)),
        seeds=[m['seed'] for m in manifest['members']],
    )
    return LoadedDualSTGFEnsemble(
        members=members,
        seeds=cfg_dataclass.seeds,
        feature_map=list(MEASUREMENT_VARS),
        cfg=cfg_dataclass,
        device=dev,
    )


# ----------------------------------------------------------------------
# Dataset construction
# ----------------------------------------------------------------------


def build_dataset_from_csv(
    csv_path: str | Path,
    feature_map: list[str],
    fc_edge_index: Any,                    # ignored (DualSTGF builds FC internally)
    slide_win: int,
    slide_stride: int,
    mode: str = 'test',
    label_override: Optional[list[int]] = None,
):
    """Build an SWaTDataset matching the calibrate/detect API contract.

    The signature mirrors `inference.build_dataset_from_csv` so the dispatch in
    calibrate.py / detect.py is uniform. `fc_edge_index` is ignored — DualSTGF's
    SWaTDataset builds the FC graph internally. `label_override` is also ignored
    (the SWaTDataset reads attack labels from the CSV directly).
    """
    if label_override is not None and any(label_override):
        # The calibration script uses this to inject a non-zero label override
        # for the train CSV (which is normal-only). For DualSTGF we don't need
        # this — we read labels from the CSV directly. Quietly ignore.
        pass

    from dualstage.src.data.swat_dataset import SWaTDataset
    return SWaTDataset(
        csv_path=str(csv_path),
        window_size=slide_win,
        stride=1 if mode == 'test' else slide_stride,
        normalize=True,                  # match training-time normalisation
        normalization_stats=None,        # uses per-CSV stats; calibrate.py wraps tmp CSV
                                         # for 𝒞 etc., so per-slice stats are reasonable.
        baseline_only=False,
    )


# ----------------------------------------------------------------------
# Forward + aggregate
# ----------------------------------------------------------------------


def run_inference(
    ensemble: LoadedDualSTGFEnsemble,
    dataset,                             # SWaTDataset
    batch_size: Optional[int] = None,
) -> InferenceOutputs:
    """Run all M members over `dataset`, return per-window aggregated outputs.

    DualSTAGE_UQ outputs `(mu, log_var)` of shape `[B*N, W]`. For compatibility
    with the GDN-style calibrate/detect pipeline, we slice the LAST timestep of
    the window before aggregating: `mu_last = mu.view(B, N, W)[..., -1]` →
    shape `(B, N)`. Same for log_var, ground truth, and attack label.
    """
    if batch_size is None:
        batch_size = ensemble.cfg.batch

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = ensemble.device
    M = ensemble.cfg.M
    V = ensemble.cfg.node_num
    W = ensemble.cfg.window_size
    T = len(dataset)

    mu_buf = np.empty((M, T, V), dtype=np.float32)
    lv_buf = np.empty((M, T, V), dtype=np.float32)
    y_buf = np.empty((T, V), dtype=np.float32)
    label_buf = np.empty((T,), dtype=np.int8)

    # Make sure the cfg singleton is configured for this run.
    from src.config import cfg
    cfg.set_dataset_params(n_nodes=V, window_size=W, ocvar_dim=0)
    cfg.device = str(device)

    pos = 0
    for batch in loader:
        batch = batch.to(device)
        b = int(batch.batch.max().item()) + 1
        # Ground truth (last timestep of each window).
        # batch.x shape: [B*N, W], reshape to [B, N, W], take last column.
        x_bnw = batch.x.view(b, V, W)
        y_last = x_bnw[..., -1]                                  # (B, N)
        labels = batch.y.view(-1)                                 # (B,)

        with torch.no_grad():
            for m_idx, member in enumerate(ensemble.members):
                mu, log_var = member(batch)
                mu_bnw = mu.view(b, V, W)
                lv_bnw = log_var.view(b, V, W)
                mu_buf[m_idx, pos:pos + b, :] = mu_bnw[..., -1].detach().cpu().numpy()
                lv_buf[m_idx, pos:pos + b, :] = lv_bnw[..., -1].detach().cpu().numpy()

        y_buf[pos:pos + b, :] = y_last.cpu().numpy()
        label_buf[pos:pos + b] = labels.cpu().numpy().astype(np.int8)
        pos += b

    assert pos == T, f"buffer underflow: filled {pos} / {T}"

    sigma2_per_member = np.exp(lv_buf)
    mu_bar = mu_buf.mean(axis=0)
    sigma2_a = sigma2_per_member.mean(axis=0)
    sigma2_e = mu_buf.var(axis=0)
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
