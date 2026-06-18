"""Mahalanobis OOD score against the training-set distribution of penultimate-layer reps.

Step 5 of the inference outline. Per-node Gaussian fit on phi_v(X_train), then
Omega_v(X) = sqrt( (phi_v(X) - mu_v)^T inv(Sigma_v) (phi_v(X) - mu_v) ) at inference.

The penultimate layer for GDN_UQ is the post-dropout per-node tensor that feeds
both heads (out tensor on models/GDN_UQ.py around line 91). We capture it via a
forward hook on the dropout module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.TimeDataset import TimeDataset
from models.GDN_UQ import GDN_UQ


@dataclass
class MahalanobisFit:
    """Per-node Gaussian fit. Shapes: mean (V, d), inv_cov (V, d, d)."""
    mean: np.ndarray             # (V, d)
    inv_cov: np.ndarray          # (V, d, d)
    log_det_cov: np.ndarray      # (V,) — for diagnostic / log-likelihood form


# ----------------------------------------------------------------------
# Penultimate-layer extraction
# ----------------------------------------------------------------------


class _PenultimateCapture:
    """Forward hook helper: captures the post-dropout per-node tensor."""

    def __init__(self):
        self.captured: torch.Tensor | None = None

    def __call__(self, module, inputs, output):
        # output of self.dp has shape (batch, node_num, dim*edge_set_num).
        self.captured = output.detach()


def _extract_phi(member: GDN_UQ, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """Run a single member over `dataloader` and collect the penultimate-layer rep.

    Returns array of shape (T, V, d).
    """
    cap = _PenultimateCapture()
    handle = member.dp.register_forward_hook(cap)
    member.eval()

    tensors: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for batch in dataloader:
                x, _y, _label, edge_index = batch
                x = x.to(device).float()
                edge_index = edge_index.to(device).long()
                cap.captured = None
                _ = member(x, edge_index)
                phi = cap.captured                    # (batch, node_num, d)
                if phi is None:
                    raise RuntimeError("forward hook did not fire")
                tensors.append(phi.cpu().numpy())
    finally:
        handle.remove()

    return np.concatenate(tensors, axis=0).astype(np.float32)


def _extract_phi_ensemble_avg(
    members: list[GDN_UQ], dataloader: DataLoader, device: torch.device,
) -> np.ndarray:
    """Run every member over `dataloader` and average the captured reps.

    Returns array of shape (T, V, d) with values averaged across the M members.
    Used when `ood_mode == 'ensemble_avg'` (RESULTS.md future-work #6) — gives
    a single-rep OOD signal that draws on the consensus of all members.
    """
    M = len(members)
    if M == 0:
        raise ValueError('members list is empty')
    accum: np.ndarray | None = None
    for member in members:
        phi_m = _extract_phi(member, dataloader, device)         # (T, V, d)
        if accum is None:
            accum = phi_m
        else:
            accum = accum + phi_m
    return (accum / float(M)).astype(np.float32)


def _members_to_phi(
    members_or_member,
    dataloader: DataLoader,
    device: torch.device,
    mode: str = 'single',
) -> np.ndarray:
    """Dispatch helper: 'single' → first member only, 'ensemble_avg' → mean."""
    if isinstance(members_or_member, (list, tuple)):
        if mode == 'ensemble_avg':
            return _extract_phi_ensemble_avg(list(members_or_member), dataloader, device)
        if mode == 'single':
            return _extract_phi(members_or_member[0], dataloader, device)
        raise ValueError(f'unknown ood mode: {mode!r}')
    # Single member passed directly (backward-compat path).
    return _extract_phi(members_or_member, dataloader, device)


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------


def fit_mahalanobis(
    member,                                # GDN_UQ OR list[GDN_UQ]
    dataset: TimeDataset,
    device: torch.device,
    batch_size: int = 64,
    eps_reg: float = 1e-3,
    ood_mode: str = 'single',              # 'single' | 'ensemble_avg'
) -> MahalanobisFit:
    """Compute per-node mean + inverse covariance of the penultimate rep on `dataset`.

    `eps_reg` is added to the diagonal of each per-node covariance before inversion
    (Tikhonov-style regularisation) so the inverse is stable for sensors whose
    representation occasionally collapses.

    When `member` is a list of GDN_UQ instances and `ood_mode='ensemble_avg'`,
    the per-(t, v) penultimate reps are averaged across the M members before
    fitting (RESULTS.md future-work #6).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    phi = _members_to_phi(member, loader, device, mode=ood_mode)    # (T, V, d)

    T, V, d = phi.shape
    mean = phi.mean(axis=0)                           # (V, d)
    centred = phi - mean[None]                        # (T, V, d)

    # Per-node covariance (V, d, d).
    # einsum: cov[v, i, j] = (1/T) Σ_t centred[t, v, i] * centred[t, v, j]
    cov = np.einsum('tvi,tvj->vij', centred, centred) / T

    inv_cov = np.empty_like(cov)
    log_det = np.empty(V, dtype=np.float64)
    eye = np.eye(d, dtype=cov.dtype)
    for v in range(V):
        c = cov[v] + eps_reg * eye
        inv_cov[v] = np.linalg.inv(c)
        sign, ld = np.linalg.slogdet(c)
        log_det[v] = ld

    return MahalanobisFit(
        mean=mean.astype(np.float32),
        inv_cov=inv_cov.astype(np.float32),
        log_det_cov=log_det.astype(np.float64),
    )


# ----------------------------------------------------------------------
# Score
# ----------------------------------------------------------------------


def score_mahalanobis(
    member,                                # GDN_UQ OR list[GDN_UQ]
    dataset: TimeDataset,
    fit: MahalanobisFit,
    device: torch.device,
    batch_size: int = 64,
    ood_mode: str = 'single',              # 'single' | 'ensemble_avg'
) -> np.ndarray:
    """Per-(t, v) Omega score. Returns (T, V) array.

    Pass `ood_mode='ensemble_avg'` with a list of members to score against the
    across-ensemble averaged representation; matches the corresponding fit.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    phi = _members_to_phi(member, loader, device, mode=ood_mode)    # (T, V, d)

    centred = phi - fit.mean[None]                    # (T, V, d)
    # quad[t, v] = centred[t, v]^T inv_cov[v] centred[t, v]
    # Use einsum: (T, V, d) x (V, d, d) -> (T, V, d), then dot with centred -> (T, V)
    tmp = np.einsum('tvi,vij->tvj', centred, fit.inv_cov)
    quad = np.einsum('tvj,tvj->tv', tmp, centred)
    quad = np.maximum(quad, 0.0)
    return np.sqrt(quad).astype(np.float32)
