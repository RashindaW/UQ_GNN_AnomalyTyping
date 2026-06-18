"""Learnable sparse adjacency for GDN_GDeltaUQ (Tank-style group-lasso, L=1).

Replaces GDN's hard top-K cosine-similarity neighbour selection with a
differentiable directed adjacency

    A[v, u] = sigmoid( phi(E_v) . psi(E_u) / tau )

where E = self.embedding.weight from the parent model. The diagonal is
forced to 1 (self-loops always survive) and excluded from the L1 penalty,
which is the natural reduction of Tank et al.'s group-lasso when there is
only one lag dimension per edge — the group collapses to a scalar.

Edge weighting (the off-diagonal A entries) is then injected into
GraphLayer's attention via `alpha += log(A)` immediately before the
per-target softmax, so `A_{vu} -> 0` cleanly suppresses that edge's
message contribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LearnableAdjacency(nn.Module):
    """Two decoupled linear projections of the parent's embedding produce
    source / target roles for a directed VxV adjacency.

    The embedding is detached before being fed into phi / psi so the L1
    penalty does not back-propagate into the embedding (which GraphLayer
    consumes separately as an attention bias). If the resulting A turns
    out to be too rigid, drop the detach upstream.
    """

    def __init__(self, embed_dim: int, num_nodes: int, tau: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.tau = float(tau)
        self.phi = nn.Linear(embed_dim, embed_dim, bias=False)
        self.psi = nn.Linear(embed_dim, embed_dim, bias=False)
        # Last A computed by forward(); set after every forward so
        # penalty() and external callers (sparsity diagnostics, the
        # lambda-search driver) can read it without re-running the model.
        self._last_A: torch.Tensor | None = None

    def forward(self, embedding_weight: torch.Tensor) -> torch.Tensor:
        """embedding_weight: (V, embed_dim) -- typically `self.embedding.weight`
        from the parent GDN_GDeltaUQ. Returns A of shape (V, V).
        """
        if embedding_weight.shape != (self.num_nodes, self.embed_dim):
            raise ValueError(
                f'embedding_weight shape {tuple(embedding_weight.shape)} != '
                f'({self.num_nodes}, {self.embed_dim})'
            )
        # detach so phi/psi gradients do not push the embedding around.
        e = embedding_weight.detach()
        phi_e = self.phi(e)                                  # (V, d)
        psi_e = self.psi(e)                                  # (V, d)
        logits = (phi_e @ psi_e.t()) / self.tau              # (V, V)
        A = torch.sigmoid(logits)
        # Force self-loops on and zero them from the off-diagonal for
        # penalty/index extraction downstream.
        eye = torch.eye(self.num_nodes, device=A.device, dtype=A.dtype)
        A = A * (1.0 - eye) + eye
        self._last_A = A
        return A

    def penalty(self) -> torch.Tensor:
        """L1 norm of the off-diagonal entries of the most recent A.

        Tank et al.'s group-lasso over lag dimensions collapses to a scalar
        L1 here because L=1 (one scalar per edge). penalty() is meaningful
        only after at least one forward() call.
        """
        if self._last_A is None:
            raise RuntimeError(
                'LearnableAdjacency.penalty() called before any forward(); '
                'call forward(self.embedding.weight) first.'
            )
        eye = torch.eye(self.num_nodes,
                        device=self._last_A.device,
                        dtype=self._last_A.dtype)
        off_diag = self._last_A * (1.0 - eye)
        return off_diag.abs().sum()

    @torch.no_grad()
    def mean_degree(self, edge_threshold: float = 1e-3) -> float:
        """Diagnostic: mean per-target out-edge count above `edge_threshold`,
        useful for the lambda-search stopping rule. Reads the last A; no
        recompute. Returns 0.0 if forward() has not run yet."""
        if self._last_A is None:
            return 0.0
        eye = torch.eye(self.num_nodes,
                        device=self._last_A.device,
                        dtype=self._last_A.dtype)
        off_diag = self._last_A * (1.0 - eye)
        return float((off_diag > edge_threshold).float().sum(dim=-1).mean().item())
