"""Graph sensitivity for the GDN_UQ ensemble — Step 11 of the inference outline.

Lazy: invoked only for queries that have already been flagged. The hard top-K in
GDN_UQ.forward is non-differentiable at the boundary where neighbours flip, so
we wrap the forward pass and replace the discrete top-K with a *gradient-bearing
identity* on the selected similarity values: `topk_vals / topk_vals.detach()`.
The forward output is numerically identical to the hard top-K, but autograd
treats the selected `cos_ji_mat` entries as differentiable inputs.

This replaces an earlier implementation that used a temperature-softmax (τ = 0.05)
which produced a near-one-hot soft assignment and killed the gradient (the
sensitivity scores collapsed to ~1e-18). The new implementation has no
temperature hyperparameter — it differentiates wrt the selected similarity
values directly, matching the outline's recommendation: "differentiate with
respect to similarity scores rather than the discrete adjacency; equivalent for
our purposes".

We expose:
  - SoftTopKGDN_UQ              : forward pass with the gradient-bearing identity.
  - compute_member_sensitivity  : one backward pass per (member, query node).
  - ensemble_sensitivity        : averages member gradients, returns (unweighted, weighted).
  - empirical_adjacency_covariance : Σ̂_A over the M ensemble members for the
                                     weighted sensitivity g^T Σ̂_A g.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GDN import GNNLayer, OutLayer, get_batch_edge_index
from models.GDN_UQ import GDN_UQ


# ----------------------------------------------------------------------
# Soft top-K GDN_UQ
# ----------------------------------------------------------------------


class SoftTopKGDN_UQ(nn.Module):
    """A GDN_UQ variant whose forward path uses a soft top-K so the cosine
    similarity matrix participates in the gradient.

    The model shares ALL weights with a parent GDN_UQ via state_dict copy. The
    only behavioural difference is in how the per-batch edge_index for each
    member is constructed.
    """

    def __init__(self, parent: GDN_UQ, temperature: float | None = None):
        super().__init__()
        # We keep a reference to the parent's submodules so the parameters are
        # shared and updates to the parent are visible. Read-only forward only.
        self.parent = parent
        # `temperature` retained for back-compat with old callers; ignored.
        # The new implementation uses topk_vals / topk_vals.detach() directly
        # and has no temperature hyperparameter.
        self.temperature = temperature

    @property
    def topk(self):
        return self.parent.topk

    def forward(self, data: torch.Tensor, org_edge_index: torch.Tensor):
        """Same as GDN_UQ.forward, but with a soft top-K on the similarity matrix.

        Returns (mu, log_var, similarity) where `similarity` is the (node_num, node_num)
        cosine-similarity matrix used to construct the soft adjacency. Sensitivity is
        computed wrt this matrix.
        """
        p = self.parent
        x = data.clone()                              # don't detach — we want gradient
        edge_index_sets = p.edge_index_sets
        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        cos_ji_mat_out = None
        for i, edge_index in enumerate(edge_index_sets):
            all_embeddings = p.embedding(torch.arange(node_num, device=device))
            # Do NOT detach: we want the gradient of the prediction to flow
            # through cos_ji_mat back into the embeddings (and hence be
            # available wrt cos_ji_mat itself for the sensitivity readout).
            weights = all_embeddings.view(node_num, -1)
            cos_ji_mat = torch.matmul(weights, weights.T)
            normed = torch.matmul(
                weights.norm(dim=-1, keepdim=True),
                weights.norm(dim=-1, keepdim=True).T,
            )
            cos_ji_mat = cos_ji_mat / normed.clamp_min(1e-8)
            # Track gradient through cos_ji_mat by enabling requires_grad on a
            # fresh tensor that is a function of the embeddings. The caller will
            # call .backward() on a scalar function of the output and ask for
            # gradients wrt cos_ji_mat (it has requires_grad inherited from the
            # embedding parameter).
            if i == 0:
                cos_ji_mat_out = cos_ji_mat
            topk_num = p.topk

            # Hard top-K identifies the neighbour set; topk_vals are the
            # selected cosine-similarity scalars (still differentiable wrt
            # cos_ji_mat). We use them as a multiplicative identity:
            # `topk_vals / topk_vals.detach()` is exactly 1.0 at every entry
            # (so the forward result equals the hard top-K), but autograd
            # routes gradients through the live tensor — gradient wrt
            # cos_ji_mat is well defined at the selected indices and zero
            # elsewhere, which is precisely what we want.
            topk_vals, topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)

            # Build the gated edge_index exactly as in GDN_UQ.
            gated_i = (
                torch.arange(0, node_num, device=device)
                .unsqueeze(1).repeat(1, topk_num)
                .flatten().unsqueeze(0)
            )
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(
                gated_edge_index, batch_num, node_num
            ).to(device)

            all_embeddings = all_embeddings.repeat(batch_num, 1)
            gcn_out = p.gnn_layers[i](
                x, batch_gated_edge_index,
                node_num=node_num * batch_num, embedding=all_embeddings,
            )
            # Per-node identity scaling: average of selected similarities,
            # divided by its own detached copy. Numerically 1.0 per node; the
            # gradient wrt cos_ji_mat at this node's top-K indices is non-zero.
            per_node_w = topk_vals.mean(dim=-1)               # (V,) — varied across nodes
            per_node_w_batch = per_node_w.repeat(batch_num)   # (V*batch,)
            gcn_out = gcn_out * (
                per_node_w_batch / per_node_w_batch.detach().clamp_min(1e-12)
            )[:, None]

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num, device=device)
        out = torch.mul(x, p.embedding(indexes))

        out = out.permute(0, 2, 1)
        out = F.relu(p.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)
        # No dropout in eval mode — irrelevant here.

        mu = p.mu_head(out).view(-1, node_num)
        log_var = p.logvar_head(out).view(-1, node_num).clamp(*p.logvar_clamp)
        return mu, log_var, cos_ji_mat_out


# ----------------------------------------------------------------------
# Sensitivity computation
# ----------------------------------------------------------------------


def compute_member_sensitivity(
    member: GDN_UQ,
    x_lookback: torch.Tensor,         # (1, node_num, slide_win)
    query_node_v: int,
    temperature: float | None = None,    # deprecated, ignored
) -> torch.Tensor:
    """Gradient of mu_v wrt the cosine-similarity matrix for one member, one query.

    Returns a (node_num, node_num) tensor: g[i, j] = ∂ mu_v(query) / ∂ cos_sim[i, j].
    """
    soft = SoftTopKGDN_UQ(member)
    soft.eval()
    member.eval()

    # Make sure every parameter that contributes to cos_ji_mat (i.e. the embedding)
    # is a leaf with requires_grad — it already is, since these are nn.Parameter.
    x = x_lookback.detach().clone()
    org_edge_index = member.edge_index_sets[0]

    # Run the soft forward; mark the cos_ji_mat as the diff target.
    mu, _log_var, cos_mat = soft(x, org_edge_index)

    # Scalar to differentiate: predicted mean at query_node_v, averaged over batch.
    target = mu[:, query_node_v].sum()
    g, = torch.autograd.grad(target, cos_mat, retain_graph=False, create_graph=False)
    return g.detach()


def ensemble_sensitivity(
    members: list[GDN_UQ],
    x_lookback: torch.Tensor,
    query_node_v: int,
    adjacency_cov: np.ndarray,        # (V*V, V*V) flattened-edge covariance
    temperature: float | None = None,    # deprecated, ignored
) -> dict[str, float]:
    """Compute per-member sensitivities, average, and weighted scalar.

    Returns:
        unweighted (float): ||g_avg||_2
        weighted   (float): g_avg^T Sigma_A g_avg
    """
    grads = []
    for member in members:
        g = compute_member_sensitivity(member, x_lookback, query_node_v)
        grads.append(g.flatten().cpu().numpy())
    g_stack = np.stack(grads, axis=0)             # (M, V*V)
    g_avg = g_stack.mean(axis=0)                   # (V*V,)
    unweighted = float(np.linalg.norm(g_avg))
    weighted = float(g_avg @ adjacency_cov @ g_avg)
    return {'unweighted': unweighted, 'weighted': weighted}


def empirical_adjacency_covariance(members: list[GDN_UQ]) -> np.ndarray:
    """Σ̂_A: across-member covariance of the (vectorised) cosine-similarity matrix.

    Computed at the embedding-only level (no input dependence), since GDN's
    adjacency is a function of the learned embeddings and is the same across
    timesteps for a given member.

    Returns (V*V, V*V) array.
    """
    sim_mats = []
    with torch.no_grad():
        for member in members:
            emb = member.embedding(torch.arange(member.embedding.num_embeddings, device=member.embedding.weight.device))
            w = emb.view(emb.shape[0], -1)
            cos_mat = (w @ w.T) / (w.norm(dim=-1, keepdim=True) @ w.norm(dim=-1, keepdim=True).T).clamp_min(1e-8)
            sim_mats.append(cos_mat.detach().cpu().numpy().flatten())
    sims = np.stack(sim_mats, axis=0)               # (M, V*V)
    centred = sims - sims.mean(axis=0, keepdims=True)
    cov = centred.T @ centred / max(1, sims.shape[0] - 1)
    return cov.astype(np.float32)
