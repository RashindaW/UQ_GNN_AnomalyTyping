"""Domain-knowledge causal-graph mask for GDN-family models.

The mask is the 51x51 binary adjacency built by
``scripts/cf_domain_causal.py`` and stored in
``data/swat/causal_adjacency.npy`` (alongside a
``causal_adjacency_features.json`` listing the feature order at save time).

Convention: ``A_src_tgt[i, j] == 1`` means "node i causes node j"
(row = source, column = target).

Usage from a model:
    causal_mask = load_causal_mask('data/swat/causal_adjacency.npy',
                                    feature_list=feature_map)
    gated_edge_index, masked_adj_tgt_src = apply_and_mask(
        topk_indices_ji, causal_mask, node_num, keep_self=True
    )

``gated_edge_index`` is ``[src_row, tgt_row]`` of shape ``(2, E)`` where E
is the number of edges that survive the AND. Variable in-degree per
target node -- downstream code that assumes a fixed in-degree K will
need to be updated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_causal_mask(npy_path: str | Path,
                     feature_list: list[str],
                     features_sidecar: str | Path | None = None,
                     ) -> torch.Tensor:
    """Load the saved binary causal matrix and reindex to ``feature_list``.

    Args:
        npy_path: path to ``causal_adjacency.npy``.
        feature_list: the feature order used by the training pipeline (e.g.
            the order in ``data/swat/list.txt`` consumed by ``main.py``).
        features_sidecar: optional path to the JSON sidecar listing the
            feature order at save time. Defaults to a sibling file named
            ``causal_adjacency_features.json``.

    Returns:
        ``torch.bool`` tensor of shape ``(V, V)`` in src->tgt convention,
        reindexed so row/column ``i`` corresponds to ``feature_list[i]``.
    """
    npy_path = Path(npy_path)
    A = np.load(npy_path)
    if features_sidecar is None:
        features_sidecar = npy_path.with_name('causal_adjacency_features.json')
    saved_features: list[str] = json.loads(Path(features_sidecar).read_text())

    if set(saved_features) != set(feature_list):
        only_saved = set(saved_features) - set(feature_list)
        only_runtime = set(feature_list) - set(saved_features)
        raise ValueError(
            f'causal mask features do not match runtime features.\n'
            f'  only in saved matrix: {sorted(only_saved)}\n'
            f'  only in runtime list: {sorted(only_runtime)}'
        )

    saved_idx = {name: i for i, name in enumerate(saved_features)}
    perm = np.array([saved_idx[name] for name in feature_list], dtype=np.int64)
    A_reindexed = A[np.ix_(perm, perm)]
    return torch.from_numpy(A_reindexed).bool()


def apply_and_mask(topk_indices_ji: torch.Tensor,
                   causal_mask_src_tgt: torch.Tensor,
                   node_num: int,
                   keep_self: bool = True,
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """AND the per-forward-pass top-K graph with the causal mask.

    Args:
        topk_indices_ji: ``(V, K)`` int tensor. Row ``i`` lists the K source
            neighbours of target node ``i`` (matches GDN's convention).
        causal_mask_src_tgt: ``(V, V)`` bool tensor in src->tgt convention.
        node_num: V.
        keep_self: if True, force the diagonal of the effective mask to True
            so every node retains its self-edge regardless of what the
            domain matrix says. Forecasting models depend on this.

    Returns:
        gated_edge_index: ``(2, E)`` ``long`` tensor, row 0 = source,
            row 1 = target (same ordering used downstream in GDN).
        masked_adj_tgt_src: ``(V, V)`` bool tensor, row = target,
            col = source, representing the masked graph. Persisted as
            ``model.learned_graph`` for inspection.
    """
    device = topk_indices_ji.device
    V = node_num

    A_learned_tgt_src = torch.zeros(V, V, dtype=torch.bool, device=device)
    A_learned_tgt_src.scatter_(1, topk_indices_ji, True)

    A_domain_tgt_src = causal_mask_src_tgt.to(device).t()
    if keep_self:
        eye = torch.eye(V, dtype=torch.bool, device=device)
        A_domain_tgt_src = A_domain_tgt_src | eye

    A_masked = A_learned_tgt_src & A_domain_tgt_src

    tgt_rows, src_cols = A_masked.nonzero(as_tuple=True)
    gated_edge_index = torch.stack([src_cols, tgt_rows], dim=0).long()
    return gated_edge_index, A_masked


def apply_causal_restrict(cos_ji_mat: torch.Tensor,
                          allowed_src_tgt: torch.Tensor,
                          topk: int,
                          mode: str = 'pure',
                          keep_self: bool = True,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-top-K causal restriction: pick each node's <=K neighbours from the
    cosine-similarity matrix, but constrained by a domain causal scaffold.

    Unlike ``apply_and_mask`` (which intersects the *already-selected* top-K
    with the mask, leaving almost nothing), this restricts the *candidate set*
    BEFORE selection, so the domain prior is injected rather than destroyed.

    Args:
        cos_ji_mat: ``(V, V)`` float cosine matrix. Row i = target node, column
            j = candidate source. ``cos_ji_mat[i, j]`` is the score for edge
            j -> i (matches GDN's convention where ``topk`` is taken per row).
        allowed_src_tgt: ``(V, V)`` bool scaffold in src->tgt convention
            (``allowed[j, i]`` True means source j is an allowed causal parent
            of target i).
        topk: K — max neighbours per target.
        mode: 'pure'   -> keep ONLY allowed parents (cosine-ranked, <=K). Nodes
                          with fewer than K allowed parents get fewer edges
                          (variable in-degree); nodes with none get only the
                          self-loop PyG adds.
              'augment' -> guarantee the allowed parents are selected first,
                          then fill the remaining slots up to K with the best
                          cosine neighbours (preserves a full-K neighbourhood,
                          so detection is unharmed; weaker causal purity).
        keep_self: force the diagonal True so each node keeps its self-edge.
            (Note: GraphLayer strips+re-adds self-loops unconditionally, so this
            only affects the stored learned_graph buffer / the pre-strip count.)

    Returns:
        gated_edge_index: ``(2, E)`` long, row 0 = source, row 1 = target.
        masked_adj_tgt_src: ``(V, V)`` bool, row = target, col = source.
    """
    device = cos_ji_mat.device
    V = cos_ji_mat.shape[0]
    allowed_tgt_src = allowed_src_tgt.to(device).bool().t()   # row=target, col=source
    if keep_self:
        eye = torch.eye(V, dtype=torch.bool, device=device)
        allowed_tgt_src = allowed_tgt_src | eye

    if mode == 'pure':
        NEG = torch.finfo(cos_ji_mat.dtype).min
        score = cos_ji_mat.masked_fill(~allowed_tgt_src, NEG)
        k = min(topk, V)
        topk_vals, topk_idx = torch.topk(score, k, dim=-1)
        valid = topk_vals > NEG                                # (V, k) bool
        A_masked = torch.zeros(V, V, dtype=torch.bool, device=device)
        rows = torch.arange(V, device=device).unsqueeze(1).expand(-1, k)
        A_masked[rows[valid], topk_idx[valid]] = True
    elif mode == 'augment':
        # Boost allowed entries so they are selected first; remaining slots
        # fall back to best cosine among the rest. BIG dominates cos in [-1, 1].
        BIG = 1e4
        score = cos_ji_mat + allowed_tgt_src.to(cos_ji_mat.dtype) * BIG
        k = min(topk, V)
        _, topk_idx = torch.topk(score, k, dim=-1)
        A_masked = torch.zeros(V, V, dtype=torch.bool, device=device)
        A_masked.scatter_(1, topk_idx, True)
    else:
        raise ValueError(f"mode must be 'pure' or 'augment', got {mode!r}")

    tgt_rows, src_cols = A_masked.nonzero(as_tuple=True)
    gated_edge_index = torch.stack([src_cols, tgt_rows], dim=0).long()
    return gated_edge_index, A_masked
