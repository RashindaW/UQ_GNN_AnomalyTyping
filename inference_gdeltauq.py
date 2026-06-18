"""K-anchor inference module for the G-DeltaUQ GDN variant.

Loads a trained GDN_GDeltaUQ + AleatoricHead + calibration artifacts, then runs
inference over a TimeDataset. Per batch:
  - Compute pre-anchor hidden representation h_pre once (no_grad).
  - For each of K anchors in the pool, run the anchored layer + head -> (mu_k,
    h_final_k, attention_k).
  - Aggregate: mu_bar = mean over K, U_par = var over K of mu, U_str = var
    over K of attention, U_dist = mean_v U_par_v.
  - Run aleatoric head on h_bar (mean over K of h_final) -> log sigma^2_ale.

Edge attention reshape (load-bearing):
  GraphLayer calls remove_self_loops + add_self_loops on the batched edge
  index. The cosine-top-k always includes the self-edge (cos(v, v) = 1.0 is
  max), so each sample loses V self-edges from its non-self block. Then PyG
  appends B*V self-loops at the end (one per batched node). Net layout:
    - positions [0, B*(topk-1)*V): non-self edges, contiguous per sample
      (sample b in [b*(topk-1)*V, (b+1)*(topk-1)*V)).
    - positions [B*(topk-1)*V, B*topk*V): self-loops, sorted by batched node
      id (b*V + v for b in [0,B), v in [0,V)).
  We use only the non-self block for U_str. Self-loops don't represent
  inter-sensor structural relations, and including them would muddy the
  position-to-(v, u) mapping. The non-self block reshapes cleanly to
  (B, (topk-1)*V) and the (target_v, source_u) mapping comes from the first
  sample's slice of GNNLayer.edge_index_1.

LSA (use_learnable_adj=True) checkpoints:
  The same probe-based path below auto-detects edges_per_sample from a
  test forward, so LSA models with V*(V-1) non-self edges per sample
  (vs (topk-1)*V for the cos-top-K path) work without modification.
  GraphLayer's log(A) injection into alpha is applied before softmax, so
  the att_weight readout already reflects the *gated* attention -- U_str
  is the K-anchor variance of those gated weights, which is the correct
  signal for the LSA epistemic story. edge_index_sample's shape becomes
  (2, V*(V-1)) and is constant across windows.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class GDeltaUQInferenceOutputs:
    mu_bar:            np.ndarray   # (T, V)
    U_par:             np.ndarray   # (T, V)
    U_str:             np.ndarray   # (T, (topk-1)*V) per non-self edge variance
    U_dist:            np.ndarray   # (T,)
    sigma2_ale:        np.ndarray   # (T, V)
    mu_per_anchor:     np.ndarray   # (K, T, V)
    h_bar:             np.ndarray   # (T, V, d) penultimate, anchor-averaged
    ground_truth:      np.ndarray   # (T, V)
    attack_label:      np.ndarray   # (T,)
    edge_index_sample: np.ndarray   # (2, (topk-1)*V) source on row 0, target on row 1
    topk:              int


@dataclass
class LoadedGDeltaUQ:
    model: torch.nn.Module
    aleatoric_head: Optional[torch.nn.Module]
    anchor_pool: torch.Tensor      # (K, V, d_in_to_anchored_layer)
    q_v: Optional[np.ndarray]      # (V,) per-sensor conformal threshold; may be None
    u_bar_norm: dict               # {'U_par': float, 'U_str': float, 'U_dist': float}
    feature_map: list
    cfg: dict                      # hyperparameters dict
    device: torch.device


def run_inference(
    loaded: LoadedGDeltaUQ,
    dataset,
    batch_size: int = 128,
) -> GDeltaUQInferenceOutputs:
    model = loaded.model.eval()
    aleatoric = (loaded.aleatoric_head.eval() if loaded.aleatoric_head is not None else None)
    anchor_pool = loaded.anchor_pool.to(loaded.device)
    K, V, _ = anchor_pool.shape

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    T = len(dataset)
    topk = int(loaded.cfg['topk'])
    d_hidden = int(loaded.cfg['dim'])

    # Probe the first batch to determine edges_per_sample. Without the
    # causal mask this collapses to topk*V (so nonself_per_sample = (topk-1)*V
    # exactly as before). With the mask in-degree is variable but the masked
    # graph is constant across all forward passes, so one probe is sufficient.
    probe_iter = iter(loader)
    probe_x, probe_y, probe_label, probe_edge = next(probe_iter)
    probe_x = probe_x.float().to(loaded.device)
    probe_edge = probe_edge.float().to(loaded.device)
    B_probe = probe_x.shape[0]
    with torch.no_grad():
        h_pre_probe = model.forward_split(probe_x, probe_edge)
        _, _, att_probe = model.forward_anchored(
            h_pre_probe, anchor_pool[0], probe_edge
        )
    edges_per_sample = att_probe.shape[0] // B_probe
    nonself_per_sample = edges_per_sample - V
    if nonself_per_sample <= 0:
        raise RuntimeError(
            f'edges_per_sample={edges_per_sample}, V={V}, '
            f'nonself_per_sample={nonself_per_sample} -- expected '
            'edges_per_sample > V (at least V self-loops + some non-self edges).'
        )

    mu_buf = np.empty((K, T, V), dtype=np.float32)
    Upar_buf = np.empty((T, V), dtype=np.float32)
    Ustr_buf = np.empty((T, nonself_per_sample), dtype=np.float32)
    Udist_buf = np.empty((T,), dtype=np.float32)
    hbar_buf = np.empty((T, V, d_hidden), dtype=np.float32)
    sigma2_ale_buf = np.empty((T, V), dtype=np.float32)
    y_buf = np.empty((T, V), dtype=np.float32)
    label_buf = np.empty((T,), dtype=np.int8)

    edge_index_sample = None
    anchored_idx = model.anchored_layer_idx

    pos = 0
    with torch.no_grad():
        for x, y, label, edge_index in loader:
            x = x.float().to(loaded.device)
            edge_index = edge_index.float().to(loaded.device)
            B = x.shape[0]

            h_pre = model.forward_split(x, edge_index)  # (B, V, d_in)

            mu_stack = torch.empty((K, B, V), device=loaded.device)
            h_stack = torch.empty((K, B, V, d_hidden), device=loaded.device)
            att_stack = torch.empty((K, B, nonself_per_sample), device=loaded.device)

            for k in range(K):
                anchor = anchor_pool[k]
                mu_k, h_k, att_k = model.forward_anchored(h_pre, anchor, edge_index)
                mu_stack[k] = mu_k
                h_stack[k] = h_k
                # att_k: (B*topk*V, heads=1, 1). Take only the non-self block:
                # positions [0, B*(topk-1)*V). Layout is per-sample contiguous
                # in this block, so reshape to (B, (topk-1)*V) is valid.
                att_flat = att_k.view(-1)
                total_nonself = B * nonself_per_sample
                att_stack[k] = att_flat[:total_nonself].view(B, nonself_per_sample)

            mu_bar = mu_stack.mean(dim=0)                    # (B, V)
            U_par = mu_stack.var(dim=0, unbiased=True)       # (B, V)
            U_str = att_stack.var(dim=0, unbiased=True)      # (B, topk*V)
            U_dist = U_par.mean(dim=-1)                      # (B,)
            h_bar = h_stack.mean(dim=0)                      # (B, V, d)

            if aleatoric is not None:
                log_sigma2 = aleatoric(h_bar)
                sigma2_ale = log_sigma2.exp()
            else:
                sigma2_ale = torch.ones_like(mu_bar)

            mu_buf[:, pos:pos + B, :] = mu_stack.cpu().numpy()
            Upar_buf[pos:pos + B, :] = U_par.cpu().numpy()
            Ustr_buf[pos:pos + B, :] = U_str.cpu().numpy()
            Udist_buf[pos:pos + B] = U_dist.cpu().numpy()
            hbar_buf[pos:pos + B, :, :] = h_bar.cpu().numpy()
            sigma2_ale_buf[pos:pos + B, :] = sigma2_ale.cpu().numpy()
            y_buf[pos:pos + B, :] = y.cpu().numpy()
            label_buf[pos:pos + B] = label.cpu().numpy().astype(np.int8)
            pos += B

            if edge_index_sample is None:
                # GNNLayer.edge_index_1 has the post-self-loop batched edge
                # index, shape (2, B*topk*V). The first B*(topk-1)*V columns
                # are the non-self edges, with per-sample contiguous layout.
                # Sample 0 occupies columns [0, (topk-1)*V); row 0 is source,
                # row 1 is target, in batched indexing (0..B*V-1). Sample 0's
                # node IDs are already in [0, V) - no de-batching shift.
                ei = model.gnn_layers[anchored_idx].edge_index_1
                ei_sample = ei[:, :nonself_per_sample].detach().cpu().numpy()
                edge_index_sample = ei_sample.astype(np.int64)

    assert pos == T, f'buffer underflow: filled {pos} / {T}'

    return GDeltaUQInferenceOutputs(
        mu_bar=mu_buf.mean(axis=0).astype(np.float32),
        U_par=Upar_buf,
        U_str=Ustr_buf,
        U_dist=Udist_buf,
        sigma2_ale=sigma2_ale_buf,
        mu_per_anchor=mu_buf,
        h_bar=hbar_buf,
        ground_truth=y_buf,
        attack_label=label_buf,
        edge_index_sample=edge_index_sample,
        topk=topk,
    )


def top_k_edges_per_timestep(U_str, edge_index_sample, top_k=3):
    """For each timestep, the top-k edges (target_v, source_u) by U_str.

    Args:
        U_str: (T, (topk-1)*V) per non-self edge variance.
        edge_index_sample: (2, (topk-1)*V) per-sample non-self edge index
            (source on row 0, target on row 1) as stored by GNNLayer.
        top_k: how many edges to return per timestep.
    Returns:
        out: (T, top_k, 2) int64 array of (target_v, source_u).
    """
    T, total_edges = U_str.shape
    src_row = edge_index_sample[0]
    tgt_row = edge_index_sample[1]
    assert src_row.shape[0] == total_edges, (
        f'edge_index width {src_row.shape[0]} != U_str width {total_edges}'
    )
    # We restricted to the non-self block, so every edge has src != tgt.
    assert (src_row != tgt_row).all(), (
        'edge_index_sample contains self-loops; expected non-self block only'
    )

    # argpartition for top_k positions (unsorted within top_k; sort each row).
    idx_part = np.argpartition(-U_str, kth=top_k, axis=1)[:, :top_k]   # (T, top_k)
    # Sort each row's top_k positions by descending variance.
    rows = np.arange(T)[:, None]
    vals = U_str[rows, idx_part]
    order = np.argsort(-vals, axis=1)
    idx_sorted = np.take_along_axis(idx_part, order, axis=1)            # (T, top_k)

    out = np.empty((T, top_k, 2), dtype=np.int64)
    for j in range(top_k):
        pos = idx_sorted[:, j]
        out[:, j, 0] = tgt_row[pos]
        out[:, j, 1] = src_row[pos]
    return out


def top_k_sensors_by_upar(U_par, top_k=3):
    """For each timestep, return the top-k sensor indices by U_par (sorted desc)."""
    idx_part = np.argpartition(-U_par, kth=top_k, axis=1)[:, :top_k]
    rows = np.arange(U_par.shape[0])[:, None]
    vals = U_par[rows, idx_part]
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx_part, order, axis=1)
