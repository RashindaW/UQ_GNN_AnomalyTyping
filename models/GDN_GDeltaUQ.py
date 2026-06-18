"""GDN + G-DeltaUQ variant.

Multi-layer GDN (default 2 layers) with hidden-layer stochastic anchoring at the
input to the final MPNN layer. During training the anchor is sampled by
batch-shuffling the pre-anchor hidden representation and detaching it. At
inference K anchors from a calibrated pool are passed in explicitly, producing
K predictions whose spread defines the parametric and structural epistemic
uncertainty signals.

The pre-anchor backbone (layers 0 .. n-2) is anchor-invariant and can be
computed once per timestep via `forward_split`; the anchored layer plus head
then runs K times.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .GDN import OutLayer, GNNLayer, get_batch_edge_index
from .causal_mask import apply_and_mask, apply_causal_restrict
from .learnable_adj import LearnableAdjacency


class GDN_GDeltaUQ(nn.Module):
    """G-DeltaUQ-anchored GDN.

    Args:
        edge_index_sets: list of edge_index tensors. Only the first is used by
            this implementation (matches existing GDN convention; SWaT uses 1).
        node_num: number of sensors V.
        dim: hidden dimension d (per GNN layer output).
        out_layer_inter_dim: hidden width of the final MLP head.
        input_dim: window length W (slide_win).
        out_layer_num: depth of the final MLP head.
        topk: number of neighbours in the learned graph.
        n_gnn_layers: number of stacked GNN layers (>= 2). Anchoring happens at
            the input to the last layer. Must be >= 2 for the structural
            epistemic signal to be non-degenerate.
    """

    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256,
                 input_dim=10, out_layer_num=1, topk=20, n_gnn_layers=2,
                 causal_mask=None, causal_mask_keep_self=True,
                 use_learnable_adj=False, lsa_tau=1.0,
                 causal_restrict=None, causal_restrict_mode='pure',
                 causal_restrict_keep_self=True):
        """
        causal_mask:           optional (V, V) bool tensor in src->tgt
                               convention (see models/causal_mask.py). If
                               provided, the per-forward-pass top-K graph
                               is AND'd with this mask before message-passing.
                               WARNING: variable in-degree per node when
                               this is enabled; downstream code that assumes
                               a fixed in-degree K (e.g. inference_gdeltauq.py)
                               must be updated.
        causal_mask_keep_self: when AND-ing, force the diagonal of the mask
                               to True so each node retains its self-edge.
        use_learnable_adj:     opt in to the Tank-style Learnable Sparse
                               Adjacency (models/learnable_adj.py). When True,
                               the hard top-K cos-sim block is replaced by
                               a complete VxV soft adjacency A whose off-diag
                               entries gate the GraphLayer attention via
                               log(A) added to the attention logits. Use with
                               train_gdeltauq_jointnll(..., lambda_adj>0) for
                               group-lasso sparsity. Mutually exclusive with
                               causal_mask.
        lsa_tau:               temperature for the sigmoid score; lower tau =
                               sharper (more 0/1-like) edges.
        causal_restrict:       optional (V, V) bool tensor in src->tgt
                               convention. If provided, the cosine-similarity
                               matrix is RESTRICTED to causally-allowed parent
                               pairs BEFORE top-K selection (unlike causal_mask
                               which AND's AFTER selection). Injects the prior
                               instead of intersecting it. Variable in-degree.
        causal_restrict_mode:  'pure' (only allowed parents survive top-K) or
                               'augment' (allowed parents guaranteed, remaining
                               slots filled to K by best cosine). See
                               models/causal_mask.py::apply_causal_restrict.
        causal_restrict_keep_self: force each node's self-edge in the restricted
                               candidate set.
        """
        super().__init__()

        if n_gnn_layers < 2:
            raise ValueError(
                f"GDN_GDeltaUQ requires n_gnn_layers >= 2 (got {n_gnn_layers}); "
                "structural epistemic signal U^str collapses for single-layer "
                "stacks because anchoring would reduce to raw-input anchoring."
            )
        if use_learnable_adj and causal_mask is not None:
            raise ValueError(
                'use_learnable_adj=True is mutually exclusive with causal_mask: '
                'LSA constructs the graph from scratch, causal_mask filters the '
                'top-K graph -- composing them needs design work that is out of '
                'scope for the LSA change.'
            )
        if causal_restrict is not None and (causal_mask is not None or use_learnable_adj):
            raise ValueError(
                'causal_restrict is mutually exclusive with causal_mask and '
                'use_learnable_adj: it is an alternative graph-construction path '
                '(restrict cosine candidates before top-K).'
            )

        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.dim = dim
        self.topk = topk
        self.n_gnn_layers = n_gnn_layers
        self.anchored_layer_idx = n_gnn_layers - 1
        self.causal_mask_keep_self = bool(causal_mask_keep_self)
        self.use_learnable_adj = bool(use_learnable_adj)
        if causal_mask is not None:
            cm = causal_mask.bool()
            if cm.shape != (node_num, node_num):
                raise ValueError(
                    f'causal_mask shape {tuple(cm.shape)} != '
                    f'({node_num}, {node_num})'
                )
            # persistent=False: the mask is static config (path lives in
            # hyperparameters.json), not a learned weight, so it stays out
            # of state_dict. Calibration/eval scripts re-attach it via the
            # constructor when loading a masked checkpoint.
            self.register_buffer('causal_mask', cm, persistent=False)
        else:
            self.causal_mask = None

        if self.use_learnable_adj:
            self.adj = LearnableAdjacency(dim, node_num, tau=lsa_tau)
        else:
            self.adj = None

        self.causal_restrict_mode = str(causal_restrict_mode)
        self.causal_restrict_keep_self = bool(causal_restrict_keep_self)
        if causal_restrict is not None:
            cr = causal_restrict.bool()
            if cr.shape != (node_num, node_num):
                raise ValueError(
                    f'causal_restrict shape {tuple(cr.shape)} != '
                    f'({node_num}, {node_num})'
                )
            # non-persistent: static config (path in hyperparameters.json),
            # re-attached by calibrate/eval/cf_engine when loading a checkpoint.
            self.register_buffer('causal_restrict', cr, persistent=False)
        else:
            self.causal_restrict = None

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        # GNN stack: layer 0 takes raw window features (W -> d); intermediate
        # layers map d -> d; the anchored final layer takes [h - C ; C] of
        # width 2d and produces d.
        gnn_layers = []
        for layer_idx in range(n_gnn_layers):
            if layer_idx == 0:
                in_ch = input_dim
            elif layer_idx == self.anchored_layer_idx:
                in_ch = 2 * dim
            else:
                in_ch = dim
            gnn_layers.append(
                GNNLayer(in_ch, dim, inter_dim=dim + embed_dim, heads=1)
            )
        self.gnn_layers = nn.ModuleList(gnn_layers)

        self.learned_graph = None

        # Single edge_index_set assumed (matches existing GDN on SWaT); the
        # head takes the per-node hidden of width d.
        self.out_layer = OutLayer(
            dim, node_num, out_layer_num, inter_num=out_layer_inter_dim
        )

        self.cache_edge_index_sets = [None] * len(edge_index_sets)
        self.dp = nn.Dropout(0.2)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    # ------------------------------------------------------------------ #
    # Internal helper: build the learned top-k graph and batched edge
    # index once per forward (matches models/GDN.py:139-160).
    # ------------------------------------------------------------------ #
    def _build_learned_graph(self, batch_num, device):
        """Build the per-forward batched edge index.

        Returns (batch_gated, broadcast_embeddings, edge_weight):
            edge_weight is None for the legacy top-K / causal-mask paths and
            a length-(B*V*(V-1)) tensor for the LSA path. Callers must
            forward edge_weight to GNNLayer; GraphLayer injects log(A) into
            the per-target softmax.
        """
        all_embeddings = self.embedding(torch.arange(self.node_num).to(device))

        if self.use_learnable_adj:
            # Tank-style learnable directed adjacency. A is V x V with rows
            # = target, cols = source (matches GDN cos_ji_mat convention).
            A = self.adj(self.embedding.weight)            # (V, V)
            self.learned_graph = A
            V = self.node_num
            # Complete non-self edge index: for each receiver v, list all
            # V-1 sources u != v in increasing u-order. The per-receiver
            # chunk layout mirrors the legacy top-K layout, so the
            # downstream non-self / self-loop slicing in inference_gdeltauq
            # still works (it probes for actual edge count).
            arange_v = torch.arange(V, device=device)
            v_idx = arange_v.repeat_interleave(V - 1)      # tgt receivers
            # For each receiver v, sources are 0..V-1 except v. Build the
            # mask of (V, V-1) non-self source indices once.
            all_pairs = arange_v.unsqueeze(0).expand(V, V)  # (V, V)
            non_self_mask = ~torch.eye(V, dtype=torch.bool, device=device)
            u_idx = all_pairs[non_self_mask].view(V, V - 1).reshape(-1)  # src
            gated_edge_index = torch.stack([u_idx, v_idx], dim=0)  # (2, V*(V-1))
            # edge_weight per edge: A[target, source] = A[v_idx, u_idx].
            edge_weight_single = A[v_idx, u_idx]            # (V*(V-1),)
            # Replicate across batch: same A applies to every sample.
            edge_weight = edge_weight_single.repeat(batch_num)
        else:
            weights = all_embeddings.detach().clone().view(self.node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed = torch.matmul(
                weights.norm(dim=-1).view(-1, 1),
                weights.norm(dim=-1).view(1, -1),
            )
            cos_ji_mat = cos_ji_mat / normed

            if self.causal_restrict is not None:
                # Restrict the cosine candidates to causally-allowed parents
                # BEFORE top-K (inject the prior rather than intersect it).
                gated_edge_index, restricted_adj = apply_causal_restrict(
                    cos_ji_mat,
                    self.causal_restrict,
                    self.topk,
                    mode=self.causal_restrict_mode,
                    keep_self=self.causal_restrict_keep_self,
                )
                self.learned_graph = restricted_adj
                edge_weight = None
                batch_gated = get_batch_edge_index(
                    gated_edge_index, batch_num, self.node_num
                ).to(device)
                broadcast_embeddings = all_embeddings.repeat(batch_num, 1)
                return batch_gated, broadcast_embeddings, edge_weight

            topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]

            if self.causal_mask is not None:
                gated_edge_index, masked_adj = apply_and_mask(
                    topk_indices_ji,
                    self.causal_mask,
                    self.node_num,
                    keep_self=self.causal_mask_keep_self,
                )
                # learned_graph now stores the masked adjacency (target-major
                # bool matrix) rather than raw topk indices. Variable in-degree
                # per node -- inference_gdeltauq.py assumes fixed (topk-1)*V
                # layout and must be updated before consuming such checkpoints.
                self.learned_graph = masked_adj
            else:
                self.learned_graph = topk_indices_ji
                gated_i = (
                    torch.arange(0, self.node_num).T.unsqueeze(1).repeat(1, self.topk)
                    .flatten().to(device).unsqueeze(0)
                )
                gated_j = topk_indices_ji.flatten().unsqueeze(0)
                gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            edge_weight = None

        batch_gated = get_batch_edge_index(
            gated_edge_index, batch_num, self.node_num
        ).to(device)
        broadcast_embeddings = all_embeddings.repeat(batch_num, 1)

        return batch_gated, broadcast_embeddings, edge_weight

    def adjacency_penalty(self) -> torch.Tensor:
        """L1 penalty on the off-diag of the most recent A. Returns a zero
        scalar (on the embedding's device) when LSA is off, so the joint-NLL
        training loop can call this unconditionally without branching."""
        if self.adj is None:
            return torch.zeros((), device=self.embedding.weight.device,
                                dtype=self.embedding.weight.dtype)
        return self.adj.penalty()

    # ------------------------------------------------------------------ #
    # Pre-anchor backbone — layers 0 .. anchored_layer_idx - 1.
    # Returns the per-node hidden representation right before the
    # anchored layer (shape (B, V, dim) for layers >= 1; (B, V, input_dim)
    # if anchored_layer_idx == 0, which is forbidden by the constructor).
    # ------------------------------------------------------------------ #
    def forward_split(self, data, org_edge_index=None):
        """Run the pre-anchor backbone only. Output is what the anchored
        layer consumes (before concatenation with the anchor). Used at
        inference to cache once per timestep across the K anchored passes
        and at calibration to build the anchor pool."""
        x = data.clone().detach()
        device = data.device
        batch_num, node_num, all_feature = x.shape
        assert node_num == self.node_num

        batch_gated, broadcast_embeddings, edge_weight = self._build_learned_graph(
            batch_num, device)

        h = x.view(-1, all_feature).contiguous()  # (B*V, W)
        for layer_idx in range(self.anchored_layer_idx):
            h = self.gnn_layers[layer_idx](
                h, batch_gated,
                node_num=node_num * batch_num,
                embedding=broadcast_embeddings,
                edge_weight=edge_weight,
            )  # (B*V, dim)

        return h.view(batch_num, node_num, -1)

    # ------------------------------------------------------------------ #
    # Anchored layer + head. Takes pre-anchor hidden and an anchor tensor.
    # ------------------------------------------------------------------ #
    def forward_anchored(self, h_pre, anchor, org_edge_index=None):
        """Run the anchored MPNN layer and the final head.

        Args:
            h_pre: (B, V, d_in) pre-anchor hidden representation, where d_in
                is dim if anchored_layer_idx > 0 (the normal case for >=2
                layers).
            anchor: (V, d_in) or (B, V, d_in). If 2D it's broadcast across
                the batch dim.
        Returns:
            mu: (B, V) predicted mean.
            h_final: (B, V, dim) post-dropout hidden representation.
            attention: per-edge attention from the anchored layer, shape
                (B * topk * V, heads=1, 1) (raw PyG layout).
        """
        device = h_pre.device
        batch_num, node_num, d_in = h_pre.shape
        assert node_num == self.node_num

        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(0).expand(batch_num, -1, -1)
        elif anchor.dim() == 3:
            if anchor.shape[0] == 1 and batch_num > 1:
                anchor = anchor.expand(batch_num, -1, -1)
        else:
            raise ValueError(f"anchor must be 2D or 3D, got {anchor.shape}")

        anchored_in = torch.cat([h_pre - anchor, anchor], dim=-1)  # (B, V, 2*d_in)
        anchored_in_flat = anchored_in.view(-1, anchored_in.shape[-1]).contiguous()

        batch_gated, broadcast_embeddings, edge_weight = self._build_learned_graph(
            batch_num, device)

        # Anchored MPNN layer.
        h_anchored = self.gnn_layers[self.anchored_layer_idx](
            anchored_in_flat, batch_gated,
            node_num=node_num * batch_num,
            embedding=broadcast_embeddings,
            edge_weight=edge_weight,
        )  # (B*V, dim)
        attention = self.gnn_layers[self.anchored_layer_idx].att_weight_1

        # Head: BN + embedding-gate + BN + Dropout + OutLayer.
        h_anchored = h_anchored.view(batch_num, node_num, -1)
        indexes = torch.arange(0, node_num).to(device)
        gated = torch.mul(h_anchored, self.embedding(indexes))

        gated = gated.permute(0, 2, 1)
        gated = F.relu(self.bn_outlayer_in(gated))
        gated = gated.permute(0, 2, 1)

        h_final = self.dp(gated)  # (B, V, dim)

        mu = self.out_layer(h_final).view(-1, node_num)  # (B, V)

        return mu, h_final, attention

    # ------------------------------------------------------------------ #
    # Full forward. At training-time the anchor is sampled by batch-shuffle
    # (detached, per plan v2 1.2). At inference the caller passes an
    # explicit anchor from the calibrated pool.
    # ------------------------------------------------------------------ #
    def forward(self, data, org_edge_index, anchor=None):
        """Full forward pass.

        Args:
            data: (B, V, W) input window.
            org_edge_index: placeholder (the model uses its own learned
                top-k graph; included for API parity with GDN).
            anchor: if None (training), sample by batch-shuffle. Otherwise
                a (V, d) or (B, V, d) tensor sourced from the anchor pool.
        Returns:
            (mu, h_final, attention)
        """
        h_pre = self.forward_split(data, org_edge_index)  # (B, V, d_in)

        if anchor is None:
            batch_num = h_pre.shape[0]
            perm = torch.randperm(batch_num, device=h_pre.device)
            anchor = h_pre[perm].detach()

        return self.forward_anchored(h_pre, anchor, org_edge_index)
