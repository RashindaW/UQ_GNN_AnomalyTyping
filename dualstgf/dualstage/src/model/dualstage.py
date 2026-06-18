import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn import ModuleList
from typing import Optional, Tuple
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import (    
    GCN, GAT, GIN,
    GCNConv,
    GATv2Conv,
    GINEConv,
    LayerNorm,
    BatchNorm,
    GraphNorm,
)
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    sort_edge_index,
    is_undirected,
    to_undirected
)

from ..utils.init import init_weights
from ..config import cfg


RNN_LAYER_DICT = {
    'gru': nn.GRU,
    'lstm': nn.LSTM,
}

ACTIVATION_DICT = {
    'leakyrelu': nn.LeakyReLU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'softplus': nn.Softplus,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity,
}

NORM_LAYER_DICT = {
    'layer': LayerNorm,
    'batch': BatchNorm,
    'graph': GraphNorm,
}

GNN_DICT = {
    'gat': GAT,
    'gcn': GCN,
    'gin': GIN,
}

GNN_CONV_LAYER_DICT = {
    'gat': GATv2Conv,
    'gcn': GCNConv,
    'gin': GINEConv,
}


class TimeEncode(nn.Module):
    """
    https://github.com/twitter-research/tgn/blob/master/model/time_encoding.py
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False


    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class IDCNN(nn.Module):
    """Enhanced 1D CNN for edge attention preprocessing."""
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1, kernel_size=3, num_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
        if num_layers > 1:
            layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class EdgeGRU(nn.Module):
    """Scalar GRU that aggregates per-timestep attention into a single edge weight."""
    def __init__(self):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_size=1, hidden_size=1)

    def forward(self, alpha_seq):
        """alpha_seq: [num_edges, W] -> edge_weights: [num_edges]"""
        num_edges, w = alpha_seq.shape
        h = torch.zeros(num_edges, 1, device=alpha_seq.device)
        for t in range(w):
            h = self.gru_cell(alpha_seq[:, t:t+1], h)
        return F.softplus(h.squeeze(-1))


class TemporalFeatureGraph(nn.Module):
    """
    Memory-efficient: Process timesteps sequentially.
    At each ti: use hj[ti], hk[ti] (scalars) + temporal encoding
    e_jk^ti = a^T · LeakyReLU(W · [hj[ti], hk[ti], emb(ti)])
    """
    def __init__(self, n_nodes, hidden_dim=64, time_dim=5, dropout=0.3, learn_sys=True, sub_window_size=1):
        super().__init__()
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.learn_sys = learn_sys
        self.sub_window_size = sub_window_size

        self.time_encode = TimeEncode(time_dim)
        # Input: [scalar_j, scalar_k, emb(ti)] = 2 + time_dim
        self.W = nn.Linear(2 + time_dim, hidden_dim)
        self.att = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.att.unsqueeze(0))
        self.edge_gru = EdgeGRU()

    def forward(self, h, batch):
        """
        h: [B, N, W] - IDCNN output (filtered signal)
        Returns: edge_index, edge_weights, alpha_seq (for divergence!)
        """
        b, n, w = h.shape
        device = h.device
        dt = self.sub_window_size

        # Sub-window mean-pooling: [B, N, W] → [B, N, num_snapshots]
        if dt > 1 and w % dt == 0:
            num_snapshots = w // dt
            h_snap = h.view(b, n, num_snapshots, dt).mean(dim=-1)
        else:
            num_snapshots = w
            h_snap = h

        # Time encodings for all snapshots
        t_idx = torch.arange(num_snapshots, device=device, dtype=torch.float32)
        time_emb = self.time_encode(t_idx)  # [num_snapshots, time_dim]

        # Collect attention over time: [B, N, N, num_snapshots]
        alpha_seq = torch.zeros(b, n, n, num_snapshots, device=device)

        for ti in range(num_snapshots):
            # Node values at snapshot ti: [B, N, 1]
            h_ti = h_snap[:, :, ti:ti+1]

            # Pairwise scalars: [B, N, N, 2]
            hj = h_ti.unsqueeze(2).expand(-1, -1, n, -1)  # [B, N, N, 1]
            hk = h_ti.unsqueeze(1).expand(-1, n, -1, -1)  # [B, N, N, 1]
            h_pair = torch.cat([hj, hk], dim=-1)  # [B, N, N, 2]

            # Add time encoding: [B, N, N, 2+time_dim]
            t_enc = time_emb[ti].view(1, 1, 1, -1).expand(b, n, n, -1)
            v = torch.cat([h_pair, t_enc], dim=-1)

            # GATv2 attention
            u = F.leaky_relu(self.W(v), 0.2)  # [B, N, N, hidden]
            e = (u * self.att).sum(dim=-1)    # [B, N, N]
            alpha = F.softmax(e, dim=2)        # Normalize over neighbors

            if self.training:
                alpha = F.dropout(alpha, p=self.dropout)

            alpha_seq[:, :, :, ti] = alpha

        # EdgeGRU: aggregate attention over time
        # [B*N*N, num_snapshots] -> [B*N*N]
        alpha_flat = alpha_seq.view(b * n * n, num_snapshots)
        edge_weights = self.edge_gru(alpha_flat)
        edge_weights_dense = edge_weights.view(b, n, n)

        # Convert to sparse (for GNN compatibility)
        edge_index, edge_weights = self._dense_to_sparse(edge_weights_dense, b, n, device)

        return edge_index, edge_weights, alpha_seq  # Return alpha_seq for divergence!

    def _dense_to_sparse(self, dense, b, n, device):
        # Symmetrize in dense form if learning symmetric system
        if self.learn_sys:
            dense = (dense + dense.transpose(-1, -2)) / 2

        # Exclude self-loops (i ≠ j) - GIN already has (1+ε)·x_i term
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        src, dst = mask.nonzero(as_tuple=True)
        single = torch.stack([src, dst], dim=0)  # [2, N*(N-1)]

        edge_indices = []
        edge_weights_list = []
        for i in range(b):
            edge_indices.append(single + i * n)
            edge_weights_list.append(dense[i][mask])

        return torch.cat(edge_indices, dim=1), torch.cat(edge_weights_list)


class WeightedGIN(nn.Module):
    """z_j = MLP((1+ε)·z_j + Σ_k A_jk·z_k) + residual"""
    def __init__(self, in_channels, out_channels, eps=0.0, train_eps=True):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps)) if train_eps else eps
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        # Residual connection if dimensions match
        self.residual = nn.Identity() if in_channels == out_channels else None

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index
        weighted_msg = x[src] * edge_weight.unsqueeze(-1)
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, dst, weighted_msg)

        self_term = (1 + self.eps) * x
        # Diagnostic: track message-vs-self contribution (eval only)
        if not self.training:
            self._last_aggr_norm = aggr.detach().norm().item()
            self._last_self_norm = self_term.detach().norm().item()

        out = self.mlp(self_term + aggr)
        if self.residual is not None:
            out = out + self.residual(x)
        return out


class BackwardOCEncoder(nn.Module):
    """Process OC in reverse: h_c^ti = ReLU(GRU(U^ti, h_c^(ti+1)))"""
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_dim, batch_first=True)

    def forward(self, u_seq):
        """u_seq: [B, W, C] -> [B, hidden_dim]"""
        u_rev = torch.flip(u_seq, dims=[1])
        _, h = self.gru(u_rev)
        return F.relu(h.squeeze(0))


class SpectralFeatureGraph(nn.Module):
    """
    Simpler graph for spectral embeddings (no temporal dimension).
    Spectral features are already window-aggregated via FFT.
    """
    def __init__(self, embed_dim, hidden_dim=64, dropout=0.3, learn_sys=True):
        super().__init__()
        self.dropout = dropout
        self.learn_sys = learn_sys

        # Input: [freq_j, freq_k] concatenated
        self.W = nn.Linear(2 * embed_dim, hidden_dim)
        self.att = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(self, h_freq, batch, n_nodes):
        """
        h_freq: [B*N, embed_dim] - spectral embeddings
        Returns: edge_index, edge_weights, alpha (for divergence!)
        """
        device = h_freq.device
        b = h_freq.shape[0] // n_nodes
        n = n_nodes

        # Reshape to [B, N, embed_dim]
        h = h_freq.view(b, n, -1)

        # Pairwise: [B, N, N, 2*embed_dim]
        hj = h.unsqueeze(2).expand(-1, -1, n, -1)
        hk = h.unsqueeze(1).expand(-1, n, -1, -1)
        h_pair = torch.cat([hj, hk], dim=-1)

        # GATv2 attention (no temporal encoding)
        u = F.leaky_relu(self.W(h_pair), 0.2)  # [B, N, N, hidden]
        e = (u * self.att).sum(dim=-1)         # [B, N, N]
        alpha = F.softmax(e, dim=2)

        if self.training:
            alpha = F.dropout(alpha, p=self.dropout)

        # Convert to sparse
        edge_index, edge_weights = self._dense_to_sparse(alpha, b, n, device)

        return edge_index, edge_weights, alpha  # Return alpha for divergence!

    def _dense_to_sparse(self, dense, b, n, device):
        # Symmetrize in dense form if learning symmetric system
        if self.learn_sys:
            dense = (dense + dense.transpose(-1, -2)) / 2

        # Exclude self-loops (i ≠ j) - GIN already has (1+ε)·x_i term
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        src, dst = mask.nonzero(as_tuple=True)
        single = torch.stack([src, dst], dim=0)  # [2, N*(N-1)]

        edge_indices = []
        edge_weights_list = []
        for i in range(b):
            edge_indices.append(single + i * n)
            edge_weights_list.append(dense[i][mask])

        return torch.cat(edge_indices, dim=1), torch.cat(edge_weights_list)


class FeatureGraph(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 n_nodes,
                 dropout=0.3,
                 reduce_type='knn',
                 topk=20,
                 learn_sys=True,
                 **kwargs
                ):
        
        super().__init__()
        self.learn_sys = learn_sys
        self.n_nodes = n_nodes
        self.in_channels = in_channels
        self.out_channels = embed_dim
        self.lin_l = nn.Linear(in_channels, embed_dim)
        self.lin_r = nn.Linear(in_channels, embed_dim)
        self.att = nn.Parameter(torch.Tensor(1, embed_dim))
        self.dropout = dropout
        self.topk = min(topk, n_nodes) # Ensure topk doesn't exceed number of nodes
        
        if self.topk == self.n_nodes:
             # If we want fully connected or 'all', we use 'none' reduction (all edges)
             # Or we can just proceed with k=N-1?
             # Original code set reduce_type='none' if topk == cfg.dataset.n_nodes
             # But cfg.dataset.n_nodes is global config, self.n_nodes is local.
             pass
             
        if self.topk >= self.n_nodes: # Safety check
             self.topk = self.n_nodes
             reduce_type = 'none'

        self.reduce_type = reduce_type
        self.apply(init_weights)


    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        nn.init.xavier_uniform_(self.att)


    def forward(self, x, edge_index, batch):
        """
        x: [batch_size, n_nodes, in_channels]
        """
        b, n, _ = x.shape
        device = x.device
        
        # x is [batch, n_nodes, dim]
        # Flatten for attention comp: [batch*n_nodes, dim]
        x_l = self.lin_l(x).reshape(-1, self.out_channels) 
        x_r = self.lin_r(x).reshape(-1, self.out_channels)
        
        # Create fully connected graph for attention
        # For efficiency, we compute attention matrix directly
        
        # [batch, n_nodes, dim]
        x_l = x_l.view(b, n, -1)
        x_r = x_r.view(b, n, -1)
        
        # [batch, n_nodes, n_nodes, dim]
        # Broadcasting: [b, n, 1, d] + [b, 1, n, d]
        x_cat = x_l.unsqueeze(2) + x_r.unsqueeze(1)
        
        # Activation
        x_cat = F.leaky_relu(x_cat, 0.2)
        
        # Attention weights: [batch, n_nodes, n_nodes]
        # (x_cat * att).sum(-1)
        alpha = (x_cat * self.att).sum(dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.reduce_type == 'knn':
            # alpha is [batch, n_nodes, n_nodes]
            # remove diagonal (self-loops) from candidate neighbors to avoid picking self
            # alpha_no_diag = alpha - torch.diag_embed(torch.diagonal(alpha, dim1=1, dim2=2) + 1e9) 
            # Actually, original code subtracted diagonal?
            # alpha = alpha - torch.diag_embed(torch.diagonal(alpha, dim1=1, dim2=2))
            
            # Just topk on the raw alpha is safer if k < N. 
            # If k >= N, indices will be out of range? No, topk(k) on dim size N works if k<=N.
            # The error "selected index k out of range" means k > N.
            
            # Safety clamping inside forward too, just in case dynamic sizing changes?
            # But n_nodes should be fixed.
            k = min(self.topk, n)
            
            attention, indices = torch.topk(alpha, k, dim=-1)
            
            # attention: [batch, n_nodes, k]
            # indices: [batch, n_nodes, k]
            
            # Normalize attention weights within selected neighbors for stability
            attention = F.softmax(attention, dim=-1)
            attention = attention.view(-1) # [batch * n_nodes * k]

            edge_num = k * n
            device = x.device
            
            # Construct edge indices for the selected top-k
            # Source nodes: 0,0,..,0, 1,1,..,1 ...
            # For each batch: 0..N-1 repeated k times? No.
            # We want for each node i, the k neighbors j.
            # Source is i. Target is j (from indices).
            
            # indices [b, n, k] -> contains index of j
            
            # Construct source index i
            # [0, 0, ... (k times), 1, 1, ... ]
            index_i = torch.arange(0, n, device=device).unsqueeze(1).repeat(1, k).flatten() # [n*k]
            index_i = index_i.unsqueeze(0).repeat(b, 1).view(1, -1) # [1, b*n*k]
            
            # Construct target index j
            index_j = indices.view(1, -1) # [1, b*n*k]
            
            # Adjust offsets for batching
            # node indices in PyG batch are usually cumulative 0..B*N-1
            # But here we are constructing edges within each batch graph 0..N-1
            # and then offsetting them? 
            # Wait, FeatureGraph takes `x` as [batch, n_nodes, dim].
            # Usually returns edge_index for the whole batch.
            
            # Offset calculation:
            # i-th batch starts at i*n_nodes
            
            # current index_i is 0..n-1 repeated b times
            # current index_j is 0..n-1 repeated b times (values from topk)
            
            # We need to add i*n to the indices of the i-th batch
            # Reshape to [b, n*k]
            index_i = index_i.view(b, -1)
            index_j = index_j.view(b, -1)
            
            offset = (torch.arange(b, device=device) * n).unsqueeze(1) # [b, 1]
            
            index_i = (index_i + offset).view(1, -1)
            index_j = (index_j + offset).view(1, -1)

            new_edge_index = torch.cat((index_i, index_j), dim=0)

        elif self.reduce_type == 'none':
            # All-to-all attention (fully connected), normalize per source node
            attention_dense = F.softmax(alpha, dim=-1)  # [b, n, n]
            attention = attention_dense.view(-1)  # [b*n*n]
            if edge_index is None:
                # Build dense edge_index for each graph in the batch
                base = torch.arange(n, device=device)
                src = base.repeat_interleave(n)
                dst = base.repeat(n)
                single = torch.stack((src, dst), dim=0)  # [2, n*n]
                batches = []
                for i in range(b):
                    offset = i * n
                    batches.append(single + offset)
                new_edge_index = torch.cat(batches, dim=1)
            else:
                new_edge_index = edge_index

        attention = F.dropout(attention, p=self.dropout, training=self.training)
        if self.learn_sys and not is_undirected(new_edge_index):
             # to_undirected might duplicate edges? 
             # For anomaly detection, usually symmetric relation is preferred?
            undirected_edge_index, undirected_attention = to_undirected(new_edge_index, attention)
            return undirected_edge_index, undirected_attention
        else:
            return new_edge_index, attention


class TemporalGraph(nn.Module):
    def __init__(
        self,
        embed_dim,
        win=5,
        kernel_size=5,
        dropout=0.,
        use_time_encoding=False,
        time_dim=5,
        **kwargs):
        
        super().__init__()
        self.win = win
        self.time_dim = time_dim
        self.use_time_encoding = use_time_encoding
        self.embed_dim = embed_dim
        
        if use_time_encoding:
            self.time_encoding = TimeEncode(time_dim)
            
        # Temporal convolution to capture local patterns before attention?
        # Paper says: 1D Conv on each node time series
        self.conv1d = nn.Conv1d(1, 16, kernel_size, padding=kernel_size//2)
        
        # Attention mechanism to compute edge weights between nodes *across time*?
        # No, "Temporal Graph" usually means edges between t and t-1, or dependencies.
        # Dynamic Edge via Graph Attention
        # This module infers dynamic edges from temporal correlations.
        
        # Input: [batch, nodes, window]
        # We want to learn A_t for each window.
        
        self.lin_q = nn.Linear(win, 64)
        self.lin_k = nn.Linear(win, 64) 
        self.att = nn.Parameter(torch.Tensor(1, 64))
        
        self.gru = nn.GRU(win, 64, batch_first=True)

    def forward(self, x):
        # x: [batch, nodes, window]
        # Simple temporal attention or correlation?
        # Placeholder logic matching original if needed
        return None

class GRUEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, norm_func=None, mode='univariate', dropout=0.0, activation='relu'):
        super().__init__()
        self.mode = mode
        if mode == 'univariate':
            self.gru = nn.GRU(in_channels, out_channels, batch_first=True)
        elif mode == 'multivariate':
            self.gru = nn.GRU(in_channels, out_channels, batch_first=True)

        self.norm = norm_func(out_channels) if norm_func else nn.Identity()
        self.feat_dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else lambda x: x
        
    def forward(self, x, h0=None):
        # x: [batch, nodes, window] (univariate) or [batch, window, feats] (multivariate)
        if self.mode == 'univariate':
            # GRU expects [batch, seq_len, features]
            # Treat each node as a sample? Or independent series?
            # If x is [batch, nodes, window], we reshape to [batch*nodes, window, 1]
            b, n, w = x.shape
            x = x.view(b * n, w, 1)
            if h0 is not None:
                h0 = h0.unsqueeze(0)
            out, h = self.gru(x, h0)
            # h: [1, b*n, out_dim]
            h = self.feat_dropout(self.activation(h.squeeze(0))).view(b, n, -1)
            if not isinstance(self.norm, nn.Identity):
                h_flat = h.reshape(b * n, -1)
                if isinstance(self.norm, GraphNorm):
                    batch = torch.arange(b, device=h.device).repeat_interleave(n)
                    h_flat = self.norm(h_flat, batch)
                else:
                    h_flat = self.norm(h_flat)
                h = h_flat.view(b, n, -1)
            return h
        elif self.mode == 'multivariate':
            # x: [batch, window, dim]
            if h0 is not None:
                h0 = h0.unsqueeze(0)
            out, h = self.gru(x, h0)
            h = self.feat_dropout(self.activation(h.squeeze(0)))
            if not isinstance(self.norm, nn.Identity):
                if isinstance(self.norm, GraphNorm):
                    batch = torch.arange(h.size(0), device=h.device)
                    h = self.norm(h, batch)
                else:
                    h = self.norm(h)
            return h

class SpectralEncoder(nn.Module):
    """
    Encodes frequency domain features using rFFT and an optional band-mixing layer.
    This forms the 'Spectral View' of the data.
    """
    def __init__(
        self,
        window_size,
        embed_dim=16,
        max_freq_bins=0,
        band_mixer="none",
        activation="relu",
        norm_func=None,
        use_log_magnitude=True,
        use_spectral_features=False,
    ):
        super().__init__()
        self.window_size = window_size
        
        # Calculate number of rFFT bins: (N/2) + 1
        full_bins = (window_size // 2) + 1
        
        # If max_freq_bins is 0 or larger than full_bins, use all
        self.n_bins = full_bins
        if max_freq_bins > 0 and max_freq_bins < full_bins:
            self.n_bins = max_freq_bins
            
        self.band_mixer_type = band_mixer
        self.use_log_magnitude = use_log_magnitude
        self.use_spectral_features = use_spectral_features
        self.feature_dim = 7 if self.use_spectral_features else 0
        
        # Layers for mixing frequency bands
        # Input shape per node: [Batch, n_bins] (magnitude)
        if band_mixer == "mlp":
            self.mixer = nn.Sequential(
                nn.Linear(self.n_bins, embed_dim * 2),
                ACTIVATION_DICT[activation](),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        elif band_mixer == "conv":
            # 1D Conv over frequency bins?
            # Input [Batch, 1, n_bins] -> [Batch, embed_dim, 1]?
            self.mixer = nn.Sequential(
                nn.Conv1d(1, embed_dim, kernel_size=3, padding=1),
                ACTIVATION_DICT[activation](),
                nn.AdaptiveAvgPool1d(1), # Pooling to get single vector
                nn.Flatten()
            )
        else:
            # Linear projection
            self.mixer = nn.Linear(self.n_bins, embed_dim)

        self.output_dim = embed_dim + self.feature_dim
        self.norm = norm_func(self.output_dim) if norm_func else nn.Identity()
        self.eps = 1e-8
        self.register_buffer("freqs", torch.linspace(0.0, 1.0, steps=self.n_bins), persistent=False)

    def _compute_spectral_features(self, power: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral shape features from power spectra.
        Returns [B, N, 7] features.
        """
        b, n, bins = power.shape
        freqs = self.freqs.to(power.device).view(1, 1, -1)
        total = power.sum(dim=-1).clamp_min(self.eps)

        centroid = (power * freqs).sum(dim=-1) / total

        diff = freqs - centroid.unsqueeze(-1)
        bandwidth = torch.sqrt((power * diff.pow(2)).sum(dim=-1) / total)

        geo_mean = torch.exp(torch.mean(torch.log(power + self.eps), dim=-1))
        arith_mean = power.mean(dim=-1).clamp_min(self.eps)
        flatness = geo_mean / arith_mean

        rolloff_ratio = 0.85
        cumulative = power.cumsum(dim=-1)
        threshold = total * rolloff_ratio
        mask = cumulative >= threshold.unsqueeze(-1)
        rolloff_idx = mask.float().argmax(dim=-1)
        rolloff = self.freqs.to(power.device)[rolloff_idx]

        band_1 = max(1, bins // 3)
        band_2 = max(band_1 + 1, (2 * bins) // 3)
        band_2 = min(band_2, bins)
        low = power[..., :band_1].sum(dim=-1)
        mid = power[..., band_1:band_2].sum(dim=-1)
        high = power[..., band_2:].sum(dim=-1)
        band_total = total.clamp_min(self.eps)
        band_low = low / band_total
        band_mid = mid / band_total
        band_high = high / band_total

        features = torch.stack(
            [centroid, bandwidth, flatness, rolloff, band_low, band_mid, band_high],
            dim=-1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, x):
        """
        Args:
            x: Temporal data [Batch, Nodes, Window]
        Returns:
            h_freq: Spectral node embeddings [Batch, Nodes, Embed_Dim]
        """
        b, n, w = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. Compute rFFT
        # x: [B, N, W] -> rFFT -> [B, N, W/2 + 1] (complex)
        fft_out = torch.fft.rfft(x, dim=-1)
        
        # 2. Compute Magnitude / Power
        # [B, N, n_bins]
        mag = torch.abs(fft_out).clamp_min(self.eps)
        power = mag.pow(2)
        if self.use_log_magnitude:
            mag = torch.log1p(mag)
        mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Truncate/Select bins
        if self.n_bins < mag.shape[-1]:
            mag = mag[..., :self.n_bins]
            power = power[..., :self.n_bins]
            
        # 4. Mix Bands -> Embedding
        # Reshape for mixer: [B*N, n_bins]
        mag_flat = mag.reshape(b * n, -1)
        
        if self.band_mixer_type == "conv":
            # [B*N, 1, n_bins]
            mag_flat = mag_flat.unsqueeze(1)
            
        h_freq = self.mixer(mag_flat)
        
        # [B, N, Embed_Dim]
        h_freq = h_freq.reshape(b, n, -1)
        if self.use_spectral_features:
            spectral_features = self._compute_spectral_features(power)
            h_freq = torch.cat([h_freq, spectral_features], dim=-1)
        
        # 5. Normalize
        # Apply norm per node embedding if configured.
        if not isinstance(self.norm, nn.Identity):
            h_flat = h_freq.reshape(b * n, -1)
            if isinstance(self.norm, GraphNorm):
                batch = torch.arange(b, device=x.device).repeat_interleave(n)
                h_flat = self.norm(h_flat, batch)
            else:
                h_flat = self.norm(h_flat)
            h_freq = h_flat.view(b, n, -1)
        
        return h_freq


class ReconstructionModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64, num_layers=1,
                 with_variance_head: bool = False):
        super().__init__()
        # GRU decoder: [batch, nodes, 1] -> [batch, nodes, window]?
        # Or reconstruct window from embedding?
        # Usually: Embedding -> GRU -> Window

        self.rnn = nn.GRU(in_channels, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, out_channels)
        # Heteroscedastic UQ extension: an independent log-variance head sharing
        # the same GRU hidden state but with its own projection. Activated when
        # the parent DualSTAGE constructs this with `with_variance_head=True`.
        # The two heads do NOT share parameters (independent Linear layers) per
        # the heteroscedastic-deep-ensemble blueprint.
        self.with_variance_head = bool(with_variance_head)
        if self.with_variance_head:
            self.out_logvar = nn.Linear(hidden_dim, out_channels)
        else:
            self.out_logvar = None
        
    def forward(self, x, batch,  c=None):
        # x will be last hidden state of the GRU layer
        # x: [batch, nodes, hidden_dim]
        # We want to reconstruct sequence of length 'window'
        # Repeat x?
        
        # Implementation depends on decoder strategy. 
        # Simple MLP decoder for window?
        # Or autoregressive?
        # Paper: "Reconstructs the time series... in reverse order"
        
        # Let's assume x is [batch*nodes, dim]
        return None
    
    def reconstruct(self, z, window_size, h0=None, flip_output=True):
        # z: [batch, nodes, dim]
        b, n, d = z.shape
        z_flat = z.view(b * n, d)

        # Repeat z for each timestep?
        # Input to GRU: [b*n, window, d]
        z_rep = z_flat.unsqueeze(1).repeat(1, window_size, 1)

        if h0 is not None:
            h0 = h0.unsqueeze(0).expand(self.rnn.num_layers, -1, -1)
        out, _ = self.rnn(z_rep, h0)
        # out: [b*n, window, hidden]

        recon = self.out(out).squeeze(-1)         # [b*n, window]

        if self.with_variance_head:
            log_var = self.out_logvar(out).squeeze(-1)  # [b*n, window]
            if flip_output:
                recon = torch.flip(recon, dims=[1])
                log_var = torch.flip(log_var, dims=[1])
            return recon, log_var

        if flip_output:
            return torch.flip(recon, dims=[1])
        return recon


ENCODER_DICT = {
    'gru': GRUEncoder,
}


class DualSTAGE(nn.Module):
    """
    Dynamic Spectral-Temporal Graph Attention Network for Anomaly Detection.

    Combines temporal and spectral views of multivariate time series data
    with dynamic edge learning via graph attention for fault detection.
    """
    def __init__(
        self,
        feat_input_node,
        feat_target_node,
        feat_input_edge,
        node_encoder_type='gru',
        node_encoder_mode='univariate',
        contr_encoder_type='gru',
        infer_temporal_edge=True,
        temp_edge_hid_dim=100,
        temp_edge_embed_dim=1,
        sub_window_size=1,
        temporal_kernel=5,
        use_time_encoding=True,
        time_dim=5,
        temp_node_embed_dim=16,
        infer_static_graph=True,
        feat_edge_hid_dim=128,
        topk=20,
        learn_sys=True,
        aug_feat_edge_attr=True,
        num_gnn_layers=1,
        gnn_embed_dim=16,
        gnn_type='gin',
        dropout=0.3,
        feat_dropout=0.0,
        do_encoder_norm=True,
        do_gnn_norm=True,
        do_decoder_norm=True,
        encoder_norm_type='batch',
        gnn_norm_type='batch',
        decoder_norm_type='batch',
        recon_hidden_dim=10,
        num_recon_layers=1,
        edge_aggr='temp',
        act='relu',
        aug_control=True,
        flip_output=True,  # Changed to True to match paper's reversed reconstruction semantics
        # Dual-view (spectral) options
        use_spectral_view=False,
        freq_node_embed_dim=None,
        freq_max_bins=0,
        freq_band_mixer="none",
        freq_use_log=True,
        freq_use_spectral_features=False,
        freq_topk=None,
        share_gnn_weights=False,
        fuse_mode="concat",  # concat | sum | gated
        divergence_type="js",  # js | kl
        topology_mode="own_error_degree",  # own_error_degree | neighbor_propagation | plain_error
        node_gru_input="raw",  # raw | filtered (IDCNN-processed)
        gru_activation="relu",  # relu | none
        topology_error="l1",  # l1 (abs) | l2 (squared)
        spectral_only=False,  # Use only spectral branch output (skip temporal GNN in fusion)
        task="reconstruction",  # reconstruction | prediction
        pred_horizon=0,
        # Heteroscedastic UQ (DualSTGF_UQ): when True, the decoder gains an
        # independent log-variance head and forward() returns (mu, log_var)
        # instead of a single reconstruction tensor. log_var is clamped to
        # logvar_clamp to prevent gradient pathology in early training.
        with_variance_head: bool = False,
        logvar_clamp: tuple = (-10.0, 10.0),
    ):
        super(DualSTAGE, self).__init__()
        self.with_variance_head = bool(with_variance_head)
        self.logvar_clamp = tuple(logvar_clamp)
        self.infer_temporal_edge = infer_temporal_edge
        self.infer_graph = infer_static_graph
        self.edge_aggr = edge_aggr
        self.aug_control = aug_control
        self.flip_output = flip_output
        self.use_spectral_view = use_spectral_view
        self.spectral_only = spectral_only
        self.share_gnn_weights = share_gnn_weights
        self.fuse_mode = fuse_mode
        self.divergence_type = divergence_type
        self.topology_mode = topology_mode
        self.node_gru_input = node_gru_input
        self.topology_error = topology_error
        self.task = task
        self.pred_horizon = pred_horizon
        self.div_eps = 1e-8
        self.register_buffer('_cal_err_mean', None)
        self.register_buffer('_cal_err_std', None)
        self.do_encoder_norm = do_encoder_norm
        self.do_gnn_norm = do_gnn_norm
        self.do_decoder_norm = do_decoder_norm
        self.feat_dropout = nn.Dropout(feat_dropout)

        # Only create control encoder if we have control variables
        if self.aug_control and cfg.dataset.ocvar_dim > 0:
            self.control_encoder = ENCODER_DICT[contr_encoder_type](
                in_channels=cfg.dataset.ocvar_dim,
                out_channels=temp_node_embed_dim,
                norm_func=NORM_LAYER_DICT[encoder_norm_type] if do_encoder_norm else None,
                mode='multivariate',
                dropout=feat_dropout,
                activation=gru_activation,
            )
            # Backward OC encoder for decoder initialization (paper requirement)
            self.backward_oc_encoder = BackwardOCEncoder(
                in_channels=cfg.dataset.ocvar_dim,
                hidden_dim=temp_node_embed_dim,
            )
            # Project if dimensions differ
            if temp_node_embed_dim != recon_hidden_dim:
                self.backward_context_proj = nn.Linear(temp_node_embed_dim, recon_hidden_dim)
            else:
                self.backward_context_proj = nn.Identity()
        else:
            self.aug_control = False  # Disable if no control variables

        self.node_encoder = ENCODER_DICT[node_encoder_type](
            in_channels=feat_input_node,
            out_channels=temp_node_embed_dim,
            norm_func=NORM_LAYER_DICT[encoder_norm_type] if do_encoder_norm else None,
            mode=node_encoder_mode,
            dropout=feat_dropout,
            activation=gru_activation,
        )
        self.idcnn = IDCNN(in_channels=1, hidden_channels=16, out_channels=1, kernel_size=3, num_layers=2)

        # TemporalFeatureGraph for paper-compliant edge construction with GATv2 + EdgeGRU
        self.temporal_feature_graph = TemporalFeatureGraph(
            n_nodes=cfg.dataset.n_nodes,
            hidden_dim=temp_edge_hid_dim,  # 100
            time_dim=time_dim,
            dropout=dropout,
            learn_sys=learn_sys,
            sub_window_size=sub_window_size,
        )

        base_freq_embed_dim = freq_node_embed_dim or temp_node_embed_dim
        self.freq_node_embed_dim = base_freq_embed_dim
        freq_topk = freq_topk if freq_topk is not None else topk
        if self.use_spectral_view:
            self.spectral_encoder = SpectralEncoder(
                window_size=cfg.dataset.window_size,
                embed_dim=base_freq_embed_dim,
                max_freq_bins=freq_max_bins,
                band_mixer=freq_band_mixer,
                activation=act,
                norm_func=NORM_LAYER_DICT[encoder_norm_type] if do_encoder_norm else None,
                use_log_magnitude=freq_use_log,
                use_spectral_features=freq_use_spectral_features,
            )
            self.freq_node_embed_dim = self.spectral_encoder.output_dim

            # SpectralFeatureGraph for spectral branch edge construction
            self.spectral_feature_graph = SpectralFeatureGraph(
                embed_dim=self.freq_node_embed_dim,
                hidden_dim=temp_edge_hid_dim,
                dropout=dropout,
                learn_sys=learn_sys,
            )

            # OC conditioning for spectral branch
            if self.aug_control and cfg.dataset.ocvar_dim > 0:
                self.spectral_oc_proj = nn.Linear(temp_node_embed_dim, self.freq_node_embed_dim)

        if self.use_spectral_view and freq_topk != topk:
            warnings.warn(
                "freq_topk differs from temporal topk; overriding to match for divergence alignment.",
                RuntimeWarning,
            )
            freq_topk = topk

        if self.infer_graph:
            self.feat_edge_layer = FeatureGraph(
                n_nodes=cfg.dataset.n_nodes,
                in_channels=temp_node_embed_dim,
                embed_dim=feat_edge_hid_dim,
                topk=topk,
                learn_sys=learn_sys,
            )
            if self.use_spectral_view:
                self.feat_edge_layer_freq = FeatureGraph(
                    n_nodes=cfg.dataset.n_nodes,
                    in_channels=self.freq_node_embed_dim,
                    embed_dim=feat_edge_hid_dim,
                    topk=freq_topk,
                    learn_sys=learn_sys,
                )

        self.gnn_edge_dim = 1
        self.gnn_edge_dim_freq = 1

        if self.infer_temporal_edge:
            # Edge features from TemporalGraph are scalar attention values per timestep
            pass
            
        # GNN Layers - WeightedGIN (edge weights as scalar multipliers)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = temp_node_embed_dim if i == 0 else gnn_embed_dim
            self.gnn_layers.append(WeightedGIN(in_dim, gnn_embed_dim, train_eps=True))

        if self.use_spectral_view:
            if self.share_gnn_weights:
                self.gnn_layers_freq = self.gnn_layers
            else:
                self.gnn_layers_freq = nn.ModuleList()
                for i in range(num_gnn_layers):
                    in_dim = self.freq_node_embed_dim if i == 0 else gnn_embed_dim
                    self.gnn_layers_freq.append(WeightedGIN(in_dim, gnn_embed_dim, train_eps=True))

        if self.do_gnn_norm:
            norm_cls = NORM_LAYER_DICT[gnn_norm_type]
            self.gnn_norms = nn.ModuleList([norm_cls(gnn_embed_dim) for _ in range(num_gnn_layers)])
            if self.use_spectral_view:
                if self.share_gnn_weights:
                    self.gnn_norms_freq = self.gnn_norms
                else:
                    self.gnn_norms_freq = nn.ModuleList([norm_cls(gnn_embed_dim) for _ in range(num_gnn_layers)])
        else:
            self.gnn_norms = None
            self.gnn_norms_freq = None

        # Fusion
        fusion_dim = gnn_embed_dim
        if self.use_spectral_view:
            if fuse_mode == "concat":
                fusion_dim = gnn_embed_dim * 2
                self.fusion_layer = nn.Linear(fusion_dim, gnn_embed_dim)
            elif fuse_mode == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(gnn_embed_dim * 2, 1),
                    nn.Sigmoid()
                )
                self.fusion_layer = nn.Identity()
            else: # sum
                self.fusion_layer = nn.Identity()

        self.decoder = ReconstructionModel(
            in_channels=gnn_embed_dim, # Input is fused Z
            out_channels=feat_target_node,
            hidden_dim=recon_hidden_dim,
            num_layers=num_recon_layers,
            with_variance_head=self.with_variance_head,
        )
        self.decoder_norm = (
            NORM_LAYER_DICT[decoder_norm_type](gnn_embed_dim)
            if self.do_decoder_norm
            else nn.Identity()
        )
        
        self.topk = topk


    def learn_graph(self, node_feat, batch, branch="temporal"):
        """
        Learn graph structure (adjacency and edge weights).
        """
        # edge_index: fully connected?
        # Dynamic edge learning starts with fully connected and prunes via attention
        # node_feat: [batch, nodes, dim] (after reshape)
        
        # We need to construct a fully connected edge_index for the batch
        # Or just let FeatureGraph handle it?
        # FeatureGraph takes 'x' and 'edge_index'.
        # If reduce_type is 'knn', it ignores input edge_index mostly?
        # But it needs it for 'none'.
        
        # Let's pass None and let FeatureGraph construct if needed?
        # But FeatureGraph signature has edge_index.
        
        # Construct dummy full edge index?
        # FeatureGraph implementation in this file assumes `edge_index` is passed 
        # but for `knn` it computes new one.
        
        # Let's pass placeholder or None if allowed. 
        # Looking at FeatureGraph.forward:
        # It calculates `alpha` (attention) from `x`.
        # If knn: computes `new_edge_index` from topk.
        # If none: uses `edge_index`.
        
        # So for knn, input edge_index is unused? 
        # Actually: "old_edge_index = edge_index[:, perm]" is commented out.
        # So yes, for KNN, we don't need input edge_index.
        
        # Reshape node_feat to [Batch, Nodes, Dim]
        # node_feat comes from GNN/Encoder as [Batch*Nodes, Dim]
        # We need to reshape.
        
        # Determine batch size B
        # batch tensor is [Batch*Nodes]
        # num_graphs = batch.max().item() + 1
        
        # We can assume fixed N_nodes from config
        n_nodes = cfg.dataset.n_nodes
        b = node_feat.shape[0] // n_nodes
        
        x_reshaped = node_feat.view(b, n_nodes, -1)
        
        layer = self.feat_edge_layer if branch == "temporal" else self.feat_edge_layer_freq
        
        # Pass dummy edge_index
        edge_index, edge_attr = layer(x_reshaped, None, batch)
        
        return edge_index, edge_attr, edge_index # Return learned graph

    @staticmethod
    def _apply_norm(norm, x, batch=None):
        if isinstance(norm, GraphNorm):
            if batch is None:
                raise ValueError("GraphNorm requires a batch vector.")
            return norm(x, batch)
        return norm(x)

    def forward(self, data, return_graph=False):
        x, c, edge_index, batch = data.x, data.c, data.edge_index, data.batch
        n_nodes = cfg.dataset.n_nodes
        context_expanded = None
        c_in = None  # Control input for backward OC encoder

        # 1. Encode Control (U) -> Context

        # Control Encoder
        if self.aug_control:
            n_ctrl = cfg.dataset.ocvar_dim
            if c.dim() == 2:
                # c: [Batch * n_ctrl, window] -> we need to reshape
                if c.shape[0] % n_ctrl == 0:
                    b = c.shape[0] // n_ctrl
                    c = c.view(b, n_ctrl, -1)
                    c_in = c.transpose(1, 2)
                else:
                     raise ValueError(f"Unexpected control tensor shape: {c.shape}")
            elif c.dim() == 3:
                if c.shape[-1] == n_ctrl:
                    c_in = c  # [B, W, C]
                elif c.shape[1] == n_ctrl:
                    c_in = c.transpose(1, 2)  # [B, C, W] -> [B, W, C]
                else:
                    raise ValueError(f"Unexpected control tensor shape: {c.shape}")
            else:
                raise ValueError(f"Unexpected control tensor shape: {c.shape}")

            context = self.control_encoder(c_in) # [B, Dim]
            
            # Broadcast context to nodes so it can initialize each node GRU state.
            # Context is [B, Dim]; nodes are [B*N].
            context_expanded = context.repeat_interleave(n_nodes, dim=0) # [B*N, Dim]

        # 2. Encode Nodes (X) -> H_temp
        # x comes in as [B*N, W]. Reshape to [B, N, W] before encoding.
        b = x.shape[0] // n_nodes
        x_nodes = x.view(b, n_nodes, -1)

        # IDCNN for edge construction only (paper requirement)
        x_idcnn = self.idcnn(x_nodes.view(b * n_nodes, 1, -1)).view(b, n_nodes, -1)

        # Node GRU input: raw signal (default) or IDCNN-filtered
        gru_input = x_idcnn if self.node_gru_input == "filtered" else x_nodes
        h_temp = self.node_encoder(
            gru_input,
            h0=context_expanded if self.aug_control else None,
        )           # [B, N, Dim]
        h_temp = torch.nan_to_num(h_temp, nan=0.0, posinf=0.0, neginf=0.0)
        h_temp = h_temp.view(b * n_nodes, -1)         # flatten for GNN input
        
        # 3. Learn Temporal Graph using TemporalFeatureGraph (paper-compliant)
        # Use IDCNN output for edge construction
        adj_temp, attn_temp, alpha_temp = self.temporal_feature_graph(x_idcnn, batch)
        attn_temp = torch.nan_to_num(attn_temp, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 4. Spectral Branch
        h_freq = None
        adj_freq, attn_freq = None, None
        alpha_freq = None
        div_loss = torch.tensor(0.0, device=x.device)
        div_score = None
        if self.use_spectral_view:
            # Input x: [B*N, W]
            # Reshape to [B, N, W] for SpectralEncoder
            n = cfg.dataset.n_nodes
            b_spectral = x.shape[0] // n

            x_reshaped = x.view(b_spectral, n, -1)

            h_freq_batch = self.spectral_encoder(x_reshaped)  # [B, N, Dim]
            h_freq_batch = torch.nan_to_num(h_freq_batch, nan=0.0, posinf=0.0, neginf=0.0)

            # Add OC conditioning to spectral embeddings
            if self.aug_control and context is not None:
                # context: [B, temp_node_embed_dim] from control_encoder
                oc_spectral = self.spectral_oc_proj(context)  # [B, freq_embed_dim]
                oc_spectral = oc_spectral.unsqueeze(1).expand(-1, n, -1)  # [B, N, freq_embed_dim]
                h_freq_batch = h_freq_batch + oc_spectral  # Additive conditioning

            h_freq = h_freq_batch.view(b_spectral * n, -1)  # Flatten

            # Use SpectralFeatureGraph (paper-compliant)
            adj_freq, attn_freq, alpha_freq = self.spectral_feature_graph(h_freq, batch, n)
            attn_freq = torch.nan_to_num(attn_freq, nan=0.0, posinf=0.0, neginf=0.0)

            # Compute divergence (YOUR NOVEL CONTRIBUTION!)
            # alpha_temp: [B, N, N, W] - temporal attention over time
            # alpha_freq: [B, N, N] - spectral attention (no temporal dim)
            # For divergence: average alpha_temp over time to match shape
            # IMPORTANT: align with post-symmetrized GNN graph (S6 fix)
            try:
                alpha_temp_avg = alpha_temp.mean(dim=-1)  # [B, N, N]

                # Apply same symmetrization as _dense_to_sparse
                if self.temporal_feature_graph.learn_sys:
                    alpha_temp_avg = (alpha_temp_avg + alpha_temp_avg.transpose(-1, -2)) / 2
                if self.spectral_feature_graph.learn_sys:
                    alpha_freq_sym = (alpha_freq + alpha_freq.transpose(-1, -2)) / 2
                else:
                    alpha_freq_sym = alpha_freq

                # Remove self-loops (matching _dense_to_sparse behavior)
                diag_mask = torch.eye(alpha_temp_avg.size(-1), device=alpha_temp_avg.device, dtype=torch.bool)
                alpha_temp_avg = alpha_temp_avg.masked_fill(diag_mask.unsqueeze(0), 0.0)
                alpha_freq_sym = alpha_freq_sym.masked_fill(diag_mask.unsqueeze(0), 0.0)

                # Safe row renormalization (zero rows stay zero)
                row_sum_t = alpha_temp_avg.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                alpha_temp_avg = alpha_temp_avg / row_sum_t
                row_sum_f = alpha_freq_sym.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                alpha_freq_sym = alpha_freq_sym / row_sum_f

                div_loss, div_score = self._compute_divergence_from_alpha(
                    alpha_temp_avg, alpha_freq_sym
                )
            except Exception:
                div_loss = torch.tensor(0.0, device=x.device)
                div_score = None
            
        # 5. GNN Layers - WeightedGIN (edge weights as scalar multipliers)
        # Temporal GNN
        z_temp = h_temp
        for i, conv in enumerate(self.gnn_layers):
            z_temp = conv(z_temp, adj_temp, attn_temp)  # WeightedGIN: edge_weight, not edge_attr
            if self.do_gnn_norm:
                z_temp = self._apply_norm(self.gnn_norms[i], z_temp, batch)
            z_temp = torch.nan_to_num(z_temp, nan=0.0, posinf=0.0, neginf=0.0)
            z_temp = F.relu(z_temp)
            z_temp = self.feat_dropout(z_temp)

        # Spectral GNN
        z_freq = None
        if self.use_spectral_view:
            z_freq = h_freq
            attn_freq_flat = attn_freq.view(-1) if attn_freq is not None else None
            for i, conv in enumerate(self.gnn_layers_freq):
                z_freq = conv(z_freq, adj_freq, attn_freq_flat)  # WeightedGIN
                if self.do_gnn_norm:
                    z_freq = self._apply_norm(self.gnn_norms_freq[i], z_freq, batch)
                z_freq = torch.nan_to_num(z_freq, nan=0.0, posinf=0.0, neginf=0.0)
                z_freq = F.relu(z_freq)
                z_freq = self.feat_dropout(z_freq)

        # 6. Fusion
        z_fused = z_temp
        gate_values = None  # For monitoring gated fusion

        if self.spectral_only and z_freq is not None:
            # Spectral-only ablation: bypass temporal GNN, use spectral output directly
            z_fused = z_freq
        elif self.use_spectral_view:
            # Fusion
            if self.fuse_mode == "concat":
                z_cat = torch.cat([z_temp, z_freq], dim=-1)
                z_fused = self.fusion_layer(z_cat)
            elif self.fuse_mode == "sum":
                z_fused = z_temp + z_freq
            elif self.fuse_mode == "gated":
                z_cat = torch.cat([z_temp, z_freq], dim=-1)
                g = self.gate(z_cat)
                z_fused = g * z_temp + (1-g) * z_freq
                gate_values = g  # Store for diagnostics

        # 7. Decode / Reconstruct
        # Reconstruct X: reshape back to [B, N, D] for decoder
        b = int(batch.max().item()) + 1
        n = cfg.dataset.n_nodes
        if self.do_decoder_norm:
            z_fused = self._apply_norm(self.decoder_norm, z_fused, batch)
        z_fused = torch.nan_to_num(z_fused, nan=0.0, posinf=0.0, neginf=0.0)
        z_fused_nodes = z_fused.view(b, n, -1)
        output_window = cfg.dataset.window_size
        if self.task == "prediction" and self.pred_horizon:
            output_window = self.pred_horizon

        # Backward OC context for decoder initialization (paper requirement)
        decoder_h0 = None
        if self.aug_control and c_in is not None:
            backward_ctx = self.backward_oc_encoder(c_in)
            backward_ctx = self.backward_context_proj(backward_ctx)
            decoder_h0 = backward_ctx.repeat_interleave(n_nodes, dim=0)

        decoder_out = self.decoder.reconstruct(
            z_fused_nodes,
            output_window,
            h0=decoder_h0,
            flip_output=False,  # We handle reversal manually
        )
        if self.with_variance_head:
            recon, log_var = decoder_out
            recon = torch.flip(recon.view(b * n_nodes, -1), dims=[1])
            log_var = torch.flip(log_var.view(b * n_nodes, -1), dims=[1])
            log_var = log_var.clamp(self.logvar_clamp[0], self.logvar_clamp[1])
        else:
            recon = decoder_out
            recon = torch.flip(recon.view(b * n_nodes, -1), dims=[1])
            log_var = None

        if return_graph:
            aux = {}
            aux["alpha_temp"] = alpha_temp  # [B, N, N, W] dense attention
            if alpha_freq is not None:
                aux["alpha_freq"] = alpha_freq  # [B, N, N] dense spectral attention
            if self.use_spectral_view:
                aux["divergence_loss"] = div_loss
                if div_score is not None:
                    aux["divergence_score"] = div_score
                aux["adj_freq"] = adj_freq
                aux["attn_freq"] = attn_freq
                # Gate diagnostics for monitoring spectral branch contribution
                if gate_values is not None:
                    aux["gate_mean"] = gate_values.mean().item()
                    aux["gate_std"] = gate_values.std().item()
                    aux["z_freq_norm"] = z_freq.norm().item() if z_freq is not None else 0.0
            if self.with_variance_head:
                return recon, log_var, adj_temp, attn_temp, aux
            return recon, adj_temp, attn_temp, aux

        if self.with_variance_head:
            return recon, log_var
        return recon

    # ===== Anomaly scoring helpers =====
    def _align_target_and_recon(self, x_true: torch.Tensor, x_recon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure target and reconstruction share the same shape for error computation.
        """
        if x_true.dim() == x_recon.dim() + 1 and x_true.size(-1) == 1:
            x_true = x_true.squeeze(-1)
        if x_recon.dim() == 3 and x_recon.size(-1) == 1:
            x_recon = x_recon.squeeze(-1)
        if x_true.shape != x_recon.shape:
            x_true = x_true.view_as(x_recon)
        x_true = torch.nan_to_num(x_true, nan=0.0, posinf=1e6, neginf=-1e6)
        x_recon = torch.nan_to_num(x_recon, nan=0.0, posinf=1e6, neginf=-1e6)
        return x_true, x_recon

    def _dense_attn(self, edge_index: torch.Tensor, attn: torch.Tensor, num_graphs: int, n_nodes: int) -> torch.Tensor:
        """
        Convert sparse attentions to dense [B, N, N] tensors, row-normalized.
        """
        attn = attn.view(-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        device = attn.device
        src = edge_index[0]
        dst = edge_index[1]
        g = src // n_nodes
        src_local = src % n_nodes
        dst_local = dst % n_nodes
        dense = torch.zeros((num_graphs, n_nodes, n_nodes), device=device)
        dense[g, src_local, dst_local] = attn
        # Row-normalize to valid distributions
        row_sum = dense.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        dense = dense / row_sum
        return dense

    def _js_divergence(self, temp_dense: torch.Tensor, freq_dense: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Jensen-Shannon divergence between two dense attention distributions.
        Returns per-graph JS divergence [B].
        """
        temp_dense = torch.nan_to_num(temp_dense, nan=0.0, posinf=0.0, neginf=0.0)
        freq_dense = torch.nan_to_num(freq_dense, nan=0.0, posinf=0.0, neginf=0.0)
        P = temp_dense.clamp_min(eps)
        Q = freq_dense.clamp_min(eps)
        # Flatten per graph
        P = P.view(P.size(0), -1)
        Q = Q.view(Q.size(0), -1)
        P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(eps)
        M = (0.5 * (P + Q)).clamp_min(eps)
        kl_PM = (P * (P / M).log()).sum(dim=1)
        kl_QM = (Q * (Q / M).log()).sum(dim=1)
        js = 0.5 * (kl_PM + kl_QM)
        return js

    def _kl_divergence(self, temp_dense: torch.Tensor, freq_dense: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        KL divergence between two dense attention distributions.
        Returns per-graph KL divergence [B] for P||Q.
        """
        temp_dense = torch.nan_to_num(temp_dense, nan=0.0, posinf=0.0, neginf=0.0)
        freq_dense = torch.nan_to_num(freq_dense, nan=0.0, posinf=0.0, neginf=0.0)
        P = temp_dense.clamp_min(eps)
        Q = freq_dense.clamp_min(eps)
        P = P.view(P.size(0), -1)
        Q = Q.view(Q.size(0), -1)
        P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(eps)
        kl = (P * (P / Q).log()).sum(dim=1)
        return kl

    def _compute_view_divergence(
        self,
        adj_temp: torch.Tensor,
        attn_temp: torch.Tensor,
        adj_freq: torch.Tensor,
        attn_freq: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Align temporal and spectral attention to a common support and compute divergence.
        Returns (loss_scalar, per_graph_scores).
        """
        if adj_temp is None or adj_freq is None:
            return torch.tensor(0.0, device=batch.device), None
        if attn_temp is None or attn_freq is None:
            return torch.tensor(0.0, device=batch.device), None

        b_graphs = int(batch.max().item()) + 1
        n_nodes = cfg.dataset.n_nodes
        dense_temp = self._dense_attn(adj_temp, attn_temp.view(-1), b_graphs, n_nodes)
        dense_freq = self._dense_attn(adj_freq, attn_freq.view(-1), b_graphs, n_nodes)

        divergence_type = (self.divergence_type or "js").lower()
        if divergence_type == "kl":
            per_graph = self._kl_divergence(dense_temp, dense_freq)
        else:
            per_graph = self._js_divergence(dense_temp, dense_freq)

        return per_graph.mean(), per_graph.detach()

    def _compute_divergence_from_alpha(
        self,
        alpha_temp: torch.Tensor,
        alpha_freq: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute divergence directly from alpha matrices (dense attention).
        alpha_temp: [B, N, N] - temporal attention (averaged over time)
        alpha_freq: [B, N, N] - spectral attention
        Returns (loss_scalar, per_graph_scores).
        """
        if alpha_temp is None or alpha_freq is None:
            device = alpha_temp.device if alpha_temp is not None else alpha_freq.device
            return torch.tensor(0.0, device=device), None

        divergence_type = (self.divergence_type or "js").lower()
        if divergence_type == "kl":
            per_graph = self._kl_divergence(alpha_temp, alpha_freq)
        else:
            per_graph = self._js_divergence(alpha_temp, alpha_freq)

        return per_graph.mean(), per_graph.detach()

    def _node_error(self, x_true: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute per-node error using configured metric (l1 or l2)."""
        if self.topology_error == "l2":
            return ((x_true - x_recon) ** 2).mean(dim=-1)
        return (x_true - x_recon).abs().mean(dim=-1)

    def set_calibration_stats(self, err_mean: torch.Tensor, err_std: torch.Tensor) -> None:
        """Store per-sensor error calibration stats for z-score normalization."""
        self._cal_err_mean = err_mean.detach().clone()
        self._cal_err_std = err_std.detach().clone().clamp_min(1e-8)

    def _calibrate_node_err(self, node_err: torch.Tensor) -> torch.Tensor:
        """Z-score normalize per-node errors using calibration stats (no-op if not set)."""
        if self._cal_err_mean is None or self._cal_err_std is None:
            return node_err
        n = cfg.dataset.n_nodes
        b = node_err.numel() // n
        err_2d = node_err.view(b, n)
        err_2d = (err_2d - self._cal_err_mean) / self._cal_err_std
        return err_2d.view(-1)

    def _topology_scores_per_graph(
        self,
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route to the appropriate topology scoring method based on self.topology_mode.
        """
        mode = getattr(self, 'topology_mode', 'own_error_degree')
        if mode == 'neighbor_propagation':
            return self._topology_scores_neighbor_propagation(x_true, x_recon, edge_index, edge_weight)
        elif mode == 'plain_error':
            return self._topology_scores_plain_error(x_true, x_recon)
        else:
            return self._topology_scores_own_error_degree(x_true, x_recon, edge_index, edge_weight)

    def _topology_scores_own_error_degree(
        self,
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Original: own error / degree (r_j = err(x̂_j, x_j) / d_j)"""
        x_true, x_recon = self._align_target_and_recon(x_true, x_recon)
        n = cfg.dataset.n_nodes
        b = x_true.shape[0] // n
        device = x_true.device

        node_err = self._node_error(x_true, x_recon)  # [B*N]
        node_err = self._calibrate_node_err(node_err)

        num_nodes = node_err.numel()
        degree = torch.zeros(num_nodes, device=device, dtype=node_err.dtype)
        if edge_index is not None and edge_weight is not None:
            src, dst = edge_index
            weights = edge_weight.abs().to(device=device, dtype=node_err.dtype)
            weights = weights.detach()
            degree.index_add_(0, src, weights)
            degree.index_add_(0, dst, weights)

        eps = 1e-8
        node_scores = node_err / (degree + eps)
        return node_scores.view(b, n).mean(dim=1)

    def _topology_scores_neighbor_propagation(
        self,
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Neighbor-error propagation: weighted in-degree of neighbor errors."""
        x_true, x_recon = self._align_target_and_recon(x_true, x_recon)
        n = cfg.dataset.n_nodes
        b = x_true.shape[0] // n
        node_err = self._node_error(x_true, x_recon)  # [B*N]
        node_err = self._calibrate_node_err(node_err)

        num_nodes = node_err.numel()
        weighted_in = torch.zeros(num_nodes, device=node_err.device)
        in_degree = torch.zeros(num_nodes, device=node_err.device)
        if edge_index is not None and edge_weight is not None:
            src, dst = edge_index
            w = edge_weight.abs().float()
            w = w.detach()
            weighted_in.index_add_(0, dst, node_err[src].float() * w)
            in_degree.index_add_(0, dst, w)

        node_scores = weighted_in / (in_degree + 1e-8)
        return node_scores.view(b, n).mean(dim=1)

    def _topology_scores_plain_error(
        self,
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
    ) -> torch.Tensor:
        """Plain error without topology weighting."""
        x_true, x_recon = self._align_target_and_recon(x_true, x_recon)
        n = cfg.dataset.n_nodes
        b = x_true.shape[0] // n
        node_err = self._node_error(x_true, x_recon)  # [B*N]
        node_err = self._calibrate_node_err(node_err)
        return node_err.view(b, n).mean(dim=1)

    def compute_topology_aware_anomaly_score(
        self,
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a scalar anomaly score averaged over the batch, weighted by graph attention.
        """
        graph_scores = self._topology_scores_per_graph(x_true, x_recon, edge_index, edge_weight)
        graph_scores = torch.nan_to_num(graph_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return graph_scores.mean()

    def compute_anomaly_scores_per_sample(
        self,
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns topology-aware anomaly scores per graph in the batch: [batch].
        """
        return self._topology_scores_per_graph(x_true, x_recon, edge_index, edge_weight)

    def compute_anomaly_scores_per_timestep(
        self,
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns per-timestep reconstruction error aggregated over nodes for each graph: [batch, window].
        Topology weighting is not applied here; this is a straightforward per-timestep MSE.
        """
        x_true, x_recon = self._align_target_and_recon(x_true, x_recon)
        n = cfg.dataset.n_nodes
        b = max(int(x_true.shape[0] // n), 1)
        x_true = x_true.view(b, n, -1)
        x_recon = x_recon.view(b, n, -1)
        per_timestep = ((x_true - x_recon) ** 2).mean(dim=1)  # [b, window]
        return per_timestep
