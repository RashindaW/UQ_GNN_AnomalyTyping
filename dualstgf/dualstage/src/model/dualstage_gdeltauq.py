"""DualSTAGE_GDeltaUQ: the ONE canonical UQ method on the DualSTAGE backbone.

G-DeltaUQ hidden-layer stochastic anchoring (trained-in, batch-shuffled anchor;
K-anchor averaging at inference), exactly the CST-GL/TopoGDN pattern:

  forward_split(data)        anchor-invariant backbone, run ONCE per batch:
                             encoders + temporal AND spectral dynamic graphs +
                             WeightedGIN stacks + gated fusion + decoder_norm.
                             Returns (h_pre [B,N,D], div_loss, adj_temp, attn_temp).
  forward_anchored(h, c)     the decoder tail on cat([h - c, c], -1) [B,N,2D]:
                             GRU(2D -> hidden) over W repeated steps + Linear,
                             output flipped to chronological order (parent parity).
                             Returns (recon [B*N, W], penultimate [B,N,hidden])
                             where penultimate = the GRU hidden at the
                             chronological LAST timestep (pre-flip index 0),
                             the feature for the aleatoric head and Omega.
  forward(data, ...)         training path: batch-shuffled detached anchor;
                             returns the same shapes/aux as the plain DualSTAGE
                             so the baseline composite loss works unchanged.

The anchor sits at z_fused, AFTER fusion and decoder_norm, BEFORE the GRU
decoder (non-linear tail -> non-degenerate epistemic variance). The GATv2
attention is computed in forward_split and is therefore anchor-invariant:
no structural channel for this backbone (3 channels, CST-GL precedent).

NOT to be confused with dualstage_uq.py (heteroscedastic single-pass variance
head, a different UQ family, unused by the campaign).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import cfg
from src.model.dualstage import DualSTAGE


class DualSTAGE_GDeltaUQ(DualSTAGE):
    def __init__(self, *args, gnn_embed_dim=40, recon_hidden_dim=16,
                 num_recon_layers=1, **kwargs):
        if kwargs.get("num_gnn_layers", 1) < 2:
            kwargs["num_gnn_layers"] = 2      # anchoring the raw input is degenerate
        if kwargs.get("with_variance_head"):
            raise ValueError("variance head is a different UQ method; keep it off")
        super().__init__(*args, gnn_embed_dim=gnn_embed_dim,
                         recon_hidden_dim=recon_hidden_dim,
                         num_recon_layers=num_recon_layers, **kwargs)
        self.anchor_dim = gnn_embed_dim
        # anchored decoder: GRU input doubles to consume cat(h - anchor, anchor)
        self.decoder.rnn = nn.GRU(2 * gnn_embed_dim, recon_hidden_dim,
                                  num_recon_layers, batch_first=True)

    # ------------------------------------------------------------------
    # anchor-invariant backbone (faithful copy of DualSTAGE.forward steps 1-6
    # + decoder_norm; see dualstage.py:1134-1306)
    # ------------------------------------------------------------------
    def forward_split(self, data):
        x, c, edge_index, batch = data.x, data.c, data.edge_index, data.batch
        n_nodes = cfg.dataset.n_nodes
        context_expanded = None
        context = None

        if self.aug_control:
            n_ctrl = cfg.dataset.ocvar_dim
            if c.dim() == 2:
                if c.shape[0] % n_ctrl == 0:
                    b = c.shape[0] // n_ctrl
                    c_in = c.view(b, n_ctrl, -1).transpose(1, 2)
                else:
                    raise ValueError(f"Unexpected control tensor shape: {c.shape}")
            elif c.dim() == 3:
                c_in = c if c.shape[-1] == n_ctrl else c.transpose(1, 2)
            else:
                raise ValueError(f"Unexpected control tensor shape: {c.shape}")
            context = self.control_encoder(c_in)
            context_expanded = context.repeat_interleave(n_nodes, dim=0)

        # 2. node encoding
        b = x.shape[0] // n_nodes
        x_nodes = x.view(b, n_nodes, -1)
        x_idcnn = self.idcnn(x_nodes.view(b * n_nodes, 1, -1)).view(b, n_nodes, -1)
        gru_input = x_idcnn if self.node_gru_input == "filtered" else x_nodes
        h_temp = self.node_encoder(
            gru_input, h0=context_expanded if self.aug_control else None)
        h_temp = torch.nan_to_num(h_temp, nan=0.0, posinf=0.0, neginf=0.0)
        h_temp = h_temp.view(b * n_nodes, -1)

        # 3. temporal dynamic graph (GATv2 + EdgeGRU); anchor-invariant
        adj_temp, attn_temp, alpha_temp = self.temporal_feature_graph(x_idcnn, batch)
        attn_temp = torch.nan_to_num(attn_temp, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. spectral branch + divergence (parent lines 1191-1253)
        h_freq = None
        adj_freq, attn_freq = None, None
        div_loss = torch.tensor(0.0, device=x.device)
        if self.use_spectral_view:
            x_reshaped = x.view(b, n_nodes, -1)
            h_freq_batch = self.spectral_encoder(x_reshaped)
            h_freq_batch = torch.nan_to_num(h_freq_batch, nan=0.0, posinf=0.0, neginf=0.0)
            if self.aug_control and context is not None:
                oc_spectral = self.spectral_oc_proj(context)
                h_freq_batch = h_freq_batch + oc_spectral.unsqueeze(1).expand(-1, n_nodes, -1)
            h_freq = h_freq_batch.view(b * n_nodes, -1)
            adj_freq, attn_freq, alpha_freq = self.spectral_feature_graph(h_freq, batch, n_nodes)
            attn_freq = torch.nan_to_num(attn_freq, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                alpha_temp_avg = alpha_temp.mean(dim=-1)
                if self.temporal_feature_graph.learn_sys:
                    alpha_temp_avg = (alpha_temp_avg + alpha_temp_avg.transpose(-1, -2)) / 2
                if self.spectral_feature_graph.learn_sys:
                    alpha_freq_sym = (alpha_freq + alpha_freq.transpose(-1, -2)) / 2
                else:
                    alpha_freq_sym = alpha_freq
                diag = torch.eye(alpha_temp_avg.size(-1), device=alpha_temp_avg.device,
                                 dtype=torch.bool)
                alpha_temp_avg = alpha_temp_avg.masked_fill(diag.unsqueeze(0), 0.0)
                alpha_freq_sym = alpha_freq_sym.masked_fill(diag.unsqueeze(0), 0.0)
                alpha_temp_avg = alpha_temp_avg / alpha_temp_avg.sum(-1, keepdim=True).clamp_min(1e-8)
                alpha_freq_sym = alpha_freq_sym / alpha_freq_sym.sum(-1, keepdim=True).clamp_min(1e-8)
                div_loss, _ = self._compute_divergence_from_alpha(alpha_temp_avg, alpha_freq_sym)
            except Exception:
                div_loss = torch.tensor(0.0, device=x.device)

        # 5. GNN stacks
        z_temp = h_temp
        for i, conv in enumerate(self.gnn_layers):
            z_temp = conv(z_temp, adj_temp, attn_temp)
            if self.do_gnn_norm:
                z_temp = self._apply_norm(self.gnn_norms[i], z_temp, batch)
            z_temp = torch.nan_to_num(z_temp, nan=0.0, posinf=0.0, neginf=0.0)
            z_temp = F.relu(z_temp)
            z_temp = self.feat_dropout(z_temp)

        z_freq = None
        if self.use_spectral_view:
            z_freq = h_freq
            attn_freq_flat = attn_freq.view(-1) if attn_freq is not None else None
            for i, conv in enumerate(self.gnn_layers_freq):
                z_freq = conv(z_freq, adj_freq, attn_freq_flat)
                if self.do_gnn_norm:
                    z_freq = self._apply_norm(self.gnn_norms_freq[i], z_freq, batch)
                z_freq = torch.nan_to_num(z_freq, nan=0.0, posinf=0.0, neginf=0.0)
                z_freq = F.relu(z_freq)
                z_freq = self.feat_dropout(z_freq)

        # 6. fusion (+ decoder_norm, so the anchor space is the normed space)
        z_fused = z_temp
        if self.spectral_only and z_freq is not None:
            z_fused = z_freq
        elif self.use_spectral_view:
            if self.fuse_mode == "concat":
                z_fused = self.fusion_layer(torch.cat([z_temp, z_freq], dim=-1))
            elif self.fuse_mode == "sum":
                z_fused = z_temp + z_freq
            elif self.fuse_mode == "gated":
                g = self.gate(torch.cat([z_temp, z_freq], dim=-1))
                z_fused = g * z_temp + (1 - g) * z_freq

        if self.do_decoder_norm:
            z_fused = self._apply_norm(self.decoder_norm, z_fused, batch)
        z_fused = torch.nan_to_num(z_fused, nan=0.0, posinf=0.0, neginf=0.0)
        h_pre = z_fused.view(b, n_nodes, -1)            # [B, N, D]
        return h_pre, div_loss, adj_temp, attn_temp

    # ------------------------------------------------------------------
    # anchored decoder tail (mirrors ReconstructionModel.reconstruct,
    # dualstage.py:785-810, with the manual flip of forward:1331)
    # ------------------------------------------------------------------
    def forward_anchored(self, h_pre, anchor):
        a_in = torch.cat([h_pre - anchor, anchor], dim=-1)   # [B, N, 2D]
        b, n, d2 = a_in.shape
        w = cfg.dataset.window_size
        z_rep = a_in.view(b * n, d2).unsqueeze(1).repeat(1, w, 1)
        out, _ = self.decoder.rnn(z_rep)                     # [B*N, W, hidden]
        recon = self.decoder.out(out).squeeze(-1)            # [B*N, W] (reversed)
        recon = torch.flip(recon, dims=[1])                  # chronological
        penult = out[:, 0, :].view(b, n, -1)                 # chronological-LAST step
        return recon, penult

    # ------------------------------------------------------------------
    # training path: trained-in batch-shuffle anchoring; same return contract
    # as the plain DualSTAGE so the baseline composite loss works unchanged
    # ------------------------------------------------------------------
    def forward(self, data, return_graph=False, anchor=None):
        h_pre, div_loss, adj_t, attn_t = self.forward_split(data)
        if anchor is None:
            perm = torch.randperm(h_pre.shape[0], device=h_pre.device)
            anchor = h_pre[perm].detach()
        recon, _ = self.forward_anchored(h_pre, anchor)
        if return_graph:
            return recon, adj_t, attn_t, {"divergence_loss": div_loss}
        return recon
