"""Aleatoric MLP head for the G-DeltaUQ GDN variant.

Takes the anchor-averaged penultimate hidden representation h_bar_v (per-sensor,
dim d) and a learnable sensor embedding e_v (dim 16), and predicts a scalar
log sigma^2_ale per (batch, sensor). Trained post-hoc with Gaussian NLL using
frozen mean predictions from K-anchor inference (plan Phase 2).
"""
import torch
import torch.nn as nn


class AleatoricHead(nn.Module):
    def __init__(self, hidden_dim, num_sensors, sensor_embed_dim=16, mlp_hidden=64,
                 logvar_clamp=(-10.0, 10.0)):
        super().__init__()
        self.num_sensors = num_sensors
        self.sensor_embedding = nn.Embedding(num_sensors, sensor_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + sensor_embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.logvar_clamp = logvar_clamp

    def forward(self, h_bar):
        """
        Args:
            h_bar: (B, V, d) anchor-averaged hidden representation.
        Returns:
            log_sigma2: (B, V) per-sensor predicted log variance.
        """
        batch_num, node_num, _ = h_bar.shape
        assert node_num == self.num_sensors, (
            f"AleatoricHead expects V={self.num_sensors} sensors, got {node_num}"
        )
        sensor_ids = torch.arange(node_num, device=h_bar.device)
        e = self.sensor_embedding(sensor_ids)  # (V, sensor_embed_dim)
        e = e.unsqueeze(0).expand(batch_num, -1, -1)  # (B, V, sensor_embed_dim)
        x = torch.cat([h_bar, e], dim=-1)  # (B, V, d + 16)
        log_sigma2 = self.mlp(x).squeeze(-1)  # (B, V)
        log_sigma2 = log_sigma2.clamp(self.logvar_clamp[0], self.logvar_clamp[1])
        return log_sigma2
