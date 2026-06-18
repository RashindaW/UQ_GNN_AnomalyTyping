"""Heteroscedastic GDN with two independent output heads (mu, log_var).

Phase 1 of gdn_uq_modification_plan.md. Reuses OutLayer / GNNLayer / GraphLayer
unchanged; only the head is split. Forward returns (mu, log_var) — no parameter
sharing between heads.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .GDN import OutLayer, GNNLayer, get_batch_edge_index


class GDN_UQ(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256,
                 input_dim=10, out_layer_num=1, topk=20,
                 logvar_clamp=(-10.0, 10.0)):
        super().__init__()

        self.edge_index_sets = edge_index_sets

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1)
            for _ in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        head_in = dim * edge_set_num
        self.mu_head = OutLayer(head_in, node_num, out_layer_num, inter_num=out_layer_inter_dim)
        self.logvar_head = OutLayer(head_in, node_num, out_layer_num, inter_num=out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.logvar_clamp = logvar_clamp

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(
                    edge_index, batch_num, node_num
                ).to(device)

            batch_edge_index = self.cache_edge_index_sets[i]

            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(
                weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1)
            )
            cos_ji_mat = cos_ji_mat / normed_mat

            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = (
                torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num)
                .flatten().to(device).unsqueeze(0)
            )
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(
                gated_edge_index, batch_num, node_num
            ).to(device)
            gcn_out = self.gnn_layers[i](
                x, batch_gated_edge_index,
                node_num=node_num * batch_num, embedding=all_embeddings
            )

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))

        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)

        mu = self.mu_head(out).view(-1, node_num)
        log_var = self.logvar_head(out).view(-1, node_num)
        log_var = log_var.clamp(self.logvar_clamp[0], self.logvar_clamp[1])

        return mu, log_var
