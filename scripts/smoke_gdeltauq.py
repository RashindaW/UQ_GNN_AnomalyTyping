"""Smoke test for the GDN_GDeltaUQ model + aleatoric head.

Builds a toy model with no SWaT data dependency, runs a forward pass and K=3
anchored passes, and asserts:
  - mu has shape (B, V)
  - h has shape (B, V, dim)
  - attention shape is (B * topk * V, 1, 1)
  - U_par > 0 and U_str > 0 across anchors  (Plan Phase 9 step 2 sanity check)
  - The learned_graph is identical across anchored passes (it depends only on
    self.embedding, not on the anchor).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.GDN_GDeltaUQ import GDN_GDeltaUQ
from models.aleatoric_head import AleatoricHead


def main():
    torch.manual_seed(0)
    V = 8
    W = 5
    d = 16
    topk = 4
    B = 6
    K = 3

    # Toy edge_index; the model ignores it for graph learning but builds one
    # to set node_num.
    src = torch.arange(V).repeat_interleave(topk)
    tgt = torch.randint(0, V, (V * topk,))
    edge_index = torch.stack([src, tgt], dim=0)

    model = GDN_GDeltaUQ(
        edge_index_sets=[edge_index], node_num=V,
        dim=d, input_dim=W, out_layer_num=1, out_layer_inter_dim=32,
        topk=topk, n_gnn_layers=2,
    )
    model.eval()

    x = torch.randn(B, V, W)

    # Single forward (training-mode anchor sampling).
    model.train()
    mu, h, att = model(x, edge_index)
    assert mu.shape == (B, V), f'mu shape {mu.shape}'
    assert h.shape == (B, V, d), f'h shape {h.shape}'
    # Attention should have B * topk * V values (after PyG remove_self_loops +
    # add_self_loops, the total count is preserved).
    assert att.numel() == B * topk * V, (
        f'attention numel {att.numel()} != {B * topk * V}'
    )
    print(f'OK: forward pass shapes mu={mu.shape} h={h.shape} att={att.shape}')

    # K-anchor inference path: cache h_pre once, run K anchored heads.
    model.eval()
    with torch.no_grad():
        h_pre = model.forward_split(x, edge_index)
        assert h_pre.shape == (B, V, d), f'h_pre shape {h_pre.shape}'
        graphs = []
        mu_list = []
        att_list = []
        nonself_per_sample = (topk - 1) * V
        for k in range(K):
            perm = torch.randperm(B)
            anchor = h_pre[perm].detach()
            mu_k, _, att_k = model.forward_anchored(h_pre, anchor, edge_index)
            mu_list.append(mu_k)
            att_flat = att_k.view(-1)[:B * nonself_per_sample]
            att_list.append(att_flat.view(B, nonself_per_sample))
            graphs.append(model.learned_graph.detach().clone())

    # learned_graph invariance across anchors.
    for k in range(1, K):
        assert torch.equal(graphs[0], graphs[k]), (
            f'learned_graph changed between pass 0 and pass {k}'
        )
    print('OK: learned_graph is invariant across K anchors')

    mu_stack = torch.stack(mu_list, dim=0)               # (K, B, V)
    att_stack = torch.stack(att_list, dim=0)             # (K, B, (topk-1)*V)
    U_par = mu_stack.var(dim=0, unbiased=True).mean().item()
    U_str = att_stack.var(dim=0, unbiased=True).mean().item()
    print(f'mean U_par={U_par:.6e} mean U_str={U_str:.6e}')
    assert U_par > 0, 'U_par is zero - anchoring not affecting predictions'
    assert U_str > 0, ('U_str is zero - anchoring not affecting attention; the '
                       'anchored layer is wired incorrectly')
    print('OK: U_par > 0 and U_str > 0 (anchoring is producing variance)')

    # Aleatoric head.
    head = AleatoricHead(hidden_dim=d, num_sensors=V, sensor_embed_dim=4, mlp_hidden=16)
    log_sigma2 = head(h)
    assert log_sigma2.shape == (B, V), f'log_sigma2 shape {log_sigma2.shape}'
    print(f'OK: AleatoricHead forward shape {log_sigma2.shape}')

    # 1-layer rejection.
    try:
        GDN_GDeltaUQ([edge_index], V, n_gnn_layers=1, dim=d, input_dim=W, topk=topk)
        raise AssertionError('expected ValueError for n_gnn_layers=1')
    except ValueError as e:
        print(f'OK: 1-layer constructor rejected: {e}')

    print('\nSMOKE TEST PASSED')


if __name__ == '__main__':
    main()
