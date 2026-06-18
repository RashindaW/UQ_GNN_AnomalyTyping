"""Training loop for the GDN + G-DeltaUQ variant.

Mirrors the structure of train.py but:
  - Loss is MSE on the anchored mean prediction (the K-anchor variance is an
    inference-time signal, not a training-time loss term).
  - Anchor sampling happens inside model.forward (anchor=None path).
  - Includes a small per-epoch diagnostic that runs K=3 anchored passes on a
    validation batch and reports mean U^par / U^str so we can confirm the
    anchoring is producing variance during training.
"""
import torch
import torch.nn.functional as F


def _val_loss(model, val_dataloader, device):
    model.eval()
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y, _, edge_index in val_dataloader:
            x = x.float().to(device)
            y = y.float().to(device)
            edge_index = edge_index.float().to(device)
            mu, _, _ = model(x, edge_index)
            total += F.mse_loss(mu, y, reduction='mean').item()
            n_batches += 1
    model.train()
    return total / max(1, n_batches)


def _anchoring_diagnostic(model, val_dataloader, device, k_diag=3):
    """Sanity check that U^par > 0 and U^str > 0 with k_diag anchors."""
    model.eval()
    try:
        batch = next(iter(val_dataloader))
    except StopIteration:
        model.train()
        return None
    x, _, _, edge_index = batch
    x = x.float().to(device)
    edge_index = edge_index.float().to(device)

    with torch.no_grad():
        h_pre = model.forward_split(x, edge_index)  # (B, V, d)
        B = h_pre.shape[0]
        if B < 2:
            model.train()
            return None
        mu_list = []
        att_list = []
        for k in range(k_diag):
            perm = torch.randperm(B, device=h_pre.device)
            anchor = h_pre[perm].detach()
            mu, _, att = model.forward_anchored(h_pre, anchor, edge_index)
            mu_list.append(mu)
            att_list.append(att.squeeze(-1).squeeze(-1))
        mu_stack = torch.stack(mu_list, dim=0)         # (K, B, V)
        att_stack = torch.stack(att_list, dim=0)       # (K, B*topk*V)
        U_par = mu_stack.var(dim=0, unbiased=True).mean().item()
        U_str = att_stack.var(dim=0, unbiased=True).mean().item()
    model.train()
    return {'U_par': U_par, 'U_str': U_str}


def train_gdeltauq(model, save_path, config, train_dataloader, val_dataloader, device):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=config.get('decay', 0.0)
    )

    epoch = config['epoch']
    early_stop_win = 15

    min_loss = float('inf')
    stop_improve_count = 0
    train_loss_history = []

    diag_every = max(1, int(config.get('diag_every', 1)))

    model.train()
    for i_epoch in range(epoch):
        acu_loss = 0.0
        n_batches = 0
        for x, y, _, edge_index in train_dataloader:
            x = x.float().to(device)
            y = y.float().to(device)
            edge_index = edge_index.float().to(device)

            optimizer.zero_grad()
            mu, _, _ = model(x, edge_index)
            loss = F.mse_loss(mu, y, reduction='mean')
            loss.backward()
            optimizer.step()

            acu_loss += loss.item()
            n_batches += 1
            train_loss_history.append(loss.item())

        train_mse = acu_loss / max(1, n_batches)
        print(
            f'epoch ({i_epoch} / {epoch}) train_mse={train_mse:.8f} acu={acu_loss:.6f}',
            flush=True,
        )

        if val_dataloader is not None:
            val_mse = _val_loss(model, val_dataloader, device)
            print(f'epoch ({i_epoch} / {epoch}) val_mse={val_mse:.8f}', flush=True)

            if (i_epoch % diag_every) == 0:
                diag = _anchoring_diagnostic(model, val_dataloader, device)
                if diag is not None:
                    print(
                        f'epoch ({i_epoch} / {epoch}) anchoring: '
                        f'U_par={diag["U_par"]:.6e} U_str={diag["U_str"]:.6e}',
                        flush=True,
                    )

            if val_mse < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_mse
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                print(f'early stopping at epoch {i_epoch}', flush=True)
                break
        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

    return train_loss_history
