import torch
from util.env import get_device
from test import test
import torch.nn.functional as F


def loss_func(out, y_true, logvar_l2: float = 0.0):
    """Auto-detect tuple output: (mu, log_var) -> Gaussian NLL; tensor -> MSE.

    `logvar_l2` (β) optionally adds `β · mean(log_var ** 2)` to the NLL — pulls
    log_var toward 0 and discourages saturation at the clamp boundaries. Set
    to 0.0 (default) for the original behaviour.
    """
    if isinstance(out, tuple):
        mu, log_var = out
        nll = F.gaussian_nll_loss(mu, y_true, log_var.exp(), reduction='mean', eps=1e-6)
        if logvar_l2 > 0.0:
            nll = nll + logvar_l2 * log_var.pow(2).mean()
        return nll
    return F.mse_loss(out, y_true, reduction='mean')


def _val_logvar_diagnostic(model, val_dataloader, device):
    """Track log_var health on validation set: flag sigma collapse / explosion early."""
    model.eval()
    log_var_chunks = []
    with torch.no_grad():
        for x, _, _, edge_index in val_dataloader:
            x = x.float().to(device)
            edge_index = edge_index.float().to(device)
            out = model(x, edge_index)
            if not isinstance(out, tuple):
                model.train()
                return None
            _, log_var = out
            log_var_chunks.append(log_var.detach().cpu())
    model.train()
    if not log_var_chunks:
        return None
    lv = torch.cat(log_var_chunks).flatten()
    return {
        'mean': float(lv.mean()),
        'median': float(lv.median()),
        'clamp_saturation_fraction': float(((lv <= -9.9) | (lv >= 9.9)).float().mean()),
    }


def train(model=None, save_path='', config={}, train_dataloader=None, val_dataloader=None,
          feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat',
          train_dataset=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    train_loss_list = []

    device = get_device()
    logvar_l2 = float(config.get('logvar_l2', 0.0))

    acu_loss = 0
    min_loss = 1e+8

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    model.train()

    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            x, labels, edge_index = [
                item.float().to(device) for item in [x, labels, edge_index]
            ]

            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = loss_func(out, labels, logvar_l2=logvar_l2)

            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()

            i += 1

        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
            i_epoch, epoch,
            acu_loss / len(dataloader), acu_loss
        ), flush=True)

        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader)

            diag = _val_logvar_diagnostic(model, val_dataloader, device)
            if diag is not None:
                print(
                    'epoch ({} / {}) sigma_health: mean(log_var)={:.4f} '
                    'median(log_var)={:.4f} clamp_saturation_fraction={:.4f}'.format(
                        i_epoch, epoch, diag['mean'], diag['median'],
                        diag['clamp_saturation_fraction']
                    ),
                    flush=True,
                )

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

    return train_loss_list
