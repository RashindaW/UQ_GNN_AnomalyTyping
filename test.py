import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from util.time import timeSincePlus
from util.env import get_device


def _eval_loss(out, y_true):
    """Same auto-detect contract as train.loss_func."""
    if isinstance(out, tuple):
        mu, log_var = out
        return F.gaussian_nll_loss(mu, y_true, log_var.exp(), reduction='mean', eps=1e-6)
    return F.mse_loss(out, y_true, reduction='mean')


def test(model, dataloader):
    device = get_device()

    test_loss_list = []
    now = time.time()

    # Baseline path accumulators (single tensor output).
    t_pred_list = []
    t_ground_list = []
    t_labels_list = []

    # UQ path accumulators (tuple output).
    t_mu_list = []
    t_logvar_list = []

    test_len = len(dataloader)

    model.eval()

    is_uq = None  # set on first batch

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [
            item.to(device).float() for item in [x, y, labels, edge_index]
        ]

        with torch.no_grad():
            out = model(x, edge_index)

            if is_uq is None:
                is_uq = isinstance(out, tuple)

            loss = _eval_loss(out, y)

            if is_uq:
                mu, log_var = out
                labels_b = labels.unsqueeze(1).repeat(1, mu.shape[1])
                if len(t_mu_list) == 0:
                    t_mu_list = mu
                    t_logvar_list = log_var
                    t_ground_list = y
                    t_labels_list = labels_b
                else:
                    t_mu_list = torch.cat((t_mu_list, mu), dim=0)
                    t_logvar_list = torch.cat((t_logvar_list, log_var), dim=0)
                    t_ground_list = torch.cat((t_ground_list, y), dim=0)
                    t_labels_list = torch.cat((t_labels_list, labels_b), dim=0)
            else:
                predicted = out
                labels_b = labels.unsqueeze(1).repeat(1, predicted.shape[1])
                if len(t_pred_list) == 0:
                    t_pred_list = predicted
                    t_ground_list = y
                    t_labels_list = labels_b
                else:
                    t_pred_list = torch.cat((t_pred_list, predicted), dim=0)
                    t_ground_list = torch.cat((t_ground_list, y), dim=0)
                    t_labels_list = torch.cat((t_labels_list, labels_b), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    if is_uq:
        return avg_loss, [
            t_mu_list.tolist(),
            t_logvar_list.tolist(),
            t_ground_list.tolist(),
            t_labels_list.tolist(),
        ]
    return avg_loss, [
        t_pred_list.tolist(),
        t_ground_list.tolist(),
        t_labels_list.tolist(),
    ]
