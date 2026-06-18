"""Train the aleatoric MLP head with the GDN_GDeltaUQ frozen.

Phase 2 of the source plan. Steps:
  1. Run K-anchor inference on the 20% slice to get mu_bar_v(t) and h_bar_v(t)
     (both frozen, no_grad through the GDN).
  2. Build a TensorDataset of (h_bar, y, mu_bar) tuples.
  3. Optimize AleatoricHead with Gaussian NLL on observed y and frozen mu_bar:
        nll = ((y - mu_bar)^2 / (2 * sigma^2_ale)) + 0.5 * log_sigma^2_ale

beta-NLL option (Seitzer et al., ICLR 2022): nll_loss / train_aleatoric accept
an optional beta in [0, 1]. beta=0.0 (the default) reproduces the vanilla
Gaussian NLL above byte-for-byte; beta>0 multiplies each per-element NLL term by
a detached sigma^2**beta, which down-weights low-variance terms and improves
heteroscedastic variance learning (typical value beta=0.5). The default keeps
all existing runs unchanged.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from inference_gdeltauq import run_inference


def precompute_frozen_predictions(loaded, dataset, batch_size=128):
    """Run K-anchor inference once, cache mu_bar / h_bar / y for training.

    Returns torch tensors on CPU (float32):
        h_bar:   (N, V, d)
        y:       (N, V)
        mu_bar:  (N, V)
    """
    out = run_inference(loaded, dataset, batch_size=batch_size)
    return (
        torch.from_numpy(out.h_bar),
        torch.from_numpy(out.ground_truth),
        torch.from_numpy(out.mu_bar),
    )


def nll_loss(y, mu_bar, log_sigma2, eps=1e-6, beta=0.0):
    sigma2 = log_sigma2.exp().clamp_min(eps)
    per = ((y - mu_bar) ** 2) / (2.0 * sigma2) + 0.5 * log_sigma2
    if beta > 0:
        per = per * (sigma2.detach() ** beta)
    return per.mean()


def train_aleatoric(
    head,
    h_bar,
    y,
    mu_bar,
    device,
    epochs=5,
    batch_size=32,
    lr=1e-3,
    beta=0.0,
):
    head = head.to(device)
    head.train()

    ds = TensorDataset(h_bar, y, mu_bar)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    for ep in range(epochs):
        running = 0.0
        n_batches = 0
        for h_bar_b, y_b, mu_b in loader:
            h_bar_b = h_bar_b.to(device).float()
            y_b = y_b.to(device).float()
            mu_b = mu_b.to(device).float()

            optimizer.zero_grad()
            log_sigma2 = head(h_bar_b)
            loss = nll_loss(y_b, mu_b, log_sigma2, beta=beta)
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1

        print(
            f'aleatoric epoch ({ep} / {epochs}) nll={running / max(1, n_batches):.6f}',
            flush=True,
        )

    head.eval()
    return head
