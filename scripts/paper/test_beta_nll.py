"""CPU unit test for the beta-NLL option in train_aleatoric_gdeltauq.py.

Checks:
  1. EQUIVALENCE: nll_loss(..., beta=0.0) is byte-identical to the original
     vanilla Gaussian NLL expression on a fixed input (asserts exact equality).
  2. beta=0 vs beta=0.5 both run end-to-end and the training NLL DECREASES.
  3. beta=0.5 actually changes the loss value vs beta=0 on a non-trivial input
     (guards against the branch being a no-op).

Uses the REAL AleatoricHead (hidden_dim, num_sensors, ...) with (B, V, d) inputs.
Tiny synthetic heteroscedastic process. Run: $PY scripts/paper/test_beta_nll.py
(exit 0 PASS / 1 FAIL).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.aleatoric_head import AleatoricHead  # noqa: E402
from train_aleatoric_gdeltauq import nll_loss, train_aleatoric  # noqa: E402


def _old_nll_expr(y, mu_bar, log_sigma2, eps=1e-6):
    """The ORIGINAL formula, copied verbatim from the pre-edit source."""
    sigma2 = log_sigma2.exp().clamp_min(eps)
    return (((y - mu_bar) ** 2) / (2.0 * sigma2) + 0.5 * log_sigma2).mean()


def _make_hetero(seed=0, N=2000, V=8, d=16):
    """y = mu_true + eps, eps ~ N(0, sigma(h)^2) with sigma a function of h.

    Shapes match the real pipeline: h_bar (N, V, d), y (N, V), mu_bar (N, V).
    We feed mu_bar = mu_true so the residual the head must explain is exactly the
    heteroscedastic noise; h drives sigma through a fixed linear map.
    """
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, V, d, generator=g)
    w = torch.randn(d, generator=g)
    log_sigma2_true = 0.5 * (h @ w)            # (N, V) learnable log-variance
    sigma_true = (0.5 * log_sigma2_true).exp()  # (N, V)
    mu_true = torch.zeros(N, V)                 # constant mean; residual == noise
    eps = torch.randn(N, V, generator=g) * sigma_true
    y = mu_true + eps
    return h, y, mu_true


def _eval_nll(head, h, y, mu, beta=0.0):
    head.eval()
    with torch.no_grad():
        ls = head(h)            # (N, V)
        return float(nll_loss(y, mu, ls, beta=beta).item())


def main():
    failures = []
    device = "cpu"

    # ----- (1) exact equivalence at beta=0 on a FIXED input -----
    g = torch.Generator().manual_seed(123)
    y = torch.randn(257, generator=g)
    mu = torch.randn(257, generator=g)
    ls = torch.randn(257, generator=g) * 2.0  # spans negative & positive log-var
    a = nll_loss(y, mu, ls, beta=0.0)
    b = _old_nll_expr(y, mu, ls)
    print("equivalence: nll_loss(beta=0)=%.10f  old_expr=%.10f  diff=%.3e"
          % (a.item(), b.item(), abs(a.item() - b.item())))
    if not torch.equal(a, b):
        failures.append("nll_loss(beta=0) not byte-identical to old expression "
                        "(diff=%.3e)" % abs(a.item() - b.item()))

    # also confirm the clamp branch matches when log_sigma2 is very negative
    ls_small = torch.full((64,), -50.0)
    if not torch.equal(nll_loss(y[:64], mu[:64], ls_small, beta=0.0),
                       _old_nll_expr(y[:64], mu[:64], ls_small)):
        failures.append("nll_loss(beta=0) mismatch under eps-clamp regime")

    # ----- (3) beta>0 actually changes the value (non-no-op) -----
    nb0 = nll_loss(y, mu, ls, beta=0.0).item()
    nb5 = nll_loss(y, mu, ls, beta=0.5).item()
    print("beta sensitivity: beta=0 -> %.6f , beta=0.5 -> %.6f" % (nb0, nb5))
    if abs(nb0 - nb5) < 1e-9:
        failures.append("beta=0.5 did not change the loss value (branch is a no-op)")
    # hand-check: beta-weighted value equals mean(per * sigma2.detach()**0.5)
    sigma2 = ls.exp().clamp_min(1e-6)
    per = ((y - mu) ** 2) / (2.0 * sigma2) + 0.5 * ls
    manual = (per * (sigma2.detach() ** 0.5)).mean().item()
    if abs(manual - nb5) > 1e-6:
        failures.append("beta=0.5 value %.6f != manual reference %.6f" % (nb5, manual))

    # ----- (2) end-to-end training, loss decreases for both betas -----
    h, yy, mu_true = _make_hetero(seed=0)
    V = h.shape[1]
    d = h.shape[2]
    for beta in (0.0, 0.5):
        torch.manual_seed(7)  # same init for a fair comparison
        head = AleatoricHead(hidden_dim=d, num_sensors=V,
                             sensor_embed_dim=16, mlp_hidden=64)
        nll_before = _eval_nll(head, h, yy, mu_true, beta=beta)
        train_aleatoric(head, h, yy, mu_true, device,
                        epochs=15, batch_size=128, lr=1e-2, beta=beta)
        nll_after = _eval_nll(head, h, yy, mu_true, beta=beta)
        print("beta=%.1f : NLL before=%.4f  after=%.4f  (delta=%.4f)"
              % (beta, nll_before, nll_after, nll_after - nll_before))
        if not (nll_after < nll_before):
            failures.append("beta=%.1f training did not decrease NLL "
                            "(%.4f -> %.4f)" % (beta, nll_before, nll_after))
        with torch.no_grad():
            out = head(h)
        if not torch.all(torch.isfinite(out)):
            failures.append("beta=%.1f head produced non-finite output" % beta)

    print("")
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    print("RESULT: PASS  (beta=0 byte-identical to old NLL; both betas train & "
          "decrease loss; beta=0.5 is non-trivial)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
