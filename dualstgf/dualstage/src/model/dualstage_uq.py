"""DualSTAGE_UQ — DualSTAGE with `with_variance_head=True` by default.

Heteroscedastic deep-ensemble variant. Reuses the upstream `DualSTAGE` class with
the `with_variance_head` flag flipped on, which adds an independent
`out_logvar` projection inside the reconstruction decoder and makes
`forward()` return `(mu, log_var)`.

The actual two-head architectural change lives in `dualstage.py`:
  - `ReconstructionModel.__init__`: optionally constructs `self.out_logvar`.
  - `ReconstructionModel.reconstruct`: returns `(recon, log_var)` tuple when
    the variance head is active.
  - `DualSTAGE.__init__`: accepts `with_variance_head` and `logvar_clamp`.
  - `DualSTAGE.forward`: clamps log_var to `logvar_clamp` and returns the tuple.

Why an alias rather than a subclass: the surface change is just one default flag;
subclassing would add no functionality and obscure the inheritance.
"""
from __future__ import annotations

from .dualstage import DualSTAGE


class DualSTAGE_UQ(DualSTAGE):
    """DualSTAGE with the heteroscedastic two-head decoder enabled by default."""

    def __init__(self, *args, with_variance_head: bool = True,
                 logvar_clamp: tuple = (-10.0, 10.0), **kwargs):
        super().__init__(
            *args,
            with_variance_head=with_variance_head,
            logvar_clamp=logvar_clamp,
            **kwargs,
        )
