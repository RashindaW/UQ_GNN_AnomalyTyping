"""Uncertainty decomposition: Variant A (variance-based) and Variant B (entropy/MI).

Both variants consume the per-member arrays produced by `inference.run_inference`.

Variant A — closed-form, cheap, scale-dependent. Default for thresholded detection.
Variant B — Monte-Carlo, scale-invariant, used additionally for triage.
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------
# Variant A — variance-based (Step 3 of the outline)
# ----------------------------------------------------------------------


def variance_decomposition(
    mu_per_member: np.ndarray,        # (M, T, V)
    logvar_per_member: np.ndarray,    # (M, T, V)
) -> dict[str, np.ndarray]:
    """Return mu_bar, sigma2_aleatoric, sigma2_epistemic, sigma2_total as (T, V) arrays."""
    sigma2_per_member = np.exp(logvar_per_member)
    mu_bar = mu_per_member.mean(axis=0)
    sigma2_a = sigma2_per_member.mean(axis=0)
    sigma2_e = mu_per_member.var(axis=0)            # ddof=0 (matches Lakshminarayanan)
    sigma2_tot = sigma2_a + sigma2_e
    return {
        'mu_bar': mu_bar.astype(np.float32),
        'sigma2_aleatoric': sigma2_a.astype(np.float32),
        'sigma2_epistemic': sigma2_e.astype(np.float32),
        'sigma2_total': sigma2_tot.astype(np.float32),
    }


# ----------------------------------------------------------------------
# Variant B — entropy/MI (Step 4 of the outline)
# ----------------------------------------------------------------------


def _gaussian_log_pdf(y: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """log N(y | mu, sigma2). Broadcasts on the leading axes; returns same shape."""
    return -0.5 * (np.log(2 * np.pi * sigma2) + (y - mu) ** 2 / sigma2)


def information_decomposition(
    mu_per_member: np.ndarray,        # (M, T, V)
    logvar_per_member: np.ndarray,    # (M, T, V)
    n_samples: int = 100,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Step 4 — entropy-based aleatoric / total / MI per (t, v).

    Aleatoric (closed form, average per-member differential entropy):
        H_a_bar_v(t) = (1/M) * Σ_m (1/2) * log(2*pi*e * sigma2_{v,m}(t))

    Total (Monte-Carlo, on the M-component Gaussian mixture):
        H_tot_v(t) ≈ -(1/N_s) * Σ_i log p_bar(y_i | t, v),
            where y_i ~ p_bar = (1/M) * Σ_m N(mu_m, sigma2_m)

    MI = max(H_tot - H_a_bar, 0). The clamp absorbs MC estimation noise.

    Sampling cost: N_s = 100 samples per (t, v); on SWaT with T = 90k, V = 51, this
    is ~459M samples and a few hundred MB of intermediate, but vectorised.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    M, T, V = mu_per_member.shape
    sigma2_per_member = np.exp(logvar_per_member)
    sigma_per_member = np.sqrt(sigma2_per_member)

    # Aleatoric (closed-form average per-member differential entropy).
    half_log_2pi_e = 0.5 * (math.log(2 * math.pi) + 1.0)
    H_a_per_member = half_log_2pi_e + 0.5 * logvar_per_member  # (M, T, V)
    H_a_bar = H_a_per_member.mean(axis=0)                       # (T, V)

    # Monte-Carlo total entropy, computed in chunks over T to bound memory.
    H_tot = np.empty((T, V), dtype=np.float32)
    chunk = max(1, min(T, 4096))
    for start in range(0, T, chunk):
        end = min(T, start + chunk)
        slice_size = end - start

        # Choose a random component per sample, per (slice_t, v).
        comp_idx = rng.integers(0, M, size=(n_samples, slice_size, V))    # (N_s, t, v)

        # Gather the chosen component's mu/sigma via advanced indexing.
        t_idx = np.broadcast_to(
            np.arange(slice_size)[None, :, None], (n_samples, slice_size, V)
        )
        v_idx = np.broadcast_to(
            np.arange(V)[None, None, :], (n_samples, slice_size, V)
        )
        mu_chosen = mu_per_member[:, start:end, :][comp_idx, t_idx, v_idx]      # (N_s, t, v)
        sigma_chosen = sigma_per_member[:, start:end, :][comp_idx, t_idx, v_idx]  # (N_s, t, v)

        # Sample.
        eps = rng.standard_normal(size=(n_samples, slice_size, V)).astype(np.float32)
        y_samples = mu_chosen + sigma_chosen * eps                                # (N_s, t, v)

        # Evaluate the mixture density at y_samples.
        # log_p_per_comp[m, n_s, t, v] = log N(y_samples[n_s, t, v] | mu_m, sigma2_m)
        mu_slice = mu_per_member[:, start:end, :]            # (M, t, v)
        sigma2_slice = sigma2_per_member[:, start:end, :]    # (M, t, v)
        # Broadcast: (M, 1, t, v) vs (1, N_s, t, v) -> (M, N_s, t, v)
        log_p_per_comp = _gaussian_log_pdf(
            y_samples[None, ...],
            mu_slice[:, None, :, :],
            sigma2_slice[:, None, :, :],
        )
        # log p_bar = log( (1/M) * Σ_m exp(log_p_per_comp) ) = logsumexp - log M
        max_lp = log_p_per_comp.max(axis=0)
        log_pbar = max_lp + np.log(np.exp(log_p_per_comp - max_lp[None]).mean(axis=0))
        # H_tot ≈ -(1/N_s) * Σ_i log_pbar
        H_tot[start:end, :] = (-log_pbar.mean(axis=0)).astype(np.float32)

    H_a_bar = H_a_bar.astype(np.float32)
    MI = np.maximum(H_tot - H_a_bar, 0.0)

    return {
        'H_a_bar': H_a_bar,
        'H_tot': H_tot,
        'MI': MI,
    }
