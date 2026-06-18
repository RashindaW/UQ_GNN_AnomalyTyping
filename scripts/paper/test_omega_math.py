"""CPU dry-test for the Omega OOD math in build_omega.py (no GPU, no model).

Validates the estimator math on SYNTHETIC data:
  TRAIN = an in-distribution Gaussian blob (per-node).
  TEST  = half in-distribution (same blob) + half an OUT-OF-DISTRIBUTION shifted
          cluster. The OOD half is labelled anomalous.
A correct Mahalanobis / kNN OOD estimator must assign HIGHER Omega to the OOD
half -> AUROC > 0.9. This proves the math without needing the GPU G-DeltaUQ model.

Run: $PY scripts/paper/test_omega_math.py
Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Make the repo root importable when run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.paper.build_omega import (  # noqa: E402
    fit_per_node_gaussian,
    score_mahalanobis_arrays,
    knn_omega,
    compute_omega_from_hbar,
    safe_auroc,
)


def _make_synthetic(seed=0, T_tr=2000, T_in=500, T_ood=500, V=5, d=8, shift=4.0):
    """Per-node Gaussian blobs.

    Returns train_hbar (T_tr,V,d), test_hbar (T_in+T_ood,V,d), labels (T,).
    The in-distribution test points are drawn from the SAME distribution as
    train; the OOD points are the same covariance with the mean shifted by
    `shift` standard deviations IN THE WHITENED BASIS of each node. Shifting in
    whitened space guarantees the OOD mean is ~`shift`*sqrt(d) Mahalanobis units
    away regardless of the (random) per-node covariance, so the estimator is
    exercised non-trivially while separation is robust by construction.
    """
    rng = np.random.default_rng(seed)
    # Give each node a distinct (random but fixed) mean + a random covariance so
    # the per-node fit is exercised non-trivially.
    node_mean = rng.normal(0.0, 2.0, size=(V, d))
    cov_factors = rng.normal(0.0, 1.0, size=(V, d, d)) * 0.5
    # cov_v = A A^T + I  (SPD); chol @ z maps unit-Gaussian z into this covariance.
    cov = np.einsum("vij,vkj->vik", cov_factors, cov_factors) + np.eye(d)[None]
    chol = np.linalg.cholesky(cov)  # (V,d,d)

    def sample(n, mean_vd):
        z = rng.normal(0.0, 1.0, size=(n, V, d))
        x = np.einsum("vij,nvj->nvi", chol, z) + mean_vd[None]
        return x.astype(np.float32)

    train = sample(T_tr, node_mean)
    test_in = sample(T_in, node_mean)
    # OOD mean: shift by `shift` in EVERY whitened dimension, then map through
    # chol so the shift respects the data manifold's covariance. The whitened
    # offset has norm shift*sqrt(d), i.e. that many Mahalanobis units.
    whitened_offset = np.full((V, d), shift, dtype=np.float64)
    ood_mean = node_mean + np.einsum("vij,vj->vi", chol, whitened_offset)
    test_ood = sample(T_ood, ood_mean)

    test = np.concatenate([test_in, test_ood], axis=0)
    labels = np.concatenate([np.zeros(T_in), np.ones(T_ood)]).astype(int)
    # shuffle so order can't leak into the metric
    perm = rng.permutation(test.shape[0])
    return train, test[perm], labels[perm]


def main():
    train, test, labels = _make_synthetic()
    print("synthetic shapes: train=%s test=%s pos=%d/%d"
          % (train.shape, test.shape, int(labels.sum()), len(labels)))

    failures = []

    # --- Mahalanobis (per-node fit/score) ---
    mean_v, inv_cov_v = fit_per_node_gaussian(train, eps_reg=1e-3)
    assert mean_v.shape == (train.shape[1], train.shape[2]), "mean_v shape"
    assert inv_cov_v.shape == (train.shape[1], train.shape[2], train.shape[2]), "inv_cov_v shape"

    omega_pernode = score_mahalanobis_arrays(test, mean_v, inv_cov_v)  # (T,V)
    assert omega_pernode.shape == (test.shape[0], test.shape[1]), "omega_pernode shape"
    assert np.all(omega_pernode >= 0.0), "Mahalanobis distance must be non-negative"

    omega_max = omega_pernode.max(axis=-1)
    omega_mean = omega_pernode.mean(axis=-1)

    auroc_max = safe_auroc(omega_max, labels)
    auroc_mean = safe_auroc(omega_mean, labels)
    print("maha omega_max  AUROC = %.4f" % auroc_max)
    print("maha omega_mean AUROC = %.4f" % auroc_mean)

    # OOD half must score higher than in-dist half (sanity on the means).
    mu_ood = omega_max[labels == 1].mean()
    mu_in = omega_max[labels == 0].mean()
    print("maha omega_max mean: OOD=%.3f  IN=%.3f" % (mu_ood, mu_in))
    if not (mu_ood > mu_in):
        failures.append("Mahalanobis omega_max not higher on OOD cluster")
    if not (auroc_max > 0.9):
        failures.append("Mahalanobis omega_max AUROC %.4f <= 0.9" % auroc_max)
    if not (auroc_mean > 0.9):
        failures.append("Mahalanobis omega_mean AUROC %.4f <= 0.9" % auroc_mean)

    # --- kNN cross-check ---
    omega_knn = knn_omega(train, test, k=10, max_train=5000, seed=1)
    assert omega_knn.shape == (test.shape[0],), "omega_knn shape"
    auroc_knn = safe_auroc(omega_knn, labels)
    print("knn  omega_knn  AUROC = %.4f" % auroc_knn)
    if not (auroc_knn > 0.9):
        failures.append("kNN omega_knn AUROC %.4f <= 0.9" % auroc_knn)

    # --- orchestrator returns all keys with correct shapes ---
    bundle = compute_omega_from_hbar(train, test, eps_reg=1e-3, k=10, max_train=5000, seed=2)
    for key, shp in [
        ("omega_pernode", (test.shape[0], test.shape[1])),
        ("omega_max", (test.shape[0],)),
        ("omega_mean", (test.shape[0],)),
        ("omega_knn", (test.shape[0],)),
    ]:
        if bundle[key].shape != shp:
            failures.append("compute_omega_from_hbar[%s] shape %s != %s"
                            % (key, bundle[key].shape, shp))

    # --- subsample path is actually exercised (train > max_train) ---
    big_train = np.concatenate([train, train, train], axis=0)  # 6000 > 5000
    omega_knn_sub = knn_omega(big_train, test[:50], k=10, max_train=5000, seed=3)
    if omega_knn_sub.shape != (50,):
        failures.append("kNN subsample path shape %s != (50,)" % (omega_knn_sub.shape,))

    print("")
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  - " + f)
        return 1
    print("RESULT: PASS  (maha_max=%.4f maha_mean=%.4f knn=%.4f all > 0.9)"
          % (auroc_max, auroc_mean, auroc_knn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
