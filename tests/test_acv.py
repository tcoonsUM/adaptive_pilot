from __future__ import annotations

import numpy as np

from adaptive_covariance.estimators.independent_acv import IndependentSampleACV


def test_independent_acv_allocation_and_variance_formula() -> None:
    covariance = np.array(
        [
            [1.0, 0.92, 0.78],
            [0.92, 1.1, 0.75],
            [0.78, 0.75, 0.9],
        ]
    )
    costs = np.array([1.0, 0.08, 0.01])
    estimator = IndependentSampleACV(minimum_base=2, minimum_extra=1)
    design = estimator.design(covariance, costs, budget=30.0)
    assert design.allocation.total_cost <= 30.0 + 1.0e-10
    assert design.variance < covariance[0, 0] / 30.0

    rng = np.random.default_rng(10)
    n_replications = 3000
    base = rng.multivariate_normal(
        np.zeros(3), covariance, size=(n_replications, design.allocation.n_base)
    )
    estimates = np.mean(base[:, :, 0], axis=1)
    base_low_means = np.mean(base[:, :, 1:], axis=1)
    extra_means = np.empty_like(base_low_means)
    for low_index, n_extra in enumerate(design.allocation.n_extra):
        low_variance = covariance[low_index + 1, low_index + 1]
        extra = rng.normal(
            scale=np.sqrt(low_variance),
            size=(n_replications, int(n_extra)),
        )
        extra_means[:, low_index] = np.mean(extra, axis=1)
    estimates += (extra_means - base_low_means) @ design.weights

    assert abs(float(np.mean(estimates))) < 0.04
    np.testing.assert_allclose(
        np.var(estimates, ddof=1),
        design.variance,
        rtol=0.12,
        atol=0.0,
    )
