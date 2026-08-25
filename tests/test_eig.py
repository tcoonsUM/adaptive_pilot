from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from adaptive_covariance.oed.eig import pointwise_information_utilities


def test_pointwise_information_utility_matches_manual_calculation() -> None:
    outer = np.array([0.2, -0.1])
    inner = np.array([-0.3, 0.0, 0.4])
    epsilon = np.array([[0.05], [-0.02]])
    sigma = 0.1
    utilities = pointwise_information_utilities(
        outer,
        inner,
        epsilon,
        noise_standard_deviation=sigma,
        outer_chunk_size=1,
    )

    expected = []
    constant = -np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
    for g_outer, error in zip(outer, epsilon[:, 0], strict=True):
        log_numerator = -0.5 * (error / sigma) ** 2 + constant
        observation = g_outer + error
        logs = -0.5 * ((observation - inner) / sigma) ** 2 + constant
        expected.append(log_numerator - (logsumexp(logs) - np.log(inner.size)))
    np.testing.assert_allclose(utilities, expected)
