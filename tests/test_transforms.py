from __future__ import annotations

import numpy as np

from adaptive_covariance.covariance.transforms import (
    batch_covariance_to_unconstrained,
    batch_unconstrained_to_covariance,
    covariance_to_unconstrained,
    gamma_forward,
    gamma_inverse,
    is_positive_definite,
    unconstrained_to_covariance,
)


def _random_covariance(seed: int, dimension: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension))
    return matrix @ matrix.T + 0.5 * np.eye(dimension)


def test_gamma_round_trip() -> None:
    covariance = _random_covariance(2, 4)
    standard_deviations = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard_deviations, standard_deviations)
    gamma = gamma_forward(correlation)
    reconstructed = gamma_inverse(gamma)
    np.testing.assert_allclose(reconstructed, correlation, rtol=1.0e-8, atol=1.0e-9)
    assert is_positive_definite(reconstructed)


def test_covariance_round_trip_and_batch() -> None:
    covariances = np.stack([_random_covariance(seed, 3) for seed in range(3)])
    gamma, log_sd = batch_covariance_to_unconstrained(covariances)
    reconstructed = batch_unconstrained_to_covariance(gamma, log_sd)
    np.testing.assert_allclose(reconstructed, covariances, rtol=1.0e-8, atol=1.0e-9)
    assert all(is_positive_definite(matrix) for matrix in reconstructed)

    transformed = covariance_to_unconstrained(covariances[0])
    single = unconstrained_to_covariance(
        transformed.gamma, transformed.log_standard_deviations
    )
    np.testing.assert_allclose(single, covariances[0], rtol=1.0e-8, atol=1.0e-9)
