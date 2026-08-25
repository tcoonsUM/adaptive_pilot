"""Finite-pilot covariance observations and transformed observation noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import wishart

from adaptive_covariance.covariance.transforms import (
    covariance_to_unconstrained,
    nearest_positive_definite,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CovarianceObservation:
    """One local covariance observation constructed from common model samples."""

    covariance: FloatArray
    gamma: FloatArray
    log_standard_deviations: FloatArray
    gamma_noise_variances: FloatArray
    log_standard_deviation_noise_variances: FloatArray
    model_means: FloatArray
    model_mean_variances: FloatArray
    n_samples: int


def sample_covariance(outputs: ArrayLike) -> FloatArray:
    """Return an unbiased model-output sample covariance matrix.

    Parameters
    ----------
    outputs:
        Matrix of shape ``(n_samples, n_models)``.
    """
    values = np.asarray(outputs, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected outputs with shape (n_samples, n_models), got {values.shape}")
    if values.shape[0] < 2:
        raise ValueError("At least two pilot samples are required for covariance estimation")
    if not np.all(np.isfinite(values)):
        raise ValueError("Pilot outputs contain NaN or infinite values")
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    return 0.5 * (covariance + covariance.T)


def wishart_pushforward_noise(
    covariance: ArrayLike,
    n_samples: int,
    *,
    n_draws: int = 1000,
    rng: np.random.Generator | None = None,
    minimum_variance: float = 1.0e-12,
    jitter_relative: float = 1.0e-10,
) -> tuple[FloatArray, FloatArray]:
    """Estimate transformed-observation variances by Wishart pushforward.

    The plug-in approximation is

    ``Sigma_hat_draw ~ Wishart(Sigma_hat, n_samples - 1) / (n_samples - 1)``.

    Draws are propagated through the gamma and log-standard-deviation transforms.
    The returned values are component-wise sample variances, not standard deviations.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if n_draws < 2:
        raise ValueError("n_draws must be at least two")
    if minimum_variance <= 0.0:
        raise ValueError("minimum_variance must be positive")
    generator = np.random.default_rng() if rng is None else rng

    sigma = np.asarray(covariance, dtype=float)
    sigma = 0.5 * (sigma + sigma.T)
    scale = max(float(np.trace(sigma) / max(sigma.shape[0], 1)), 1.0)
    sigma = nearest_positive_definite(
        sigma + jitter_relative * scale * np.eye(sigma.shape[0])
    )
    degrees_of_freedom = n_samples - 1
    if degrees_of_freedom < sigma.shape[0]:
        raise ValueError(
            "Wishart pushforward requires n_samples - 1 >= n_models so the "
            "sample covariance has full rank"
        )

    # scipy accepts a Generator through random_state.
    draws = wishart.rvs(
        df=degrees_of_freedom,
        scale=sigma,
        size=n_draws,
        random_state=generator,
    )
    draws = np.asarray(draws, dtype=float) / degrees_of_freedom
    if draws.ndim == 2:
        draws = draws[None, :, :]

    gamma_draws: list[FloatArray] = []
    log_sd_draws: list[FloatArray] = []
    for draw in draws:
        transformed = covariance_to_unconstrained(draw)
        gamma_draws.append(transformed.gamma)
        log_sd_draws.append(transformed.log_standard_deviations)

    gamma_array = np.stack(gamma_draws, axis=0)
    log_sd_array = np.stack(log_sd_draws, axis=0)
    gamma_variance = np.var(gamma_array, axis=0, ddof=1)
    log_sd_variance = np.var(log_sd_array, axis=0, ddof=1)
    return (
        np.maximum(gamma_variance, minimum_variance),
        np.maximum(log_sd_variance, minimum_variance),
    )


def observation_from_outputs(
    outputs: ArrayLike,
    *,
    wishart_draws: int = 1000,
    rng: np.random.Generator | None = None,
    minimum_variance: float = 1.0e-12,
) -> CovarianceObservation:
    """Build a transformed covariance observation from a pilot-output matrix."""
    values = np.asarray(outputs, dtype=float)
    covariance = sample_covariance(values)
    transformed = covariance_to_unconstrained(covariance)
    gamma_noise, log_sd_noise = wishart_pushforward_noise(
        covariance,
        values.shape[0],
        n_draws=wishart_draws,
        rng=rng,
        minimum_variance=minimum_variance,
    )
    model_variances = np.diag(covariance)
    return CovarianceObservation(
        covariance=covariance,
        gamma=transformed.gamma,
        log_standard_deviations=transformed.log_standard_deviations,
        gamma_noise_variances=gamma_noise,
        log_standard_deviation_noise_variances=log_sd_noise,
        model_means=np.mean(values, axis=0),
        model_mean_variances=model_variances / values.shape[0],
        n_samples=values.shape[0],
    )
