"""Nested-Monte-Carlo expected-information-gain utilities."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]
ForwardModel = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True)
class NestedEIGEstimate:
    value: float
    variance: float
    utilities: FloatArray


def sample_observation_noise(
    n_samples: int,
    rng: np.random.Generator,
    *,
    standard_deviation: float = 0.1,
    n_sensors: int = 1,
) -> FloatArray:
    """Draw additive Gaussian observation noise."""
    return rng.normal(0.0, standard_deviation, size=(n_samples, n_sensors))


def _as_sensor_matrix(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional, got shape {array.shape}")
    return array


def _normal_log_density(residuals: FloatArray, standard_deviation: float) -> FloatArray:
    if standard_deviation <= 0.0:
        raise ValueError("Observation-noise standard deviation must be positive")
    per_component = (
        -0.5 * (residuals / standard_deviation) ** 2
        - np.log(standard_deviation)
        - 0.5 * np.log(2.0 * np.pi)
    )
    return np.sum(per_component, axis=-1)


def pointwise_information_utilities(
    g_outer: ArrayLike,
    g_inner: ArrayLike,
    observation_noise: ArrayLike,
    *,
    noise_standard_deviation: float = 0.1,
    outer_chunk_size: int = 256,
) -> FloatArray:
    """Compute pointwise log likelihood-to-evidence utilities.

    The evidence uses a shared inner prior sample set. Computation is chunked over
    outer samples to avoid materializing the full array for very large nested runs.
    """
    outer = _as_sensor_matrix(g_outer, "g_outer")
    inner = _as_sensor_matrix(g_inner, "g_inner")
    epsilon = _as_sensor_matrix(observation_noise, "observation_noise")
    if outer.shape != epsilon.shape:
        raise ValueError("g_outer and observation_noise must have equal shape")
    if inner.shape[1] != outer.shape[1]:
        raise ValueError("Inner and outer predictions must have equal sensor dimension")
    if outer_chunk_size < 1:
        raise ValueError("outer_chunk_size must be positive")

    log_likelihood = _normal_log_density(epsilon, noise_standard_deviation)
    utilities = np.empty(outer.shape[0], dtype=float)
    log_n_inner = np.log(inner.shape[0])
    for start in range(0, outer.shape[0], outer_chunk_size):
        stop = min(start + outer_chunk_size, outer.shape[0])
        observations = outer[start:stop] + epsilon[start:stop]
        residuals = observations[:, None, :] - inner[None, :, :]
        inner_log_likelihood = _normal_log_density(
            residuals, noise_standard_deviation
        )
        log_evidence = logsumexp(inner_log_likelihood, axis=1) - log_n_inner
        utilities[start:stop] = log_likelihood[start:stop] - log_evidence
    return utilities


def estimate_nested_eig(
    g_outer: ArrayLike,
    g_inner: ArrayLike,
    observation_noise: ArrayLike,
    *,
    noise_standard_deviation: float = 0.1,
    outer_chunk_size: int = 256,
) -> NestedEIGEstimate:
    """Return a nested-Monte-Carlo EIG estimate and variance of its sample mean."""
    utilities = pointwise_information_utilities(
        g_outer,
        g_inner,
        observation_noise,
        noise_standard_deviation=noise_standard_deviation,
        outer_chunk_size=outer_chunk_size,
    )
    variance = float(np.var(utilities, ddof=1) / utilities.size) if utilities.size > 1 else 0.0
    return NestedEIGEstimate(
        value=float(np.mean(utilities)),
        variance=variance,
        utilities=utilities,
    )


def evaluate_utility_model(
    *,
    forward_model: Callable[[FloatArray, FloatArray], FloatArray],
    design: FloatArray,
    theta_outer: FloatArray,
    observation_noise: FloatArray,
    theta_inner: FloatArray,
    noise_standard_deviation: float = 0.1,
    cost: float = 1.0,
    outer_chunk_size: int = 256,
) -> FloatArray:
    """Evaluate pointwise utilities for one forward-model fidelity."""
    if cost <= 0.0:
        raise ValueError("Experimental cost must be positive")
    g_outer = forward_model(design, theta_outer)
    g_inner = forward_model(design, theta_inner)
    return pointwise_information_utilities(
        g_outer,
        g_inner,
        observation_noise,
        noise_standard_deviation=noise_standard_deviation,
        outer_chunk_size=outer_chunk_size,
    ) / cost
