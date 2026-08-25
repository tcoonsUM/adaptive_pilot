"""Single-fidelity Monte Carlo utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from adaptive_covariance.types import ModelCallable, RandomInputSampler

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MonteCarloEstimate:
    value: float
    variance: float
    n_samples: int
    cost: float


def estimate_monte_carlo(
    *,
    model: ModelCallable,
    design: FloatArray,
    sample_random_inputs: RandomInputSampler,
    budget: float,
    model_cost: float,
    rng: np.random.Generator,
    minimum_samples: int = 2,
) -> MonteCarloEstimate:
    """Estimate a model mean using the largest integer sample count within budget."""
    if model_cost <= 0.0:
        raise ValueError("model_cost must be positive")
    n_samples = int(np.floor(budget / model_cost))
    if n_samples < minimum_samples:
        raise ValueError(
            f"Budget {budget:g} affords only {n_samples} samples at cost {model_cost:g}"
        )
    random_inputs = sample_random_inputs(n_samples, rng)
    values = np.asarray(model(np.asarray(design, dtype=float), random_inputs), dtype=float).reshape(-1)
    if values.size != n_samples:
        raise ValueError("Model output length does not match allocated Monte Carlo samples")
    sample_variance = float(np.var(values, ddof=1))
    return MonteCarloEstimate(
        value=float(np.mean(values)),
        variance=sample_variance / n_samples,
        n_samples=n_samples,
        cost=n_samples * model_cost,
    )
