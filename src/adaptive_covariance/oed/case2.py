"""Reusable utility-model ensemble for OED Case 2 (measurement-time design).

The random input to each utility model is the concatenation of the five physical
parameters and one additive observation-noise draw.  Inner-loop prior samples are
fixed for the life of the object, which mirrors the research scripts and avoids
injecting an additional source of randomness into covariance learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from adaptive_covariance.oed.eig import pointwise_information_utilities
from adaptive_covariance.oed.ensemble import OEDSurrogateEnsemble
from adaptive_covariance.oed.physics import OEDPrior, sample_oed_prior
from adaptive_covariance.types import ModelCallable

FloatArray = NDArray[np.float64]
CostScaling = Literal["source_normalized", "forward", "relative"]


@dataclass(frozen=True)
class Case2UtilitySettings:
    """Settings for the three cost-aware EIG utility models."""

    sensor_location: tuple[float, float] = (-0.8, -0.2)
    inner_sample_sizes: tuple[int, int, int] = (1000, 500, 100)
    inner_seed: int = 41
    noise_standard_deviation: float = 0.1
    cost_offset: float = 0.1
    fno_batch_size: int = 32
    outer_chunk_size: int = 256
    cost_scaling: CostScaling = "source_normalized"

    def __post_init__(self) -> None:
        if len(self.inner_sample_sizes) != 3:
            raise ValueError("Exactly three inner-loop sample sizes are required")
        if any(value < 1 for value in self.inner_sample_sizes):
            raise ValueError("Inner-loop sample sizes must be positive")
        if self.noise_standard_deviation <= 0.0:
            raise ValueError("noise_standard_deviation must be positive")
        if self.cost_offset <= 0.0:
            raise ValueError("cost_offset must be positive")
        if self.fno_batch_size < 1 or self.outer_chunk_size < 1:
            raise ValueError("Batch and chunk sizes must be positive")


class Case2UtilityEnsemble:
    """Cost-aware NMC utility models backed by the supplied MLP/FNO ensemble.

    The three inner sample sets use prefixes of one common prior sample.  This
    preserves the sample sharing used in the paper while allowing different inner
    sizes for the three fidelities.
    """

    def __init__(
        self,
        surrogate_ensemble: OEDSurrogateEnsemble,
        settings: Case2UtilitySettings | None = None,
        *,
        prior: OEDPrior | None = None,
    ) -> None:
        self.surrogates = surrogate_ensemble
        self.settings = settings or Case2UtilitySettings()
        self.prior = prior or OEDPrior()
        rng = np.random.default_rng(self.settings.inner_seed)
        maximum_inner = max(self.settings.inner_sample_sizes)
        common_inner = sample_oed_prior(maximum_inner, rng, self.prior)
        self.inner_parameters = tuple(
            common_inner[:count].copy() for count in self.settings.inner_sample_sizes
        )
        self._inner_prediction_cache: dict[tuple[int, float], FloatArray] = {}
        self.models: list[ModelCallable] = [
            partial(self.evaluate, fidelity=fidelity) for fidelity in range(3)
        ]
        self.costs = self._make_costs()

    def _make_costs(self) -> FloatArray:
        forward_costs = np.asarray(self.surrogates.costs, dtype=float)
        sizes = np.asarray(self.settings.inner_sample_sizes, dtype=float)
        if self.settings.cost_scaling == "forward":
            return forward_costs.copy()
        if self.settings.cost_scaling == "relative":
            values = forward_costs * sizes
            return values / values[0]
        if self.settings.cost_scaling == "source_normalized":
            # This is the cost convention used in the supplied Case 2 source:
            # w_m = runtime_m * N_in,m / N_in,0.
            return forward_costs * (sizes / sizes[0])
        raise ValueError(f"Unknown cost scaling: {self.settings.cost_scaling}")

    def sample_random_inputs(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Draw physical parameters and one additive-noise realization per row."""
        theta = sample_oed_prior(n_samples, rng, self.prior)
        epsilon = rng.normal(
            0.0,
            self.settings.noise_standard_deviation,
            size=(n_samples, 1),
        )
        return np.column_stack((theta, epsilon))

    @staticmethod
    def _time_from_design(design: ArrayLike) -> float:
        values = np.asarray(design, dtype=float).reshape(-1)
        if values.size != 1:
            raise ValueError("Case 2 has one design variable: measurement time")
        time = float(values[0])
        if not 0.0 < time <= 0.6 + 1.0e-12:
            raise ValueError(f"Measurement time {time:g} lies outside the surrogate horizon")
        return time

    def _predict(
        self,
        fidelity: int,
        parameters: FloatArray,
        time: float,
    ) -> FloatArray:
        return self.surrogates.predict_at_sensor(
            fidelity,  # type: ignore[arg-type]
            parameters,
            self.settings.sensor_location,
            time,
            batch_size=self.settings.fno_batch_size,
        )

    def _inner_predictions(self, fidelity: int, time: float) -> FloatArray:
        key = (int(fidelity), round(float(time), 12))
        cached = self._inner_prediction_cache.get(key)
        if cached is None:
            cached = self._predict(fidelity, self.inner_parameters[fidelity], time)
            self._inner_prediction_cache[key] = np.asarray(cached, dtype=float)
        return cached

    def clear_cache(self) -> None:
        """Discard cached inner forward predictions."""
        self._inner_prediction_cache.clear()

    def evaluate(
        self,
        design: FloatArray,
        random_inputs: FloatArray,
        *,
        fidelity: int,
    ) -> FloatArray:
        """Return one cost-aware pointwise utility for each random-input row."""
        if fidelity not in (0, 1, 2):
            raise ValueError("fidelity must be 0, 1, or 2")
        inputs = np.asarray(random_inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != 6:
            raise ValueError(
                f"Expected random_inputs shape (n, 6) = [theta, epsilon], got {inputs.shape}"
            )
        time = self._time_from_design(design)
        theta_outer = inputs[:, :5]
        epsilon = inputs[:, 5:6]
        g_outer = self._predict(fidelity, theta_outer, time)
        g_inner = self._inner_predictions(fidelity, time)
        utilities = pointwise_information_utilities(
            g_outer,
            g_inner,
            epsilon,
            noise_standard_deviation=self.settings.noise_standard_deviation,
            outer_chunk_size=self.settings.outer_chunk_size,
        )
        return utilities / (self.settings.cost_offset + time)

    def reference_mean(
        self,
        design: ArrayLike,
        *,
        n_outer: int,
        seed: int,
        fidelity: int = 0,
    ) -> tuple[float, float]:
        """Estimate a utility mean and the variance of its sample mean."""
        rng = np.random.default_rng(seed)
        random_inputs = self.sample_random_inputs(n_outer, rng)
        values = self.evaluate(
            np.asarray(design, dtype=float).reshape(-1),
            random_inputs,
            fidelity=fidelity,
        )
        variance = float(np.var(values, ddof=1) / n_outer) if n_outer > 1 else 0.0
        return float(np.mean(values)), variance
