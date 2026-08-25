"""All-at-once utility-field helpers for OED Case 1 (sensor location)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from adaptive_covariance.oed.eig import pointwise_information_utilities
from adaptive_covariance.oed.ensemble import OEDSurrogateEnsemble
from adaptive_covariance.oed.physics import OEDPrior, sample_oed_prior

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Case1UtilitySettings:
    """Settings for fixed-time, spatial sensor-placement utilities."""

    measurement_time: float = 0.325
    inner_sample_sizes: tuple[int, int, int] = (1000, 1000, 1000)
    inner_seed: int = 41
    noise_standard_deviation: float = 0.1
    fno_batch_size: int = 32
    outer_chunk_size: int = 256

    def __post_init__(self) -> None:
        if len(self.inner_sample_sizes) != 3:
            raise ValueError("Exactly three inner-loop sample sizes are required")
        if any(value < 1 for value in self.inner_sample_sizes):
            raise ValueError("Inner-loop sample sizes must be positive")
        if not 0.0 < self.measurement_time <= 0.6:
            raise ValueError("measurement_time must lie in (0, 0.6]")
        if self.noise_standard_deviation <= 0.0:
            raise ValueError("noise_standard_deviation must be positive")


class Case1UtilityFieldEvaluator:
    """Evaluate one utility sample over every location on the 21-by-21 grid."""

    def __init__(
        self,
        surrogate_ensemble: OEDSurrogateEnsemble,
        settings: Case1UtilitySettings | None = None,
        *,
        prior: OEDPrior | None = None,
    ) -> None:
        self.surrogates = surrogate_ensemble
        self.settings = settings or Case1UtilitySettings()
        self.prior = prior or OEDPrior()
        rng = np.random.default_rng(self.settings.inner_seed)
        maximum_inner = max(self.settings.inner_sample_sizes)
        shared = sample_oed_prior(maximum_inner, rng, self.prior)
        self.inner_parameters = tuple(
            shared[:count].copy() for count in self.settings.inner_sample_sizes
        )
        self._inner_fields: dict[int, FloatArray] = {}
        self.grid_1d = self.surrogates.grid_1d.copy()
        self.sensor_locations = np.column_stack(
            (
                self.surrogates.x_grid.reshape(-1),
                self.surrogates.y_grid.reshape(-1),
            )
        )

    def sample_random_inputs(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> FloatArray:
        theta = sample_oed_prior(n_samples, rng, self.prior)
        epsilon = rng.normal(
            0.0,
            self.settings.noise_standard_deviation,
            size=(n_samples, 1),
        )
        return np.column_stack((theta, epsilon))

    def _get_inner_fields(self, fidelity: int) -> FloatArray:
        cached = self._inner_fields.get(fidelity)
        if cached is None:
            cached = self.surrogates.predict_fields(
                fidelity,  # type: ignore[arg-type]
                self.inner_parameters[fidelity],
                self.settings.measurement_time,
                batch_size=self.settings.fno_batch_size,
            )
            self._inner_fields[fidelity] = np.asarray(cached, dtype=float)
        return cached

    def evaluate(
        self,
        fidelity: int,
        random_inputs: ArrayLike,
    ) -> FloatArray:
        """Return utilities with shape ``(n_outer, n_grid_locations)``."""
        if fidelity not in (0, 1, 2):
            raise ValueError("fidelity must be 0, 1, or 2")
        inputs = np.asarray(random_inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != 6:
            raise ValueError("random_inputs must have shape (n, 6)")
        outer_fields = self.surrogates.predict_fields(
            fidelity,  # type: ignore[arg-type]
            inputs[:, :5],
            self.settings.measurement_time,
            batch_size=self.settings.fno_batch_size,
        ).reshape(inputs.shape[0], -1)
        inner_fields = self._get_inner_fields(fidelity).reshape(
            self.settings.inner_sample_sizes[fidelity], -1
        )
        epsilon = inputs[:, 5:6]
        utilities = np.empty_like(outer_fields, dtype=float)
        for location in range(outer_fields.shape[1]):
            utilities[:, location] = pointwise_information_utilities(
                outer_fields[:, location],
                inner_fields[:, location],
                epsilon,
                noise_standard_deviation=self.settings.noise_standard_deviation,
                outer_chunk_size=self.settings.outer_chunk_size,
            )
        return utilities

    def evaluate_ensemble(self, random_inputs: ArrayLike) -> FloatArray:
        """Return ``(n_outer, n_locations, 3)`` aligned utility samples."""
        values = [self.evaluate(fidelity, random_inputs) for fidelity in range(3)]
        return np.stack(values, axis=-1)

    @staticmethod
    def local_covariances(utilities: ArrayLike) -> FloatArray:
        """Compute one 3-by-3 covariance matrix at every sensor location."""
        values = np.asarray(utilities, dtype=float)
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("utilities must have shape (n_samples, n_locations, 3)")
        if values.shape[0] < 2:
            raise ValueError("At least two samples are required")
        return np.stack(
            [np.cov(values[:, location, :], rowvar=False, ddof=1) for location in range(values.shape[1])],
            axis=0,
        )

    @property
    def source_normalized_costs(self) -> FloatArray:
        """Utility costs following the convention in the supplied OED source."""
        sizes = np.asarray(self.settings.inner_sample_sizes, dtype=float)
        return np.asarray(self.surrogates.costs, dtype=float) * (sizes / sizes[0])
