"""Noisy Bayesian optimization over a finite candidate set.

The paper uses BoTorch's qLogNoisyExpectedImprovement.  This module provides a
small core-dependency alternative that is easy to run and adapt.  The optional
paper environment can replace it with BoTorch without changing the pilot or
estimator interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from adaptive_covariance.types import as_2d_designs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NoisyBayesianOptimizerConfig:
    random_seed: int = 42
    n_restarts_optimizer: int = 1
    initial_length_scale: float = 0.25
    allow_repeated_candidates: bool = True
    duplicate_tolerance: float = 1.0e-12
    exploration_jitter: float = 0.0


@dataclass(frozen=True)
class PosteriorPrediction:
    mean: FloatArray
    standard_deviation: FloatArray


class NoisyBayesianOptimizer:
    """Heteroscedastic GP and expected-improvement acquisition on a candidate set."""

    def __init__(
        self,
        candidate_designs: ArrayLike,
        config: NoisyBayesianOptimizerConfig | None = None,
    ) -> None:
        self.candidates = as_2d_designs(np.asarray(candidate_designs, dtype=float))
        self.config = config or NoisyBayesianOptimizerConfig()
        self.designs = np.empty((0, self.candidates.shape[1]), dtype=float)
        self.values = np.empty(0, dtype=float)
        self.noise_variances = np.empty(0, dtype=float)
        self._model: GaussianProcessRegressor | None = None
        self._minimum = np.min(self.candidates, axis=0)
        self._span = np.max(self.candidates, axis=0) - self._minimum
        self._span[self._span <= np.finfo(float).eps] = 1.0

    def _normalize(self, designs: FloatArray) -> FloatArray:
        return (designs - self._minimum) / self._span

    def observe(
        self,
        design: ArrayLike,
        value: float,
        noise_variance: float,
        *,
        refit: bool = True,
    ) -> None:
        x = np.asarray(design, dtype=float).reshape(1, -1)
        if x.shape[1] != self.candidates.shape[1]:
            raise ValueError("Observed design has the wrong dimension")
        if not np.isfinite(value) or not np.isfinite(noise_variance):
            raise ValueError("Observation value and variance must be finite")
        if noise_variance <= 0.0:
            noise_variance = 1.0e-12
        self.designs = np.vstack((self.designs, x))
        self.values = np.append(self.values, float(value))
        self.noise_variances = np.append(self.noise_variances, float(noise_variance))
        if refit:
            self.fit()

    def observe_many(
        self,
        designs: ArrayLike,
        values: ArrayLike,
        noise_variances: ArrayLike,
    ) -> None:
        x = as_2d_designs(np.asarray(designs, dtype=float))
        y = np.asarray(values, dtype=float).reshape(-1)
        noise = np.asarray(noise_variances, dtype=float).reshape(-1)
        if x.shape[0] != y.size or y.shape != noise.shape:
            raise ValueError("Batch observation arrays have incompatible shapes")
        self.designs = np.vstack((self.designs, x))
        self.values = np.concatenate((self.values, y))
        self.noise_variances = np.concatenate((self.noise_variances, np.maximum(noise, 1e-12)))
        self.fit()

    def fit(self) -> None:
        if self.values.size < 2:
            raise RuntimeError("At least two observations are required to fit the objective GP")
        dimension = self.candidates.shape[1]
        kernel = ConstantKernel(1.0, (1.0e-4, 1.0e4)) * RBF(
            np.full(dimension, self.config.initial_length_scale),
            (1.0e-3, 1.0e3),
        )
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=np.maximum(self.noise_variances, 1.0e-12),
            normalize_y=True,
            n_restarts_optimizer=self.config.n_restarts_optimizer,
            random_state=self.config.random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self._model.fit(self._normalize(self.designs), self.values)

    def posterior(self, designs: ArrayLike | None = None) -> PosteriorPrediction:
        if self._model is None:
            raise RuntimeError("Objective GP has not been fitted")
        x = self.candidates if designs is None else as_2d_designs(np.asarray(designs, dtype=float))
        mean, standard_deviation = self._model.predict(
            self._normalize(x), return_std=True
        )
        return PosteriorPrediction(
            mean=np.asarray(mean, dtype=float),
            standard_deviation=np.maximum(np.asarray(standard_deviation, dtype=float), 0.0),
        )

    def expected_improvement(self) -> FloatArray:
        prediction = self.posterior()
        best = float(np.max(self.values))
        improvement = prediction.mean - best - self.config.exploration_jitter
        std = prediction.standard_deviation
        scores = np.zeros_like(improvement)
        positive = std > 0.0
        z = np.zeros_like(improvement)
        z[positive] = improvement[positive] / std[positive]
        scores[positive] = (
            improvement[positive] * norm.cdf(z[positive])
            + std[positive] * norm.pdf(z[positive])
        )
        scores[~positive] = np.maximum(improvement[~positive], 0.0)
        if not self.config.allow_repeated_candidates:
            for observed in self.designs:
                distance = np.linalg.norm(self.candidates - observed, axis=1)
                scores[distance <= self.config.duplicate_tolerance] = -np.inf
        return scores

    def suggest(self, *, final: bool = False) -> FloatArray:
        """Return posterior-mean maximizer for ``final`` or EI maximizer otherwise."""
        prediction = self.posterior()
        if final:
            index = int(np.argmax(prediction.mean))
        else:
            index = int(np.argmax(self.expected_improvement()))
        return self.candidates[index].copy()

    def predicted_maximizer(self) -> tuple[FloatArray, float]:
        prediction = self.posterior()
        index = int(np.argmax(prediction.mean))
        return self.candidates[index].copy(), float(prediction.mean[index])
