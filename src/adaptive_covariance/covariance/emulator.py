"""Independent Gaussian-process emulator for transformed covariance parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from adaptive_covariance.covariance.transforms import (
    batch_unconstrained_to_covariance,
    infer_n_models,
)
from adaptive_covariance.types import as_2d_designs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianProcessSettings:
    """Settings shared by the independent scalar Gaussian processes."""

    normalize_designs: bool = True
    normalize_outputs: bool = True
    initial_length_scale: float = 0.25
    length_scale_lower: float = 1.0e-3
    length_scale_upper: float = 1.0e3
    constant_lower: float = 1.0e-4
    constant_upper: float = 1.0e4
    n_restarts_optimizer: int = 1
    optimize_hyperparameters: bool = True
    random_seed: int = 0
    minimum_noise_variance: float = 1.0e-12


@dataclass(frozen=True)
class TransformedPrediction:
    """Posterior means and marginal standard deviations in transformed space."""

    gamma_mean: FloatArray
    gamma_standard_deviation: FloatArray
    log_standard_deviation_mean: FloatArray
    log_standard_deviation_standard_deviation: FloatArray

    @property
    def total_standard_deviation(self) -> FloatArray:
        """Sum of scalar posterior standard deviations at each design."""
        return np.sum(self.gamma_standard_deviation, axis=1) + np.sum(
            self.log_standard_deviation_standard_deviation, axis=1
        )


class CovarianceEmulator:
    """Independent GPs for gamma and log-standard-deviation components.

    The class deliberately models transformed components independently. Mapping
    posterior means back through the inverse transform always yields a
    positive-definite covariance matrix.
    """

    def __init__(self, settings: GaussianProcessSettings | None = None) -> None:
        self.settings = settings or GaussianProcessSettings()
        self._gamma_models: list[GaussianProcessRegressor] = []
        self._log_sd_models: list[GaussianProcessRegressor] = []
        self._design_min: FloatArray | None = None
        self._design_span: FloatArray | None = None
        self._training_state: dict[str, FloatArray] | None = None
        self.n_models: int | None = None
        self.design_dimension: int | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(self._gamma_models or self._log_sd_models)

    def _normalize_x(self, designs: FloatArray, *, fitting: bool) -> FloatArray:
        if not self.settings.normalize_designs:
            return designs
        if fitting:
            self._design_min = np.min(designs, axis=0)
            maximum = np.max(designs, axis=0)
            span = maximum - self._design_min
            span[span <= np.finfo(float).eps] = 1.0
            self._design_span = span
        if self._design_min is None or self._design_span is None:
            raise RuntimeError("Design normalization state has not been initialized")
        return (designs - self._design_min) / self._design_span

    def _make_gp(self, alpha: FloatArray, component_seed: int) -> GaussianProcessRegressor:
        dimension = int(self.design_dimension or 1)
        initial = np.full(dimension, self.settings.initial_length_scale, dtype=float)
        kernel = ConstantKernel(
            1.0,
            constant_value_bounds=(
                self.settings.constant_lower,
                self.settings.constant_upper,
            ),
        ) * RBF(
            length_scale=initial,
            length_scale_bounds=(
                self.settings.length_scale_lower,
                self.settings.length_scale_upper,
            ),
        )
        optimizer: str | None = "fmin_l_bfgs_b" if self.settings.optimize_hyperparameters else None
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=np.maximum(alpha, self.settings.minimum_noise_variance),
            optimizer=optimizer,
            n_restarts_optimizer=(
                self.settings.n_restarts_optimizer if optimizer is not None else 0
            ),
            normalize_y=self.settings.normalize_outputs,
            random_state=component_seed,
        )

    def fit(
        self,
        designs: ArrayLike,
        gammas: ArrayLike,
        log_standard_deviations: ArrayLike,
        gamma_noise_variances: ArrayLike,
        log_standard_deviation_noise_variances: ArrayLike,
    ) -> "CovarianceEmulator":
        """Fit all transformed-component GPs."""
        x = as_2d_designs(np.asarray(designs, dtype=float))
        gamma_values = np.asarray(gammas, dtype=float)
        log_sd_values = np.asarray(log_standard_deviations, dtype=float)
        gamma_noise = np.asarray(gamma_noise_variances, dtype=float)
        log_sd_noise = np.asarray(log_standard_deviation_noise_variances, dtype=float)

        for name, values in {
            "gammas": gamma_values,
            "log_standard_deviations": log_sd_values,
            "gamma_noise_variances": gamma_noise,
            "log_standard_deviation_noise_variances": log_sd_noise,
        }.items():
            if values.ndim != 2 or values.shape[0] != x.shape[0]:
                raise ValueError(
                    f"{name} must have shape (n_designs, n_components); got {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains NaN or infinite values")

        if gamma_values.shape != gamma_noise.shape:
            raise ValueError("Gamma observations and noise variances must have equal shape")
        if log_sd_values.shape != log_sd_noise.shape:
            raise ValueError("Log-standard-deviation observations and noise must have equal shape")
        inferred_models = infer_n_models(gamma_values.shape[1])
        if log_sd_values.shape[1] != inferred_models:
            raise ValueError(
                f"Gamma coordinates imply {inferred_models} models, but log-SD data have "
                f"{log_sd_values.shape[1]} columns"
            )

        self.n_models = inferred_models
        self.design_dimension = x.shape[1]
        x_fit = self._normalize_x(x, fitting=True)
        self._gamma_models = []
        self._log_sd_models = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for component in range(gamma_values.shape[1]):
                gp = self._make_gp(
                    gamma_noise[:, component],
                    self.settings.random_seed + component,
                )
                gp.fit(x_fit, gamma_values[:, component])
                self._gamma_models.append(gp)
            offset = gamma_values.shape[1]
            for component in range(log_sd_values.shape[1]):
                gp = self._make_gp(
                    log_sd_noise[:, component],
                    self.settings.random_seed + offset + component,
                )
                gp.fit(x_fit, log_sd_values[:, component])
                self._log_sd_models.append(gp)

        self._training_state = {
            "designs": x.copy(),
            "gammas": gamma_values.copy(),
            "log_standard_deviations": log_sd_values.copy(),
            "gamma_noise_variances": gamma_noise.copy(),
            "log_standard_deviation_noise_variances": log_sd_noise.copy(),
        }
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted or self.n_models is None:
            raise RuntimeError("CovarianceEmulator must be fitted before prediction")

    def predict_transformed(self, designs: ArrayLike) -> TransformedPrediction:
        """Return transformed posterior means and marginal standard deviations."""
        self._require_fitted()
        x = as_2d_designs(np.asarray(designs, dtype=float))
        if x.shape[1] != self.design_dimension:
            raise ValueError(
                f"Expected design dimension {self.design_dimension}, got {x.shape[1]}"
            )
        x_predict = self._normalize_x(x, fitting=False)

        gamma_mean = np.empty((x.shape[0], len(self._gamma_models)), dtype=float)
        gamma_std = np.empty_like(gamma_mean)
        for component, gp in enumerate(self._gamma_models):
            mean, std = gp.predict(x_predict, return_std=True)
            gamma_mean[:, component] = mean
            gamma_std[:, component] = np.maximum(std, 0.0)

        log_sd_mean = np.empty((x.shape[0], len(self._log_sd_models)), dtype=float)
        log_sd_std = np.empty_like(log_sd_mean)
        for component, gp in enumerate(self._log_sd_models):
            mean, std = gp.predict(x_predict, return_std=True)
            log_sd_mean[:, component] = mean
            log_sd_std[:, component] = np.maximum(std, 0.0)

        return TransformedPrediction(
            gamma_mean=gamma_mean,
            gamma_standard_deviation=gamma_std,
            log_standard_deviation_mean=log_sd_mean,
            log_standard_deviation_standard_deviation=log_sd_std,
        )

    def predict_covariance(self, designs: ArrayLike) -> FloatArray:
        """Map transformed posterior means to positive-definite covariance predictions."""
        prediction = self.predict_transformed(designs)
        return batch_unconstrained_to_covariance(
            prediction.gamma_mean,
            prediction.log_standard_deviation_mean,
        )

    def uncertainty_score(self, designs: ArrayLike) -> FloatArray:
        """Return the sum of transformed posterior standard deviations."""
        return self.predict_transformed(designs).total_standard_deviation

    def high_fidelity_standard_deviation_upper(
        self,
        designs: ArrayLike,
        *,
        standard_deviation_multiplier: float = 2.0,
    ) -> FloatArray:
        """Upper lognormal plug-in used for conservative ACV variance reporting."""
        prediction = self.predict_transformed(designs)
        upper_log_sd = (
            prediction.log_standard_deviation_mean[:, 0]
            + standard_deviation_multiplier
            * prediction.log_standard_deviation_standard_deviation[:, 0]
        )
        return np.exp(upper_log_sd)

    def state_dict(self) -> dict[str, Any]:
        """Return a portable training state from which the emulator can be refitted."""
        self._require_fitted()
        assert self._training_state is not None
        return {
            "settings": asdict(self.settings),
            **{key: value.copy() for key, value in self._training_state.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "CovarianceEmulator":
        """Refit an emulator from ``state_dict`` output."""
        settings_raw = state.get("settings", {})
        settings = GaussianProcessSettings(**settings_raw)
        emulator = cls(settings=settings)
        return emulator.fit(
            state["designs"],
            state["gammas"],
            state["log_standard_deviations"],
            state["gamma_noise_variances"],
            state["log_standard_deviation_noise_variances"],
        )
