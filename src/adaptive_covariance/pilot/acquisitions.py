"""Acquisition functions for active covariance learning."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def covariance_uncertainty(
    gamma_standard_deviations: ArrayLike,
    log_standard_deviation_standard_deviations: ArrayLike,
) -> FloatArray:
    """Sum transformed posterior standard deviations at every candidate design."""
    gamma_std = np.asarray(gamma_standard_deviations, dtype=float)
    log_sd_std = np.asarray(log_standard_deviation_standard_deviations, dtype=float)
    if gamma_std.ndim != 2 or log_sd_std.ndim != 2:
        raise ValueError("Posterior standard-deviation arrays must be two-dimensional")
    if gamma_std.shape[0] != log_sd_std.shape[0]:
        raise ValueError("Transformed posterior arrays must have equal candidate counts")
    return np.sum(gamma_std, axis=1) + np.sum(log_sd_std, axis=1)


def objective_importance_weights(
    objective_prediction: ArrayLike,
    *,
    mode: str = "auto",
    epsilon: float = 1.0e-12,
) -> FloatArray:
    """Convert objective predictions to nonnegative pilot-importance weights.

    ``raw`` uses the prediction directly and requires nonnegative values. ``normalized``
    applies the shifted min-max transformation from the manuscript. ``auto`` uses raw
    values when all are nonnegative and otherwise uses the normalized transformation.
    ``positive`` clips negative predictions to zero and adds ``epsilon``.
    """
    values = np.asarray(objective_prediction, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Objective predictions contain NaN or infinite values")
    selected_mode = mode
    if mode == "auto":
        selected_mode = "raw" if np.min(values) >= 0.0 else "normalized"
    if selected_mode == "raw":
        if np.min(values) < 0.0:
            raise ValueError("Raw objective weights require nonnegative predictions")
        return values
    if selected_mode == "positive":
        return np.maximum(values, 0.0) + epsilon
    if selected_mode == "normalized":
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        return (values - minimum) / (maximum - minimum + epsilon)
    raise ValueError(f"Unknown objective-weight mode: {mode}")


def mean_weighted_covariance_uncertainty(
    covariance_uncertainty_score: ArrayLike,
    objective_prediction: ArrayLike,
    *,
    weight_mode: str = "auto",
) -> FloatArray:
    """Multiply covariance uncertainty by objective-derived importance weights."""
    uncertainty = np.asarray(covariance_uncertainty_score, dtype=float).reshape(-1)
    weights = objective_importance_weights(objective_prediction, mode=weight_mode)
    if uncertainty.shape != weights.shape:
        raise ValueError("Uncertainty scores and objective predictions must have equal shape")
    return uncertainty * weights
