"""Bijective covariance parameterization used by the covariance emulator.

A positive-definite covariance matrix is written as ``Sigma = D R D``.  The
standard deviations are represented by their logarithms and the correlation
matrix is represented by the strict upper triangle of ``log(R)``.  The inverse
correlation mapping follows the fixed-point diagonal correction described by
Archakov and Hansen (2021).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class UnconstrainedCovariance:
    """Unconstrained coordinates of a positive-definite covariance matrix."""

    gamma: FloatArray
    log_standard_deviations: FloatArray

    def concatenate(self) -> FloatArray:
        """Return ``[gamma, log_standard_deviations]``."""
        return np.concatenate((self.gamma, self.log_standard_deviations))


def n_gamma(n_models: int) -> int:
    """Number of strict triangular entries for an ``n_models`` correlation matrix."""
    if n_models < 1:
        raise ValueError("n_models must be positive")
    return n_models * (n_models - 1) // 2


def infer_n_models(n_gamma_parameters: int) -> int:
    """Infer matrix dimension from a triangular-vector length."""
    if n_gamma_parameters < 0:
        raise ValueError("Gamma-vector length cannot be negative")
    n_float = 0.5 * (1.0 + np.sqrt(1.0 + 8.0 * n_gamma_parameters))
    n = int(round(n_float))
    if n_gamma(n) != n_gamma_parameters:
        raise ValueError(
            f"Length {n_gamma_parameters} is not n(n-1)/2 for an integer n"
        )
    return n


def _symmetric(matrix: ArrayLike) -> FloatArray:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Matrix contains NaN or infinite values")
    return 0.5 * (value + value.T)


def is_positive_definite(matrix: ArrayLike, *, tolerance: float = 0.0) -> bool:
    """Return whether all eigenvalues exceed ``tolerance``."""
    try:
        eigenvalues = np.linalg.eigvalsh(_symmetric(matrix))
    except (ValueError, np.linalg.LinAlgError):
        return False
    return bool(np.min(eigenvalues) > tolerance)


def nearest_positive_definite(matrix: ArrayLike) -> FloatArray:
    """Return a nearby symmetric positive-definite matrix.

    This is the standard Higham/D'Errico projection followed by diagonal jitter
    when the projected matrix is only positive semidefinite.
    """
    a = _symmetric(matrix)
    _, singular_values, v_transpose = np.linalg.svd(a)
    h = v_transpose.T @ np.diag(singular_values) @ v_transpose
    projected = 0.5 * (a + h)
    projected = 0.5 * (projected + projected.T)
    if is_positive_definite(projected):
        return projected

    spacing = np.spacing(np.linalg.norm(a))
    identity = np.eye(a.shape[0])
    iteration = 1
    while not is_positive_definite(projected):
        minimum_eigenvalue = float(np.min(np.real(np.linalg.eigvals(projected))))
        projected += identity * (-minimum_eigenvalue * iteration**2 + spacing)
        iteration += 1
        if iteration > 100:
            raise np.linalg.LinAlgError("Nearest-PD projection failed to converge")
    return projected


def covariance_to_correlation(
    covariance: ArrayLike,
    *,
    jitter: float = 1.0e-12,
) -> tuple[FloatArray, FloatArray]:
    """Return the correlation matrix and marginal standard deviations."""
    sigma = _symmetric(covariance)
    diagonal = np.diag(sigma)
    if np.any(diagonal <= 0.0):
        sigma = nearest_positive_definite(sigma + jitter * np.eye(sigma.shape[0]))
        diagonal = np.diag(sigma)
    standard_deviations = np.sqrt(diagonal)
    correlation = sigma / np.outer(standard_deviations, standard_deviations)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    if not is_positive_definite(correlation):
        correlation = nearest_positive_definite(correlation)
        scale = np.sqrt(np.diag(correlation))
        correlation = correlation / np.outer(scale, scale)
        np.fill_diagonal(correlation, 1.0)
    return correlation, standard_deviations


def gamma_forward(
    correlation: ArrayLike,
    *,
    eigenvalue_floor: float = 1.0e-12,
) -> FloatArray:
    """Map a positive-definite correlation matrix to unconstrained gamma values."""
    if eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be positive")
    corr = _symmetric(correlation)
    if not np.allclose(np.diag(corr), 1.0, atol=1.0e-8):
        raise ValueError("gamma_forward expects a correlation matrix with unit diagonal")
    if not is_positive_definite(corr):
        corr = nearest_positive_definite(corr)
        scaling = np.sqrt(np.diag(corr))
        corr = corr / np.outer(scaling, scaling)
        np.fill_diagonal(corr, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
    log_correlation = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    indices = np.triu_indices(corr.shape[0], k=1)
    return np.asarray(log_correlation[indices], dtype=np.float64)


def gamma_inverse(
    gamma: ArrayLike,
    *,
    tolerance: float = 1.0e-10,
    max_iterations: int = 10_000,
) -> FloatArray:
    """Map unconstrained gamma values to a positive-definite correlation matrix."""
    values = np.asarray(gamma, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Gamma vector contains NaN or infinite values")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    n_models = infer_n_models(values.size)
    if n_models == 1:
        return np.ones((1, 1), dtype=float)

    log_correlation = np.zeros((n_models, n_models), dtype=float)
    upper = np.triu_indices(n_models, k=1)
    log_correlation[upper] = values
    log_correlation = log_correlation + log_correlation.T
    diagonal_indices = np.diag_indices(n_models)
    diagonal = np.zeros(n_models, dtype=float)

    target_distance = np.sqrt(n_models) * tolerance
    for _ in range(max_iterations):
        current = expm(log_correlation)
        current_diagonal = np.diag(current)
        if np.any(current_diagonal <= 0.0) or not np.all(np.isfinite(current_diagonal)):
            raise np.linalg.LinAlgError("Invalid diagonal encountered in gamma inverse mapping")
        delta = np.log(current_diagonal)
        diagonal -= delta
        log_correlation[diagonal_indices] = diagonal
        if np.linalg.norm(delta) <= target_distance:
            correlation = expm(log_correlation)
            correlation = 0.5 * (correlation + correlation.T)
            np.fill_diagonal(correlation, 1.0)
            return correlation
    raise RuntimeError(
        f"gamma_inverse did not converge within {max_iterations} iterations"
    )


def covariance_to_unconstrained(covariance: ArrayLike) -> UnconstrainedCovariance:
    """Map a covariance matrix to gamma and log-standard-deviation coordinates."""
    correlation, standard_deviations = covariance_to_correlation(covariance)
    return UnconstrainedCovariance(
        gamma=gamma_forward(correlation),
        log_standard_deviations=np.log(standard_deviations),
    )


def unconstrained_to_covariance(
    gamma: ArrayLike,
    log_standard_deviations: ArrayLike,
) -> FloatArray:
    """Reconstruct a positive-definite covariance matrix."""
    log_sds = np.asarray(log_standard_deviations, dtype=float).reshape(-1)
    corr = gamma_inverse(gamma)
    if corr.shape[0] != log_sds.size:
        raise ValueError(
            f"Gamma implies {corr.shape[0]} models but {log_sds.size} log standard deviations were supplied"
        )
    if not np.all(np.isfinite(log_sds)):
        raise ValueError("Log standard deviations contain NaN or infinite values")
    standard_deviations = np.exp(log_sds)
    covariance = np.diag(standard_deviations) @ corr @ np.diag(standard_deviations)
    return 0.5 * (covariance + covariance.T)


def batch_covariance_to_unconstrained(
    covariances: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Transform an ``(n, m, m)`` covariance array."""
    values = np.asarray(covariances, dtype=float)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"Expected shape (n, m, m), got {values.shape}")
    transformed = [covariance_to_unconstrained(matrix) for matrix in values]
    return (
        np.stack([item.gamma for item in transformed], axis=0),
        np.stack([item.log_standard_deviations for item in transformed], axis=0),
    )


def batch_unconstrained_to_covariance(
    gammas: ArrayLike,
    log_standard_deviations: ArrayLike,
) -> FloatArray:
    """Reconstruct an ``(n, m, m)`` covariance array."""
    gamma_values = np.asarray(gammas, dtype=float)
    log_sd_values = np.asarray(log_standard_deviations, dtype=float)
    if gamma_values.ndim != 2 or log_sd_values.ndim != 2:
        raise ValueError("Batched transformed coordinates must both be two-dimensional")
    if gamma_values.shape[0] != log_sd_values.shape[0]:
        raise ValueError("Gamma and log-standard-deviation batches must have equal length")
    return np.stack(
        [
            unconstrained_to_covariance(gamma_values[i], log_sd_values[i])
            for i in range(gamma_values.shape[0])
        ],
        axis=0,
    )
