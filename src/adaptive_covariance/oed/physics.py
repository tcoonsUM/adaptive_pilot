"""Physical parameterization for the two-source convection–diffusion problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OEDPrior:
    """Independent uniform prior bounds in paper parameter order."""

    lower: tuple[float, ...] = (-1.0, -1.0, 0.0, -1.0, 5.0)
    upper: tuple[float, ...] = (0.0, 1.0, 1.0, 1.0, 12.0)

    @property
    def dimension(self) -> int:
        return len(self.lower)


def sample_oed_prior(
    n_samples: int,
    rng: np.random.Generator,
    prior: OEDPrior | None = None,
) -> FloatArray:
    """Draw source locations and velocity strength from the paper prior."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    distribution = prior or OEDPrior()
    return rng.uniform(
        np.asarray(distribution.lower, dtype=float),
        np.asarray(distribution.upper, dtype=float),
        size=(n_samples, distribution.dimension),
    )


def source_field(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    parameters: ArrayLike,
    *,
    source_width: float = 0.05,
    source_strength: float = 2.0,
) -> FloatArray:
    """Evaluate the two Gaussian contaminant sources for one parameter vector."""
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    theta = np.asarray(parameters, dtype=float).reshape(-1)
    if theta.size < 4:
        raise ValueError("At least four source-location parameters are required")
    result = np.zeros_like(x, dtype=float)
    for source_x, source_y in ((theta[0], theta[1]), (theta[2], theta[3])):
        exponent = -(
            (x - source_x) ** 2 + (y - source_y) ** 2
        ) / (2.0 * source_width**2)
        result += source_strength / (2.0 * np.pi * source_width**2) * np.exp(exponent)
    return result


def source_fields(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    parameters: ArrayLike,
    *,
    source_width: float = 0.05,
    source_strength: float = 2.0,
) -> FloatArray:
    """Vectorized wrapper returning ``(n_samples, nx, ny)`` source fields."""
    theta = np.asarray(parameters, dtype=float)
    if theta.ndim != 2:
        raise ValueError("parameters must have shape (n_samples, 5)")
    return np.stack(
        [
            source_field(
                x_grid,
                y_grid,
                row,
                source_width=source_width,
                source_strength=source_strength,
            )
            for row in theta
        ],
        axis=0,
    )


def velocity_field(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    beta: float,
    *,
    sink_strength: float = 2.5,
) -> FloatArray:
    """Return the paper's rotational-plus-inward velocity field."""
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    velocity_x = beta * (np.pi / 2.0) * (1.0 - x**2) * y - sink_strength * x
    velocity_y = -beta * (np.pi / 2.0) * (1.0 - y**2) * x - sink_strength * y
    return np.stack((velocity_x, velocity_y), axis=-1)


def normalized_velocity_fields(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    beta_values: ArrayLike,
    *,
    sink_strength: float = 2.5,
) -> FloatArray:
    """Return unit-vector velocity fields used as FNO input channels."""
    fields = []
    for beta in np.asarray(beta_values, dtype=float).reshape(-1):
        field = velocity_field(
            x_grid, y_grid, float(beta), sink_strength=sink_strength
        )
        magnitude = np.linalg.norm(field, axis=-1, keepdims=True)
        fields.append(field / np.maximum(magnitude, 1.0e-10))
    return np.stack(fields, axis=0)
