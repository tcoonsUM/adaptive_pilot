"""Portable numerical transforms used by the supplied OED surrogate assets.

The collaborative archive stored fitted scikit-learn transformers with joblib.  The three
input transformers are identities, so the public repository does not serialize them.  The
only nontrivial transform is the fitted one-feature Yeo--Johnson output transform for the MLP;
its numeric parameters are stored in a safe ``.npz`` archive and evaluated here directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def yeo_johnson_transform(values: ArrayLike, power: float) -> FloatArray:
    """Apply the scalar Yeo--Johnson transform componentwise."""
    x = np.asarray(values, dtype=float)
    result = np.empty_like(x, dtype=float)
    positive = x >= 0.0
    tolerance = np.spacing(1.0)

    if abs(power) < tolerance:
        result[positive] = np.log1p(x[positive])
    else:
        result[positive] = (
            np.power(x[positive] + 1.0, power) - 1.0
        ) / power

    if abs(power - 2.0) > tolerance:
        result[~positive] = -(
            np.power(1.0 - x[~positive], 2.0 - power) - 1.0
        ) / (2.0 - power)
    else:
        result[~positive] = -np.log1p(-x[~positive])
    return result


def yeo_johnson_inverse(values: ArrayLike, power: float) -> FloatArray:
    """Invert the scalar Yeo--Johnson transform componentwise."""
    transformed = np.asarray(values, dtype=float)
    result = np.empty_like(transformed, dtype=float)
    positive = transformed >= 0.0
    tolerance = np.spacing(1.0)

    if abs(power) < tolerance:
        result[positive] = np.expm1(transformed[positive])
    else:
        base = power * transformed[positive] + 1.0
        if np.any(base <= 0.0):
            raise ValueError("Yeo-Johnson inverse received values outside its valid range")
        result[positive] = np.power(base, 1.0 / power) - 1.0

    if abs(power - 2.0) > tolerance:
        base = 1.0 - (2.0 - power) * transformed[~positive]
        if np.any(base <= 0.0):
            raise ValueError("Yeo-Johnson inverse received values outside its valid range")
        result[~positive] = 1.0 - np.power(base, 1.0 / (2.0 - power))
    else:
        result[~positive] = 1.0 - np.exp(-transformed[~positive])
    return result


@dataclass(frozen=True)
class StandardizedYeoJohnsonTransform:
    """One-feature Yeo--Johnson transform followed by standardization."""

    power: float
    mean: float
    scale: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.power):
            raise ValueError("power must be finite")
        if not np.isfinite(self.mean):
            raise ValueError("mean must be finite")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and positive")

    @classmethod
    def load(cls, path: str | Path) -> "StandardizedYeoJohnsonTransform":
        """Load numeric transform parameters from a non-pickle NumPy archive."""
        with np.load(Path(path), allow_pickle=False) as archive:
            method = str(archive["method"].item())
            standardized = bool(archive["standardize"].item())
            powers = np.asarray(archive["lambdas"], dtype=float).reshape(-1)
            means = np.asarray(archive["mean"], dtype=float).reshape(-1)
            scales = np.asarray(archive["scale"], dtype=float).reshape(-1)
        if method != "yeo-johnson" or not standardized:
            raise ValueError(
                "Only standardized one-feature Yeo-Johnson assets are supported"
            )
        if powers.size != 1 or means.size != 1 or scales.size != 1:
            raise ValueError("Expected exactly one fitted output feature")
        return cls(float(powers[0]), float(means[0]), float(scales[0]))

    def transform(self, values: ArrayLike) -> FloatArray:
        """Apply Yeo--Johnson transformation and fitted standardization."""
        transformed = yeo_johnson_transform(values, self.power)
        return (transformed - self.mean) / self.scale

    def inverse_transform(self, values: ArrayLike) -> FloatArray:
        """Undo fitted standardization and the Yeo--Johnson transformation."""
        standardized = np.asarray(values, dtype=float)
        transformed = standardized * self.scale + self.mean
        return yeo_johnson_inverse(transformed, self.power)
