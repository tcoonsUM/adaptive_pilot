"""Shared protocols and array validation helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class ModelCallable(Protocol):
    """A scalar-output stochastic model evaluated at one deterministic design."""

    def __call__(self, design: FloatArray, random_inputs: FloatArray) -> FloatArray:
        """Return one scalar model output for each row of ``random_inputs``."""


RandomInputSampler: TypeAlias = Callable[[int, np.random.Generator], FloatArray]
ModelSequence: TypeAlias = Sequence[ModelCallable]


def as_2d_designs(values: np.ndarray | Sequence[float]) -> FloatArray:
    """Convert design values to a finite two-dimensional ``float64`` array."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    elif array.ndim != 2:
        raise ValueError(f"Design array must have 1 or 2 dimensions, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Design array contains NaN or infinite values")
    return array.astype(np.float64, copy=False)


def evaluate_ensemble(
    models: ModelSequence,
    design: np.ndarray,
    random_inputs: FloatArray,
) -> FloatArray:
    """Evaluate all models and return an ``(n_samples, n_models)`` matrix."""
    if len(models) < 1:
        raise ValueError("At least one model is required")
    x = np.asarray(design, dtype=float).reshape(-1)
    z = np.asarray(random_inputs, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    outputs: list[FloatArray] = []
    for index, model in enumerate(models):
        values = np.asarray(model(x, z), dtype=float).reshape(-1)
        if values.shape[0] != z.shape[0]:
            raise ValueError(
                f"Model {index} returned {values.shape[0]} values for {z.shape[0]} inputs"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Model {index} returned NaN or infinite values")
        outputs.append(values)
    return np.column_stack(outputs)
