"""Three-fidelity one-dimensional four-branch benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FourBranchParameters:
    a: float
    b: float
    c: float
    d: float
    cost: float


PAPER_PARAMETERS = (
    FourBranchParameters(a=10.0, b=2.0, c=7.0, d=1.4, cost=1.0),
    FourBranchParameters(a=9.0, b=1.75, c=6.0, d=1.2, cost=1.0e-2),
    FourBranchParameters(a=5.0, b=1.5, c=5.0, d=1.0, cost=1.0e-4),
)


def signed_power(value: ArrayLike, exponent: float) -> FloatArray:
    """Return ``sign(value) * abs(value)**exponent`` for real noninteger powers."""
    array = np.asarray(value, dtype=float)
    return np.sign(array) * np.abs(array) ** exponent


def four_branch_value(
    design: ArrayLike,
    random_input: ArrayLike,
    parameters: FourBranchParameters,
) -> FloatArray:
    """Evaluate one fidelity of the modified four-branch function."""
    xi = float(np.asarray(design, dtype=float).reshape(-1)[0])
    theta = np.asarray(random_input, dtype=float)
    if theta.ndim > 1:
        theta = theta[:, 0]
    theta = theta.reshape(-1)

    difference = xi - theta
    opposite = theta - xi
    branch_1 = 3.0 + signed_power(difference, parameters.b) / parameters.a - (
        xi + theta
    ) / np.sqrt(2.0)
    branch_2 = 3.0 + signed_power(opposite, parameters.b) / parameters.a + (
        xi + theta
    ) / np.sqrt(2.0)
    branch_3 = signed_power(difference, parameters.d) + parameters.c / np.sqrt(2.0)
    branch_4 = signed_power(opposite, parameters.d) + parameters.c / np.sqrt(2.0)
    return np.minimum.reduce((branch_1, branch_2, branch_3, branch_4)) + 3.0


class FourBranchEnsemble:
    """Paper model ensemble with a uniform uncertain input on ``[-6, 6]``."""

    design_bounds = (-3.0, 3.0)
    random_input_bounds = (-6.0, 6.0)
    true_optimizer = 0.0

    def __init__(
        self,
        parameters: tuple[FourBranchParameters, ...] = PAPER_PARAMETERS,
    ) -> None:
        if len(parameters) < 2:
            raise ValueError("The benchmark requires at least two fidelities")
        self.parameters = parameters
        self.models: list[Callable[[FloatArray, FloatArray], FloatArray]] = [
            partial(four_branch_value, parameters=parameters_i)
            for parameters_i in parameters
        ]
        self.costs = np.asarray([item.cost for item in parameters], dtype=float)

    @staticmethod
    def sample_random_inputs(n: int, rng: np.random.Generator) -> FloatArray:
        if n < 1:
            raise ValueError("n must be positive")
        return rng.uniform(-6.0, 6.0, size=(n, 1))

    def evaluate(self, design: float, random_inputs: ArrayLike) -> FloatArray:
        values = [model(np.array([design]), np.asarray(random_inputs)) for model in self.models]
        return np.column_stack(values)

    def mean_by_quadrature(self, design: float, fidelity: int = 0) -> float:
        """Compute a deterministic uniform-input mean for diagnostics."""
        model = self.models[fidelity]

        def integrand(theta: float) -> float:
            return float(model(np.array([design]), np.array([[theta]]))[0]) / 12.0

        value, _ = quad(integrand, -6.0, 6.0, epsabs=1.0e-10, epsrel=1.0e-10, limit=400)
        return float(value)

    def reference_statistics(
        self,
        designs: ArrayLike,
        *,
        n_samples: int = 200_000,
        seed: int = 42,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return high-fidelity means, covariances, and correlations on a design grid."""
        xi = np.asarray(designs, dtype=float).reshape(-1)
        rng = np.random.default_rng(seed)
        theta = self.sample_random_inputs(n_samples, rng)
        means = np.empty(xi.size, dtype=float)
        covariances = np.empty((xi.size, len(self.models), len(self.models)), dtype=float)
        correlations = np.empty_like(covariances)
        for index, design in enumerate(xi):
            outputs = self.evaluate(float(design), theta)
            means[index] = float(np.mean(outputs[:, 0]))
            covariances[index] = np.cov(outputs, rowvar=False, ddof=1)
            correlations[index] = np.corrcoef(outputs, rowvar=False)
        return means, covariances, correlations
