"""Self-contained independent-sample approximate control-variate estimator.

This implementation is intentionally small and dependency-light.  All models are
sampled together on a common base set.  Each low-fidelity model receives an
additional independent sample set.  The zero-mean control variates are the
additional-sample means minus the corresponding base-sample means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from adaptive_covariance.types import ModelSequence, RandomInputSampler, evaluate_ensemble

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ACVAllocation:
    """Base and independent low-fidelity sample counts."""

    n_base: int
    n_extra: IntArray
    costs: FloatArray
    single_fidelity: bool = False

    @property
    def total_cost(self) -> float:
        if self.single_fidelity:
            return self.n_base * float(self.costs[0])
        return self.n_base * float(np.sum(self.costs)) + float(
            np.dot(self.n_extra, self.costs[1:])
        )

    @property
    def samples_per_model(self) -> IntArray:
        if self.single_fidelity:
            counts = np.zeros_like(self.costs, dtype=np.int64)
            counts[0] = self.n_base
            return counts
        return np.concatenate(
            (np.array([self.n_base], dtype=np.int64), self.n_base + self.n_extra)
        )


@dataclass(frozen=True)
class ACVDesign:
    """Allocation, optimal control-variate weights, and predicted variance."""

    allocation: ACVAllocation
    weights: FloatArray
    variance: float
    covariance: FloatArray
    method: str = "independent-sample-acv"


@dataclass(frozen=True)
class ACVEstimate:
    value: float
    variance: float
    design: ACVDesign
    base_means: FloatArray
    extra_means: FloatArray


def _delta_covariances(
    covariance: FloatArray,
    n_base: float,
    n_extra: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    low_covariance = covariance[1:, 1:]
    k_matrix = low_covariance / n_base
    diagonal = np.diag(low_covariance) * (1.0 / n_extra)
    k_matrix = k_matrix.copy()
    k_matrix[np.diag_indices_from(k_matrix)] += diagonal
    q_delta = -covariance[0, 1:] / n_base
    return k_matrix, q_delta


def optimal_acv_weights(
    covariance: ArrayLike,
    n_base: float,
    n_extra: ArrayLike,
) -> FloatArray:
    """Return the variance-minimizing weights for a fixed allocation."""
    sigma = np.asarray(covariance, dtype=float)
    extras = np.asarray(n_extra, dtype=float).reshape(-1)
    k_matrix, q_delta = _delta_covariances(sigma, n_base, extras)
    return -np.linalg.pinv(k_matrix, hermitian=True) @ q_delta


def acv_variance(
    covariance: ArrayLike,
    n_base: float,
    n_extra: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    """Evaluate the independent-sample ACV variance formula."""
    sigma = np.asarray(covariance, dtype=float)
    extras = np.asarray(n_extra, dtype=float).reshape(-1)
    k_matrix, q_delta = _delta_covariances(sigma, n_base, extras)
    alpha = (
        optimal_acv_weights(sigma, n_base, extras)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    variance = (
        sigma[0, 0] / n_base
        + float(alpha @ k_matrix @ alpha)
        + 2.0 * float(alpha @ q_delta)
    )
    return max(float(variance), 0.0)


def _base_candidates(minimum: int, maximum: int, maximum_count: int = 240) -> IntArray:
    if maximum < minimum:
        return np.empty(0, dtype=np.int64)
    if maximum - minimum + 1 <= maximum_count:
        return np.arange(minimum, maximum + 1, dtype=np.int64)
    linear = np.linspace(minimum, maximum, maximum_count // 2)
    geometric = np.geomspace(max(minimum, 1), maximum, maximum_count // 2)
    return np.unique(np.rint(np.concatenate((linear, geometric))).astype(np.int64))


def _optimize_extras_for_base(
    covariance: FloatArray,
    low_costs: FloatArray,
    n_base: int,
    remaining_budget: float,
    minimum_extra: int,
) -> IntArray | None:
    mandatory = minimum_extra * float(np.sum(low_costs))
    if remaining_budget + 1.0e-12 < mandatory:
        return None
    n_low = low_costs.size
    available = max(remaining_budget - mandatory, 0.0)
    start = np.full(n_low, minimum_extra, dtype=float)
    if available > 0.0:
        start += available / (n_low * low_costs)

    bounds = [
        (float(minimum_extra), float(max(minimum_extra, remaining_budget / cost)))
        for cost in low_costs
    ]

    def objective(extras: FloatArray) -> float:
        return acv_variance(covariance, float(n_base), extras)

    constraints = {
        "type": "ineq",
        "fun": lambda extras: remaining_budget - float(np.dot(extras, low_costs)),
    }
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1.0e-12, "disp": False},
    )
    continuous = result.x if result.success and np.all(np.isfinite(result.x)) else start
    extras = np.maximum(np.floor(continuous).astype(np.int64), minimum_extra)

    # A floor loses less than one sample per low-fidelity model. Spend the small
    # remainder greedily according to marginal variance reduction per unit cost.
    leftover = remaining_budget - float(np.dot(extras, low_costs))
    iterations = 0
    while leftover + 1.0e-12 >= float(np.min(low_costs)) and iterations < 100_000:
        current_variance = acv_variance(covariance, float(n_base), extras)
        best_index: int | None = None
        best_efficiency = -np.inf
        for index, cost in enumerate(low_costs):
            if cost <= leftover + 1.0e-12:
                proposal = extras.copy()
                proposal[index] += 1
                reduction = current_variance - acv_variance(
                    covariance, float(n_base), proposal
                )
                efficiency = reduction / cost
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_index = index
        if best_index is None:
            break
        extras[best_index] += 1
        leftover -= float(low_costs[best_index])
        iterations += 1
    return extras


class IndependentSampleACV:
    """Optimize and evaluate the independent-sample ACV estimator."""

    def __init__(self, *, minimum_base: int = 2, minimum_extra: int = 2) -> None:
        if minimum_base < 2 or minimum_extra < 1:
            raise ValueError("Invalid minimum sample count")
        self.minimum_base = minimum_base
        self.minimum_extra = minimum_extra

    def design(
        self,
        covariance: ArrayLike,
        costs: ArrayLike,
        budget: float,
        *,
        fixed_allocation: ACVAllocation | None = None,
        allow_single_fidelity_fallback: bool = True,
    ) -> ACVDesign:
        """Choose an allocation and optimal weights for a covariance prediction."""
        sigma = np.asarray(covariance, dtype=float)
        model_costs = np.asarray(costs, dtype=float).reshape(-1)
        if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
            raise ValueError("covariance must be square")
        if sigma.shape[0] != model_costs.size:
            raise ValueError("Covariance dimension and model-cost count do not match")
        if model_costs.size < 2:
            raise ValueError("ACV requires at least one low-fidelity model")
        if np.any(model_costs <= 0.0) or budget <= 0.0:
            raise ValueError("Costs and budget must be positive")

        if fixed_allocation is not None:
            if fixed_allocation.single_fidelity:
                variance = sigma[0, 0] / fixed_allocation.n_base
                return ACVDesign(
                    allocation=fixed_allocation,
                    weights=np.zeros(model_costs.size - 1),
                    variance=float(variance),
                    covariance=sigma,
                    method="single-fidelity",
                )
            weights = optimal_acv_weights(
                sigma, fixed_allocation.n_base, fixed_allocation.n_extra
            )
            variance = acv_variance(
                sigma, fixed_allocation.n_base, fixed_allocation.n_extra, weights
            )
            return ACVDesign(
                allocation=fixed_allocation,
                weights=weights,
                variance=variance,
                covariance=sigma,
            )

        low_costs = model_costs[1:]
        mandatory_extra_cost = self.minimum_extra * float(np.sum(low_costs))
        maximum_base = int(
            np.floor((budget - mandatory_extra_cost) / float(np.sum(model_costs)))
        )
        best: ACVDesign | None = None
        for n_base in _base_candidates(self.minimum_base, maximum_base):
            remaining = budget - n_base * float(np.sum(model_costs))
            extras = _optimize_extras_for_base(
                sigma,
                low_costs,
                int(n_base),
                remaining,
                self.minimum_extra,
            )
            if extras is None:
                continue
            allocation = ACVAllocation(
                n_base=int(n_base),
                n_extra=extras,
                costs=model_costs,
            )
            weights = optimal_acv_weights(sigma, allocation.n_base, allocation.n_extra)
            variance = acv_variance(
                sigma, allocation.n_base, allocation.n_extra, weights
            )
            candidate = ACVDesign(
                allocation=allocation,
                weights=weights,
                variance=variance,
                covariance=sigma,
            )
            if best is None or candidate.variance < best.variance:
                best = candidate

        n_single = int(np.floor(budget / model_costs[0]))
        single_variance = sigma[0, 0] / n_single if n_single >= 2 else np.inf
        if best is None or (allow_single_fidelity_fallback and single_variance <= best.variance):
            if n_single < 2:
                raise ValueError("Budget is too small for either ACV or single-fidelity MC")
            allocation = ACVAllocation(
                n_base=n_single,
                n_extra=np.zeros(model_costs.size - 1, dtype=np.int64),
                costs=model_costs,
                single_fidelity=True,
            )
            return ACVDesign(
                allocation=allocation,
                weights=np.zeros(model_costs.size - 1),
                variance=float(single_variance),
                covariance=sigma,
                method="single-fidelity",
            )
        return best

    def estimate(
        self,
        *,
        models: ModelSequence,
        point: FloatArray,
        sample_random_inputs: RandomInputSampler,
        design: ACVDesign,
        rng: np.random.Generator,
    ) -> ACVEstimate:
        """Evaluate a previously designed estimator."""
        allocation = design.allocation
        if len(models) != allocation.costs.size:
            raise ValueError("Model count does not match ACV allocation")
        if allocation.single_fidelity:
            inputs = sample_random_inputs(allocation.n_base, rng)
            values = np.asarray(models[0](point, inputs), dtype=float).reshape(-1)
            return ACVEstimate(
                value=float(np.mean(values)),
                variance=design.variance,
                design=design,
                base_means=np.array([np.mean(values)], dtype=float),
                extra_means=np.empty(0, dtype=float),
            )

        base_inputs = sample_random_inputs(allocation.n_base, rng)
        base_outputs = evaluate_ensemble(models, point, base_inputs)
        base_means = np.mean(base_outputs, axis=0)
        extra_means = np.empty(len(models) - 1, dtype=float)
        for low_index, n_extra in enumerate(allocation.n_extra, start=1):
            extra_inputs = sample_random_inputs(int(n_extra), rng)
            values = np.asarray(models[low_index](point, extra_inputs), dtype=float).reshape(-1)
            extra_means[low_index - 1] = np.mean(values)
        deltas = extra_means - base_means[1:]
        value = float(base_means[0] + design.weights @ deltas)
        return ACVEstimate(
            value=value,
            variance=design.variance,
            design=design,
            base_means=base_means,
            extra_means=extra_means,
        )

    def estimate_with_budget(
        self,
        *,
        models: ModelSequence,
        point: FloatArray,
        sample_random_inputs: RandomInputSampler,
        covariance: ArrayLike,
        costs: ArrayLike,
        budget: float,
        rng: np.random.Generator,
        fixed_allocation: ACVAllocation | None = None,
    ) -> ACVEstimate:
        """Design and evaluate an ACV estimator in one call."""
        acv_design = self.design(
            covariance,
            costs,
            budget,
            fixed_allocation=fixed_allocation,
        )
        return self.estimate(
            models=models,
            point=point,
            sample_random_inputs=sample_random_inputs,
            design=acv_design,
            rng=rng,
        )
