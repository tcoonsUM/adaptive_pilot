"""Optional MXMCPy allocation backend used by the original research scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from adaptive_covariance.types import ModelSequence, RandomInputSampler

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MXMCPyDesign:
    algorithm: str
    variance: float
    allocation: Any
    estimator: Any
    covariance: FloatArray
    costs: FloatArray


@dataclass(frozen=True)
class MXMCPyEstimate:
    value: float
    variance: float
    design: MXMCPyDesign


def _import_mxmc() -> tuple[Any, Any]:
    try:
        from mxmc import Estimator, Optimizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "The MXMCPy backend requires NASA's optional MXMCPy package. "
            "Install this project with `python -m pip install -e .[mxmcpy]`."
        ) from error
    return Optimizer, Estimator


class MXMCPyBackend:
    """Search MXMCPy's allocation families and evaluate the selected estimator."""

    def __init__(
        self,
        *,
        exclude_algorithms: Sequence[str] = ("mfmc", "mlmc"),
        algorithms: Sequence[str] | None = None,
    ) -> None:
        self.exclude_algorithms = {name.lower() for name in exclude_algorithms}
        self.algorithms = None if algorithms is None else list(algorithms)

    def design(
        self,
        covariance: ArrayLike,
        costs: ArrayLike,
        budget: float,
    ) -> MXMCPyDesign:
        Optimizer, Estimator = _import_mxmc()
        sigma = np.asarray(covariance, dtype=float)
        model_costs = np.asarray(costs, dtype=float).reshape(-1)
        optimizer = Optimizer(model_costs, sigma)
        algorithms = self.algorithms or list(Optimizer.get_algorithm_names())
        best_name: str | None = None
        best_result: Any = None
        for algorithm in algorithms:
            if algorithm.lower() in self.exclude_algorithms:
                continue
            try:
                result = optimizer.optimize(algorithm, budget)
            except Exception:
                continue
            if best_result is None or float(result.variance) < float(best_result.variance):
                best_name = algorithm
                best_result = result
        if best_name is None or best_result is None:
            raise RuntimeError("MXMCPy did not return a feasible allocation")
        estimator = Estimator(best_result.allocation, sigma)
        return MXMCPyDesign(
            algorithm=best_name,
            variance=float(best_result.variance),
            allocation=best_result.allocation,
            estimator=estimator,
            covariance=sigma,
            costs=model_costs,
        )

    def estimate(
        self,
        *,
        models: ModelSequence,
        point: FloatArray,
        sample_random_inputs: RandomInputSampler,
        design: MXMCPyDesign,
        rng: np.random.Generator,
    ) -> MXMCPyEstimate:
        total_samples = int(design.allocation.num_total_samples)
        random_inputs = sample_random_inputs(total_samples, rng)
        allocated = design.allocation.allocate_samples_to_models(random_inputs)
        outputs = [
            np.asarray(model(point, samples), dtype=float).reshape(-1)
            for model, samples in zip(models, allocated, strict=True)
        ]
        return MXMCPyEstimate(
            value=float(design.estimator.get_estimate(outputs)),
            variance=design.variance,
            design=design,
        )
