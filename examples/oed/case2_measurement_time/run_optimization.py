#!/usr/bin/env python3
"""Run measurement-time optimization for OED Case 2.

The proposed method uses the frozen covariance emulator produced by
``run_pilot.py`` to configure a fresh ACV estimator at every measurement time.
A single-fidelity nested-Monte-Carlo baseline is also available.  The public
optimizer is a lightweight heteroscedastic Gaussian-process expected-
improvement implementation over a configurable candidate grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from adaptive_covariance.config import load_yaml, repository_root
from adaptive_covariance.estimators import IndependentSampleACV, estimate_monte_carlo
from adaptive_covariance.oed import (
    Case2UtilityEnsemble,
    Case2UtilitySettings,
    OEDSurrogateEnsemble,
)
from adaptive_covariance.optimization import (
    NoisyBayesianOptimizer,
    NoisyBayesianOptimizerConfig,
)
from adaptive_covariance.pilot import PilotStudyResult

FloatArray = NDArray[np.float64]
METHODS = ("mfeig-gamma-opt", "mfeig-flat", "nmc-sf")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _utility_settings(raw: dict[str, Any]) -> Case2UtilitySettings:
    values = dict(raw)
    values["sensor_location"] = tuple(
        float(value) for value in values.get("sensor_location", (-0.8, -0.2))
    )
    values["inner_sample_sizes"] = tuple(
        int(value) for value in values.get("inner_sample_sizes", (1000, 500, 100))
    )
    return Case2UtilitySettings(**values)


def _reference_curve(path: Path) -> tuple[FloatArray, FloatArray]:
    with np.load(path, allow_pickle=False) as archive:
        designs = np.asarray(archive["xi"], dtype=float).reshape(-1)
        means = np.asarray(archive["cost_aware_utility_means"], dtype=float)[:, 0]
    order = np.argsort(designs)
    return designs[order], means[order]


def _initialization_cost(
    method: str,
    pilot: PilotStudyResult,
    costs: FloatArray,
) -> tuple[slice, float]:
    if method in {"mfeig-gamma-opt", "mfeig-flat"}:
        selection = slice(None)
        cost = float(np.sum(pilot.sample_counts) * np.sum(costs))
    else:
        selection = slice(0, pilot.initial_design_count)
        cost = float(
            np.sum(pilot.sample_counts[: pilot.initial_design_count]) * costs[0]
        )
    return selection, cost


def _per_iteration_budget(
    raw: dict[str, Any],
    *,
    method: str,
    costs: FloatArray,
    initialization_cost: float,
    n_iterations: int,
) -> float:
    if "per_iteration_budget" in raw:
        budget = float(raw["per_iteration_budget"])
    elif "per_iteration_ensemble_samples" in raw:
        budget = float(raw["per_iteration_ensemble_samples"]) * float(np.sum(costs))
    elif "total_budget_hours" in raw:
        total = float(raw["total_budget_hours"]) * 3600.0
        remaining = total - initialization_cost
        if remaining <= 0.0:
            raise ValueError(
                f"Initialization cost exhausts total budget for method {method}"
            )
        budget = remaining / n_iterations
    else:
        raise ValueError(
            "optimization must define per_iteration_budget, "
            "per_iteration_ensemble_samples, or total_budget_hours"
        )
    if budget <= 0.0:
        raise ValueError("Per-iteration budget must be positive")
    return budget


def run_experiment(
    *,
    config_path: Path,
    pilot_path: Path,
    method: str,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    raw = load_yaml(config_path)
    root = repository_root()
    seed = int(raw.get("random_seed", 42) if seed_override is None else seed_override)
    rng = np.random.default_rng(seed)

    asset_raw = raw.get("assets", {})
    surrogate = OEDSurrogateEnsemble.from_assets(
        _resolve(root, asset_raw.get("root", "assets/oed")),
        device=str(asset_raw.get("device", "cpu")),
    )
    utilities = Case2UtilityEnsemble(
        surrogate,
        _utility_settings(raw.get("utility", {})),
    )
    pilot = PilotStudyResult.load(pilot_path)

    optimization_raw = raw["optimization"]
    n_iterations = int(optimization_raw.get("n_iterations", 20))
    candidate_count = int(optimization_raw.get("candidate_count", 101))
    candidates = np.linspace(0.1, 0.6, candidate_count)[:, None]
    objective_raw = optimization_raw.get("objective_gp", {})
    optimizer = NoisyBayesianOptimizer(
        candidates,
        NoisyBayesianOptimizerConfig(
            random_seed=seed,
            n_restarts_optimizer=int(objective_raw.get("n_restarts_optimizer", 1)),
            allow_repeated_candidates=bool(
                objective_raw.get("allow_repeated_candidates", True)
            ),
            exploration_jitter=float(objective_raw.get("exploration_jitter", 0.0)),
        ),
    )

    initialization_slice, cumulative_cost = _initialization_cost(
        method,
        pilot,
        utilities.costs,
    )
    optimizer.observe_many(
        pilot.designs[initialization_slice],
        pilot.high_fidelity_means[initialization_slice],
        pilot.high_fidelity_mean_variances[initialization_slice],
    )
    iteration_budget = _per_iteration_budget(
        optimization_raw,
        method=method,
        costs=utilities.costs,
        initialization_cost=cumulative_cost,
        n_iterations=n_iterations,
    )

    estimator_raw = optimization_raw.get("estimator", {})
    estimator = IndependentSampleACV(
        minimum_base=int(estimator_raw.get("minimum_base", 2)),
        minimum_extra=int(estimator_raw.get("minimum_extra", 2)),
    )
    inflation_multiplier = float(
        optimization_raw.get("variance_inflation_standard_deviations", 2.0)
    )

    reference_path = _resolve(
        root,
        raw.get("reference", {}).get(
            "path", "paper_results/oed_case2/reference_quantities.npz"
        ),
    )
    reference_designs, reference_means = _reference_curve(reference_path)
    reference_maximum = float(np.max(reference_means))
    reference_optimizer = float(reference_designs[int(np.argmax(reference_means))])

    query_designs = np.empty((n_iterations, 1), dtype=float)
    estimates = np.empty(n_iterations, dtype=float)
    estimate_variances = np.empty(n_iterations, dtype=float)
    predicted_maximizers = np.empty((n_iterations + 1, 1), dtype=float)
    predicted_maximum_values = np.empty(n_iterations + 1, dtype=float)
    regrets = np.empty(n_iterations + 1, dtype=float)
    cumulative_costs = np.empty(n_iterations + 1, dtype=float)
    sample_counts = np.zeros((n_iterations, 3), dtype=np.int64)
    weights = np.full((n_iterations, 2), np.nan, dtype=float)
    allocation_methods: list[str] = []

    predicted_maximizers[0], predicted_maximum_values[0] = optimizer.predicted_maximizer()
    cumulative_costs[0] = cumulative_cost
    regrets[0] = reference_maximum - float(
        np.interp(predicted_maximizers[0, 0], reference_designs, reference_means)
    )
    cached_flat_allocation = None

    for iteration in range(n_iterations):
        point = optimizer.suggest()
        query_designs[iteration] = point
        if method == "nmc-sf":
            estimate = estimate_monte_carlo(
                model=utilities.models[0],
                design=point,
                sample_random_inputs=utilities.sample_random_inputs,
                budget=iteration_budget,
                model_cost=utilities.costs[0],
                rng=rng,
            )
            value = estimate.value
            variance = estimate.variance
            iteration_cost = estimate.cost
            sample_counts[iteration, 0] = estimate.n_samples
            allocation_methods.append("single-fidelity")
        else:
            covariance = (
                pilot.emulator.predict_covariance(point[None, :])[0]
                if method == "mfeig-gamma-opt"
                else pilot.flat_covariance
            )
            fixed_allocation = (
                cached_flat_allocation if method == "mfeig-flat" else None
            )
            acv_design = estimator.design(
                covariance,
                utilities.costs,
                iteration_budget,
                fixed_allocation=fixed_allocation,
            )
            if method == "mfeig-flat" and cached_flat_allocation is None:
                cached_flat_allocation = acv_design.allocation
            acv_estimate = estimator.estimate(
                models=utilities.models,
                point=point,
                sample_random_inputs=utilities.sample_random_inputs,
                design=acv_design,
                rng=rng,
            )
            value = acv_estimate.value
            variance = acv_estimate.variance
            if method == "mfeig-gamma-opt" and covariance[0, 0] > 0.0:
                upper_hf_sd = pilot.emulator.high_fidelity_standard_deviation_upper(
                    point[None, :],
                    standard_deviation_multiplier=inflation_multiplier,
                )[0]
                variance *= float(upper_hf_sd**2 / covariance[0, 0])
            iteration_cost = acv_design.allocation.total_cost
            sample_counts[iteration] = acv_design.allocation.samples_per_model
            weights[iteration] = acv_design.weights
            allocation_methods.append(acv_design.method)

        estimates[iteration] = value
        estimate_variances[iteration] = max(float(variance), 1.0e-12)
        optimizer.observe(point, value, estimate_variances[iteration])
        cumulative_cost += float(iteration_cost)
        cumulative_costs[iteration + 1] = cumulative_cost
        predicted_maximizers[iteration + 1], predicted_maximum_values[iteration + 1] = (
            optimizer.predicted_maximizer()
        )
        regrets[iteration + 1] = reference_maximum - float(
            np.interp(
                predicted_maximizers[iteration + 1, 0],
                reference_designs,
                reference_means,
            )
        )
        print(
            f"Iteration {iteration + 1:02d}/{n_iterations}: "
            f"t={point[0]:.4f}, estimate={value:.5g}, "
            f"predicted t*={predicted_maximizers[iteration + 1, 0]:.4f}, "
            f"regret={regrets[iteration + 1]:.5g}"
        )

    return {
        "method": method,
        "seed": seed,
        "query_designs": query_designs,
        "estimates": estimates,
        "estimate_variances": estimate_variances,
        "predicted_maximizers": predicted_maximizers,
        "predicted_maximum_values": predicted_maximum_values,
        "regrets": regrets,
        "cumulative_costs": cumulative_costs,
        "runtime_hours": cumulative_costs / 3600.0,
        "sample_counts": sample_counts,
        "weights": weights,
        "allocation_methods": allocation_methods,
        "iteration_budget": np.array(iteration_budget),
        "reference_optimizer": np.array([reference_optimizer]),
        "reference_maximum": np.array(reference_maximum),
        "reference_designs": reference_designs,
        "reference_means": reference_means,
        "final_predicted_optimizer": predicted_maximizers[-1],
        "config": raw,
    }


def save_result(result: dict[str, Any], output_path: Path) -> Path:
    metadata = {
        "method": result["method"],
        "seed": result["seed"],
        "allocation_methods": result["allocation_methods"],
        "config": result["config"],
    }
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        metadata=np.array(json.dumps(metadata)),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="mfeig-gamma-opt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    result = run_experiment(
        config_path=arguments.config,
        pilot_path=arguments.pilot,
        method=arguments.method,
        seed_override=arguments.seed,
    )
    destination = save_result(result, arguments.output)
    print(f"Saved Case 2 optimization result to {destination}")
    print(f"Final predicted measurement time: {result['final_predicted_optimizer'][0]:.6f}")


if __name__ == "__main__":
    main()
