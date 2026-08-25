#!/usr/bin/env python3
"""Run one four-branch stochastic-optimization trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_covariance.config import load_yaml
from adaptive_covariance.covariance.observations import sample_covariance
from adaptive_covariance.estimators.independent_acv import IndependentSampleACV
from adaptive_covariance.estimators.monte_carlo import estimate_monte_carlo
from adaptive_covariance.estimators.mxmcpy_backend import MXMCPyBackend
from adaptive_covariance.models.four_branch import FourBranchEnsemble
from adaptive_covariance.optimization import (
    NoisyBayesianOptimizer,
    NoisyBayesianOptimizerConfig,
)
from adaptive_covariance.pilot import PilotStudyResult
from adaptive_covariance.types import evaluate_ensemble

METHODS = ("mc-sf", "acv-emulator", "acv-flat", "acv-independent")


def _make_estimator(raw: dict[str, Any]):
    backend = raw.get("backend", "independent")
    if backend == "independent":
        return IndependentSampleACV(
            minimum_base=int(raw.get("minimum_base", 2)),
            minimum_extra=int(raw.get("minimum_extra", 2)),
        )
    if backend == "mxmcpy":
        return MXMCPyBackend()
    raise ValueError(f"Unknown estimator backend: {backend}")


def run_experiment(
    *,
    config_path: Path,
    pilot_path: Path,
    method: str,
    flat_pilot_path: Path | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    raw = load_yaml(config_path)
    optimization_raw = raw["optimization"]
    seed = int(raw.get("random_seed", 42) if seed_override is None else seed_override)
    rng = np.random.default_rng(seed)
    ensemble = FourBranchEnsemble()
    pilot = PilotStudyResult.load(pilot_path)
    flat_pilot = (
        PilotStudyResult.load(flat_pilot_path)
        if flat_pilot_path is not None
        else pilot
    )
    method_pilot = flat_pilot if method == "acv-flat" else pilot

    n_candidates = int(optimization_raw["candidate_count"])
    candidates = np.linspace(*ensemble.design_bounds, n_candidates)[:, None]
    objective_gp_raw = optimization_raw.get("objective_gp", {})
    optimizer = NoisyBayesianOptimizer(
        candidates,
        NoisyBayesianOptimizerConfig(
            random_seed=seed,
            n_restarts_optimizer=int(objective_gp_raw.get("n_restarts_optimizer", 1)),
            allow_repeated_candidates=bool(
                objective_gp_raw.get("allow_repeated_candidates", True)
            ),
        ),
    )

    if method in {"acv-emulator", "acv-flat"}:
        initialization_slice = slice(None)
    else:
        initialization_slice = slice(0, method_pilot.initial_design_count)
    optimizer.observe_many(
        method_pilot.designs[initialization_slice],
        method_pilot.high_fidelity_means[initialization_slice],
        method_pilot.high_fidelity_mean_variances[initialization_slice],
    )

    if method in {"mc-sf", "acv-independent"}:
        cumulative_cost = float(
            np.sum(method_pilot.sample_counts[initialization_slice]) * ensemble.costs[0]
        )
    else:
        cumulative_cost = float(
            np.sum(method_pilot.sample_counts[initialization_slice])
            * np.sum(ensemble.costs)
        )

    n_iterations = int(optimization_raw["n_iterations"])
    if "total_budget" in optimization_raw:
        total_budget = float(optimization_raw["total_budget"])
        remaining_budget = total_budget - cumulative_cost
        if remaining_budget <= 0.0:
            raise ValueError(
                f"Initialization/pilot cost {cumulative_cost:g} exhausts total budget "
                f"{total_budget:g}"
            )
        per_iteration_budget = remaining_budget / n_iterations
    else:
        total_budget = np.nan
        per_iteration_budget = float(optimization_raw["per_iteration_budget"])
    local_pilot_samples = int(optimization_raw.get("local_pilot_samples", 0))
    inflation_multiplier = float(
        optimization_raw.get("variance_inflation_standard_deviations", 2.0)
    )
    estimator = _make_estimator(optimization_raw.get("estimator", {}))

    query_designs = np.empty((n_iterations, 1), dtype=float)
    estimates = np.empty(n_iterations, dtype=float)
    estimate_variances = np.empty(n_iterations, dtype=float)
    predicted_maximizers = np.empty((n_iterations + 1, 1), dtype=float)
    predicted_maximum_values = np.empty(n_iterations + 1, dtype=float)
    cumulative_costs = np.empty(n_iterations + 1, dtype=float)
    sample_counts = np.zeros((n_iterations, len(ensemble.models)), dtype=np.int64)
    allocation_names: list[str] = []

    predicted_maximizers[0], predicted_maximum_values[0] = optimizer.predicted_maximizer()
    cumulative_costs[0] = cumulative_cost
    cached_flat_design: Any = None

    for iteration in range(n_iterations):
        point = optimizer.suggest()
        query_designs[iteration] = point

        if method == "mc-sf":
            estimate = estimate_monte_carlo(
                model=ensemble.models[0],
                design=point,
                sample_random_inputs=ensemble.sample_random_inputs,
                budget=per_iteration_budget,
                model_cost=ensemble.costs[0],
                rng=rng,
            )
            value = estimate.value
            variance = estimate.variance
            sample_counts[iteration, 0] = estimate.n_samples
            iteration_cost = estimate.cost
            allocation_names.append("single-fidelity")
        else:
            estimator_budget = per_iteration_budget
            if method == "acv-emulator":
                covariance = pilot.emulator.predict_covariance(point[None, :])[0]
            elif method == "acv-flat":
                covariance = flat_pilot.flat_covariance
            else:
                if local_pilot_samples < 2:
                    raise ValueError("acv-independent requires local_pilot_samples >= 2")
                local_inputs = ensemble.sample_random_inputs(local_pilot_samples, rng)
                local_outputs = evaluate_ensemble(ensemble.models, point, local_inputs)
                covariance = sample_covariance(local_outputs)
                local_cost = local_pilot_samples * float(np.sum(ensemble.costs))
                estimator_budget -= local_cost
                if estimator_budget <= 0.0:
                    raise ValueError("Local pilot cost exhausts the per-iteration budget")

            if isinstance(estimator, IndependentSampleACV):
                fixed_allocation = None
                if method == "acv-flat" and cached_flat_design is not None:
                    fixed_allocation = cached_flat_design.allocation
                acv_design = estimator.design(
                    covariance,
                    ensemble.costs,
                    estimator_budget,
                    fixed_allocation=fixed_allocation,
                )
                if method == "acv-flat" and cached_flat_design is None:
                    cached_flat_design = acv_design
                acv_estimate = estimator.estimate(
                    models=ensemble.models,
                    point=point,
                    sample_random_inputs=ensemble.sample_random_inputs,
                    design=acv_design,
                    rng=rng,
                )
                value = acv_estimate.value
                variance = acv_estimate.variance
                sample_counts[iteration] = acv_design.allocation.samples_per_model
                iteration_cost = acv_design.allocation.total_cost
                allocation_names.append(acv_design.method)
            else:
                acv_design = estimator.design(covariance, ensemble.costs, estimator_budget)
                acv_estimate = estimator.estimate(
                    models=ensemble.models,
                    point=point,
                    sample_random_inputs=ensemble.sample_random_inputs,
                    design=acv_design,
                    rng=rng,
                )
                value = acv_estimate.value
                variance = acv_estimate.variance
                counts = np.asarray(
                    acv_design.allocation.get_number_of_samples_per_model(), dtype=np.int64
                )
                sample_counts[iteration, : counts.size] = counts
                iteration_cost = estimator_budget
                allocation_names.append(acv_design.algorithm)

            if method == "acv-independent":
                iteration_cost += local_pilot_samples * float(np.sum(ensemble.costs))
            if method == "acv-emulator":
                upper_hf_sd = pilot.emulator.high_fidelity_standard_deviation_upper(
                    point[None, :],
                    standard_deviation_multiplier=inflation_multiplier,
                )[0]
                variance *= float(upper_hf_sd**2 / covariance[0, 0])

        estimates[iteration] = value
        estimate_variances[iteration] = max(float(variance), 1.0e-12)
        optimizer.observe(point, value, estimate_variances[iteration])
        cumulative_cost += float(iteration_cost)
        cumulative_costs[iteration + 1] = cumulative_cost
        predicted_maximizers[iteration + 1], predicted_maximum_values[iteration + 1] = (
            optimizer.predicted_maximizer()
        )

    return {
        "method": method,
        "seed": seed,
        "query_designs": query_designs,
        "estimates": estimates,
        "estimate_variances": estimate_variances,
        "predicted_maximizers": predicted_maximizers,
        "predicted_maximum_values": predicted_maximum_values,
        "cumulative_costs": cumulative_costs,
        "sample_counts": sample_counts,
        "allocation_names": allocation_names,
        "final_predicted_optimizer": predicted_maximizers[-1],
        "source_reference_optimizer": np.array([ensemble.true_optimizer]),
        "per_iteration_budget": np.array(per_iteration_budget),
        "configured_total_budget": np.array(total_budget),
    }


def save_result(result: dict[str, Any], output_path: Path) -> None:
    metadata = {
        "method": result["method"],
        "seed": result["seed"],
        "allocation_names": result["allocation_names"],
    }
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        metadata=np.array(json.dumps(metadata)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument(
        "--flat-pilot",
        type=Path,
        help="Optional fixed space-filling pilot archive used by acv-flat.",
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    result = run_experiment(
        config_path=arguments.config,
        pilot_path=arguments.pilot,
        method=arguments.method,
        flat_pilot_path=arguments.flat_pilot,
        seed_override=arguments.seed,
    )
    save_result(result, arguments.output)
    print(f"Saved optimization result to {arguments.output.resolve()}")
    print(f"Predicted optimizer: {result['final_predicted_optimizer'][0]:.8f}")


if __name__ == "__main__":
    main()
