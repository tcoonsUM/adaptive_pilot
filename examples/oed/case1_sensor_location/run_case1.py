#!/usr/bin/env python3
"""Run the all-at-once OED sensor-location comparison (paper Case 1).

The expensive forward-model work is performed through the supplied MLP/FNO
checkpoints.  A single outer sample produces a utility field over the full
21-by-21 sensor grid.  The script compares

* high-fidelity nested Monte Carlo (NMC-SF),
* fixed-allocation ACV with location-dependent weights (MFEIG-ADAPT), and
* fixed-allocation ACV with one domain-averaged weight vector (MFEIG-FLAT).

The public implementation uses the dependency-light independent-sample ACV
family.  The fixed allocation is optimized once from the domain-averaged pilot
covariance, matching the structural restriction described for Case 1 in the
manuscript; only the weights are adapted across sensor locations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from adaptive_covariance.config import load_yaml, repository_root
from adaptive_covariance.covariance.transforms import (
    covariance_to_correlation,
    nearest_positive_definite,
)
from adaptive_covariance.estimators.independent_acv import (
    IndependentSampleACV,
    acv_variance,
    optimal_acv_weights,
)
from adaptive_covariance.oed import (
    Case1UtilityFieldEvaluator,
    Case1UtilitySettings,
    OEDSurrogateEnsemble,
)

FloatArray = NDArray[np.float64]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _regularize_covariances(covariances: FloatArray) -> FloatArray:
    result = np.empty_like(covariances, dtype=float)
    for index, covariance in enumerate(covariances):
        scale = max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
        result[index] = nearest_positive_definite(
            covariance + 1.0e-12 * scale * np.eye(covariance.shape[0])
        )
    return result


def _correlations(covariances: FloatArray) -> FloatArray:
    return np.stack(
        [covariance_to_correlation(covariance)[0] for covariance in covariances],
        axis=0,
    )


def _load_oracle(path: Path, expected_locations: int) -> tuple[FloatArray, int]:
    with np.load(path, allow_pickle=False) as archive:
        oracle = np.asarray(archive["oracle_eig"], dtype=float).reshape(-1)
    if oracle.size != expected_locations:
        raise ValueError(
            f"Oracle contains {oracle.size} locations; expected {expected_locations}"
        )
    return oracle, int(np.argmax(oracle))


def _acv_trial(
    evaluator: Case1UtilityFieldEvaluator,
    allocation_design: Any,
    flat_covariance: FloatArray,
    local_covariances: FloatArray,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    allocation = allocation_design.allocation
    if allocation.single_fidelity:
        raise RuntimeError("Case 1 ACV allocation unexpectedly fell back to single fidelity")

    base_inputs = evaluator.sample_random_inputs(allocation.n_base, rng)
    base_utilities = evaluator.evaluate_ensemble(base_inputs)
    base_means = np.mean(base_utilities, axis=0)  # (n_locations, 3)

    extra_means = np.empty((base_means.shape[0], 2), dtype=float)
    for low_fidelity, n_extra in enumerate(allocation.n_extra, start=1):
        extra_inputs = evaluator.sample_random_inputs(int(n_extra), rng)
        extra_means[:, low_fidelity - 1] = np.mean(
            evaluator.evaluate(low_fidelity, extra_inputs), axis=0
        )
    deltas = extra_means - base_means[:, 1:]

    flat_weights = optimal_acv_weights(
        flat_covariance,
        allocation.n_base,
        allocation.n_extra,
    )
    flat_estimate = base_means[:, 0] + deltas @ flat_weights
    flat_variance = np.full(
        base_means.shape[0],
        acv_variance(
            flat_covariance,
            allocation.n_base,
            allocation.n_extra,
            flat_weights,
        ),
        dtype=float,
    )

    adapted_estimate = np.empty(base_means.shape[0], dtype=float)
    adapted_variance = np.empty(base_means.shape[0], dtype=float)
    for location, covariance in enumerate(local_covariances):
        weights = optimal_acv_weights(
            covariance,
            allocation.n_base,
            allocation.n_extra,
        )
        adapted_estimate[location] = base_means[location, 0] + float(
            weights @ deltas[location]
        )
        adapted_variance[location] = acv_variance(
            covariance,
            allocation.n_base,
            allocation.n_extra,
            weights,
        )
    return adapted_estimate, adapted_variance, flat_estimate, flat_variance


def run(config_path: Path, output_path: Path) -> Path:
    raw = load_yaml(config_path)
    root = repository_root()
    seed = int(raw.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    asset_raw = raw.get("assets", {})
    asset_root = _resolve(root, asset_raw.get("root", "assets/oed"))
    surrogate = OEDSurrogateEnsemble.from_assets(
        asset_root,
        device=str(asset_raw.get("device", "cpu")),
    )

    utility_settings = Case1UtilitySettings(
        **{
            **raw.get("utility", {}),
            "inner_sample_sizes": tuple(
                int(value)
                for value in raw.get("utility", {}).get(
                    "inner_sample_sizes", (1000, 1000, 1000)
                )
            ),
        }
    )
    evaluator = Case1UtilityFieldEvaluator(surrogate, utility_settings)

    pilot_raw = raw.get("pilot", {})
    pilot_samples = int(pilot_raw.get("n_outer_samples", 300))
    pilot_seed = int(pilot_raw.get("seed", seed + 1))
    pilot_rng = np.random.default_rng(pilot_seed)
    pilot_inputs = evaluator.sample_random_inputs(pilot_samples, pilot_rng)
    pilot_utilities = evaluator.evaluate_ensemble(pilot_inputs)
    local_covariances = _regularize_covariances(
        evaluator.local_covariances(pilot_utilities)
    )
    flat_covariance = nearest_positive_definite(np.mean(local_covariances, axis=0))

    estimation_raw = raw.get("estimation", {})
    n_trials = int(estimation_raw.get("n_trials", 60))
    nmc_outer_samples = int(estimation_raw.get("nmc_outer_samples", 25))
    if n_trials < 1 or nmc_outer_samples < 2:
        raise ValueError("n_trials must be positive and nmc_outer_samples at least two")

    costs = evaluator.source_normalized_costs
    estimator_budget = float(
        estimation_raw.get("estimator_budget", nmc_outer_samples * costs[0])
    )
    acv = IndependentSampleACV(
        minimum_base=int(estimation_raw.get("minimum_base", 2)),
        minimum_extra=int(estimation_raw.get("minimum_extra", 2)),
    )
    allocation_design = acv.design(
        flat_covariance,
        costs,
        estimator_budget,
        allow_single_fidelity_fallback=False,
    )

    oracle_raw = raw.get("oracle", {})
    oracle_path = _resolve(
        root,
        oracle_raw.get(
            "path", "paper_results/oed_case1/case1_precomputed.npz"
        ),
    )
    oracle, oracle_index = _load_oracle(
        oracle_path,
        evaluator.sensor_locations.shape[0],
    )
    oracle_maximum = float(oracle[oracle_index])

    n_locations = evaluator.sensor_locations.shape[0]
    nmc_estimates = np.empty((n_trials, n_locations), dtype=float)
    adapted_estimates = np.empty_like(nmc_estimates)
    flat_estimates = np.empty_like(nmc_estimates)
    nmc_variances = np.empty_like(nmc_estimates)
    adapted_variances = np.empty_like(nmc_estimates)
    flat_variances = np.empty_like(nmc_estimates)
    predicted_indices = np.empty((n_trials, 3), dtype=np.int64)
    regrets = np.empty((n_trials, 3), dtype=float)

    for trial in range(n_trials):
        nmc_inputs = evaluator.sample_random_inputs(nmc_outer_samples, rng)
        nmc_values = evaluator.evaluate(0, nmc_inputs)
        nmc_estimates[trial] = np.mean(nmc_values, axis=0)
        nmc_variances[trial] = np.var(nmc_values, axis=0, ddof=1) / nmc_outer_samples

        (
            adapted_estimates[trial],
            adapted_variances[trial],
            flat_estimates[trial],
            flat_variances[trial],
        ) = _acv_trial(
            evaluator,
            allocation_design,
            flat_covariance,
            local_covariances,
            rng,
        )

        predicted_indices[trial] = (
            int(np.argmax(nmc_estimates[trial])),
            int(np.argmax(adapted_estimates[trial])),
            int(np.argmax(flat_estimates[trial])),
        )
        regrets[trial] = oracle_maximum - oracle[predicted_indices[trial]]
        print(
            f"Trial {trial + 1:03d}/{n_trials}: "
            f"regret NMC={regrets[trial, 0]:.5g}, "
            f"ADAPT={regrets[trial, 1]:.5g}, "
            f"FLAT={regrets[trial, 2]:.5g}"
        )

    metadata = {
        "configuration": raw,
        "utility_settings": asdict(utility_settings),
        "allocation_method": allocation_design.method,
        "allocation_total_cost": allocation_design.allocation.total_cost,
        "model_costs": costs.tolist(),
        "oracle_path": str(oracle_path),
        "method_order": ["NMC-SF", "MFEIG-ADAPT", "MFEIG-FLAT"],
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        grid_1d=evaluator.grid_1d,
        sensor_locations=evaluator.sensor_locations,
        oracle_eig=oracle,
        oracle_index=np.array(oracle_index),
        pilot_local_covariances=local_covariances,
        pilot_local_correlations=_correlations(local_covariances),
        pilot_flat_covariance=flat_covariance,
        allocation_n_base=np.array(allocation_design.allocation.n_base),
        allocation_n_extra=allocation_design.allocation.n_extra,
        allocation_samples_per_model=allocation_design.allocation.samples_per_model,
        allocation_weights=allocation_design.weights,
        allocation_predicted_variance=np.array(allocation_design.variance),
        nmc_estimates=nmc_estimates,
        mfeig_adapt_estimates=adapted_estimates,
        mfeig_flat_estimates=flat_estimates,
        nmc_variances=nmc_variances,
        mfeig_adapt_variances=adapted_variances,
        mfeig_flat_variances=flat_variances,
        predicted_indices=predicted_indices,
        predicted_locations=evaluator.sensor_locations[predicted_indices],
        regrets=regrets,
        metadata=np.array(json.dumps(metadata)),
    )
    print(f"Saved Case 1 result to {output_path}")
    print(
        "Mean regret: "
        f"NMC-SF={np.mean(regrets[:, 0]):.6g}, "
        f"MFEIG-ADAPT={np.mean(regrets[:, 1]):.6g}, "
        f"MFEIG-FLAT={np.mean(regrets[:, 2]):.6g}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
