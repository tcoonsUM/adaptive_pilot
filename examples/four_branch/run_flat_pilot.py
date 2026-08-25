#!/usr/bin/env python3
"""Run the fixed space-filling pilot baseline for the four-branch problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adaptive_covariance.config import load_yaml
from adaptive_covariance.models.four_branch import FourBranchEnsemble
from adaptive_covariance.pilot import PilotStudyConfig, run_pilot_at_designs


def run(config_path: Path, output_path: Path, *, seed_override: int | None = None) -> Path:
    raw = load_yaml(config_path)
    pilot_raw = dict(raw["pilot"])
    initial_designs = np.asarray(pilot_raw.pop("initial_designs"), dtype=float).reshape(-1)
    pilot_raw.pop("candidate_count", None)
    seed = int(raw.get("random_seed", 42) if seed_override is None else seed_override)
    pilot_raw["random_seed"] = seed
    pilot_raw.setdefault("gp_settings", {})["random_seed"] = seed
    config = PilotStudyConfig.from_mapping(pilot_raw)

    ensemble = FourBranchEnsemble()
    low, high = ensemble.design_bounds
    if config.n_active_designs > 0:
        active_designs = np.linspace(low, high, config.n_active_designs + 2)[1:-1]
        designs = np.concatenate((initial_designs, active_designs))[:, None]
    else:
        designs = initial_designs[:, None]

    result = run_pilot_at_designs(
        models=ensemble.models,
        sample_random_inputs=ensemble.sample_random_inputs,
        designs=designs,
        initial_design_count=initial_designs.size,
        config=config,
    )
    result.save(output_path)
    print(f"Saved space-filling pilot study to {output_path.resolve()}")
    print("Pilot designs:", ", ".join(f"{x:.4g}" for x in result.designs[:, 0]))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    run(arguments.config, arguments.output, seed_override=arguments.seed)


if __name__ == "__main__":
    main()
