#!/usr/bin/env python3
"""Run the offline active pilot stage for the four-branch benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adaptive_covariance.config import load_yaml
from adaptive_covariance.models.four_branch import FourBranchEnsemble
from adaptive_covariance.pilot import PilotStudyConfig, run_pilot_study


def run(config_path: Path, output_path: Path, *, seed_override: int | None = None) -> Path:
    raw = load_yaml(config_path)
    pilot_raw = dict(raw["pilot"])
    initial_designs = np.asarray(pilot_raw.pop("initial_designs"), dtype=float)[:, None]
    candidate_count = int(pilot_raw.pop("candidate_count"))
    if seed_override is not None:
        pilot_raw["random_seed"] = seed_override
        pilot_raw.setdefault("gp_settings", {})["random_seed"] = seed_override
    else:
        pilot_raw.setdefault("random_seed", int(raw.get("random_seed", 42)))
    config = PilotStudyConfig.from_mapping(pilot_raw)
    ensemble = FourBranchEnsemble()
    candidates = np.linspace(*ensemble.design_bounds, candidate_count)[:, None]
    result = run_pilot_study(
        models=ensemble.models,
        sample_random_inputs=ensemble.sample_random_inputs,
        candidate_designs=candidates,
        initial_designs=initial_designs,
        config=config,
    )
    result.save(output_path)
    print(f"Saved pilot study to {output_path.resolve()}")
    print("Selected active designs:")
    for record in result.records:
        print(
            f"  {record.iteration:02d} {record.acquisition:27s} "
            f"xi={record.selected_design[0]: .6f} score={record.acquisition_value:.6g}"
        )
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
