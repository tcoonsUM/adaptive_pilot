#!/usr/bin/env python3
"""Run active covariance pilot sampling for OED Case 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adaptive_covariance.config import load_yaml, repository_root
from adaptive_covariance.oed import (
    Case2UtilityEnsemble,
    Case2UtilitySettings,
    OEDSurrogateEnsemble,
)
from adaptive_covariance.pilot import PilotStudyConfig, run_pilot_study


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _utility_settings(raw: dict) -> Case2UtilitySettings:
    values = dict(raw)
    values["sensor_location"] = tuple(
        float(value) for value in values.get("sensor_location", (-0.8, -0.2))
    )
    values["inner_sample_sizes"] = tuple(
        int(value) for value in values.get("inner_sample_sizes", (1000, 500, 100))
    )
    return Case2UtilitySettings(**values)


def run(config_path: Path, output_path: Path, *, seed_override: int | None = None) -> Path:
    raw = load_yaml(config_path)
    root = repository_root()
    seed = int(raw.get("random_seed", 42) if seed_override is None else seed_override)

    asset_raw = raw.get("assets", {})
    surrogate = OEDSurrogateEnsemble.from_assets(
        _resolve(root, asset_raw.get("root", "assets/oed")),
        device=str(asset_raw.get("device", "cpu")),
    )
    utilities = Case2UtilityEnsemble(
        surrogate,
        _utility_settings(raw.get("utility", {})),
    )

    pilot_raw = dict(raw["pilot"])
    initial_designs = np.asarray(pilot_raw.pop("initial_designs"), dtype=float)[:, None]
    candidate_count = int(pilot_raw.pop("candidate_count"))
    pilot_raw["random_seed"] = seed
    pilot_raw.setdefault("gp_settings", {})["random_seed"] = seed
    config = PilotStudyConfig.from_mapping(pilot_raw)
    candidates = np.linspace(0.1, 0.6, candidate_count)[:, None]

    result = run_pilot_study(
        models=utilities.models,
        sample_random_inputs=utilities.sample_random_inputs,
        candidate_designs=candidates,
        initial_designs=initial_designs,
        config=config,
    )
    result.save(output_path)

    nominal_seconds = float(np.sum(result.sample_counts) * np.sum(utilities.costs))
    print(f"Saved Case 2 pilot study to {output_path.resolve()}")
    print(
        f"Nominal source-normalized pilot cost: {nominal_seconds / 3600.0:.3f} hours"
    )
    for record in result.records:
        print(
            f"  {record.iteration:02d} {record.acquisition:27s} "
            f"t={record.selected_design[0]:.6f} score={record.acquisition_value:.6g}"
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
