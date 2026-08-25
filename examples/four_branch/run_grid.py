#!/usr/bin/env python3
"""Create or execute the full four-branch experiment grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from adaptive_covariance.config import load_yaml
from adaptive_covariance.models.four_branch import FourBranchEnsemble
from run_flat_pilot import run as run_flat_pilot
from run_optimization import run_experiment, save_result
from run_pilot import run as run_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run jobs sequentially. Without this flag, only write manifest.jsonl.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    arguments = parser.parse_args()

    grid = load_yaml(arguments.config)
    model_cost = float(FourBranchEnsemble().costs.sum())
    base_path = arguments.config.parent / grid["base_config"]
    base = load_yaml(base_path)
    jobs = []
    for budget in grid["per_iteration_budgets"]:
        for n_designs in grid["pilot_design_counts"]:
            for n_samples in grid["pilot_samples_per_design"]:
                for trial in range(int(grid["n_trials"])):
                    seed = int(grid.get("seed_start", 0)) + trial
                    job_id = (
                        f"b{int(budget)}_npilot{int(n_samples)}_npd{int(n_designs)}_seed{seed}"
                    )
                    jobs.append(
                        {
                            "job_id": job_id,
                            "budget": float(budget),
                            "n_pilot_designs": int(n_designs),
                            "n_pilot_samples": int(n_samples),
                            "seed": seed,
                            "methods": list(grid["methods"]),
                        }
                    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for index, job in enumerate(jobs):
            stream.write(json.dumps({"index": index, **job}) + "\n")
    print(f"Wrote {len(jobs)} jobs to {manifest_path}")
    if not arguments.execute:
        return

    stop = len(jobs) if arguments.stop is None else min(arguments.stop, len(jobs))
    for index in range(arguments.start, stop):
        job = jobs[index]
        job_dir = arguments.output_dir / job["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        job_config = yaml.safe_load(yaml.safe_dump(base))
        job_config["random_seed"] = job["seed"]
        job_config["pilot"]["n_active_designs"] = job["n_pilot_designs"]
        job_config["pilot"]["n_samples_per_design"] = job["n_pilot_samples"]
        job_config["pilot"]["random_seed"] = job["seed"]
        job_config["pilot"]["gp_settings"]["random_seed"] = job["seed"]
        n_initial = len(job_config["pilot"]["initial_designs"])
        initial_cost = (
            n_initial
            * int(job_config["pilot"]["n_initial_samples"])
            * model_cost
        )
        active_cost = (
            int(job["n_pilot_designs"])
            * int(job["n_pilot_samples"])
            * model_cost
        )
        n_iterations = int(job_config["optimization"]["n_iterations"])
        job_config["optimization"].pop("per_iteration_budget", None)
        job_config["optimization"]["total_budget"] = (
            initial_cost + active_cost + n_iterations * float(job["budget"])
        )
        config_path = job_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(job_config, stream, sort_keys=False)
        pilot_path = job_dir / "pilot_active.npz"
        flat_pilot_path = job_dir / "pilot_space_filling.npz"
        if not pilot_path.exists():
            run_pilot(config_path, pilot_path, seed_override=job["seed"])
        if not flat_pilot_path.exists():
            run_flat_pilot(config_path, flat_pilot_path, seed_override=job["seed"])
        for method in job["methods"]:
            output = job_dir / f"{method}.npz"
            if output.exists():
                continue
            result = run_experiment(
                config_path=config_path,
                pilot_path=pilot_path,
                method=method,
                flat_pilot_path=flat_pilot_path,
                seed_override=job["seed"],
            )
            save_result(result, output)
            print(f"[{index + 1}/{stop}] {job['job_id']} {method}")


if __name__ == "__main__":
    main()
