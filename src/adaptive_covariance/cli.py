"""Installed command-line entry points."""

from __future__ import annotations

import argparse
import subprocess
import sys

from adaptive_covariance.config import repository_root


def _run_script(relative_path: str, argv: list[str]) -> None:
    script = repository_root() / relative_path
    completed = subprocess.run([sys.executable, str(script), *argv], check=False)
    raise SystemExit(completed.returncode)


def four_branch_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the publication four-branch pilot/optimization scripts."
    )
    parser.add_argument("stage", choices=["pilot", "flat-pilot", "optimization", "grid"])
    args, remaining = parser.parse_known_args()
    mapping = {
        "pilot": "examples/four_branch/run_pilot.py",
        "flat-pilot": "examples/four_branch/run_flat_pilot.py",
        "optimization": "examples/four_branch/run_optimization.py",
        "grid": "examples/four_branch/run_grid.py",
    }
    _run_script(mapping[args.stage], remaining)


def paper_plots_main() -> None:
    _run_script("scripts/reproduce_precomputed.py", sys.argv[1:])
