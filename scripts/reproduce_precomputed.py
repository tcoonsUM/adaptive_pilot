#!/usr/bin/env python3
"""Generate all inexpensive paper-result plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_covariance.plotting.paper import reproduce_all_precomputed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/generated"),
    )
    arguments = parser.parse_args()
    repository_root = arguments.repository_root.resolve()
    output_dir = (
        arguments.output_dir
        if arguments.output_dir.is_absolute()
        else repository_root / arguments.output_dir
    )
    paths = reproduce_all_precomputed(repository_root, output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
