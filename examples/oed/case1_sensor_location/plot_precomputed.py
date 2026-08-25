#!/usr/bin/env python3
"""Reproduce Case 1 figures from compact supplied paper arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_covariance.config import repository_root
from adaptive_covariance.plotting.paper import plot_case1_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures/generated"))
    arguments = parser.parse_args()
    root = repository_root()
    output = arguments.output_dir
    if not output.is_absolute():
        output = root / output
    for path in plot_case1_reference(root / "paper_results", output):
        print(path)


if __name__ == "__main__":
    main()
