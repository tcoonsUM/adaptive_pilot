#!/usr/bin/env python3
"""Convenience wrapper for the supplied four-branch reference plot."""

from pathlib import Path

from adaptive_covariance.plotting.paper import plot_four_branch_reference

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "figures/generated"
OUTPUT.mkdir(parents=True, exist_ok=True)
print(plot_four_branch_reference(ROOT / "paper_results", OUTPUT))
