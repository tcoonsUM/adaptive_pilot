"""Lightweight stochastic-optimization utilities."""

from adaptive_covariance.optimization.discrete_bo import (
    NoisyBayesianOptimizer,
    NoisyBayesianOptimizerConfig,
)

__all__ = ["NoisyBayesianOptimizer", "NoisyBayesianOptimizerConfig"]
