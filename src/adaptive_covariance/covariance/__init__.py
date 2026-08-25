"""Positive-definite covariance representations and emulation."""

from adaptive_covariance.covariance.emulator import CovarianceEmulator
from adaptive_covariance.covariance.observations import (
    CovarianceObservation,
    observation_from_outputs,
)
from adaptive_covariance.covariance.transforms import (
    covariance_to_unconstrained,
    gamma_forward,
    gamma_inverse,
    unconstrained_to_covariance,
)

__all__ = [
    "CovarianceEmulator",
    "CovarianceObservation",
    "covariance_to_unconstrained",
    "gamma_forward",
    "gamma_inverse",
    "observation_from_outputs",
    "unconstrained_to_covariance",
]
