"""Active covariance learning for multi-fidelity stochastic optimization."""

from adaptive_covariance.covariance.emulator import CovarianceEmulator
from adaptive_covariance.pilot.study import PilotStudyConfig, PilotStudyResult, run_pilot_study

__all__ = [
    "CovarianceEmulator",
    "PilotStudyConfig",
    "PilotStudyResult",
    "run_pilot_study",
]

__version__ = "0.1.0"
