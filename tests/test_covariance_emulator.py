from __future__ import annotations

import numpy as np

from adaptive_covariance.covariance.emulator import (
    CovarianceEmulator,
    GaussianProcessSettings,
)
from adaptive_covariance.covariance.observations import observation_from_outputs
from adaptive_covariance.covariance.transforms import is_positive_definite


def _outputs(design: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    latent = rng.normal(size=n_samples)
    independent = rng.normal(size=(n_samples, 3))
    scales = np.array([1.0 + 0.2 * design, 0.8, 0.55 + 0.1 * design])
    loadings = np.array([1.0, 0.9 - 0.1 * design, 0.65 + 0.1 * design])
    return latent[:, None] * loadings + independent * scales * 0.15


def test_observation_and_emulator_predictions_are_finite_and_pd() -> None:
    rng = np.random.default_rng(4)
    designs = np.linspace(-1.0, 1.0, 5)[:, None]
    observations = [
        observation_from_outputs(
            _outputs(float(design), 24, rng),
            wishart_draws=40,
            rng=rng,
        )
        for design in designs[:, 0]
    ]
    emulator = CovarianceEmulator(
        GaussianProcessSettings(
            optimize_hyperparameters=False,
            n_restarts_optimizer=0,
            random_seed=4,
        )
    ).fit(
        designs,
        np.stack([item.gamma for item in observations]),
        np.stack([item.log_standard_deviations for item in observations]),
        np.stack([item.gamma_noise_variances for item in observations]),
        np.stack([item.log_standard_deviation_noise_variances for item in observations]),
    )

    prediction_designs = np.linspace(-0.9, 0.9, 7)[:, None]
    covariances = emulator.predict_covariance(prediction_designs)
    uncertainty = emulator.uncertainty_score(prediction_designs)
    assert covariances.shape == (7, 3, 3)
    assert np.all(np.isfinite(covariances))
    assert np.all(uncertainty >= 0.0)
    assert all(is_positive_definite(matrix) for matrix in covariances)

    restored = CovarianceEmulator.from_state_dict(emulator.state_dict())
    np.testing.assert_allclose(
        restored.predict_covariance(prediction_designs),
        covariances,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
