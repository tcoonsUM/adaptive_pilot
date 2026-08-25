from __future__ import annotations

from functools import partial

import numpy as np

from adaptive_covariance.covariance.emulator import GaussianProcessSettings
from adaptive_covariance.pilot import PilotStudyConfig, PilotStudyResult, run_pilot_study


def _model(design: np.ndarray, random_inputs: np.ndarray, *, slope: float, noise: float) -> np.ndarray:
    x = float(design[0])
    z = random_inputs[:, 0]
    return (1.0 + slope * x) * z + noise * z**2 + 2.0 - x**2


def _sample(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(n, 1))


def test_active_pilot_loop_save_and_load(tmp_path) -> None:
    models = [
        partial(_model, slope=0.2, noise=0.1),
        partial(_model, slope=0.15, noise=0.08),
        partial(_model, slope=-0.05, noise=0.05),
    ]
    config = PilotStudyConfig(
        n_active_designs=3,
        n_samples_per_design=10,
        n_initial_samples=10,
        wishart_draws=30,
        random_seed=9,
        gp_settings=GaussianProcessSettings(
            optimize_hyperparameters=False,
            n_restarts_optimizer=0,
            random_seed=9,
        ),
    )
    result = run_pilot_study(
        models=models,
        sample_random_inputs=_sample,
        candidate_designs=np.linspace(-1.0, 1.0, 31)[:, None],
        initial_designs=np.array([[-1.0], [1.0]]),
        config=config,
    )
    assert result.designs.shape == (5, 1)
    assert len(result.records) == 3
    assert {record.acquisition for record in result.records} == {
        "uncertainty",
        "mean_weighted_uncertainty",
    }
    assert result.flat_covariance.shape == (3, 3)

    path = result.save(tmp_path / "pilot.npz")
    loaded = PilotStudyResult.load(path)
    np.testing.assert_allclose(loaded.designs, result.designs)
    np.testing.assert_allclose(
        loaded.emulator.predict_covariance([[0.1]]),
        result.emulator.predict_covariance([[0.1]]),
    )


def test_fixed_design_pilot_uses_requested_sample_counts() -> None:
    from adaptive_covariance.pilot import run_pilot_at_designs

    models = [
        partial(_model, slope=0.2, noise=0.1),
        partial(_model, slope=0.15, noise=0.08),
        partial(_model, slope=-0.05, noise=0.05),
    ]
    config = PilotStudyConfig(
        n_active_designs=2,
        n_samples_per_design=8,
        n_initial_samples=6,
        wishart_draws=20,
        random_seed=11,
        gp_settings=GaussianProcessSettings(
            optimize_hyperparameters=False,
            n_restarts_optimizer=0,
            random_seed=11,
        ),
    )
    result = run_pilot_at_designs(
        models=models,
        sample_random_inputs=_sample,
        designs=np.array([[-1.0], [1.0], [0.0], [0.5]]),
        initial_design_count=2,
        config=config,
    )
    np.testing.assert_array_equal(result.sample_counts, [6, 6, 8, 8])
    assert result.records == []
    assert result.designs.shape == (4, 1)
    assert result.emulator.predict_covariance([[0.25]]).shape == (1, 3, 3)
