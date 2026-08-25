from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adaptive_covariance.oed import OEDSurrogateEnsemble


def test_supplied_oed_checkpoints_load_and_predict() -> None:
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1] / "assets" / "oed"
    ensemble = OEDSurrogateEnsemble.from_assets(root)
    theta = np.array(
        [
            [-0.8, -0.2, 0.4, 0.3, 7.0],
            [-0.4, 0.5, 0.7, -0.6, 10.0],
        ]
    )
    predictions = [
        ensemble.predict_at_sensor(
            fidelity,
            theta,
            (-0.8, -0.2),
            0.3,
            batch_size=2,
        )
        for fidelity in range(3)
    ]
    assert all(values.shape == (2,) for values in predictions)
    assert np.all(np.isfinite(np.stack(predictions)))
