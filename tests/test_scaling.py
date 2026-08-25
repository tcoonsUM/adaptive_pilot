from __future__ import annotations

from pathlib import Path

import numpy as np

from adaptive_covariance.oed.scaling import (
    StandardizedYeoJohnsonTransform,
    yeo_johnson_inverse,
    yeo_johnson_transform,
)


def test_yeo_johnson_round_trip_for_positive_and_negative_values() -> None:
    values = np.array([-3.0, -0.5, 0.0, 0.25, 2.0])
    for power in (-2.88937282, 0.0, 1.5, 2.0):
        transformed = yeo_johnson_transform(values, power)
        recovered = yeo_johnson_inverse(transformed, power)
        np.testing.assert_allclose(recovered, values, rtol=1.0e-12, atol=1.0e-12)


def test_portable_mlp_output_transform_loads_and_round_trips() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "oed"
        / "scalers"
        / "mlp_output_transform.npz"
    )
    transform = StandardizedYeoJohnsonTransform.load(path)
    values = np.array([[0.0], [0.01], [0.1], [0.5]])
    np.testing.assert_allclose(
        transform.inverse_transform(transform.transform(values)),
        values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
