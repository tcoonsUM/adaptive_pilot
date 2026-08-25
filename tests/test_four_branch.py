from __future__ import annotations

import numpy as np

from adaptive_covariance.models.four_branch import FourBranchEnsemble


def test_four_branch_symmetry_and_reference_mean() -> None:
    ensemble = FourBranchEnsemble()
    mean_at_zero = ensemble.mean_by_quadrature(0.0)
    np.testing.assert_allclose(mean_at_zero, 2.1357906640170636, rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(
        ensemble.mean_by_quadrature(-0.7),
        ensemble.mean_by_quadrature(0.7),
        rtol=0.0,
        atol=2.0e-8,
    )
    assert ensemble.costs.tolist() == [1.0, 1.0e-2, 1.0e-4]
