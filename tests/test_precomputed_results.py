from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_compact_paper_result_files_are_numeric_and_consistent() -> None:
    root = Path(__file__).resolve().parents[1] / "paper_results"
    for archive_path in root.rglob("*.npz"):
        with np.load(archive_path, allow_pickle=False) as archive:
            assert archive.files
            for key in archive.files:
                assert archive[key].dtype != object, (archive_path, key)

    table = pd.read_csv(root / "oed_case1" / "table5_summary.csv")
    assert set(table["estimator"]) == {"NMC-SF", "MFEIG-ADAPT", "MFEIG-FLAT"}


def test_reported_summary_tables_match_manuscript_values() -> None:
    root = Path(__file__).resolve().parents[1] / "paper_results"

    table2 = pd.read_csv(root / "four_branch" / "table2_regret_reduction.csv")
    gamma_12 = table2[
        (table2["n_pilot_designs"] == 12)
        & (table2["method"] == "ACV-gamma-OPT")
    ].iloc[0]
    np.testing.assert_allclose(
        gamma_12[
            [
                "20_percent_budget_factor",
                "50_percent_budget_factor",
                "100_percent_budget_factor",
            ]
        ].to_numpy(dtype=float),
        [2.6, 13.6, 35.8],
    )

    table5 = pd.read_csv(root / "oed_case1" / "table5_summary.csv").set_index(
        "estimator"
    )
    np.testing.assert_allclose(
        table5.loc["MFEIG-ADAPT"].to_numpy(dtype=float),
        [0.0455, 0.0227, 0.0240],
    )


def test_reference_archives_have_expected_problem_shapes() -> None:
    root = Path(__file__).resolve().parents[1] / "paper_results"
    with np.load(root / "four_branch" / "reference_quantities.npz") as archive:
        assert archive["xi"].shape == (101,)
        assert archive["covariance"].shape == (101, 3, 3)
        assert float(archive["true_optimizer"]) == 0.0

    with np.load(root / "oed_case1" / "case1_precomputed.npz") as archive:
        assert archive["oracle_eig"].shape == (441,)
        assert archive["regret_nmc"].shape == (60,)
        assert int(archive["true_optimum_flat_index"]) == 50

    with np.load(root / "oed_case2" / "regret_curves.npz") as archive:
        assert archive["regret_trials_acv"].shape == (10, 21)
        assert archive["regret_trials_sf"].shape == (50, 20)
