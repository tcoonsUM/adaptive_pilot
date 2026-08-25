"""Reproduce the principal numerical plots from compact supplied arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Plot reproduction requires the `plots` optional dependency") from error
    return plt


def plot_four_branch_reference(result_root: Path, output_dir: Path) -> Path:
    plt = _matplotlib()
    with np.load(result_root / "four_branch/reference_quantities.npz", allow_pickle=False) as data:
        xi = data["xi"]
        mean = data["high_fidelity_mean"]
        correlation = data["correlation"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xi, mean)
    axes[0].axvline(0.0, linestyle="--", linewidth=1)
    axes[0].set_xlabel(r"Design $\xi$")
    axes[0].set_ylabel(r"$U_0(\xi)$")
    axes[0].set_title("High-fidelity mean")
    axes[0].grid(alpha=0.3)
    axes[1].plot(xi, correlation[:, 0, 1], label=r"$\rho_{0,1}$")
    axes[1].plot(xi, correlation[:, 0, 2], label=r"$\rho_{0,2}$")
    axes[1].plot(xi, correlation[:, 1, 2], label=r"$\rho_{1,2}$")
    axes[1].set_xlabel(r"Design $\xi$")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Model-output correlations")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    destination = output_dir / "four_branch_reference.png"
    figure.savefig(destination, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_case1_reference(result_root: Path, output_dir: Path) -> list[Path]:
    plt = _matplotlib()
    with np.load(result_root / "oed_case1/case1_precomputed.npz", allow_pickle=False) as data:
        grid_x = data["grid_x"]
        grid_y = data["grid_y"]
        oracle = data["oracle_eig"].reshape(grid_x.size, grid_y.size)
        correlations = data["local_utility_correlation"]
        regrets = {
            "NMC-SF": data["regret_nmc"],
            "MFEIG-ADAPT": data["regret_mfeig_adapt"],
            "MFEIG-FLAT": data["regret_mfeig_flat"],
        }

    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    image = axis.imshow(
        oracle.T,
        origin="lower",
        extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]),
        aspect="equal",
    )
    optimum = np.unravel_index(int(np.argmax(oracle)), oracle.shape)
    axis.plot(grid_x[optimum[0]], grid_y[optimum[1]], marker="x", markersize=9)
    axis.set_xlabel(r"$x_1$")
    axis.set_ylabel(r"$x_2$")
    axis.set_title("Case 1 reference EIG")
    figure.colorbar(image, ax=axis, label="EIG")
    figure.tight_layout()
    oracle_path = output_dir / "oed_case1_reference_eig.png"
    figure.savefig(oracle_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    pairs = ((0, 1), (0, 2), (1, 2))
    last_image = None
    for axis, (left, right) in zip(axes, pairs, strict=True):
        field = correlations[left, right].reshape(grid_x.size, grid_y.size)
        last_image = axis.imshow(
            field.T,
            origin="lower",
            extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]),
            vmin=np.min(correlations),
            vmax=np.max(correlations),
            aspect="equal",
        )
        axis.set_title(rf"$\rho_{{{left},{right}}}$")
        axis.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    if last_image is not None:
        figure.colorbar(last_image, ax=axes, shrink=0.82, label="Correlation")
    figure.subplots_adjust(wspace=0.18, right=0.88)
    correlation_path = output_dir / "oed_case1_utility_correlations.png"
    figure.savefig(correlation_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    labels = list(regrets)
    data_values = [regrets[label] for label in labels]
    axis.boxplot(data_values, tick_labels=labels, showfliers=False)
    axis.set_ylabel("Cost-aware EIG regret")
    axis.set_title("Case 1 trial-level regret")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    regret_path = output_dir / "oed_case1_regret_boxplot.png"
    figure.savefig(regret_path, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return [oracle_path, correlation_path, regret_path]


def plot_case2_reference(result_root: Path, output_dir: Path) -> Path:
    plt = _matplotlib()
    with np.load(result_root / "oed_case2/reference_quantities.npz", allow_pickle=False) as data:
        xi = data["xi"]
        utility_means = data["cost_aware_utility_means"]
        correlations = data["utility_correlations"]
        gamma = data["gamma"]
        log_sd = data["log_standard_deviations"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    labels = ("MLP", "FNO-M", "FNO-L")
    for index, label in enumerate(labels):
        axes[0, 0].plot(xi, utility_means[:, index], label=label)
    axes[0, 0].set_ylabel(r"$U_c(\xi)$")
    axes[0, 0].set_title("Cost-aware utility means")
    axes[0, 0].legend()
    axes[0, 1].plot(xi, correlations[:, 0, 1], label=r"$\rho_{0,1}$")
    axes[0, 1].plot(xi, correlations[:, 0, 2], label=r"$\rho_{0,2}$")
    axes[0, 1].plot(xi, correlations[:, 1, 2], label=r"$\rho_{1,2}$")
    axes[0, 1].set_ylabel(r"$\rho$")
    axes[0, 1].set_title("Utility-model correlations")
    axes[0, 1].legend()
    for index in range(gamma.shape[1]):
        axes[1, 0].plot(xi, gamma[:, index], label=rf"$\gamma_{index + 1}$")
    axes[1, 0].set_ylabel(r"$\gamma$")
    axes[1, 0].set_title("Correlation parameters")
    axes[1, 0].legend()
    for index in range(log_sd.shape[1]):
        axes[1, 1].plot(xi, log_sd[:, index], label=rf"$\ell_{index}$")
    axes[1, 1].set_ylabel(r"$\ell$")
    axes[1, 1].set_title("Log standard deviations")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel(r"Measurement time $\xi$")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    destination = output_dir / "oed_case2_reference_quantities.png"
    figure.savefig(destination, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_case2_regret(result_root: Path, output_dir: Path) -> Path:
    plt = _matplotlib()
    with np.load(result_root / "oed_case2/regret_curves.npz", allow_pickle=False) as data:
        runtime_acv = data["runtime_hours_acv"]
        runtime_sf = data["runtime_hours_sf"]
        regret_acv = data["regret_trials_acv"]
        regret_sf = data["regret_trials_sf"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(runtime_acv, np.mean(regret_acv, axis=0), marker="o", label=r"MFEIG-$\gamma$-OPT")
    axis.plot(runtime_sf, np.mean(regret_sf, axis=0), marker="o", label="NMC-SF")
    axis.set_xlabel("Run time (hours)")
    axis.set_ylabel("Cost-aware EIG regret")
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    destination = output_dir / "oed_case2_regret_runtime.png"
    figure.savefig(destination, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return destination


def reproduce_all_precomputed(repository_root: Path, output_dir: Path) -> list[Path]:
    result_root = repository_root / "paper_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_four_branch_reference(result_root, output_dir),
        *plot_case1_reference(result_root, output_dir),
        plot_case2_reference(result_root, output_dir),
        plot_case2_regret(result_root, output_dir),
    ]
    return paths
