"""Configurable active pilot-sampling loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from adaptive_covariance.covariance.emulator import (
    CovarianceEmulator,
    GaussianProcessSettings,
)
from adaptive_covariance.covariance.observations import (
    CovarianceObservation,
    observation_from_outputs,
)
from adaptive_covariance.pilot.acquisitions import (
    covariance_uncertainty,
    mean_weighted_covariance_uncertainty,
)
from adaptive_covariance.types import (
    ModelSequence,
    RandomInputSampler,
    as_2d_designs,
    evaluate_ensemble,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PilotStudyConfig:
    """Hyperparameters for an offline active pilot study."""

    n_active_designs: int
    n_samples_per_design: int
    n_initial_samples: int = 20
    acquisition_switch_fraction: float = 0.5
    wishart_draws: int = 1000
    objective_weight_mode: str = "auto"
    candidate_exclusion_radius: float = 1.0e-10
    random_seed: int = 42
    common_random_numbers_across_initial_designs: bool = False
    gp_settings: GaussianProcessSettings = field(default_factory=GaussianProcessSettings)

    def __post_init__(self) -> None:
        if self.n_active_designs < 0:
            raise ValueError("n_active_designs cannot be negative")
        if self.n_samples_per_design < 2 or self.n_initial_samples < 2:
            raise ValueError("Pilot sample counts must be at least two")
        if not 0.0 <= self.acquisition_switch_fraction <= 1.0:
            raise ValueError("acquisition_switch_fraction must be in [0, 1]")
        if self.wishart_draws < 2:
            raise ValueError("wishart_draws must be at least two")
        if self.candidate_exclusion_radius < 0.0:
            raise ValueError("candidate_exclusion_radius cannot be negative")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PilotStudyConfig":
        values = dict(data)
        gp_data = values.pop("gp_settings", {})
        return cls(gp_settings=GaussianProcessSettings(**gp_data), **values)


@dataclass(frozen=True)
class PilotIterationRecord:
    """Selection diagnostic for one active-learning iteration."""

    iteration: int
    acquisition: str
    selected_design: FloatArray
    acquisition_value: float
    predicted_objective: float
    covariance_uncertainty: float


class _ObjectiveEmulator:
    """Small heteroscedastic GP for the high-fidelity pilot means."""

    def __init__(self, random_seed: int) -> None:
        self._model: GaussianProcessRegressor | None = None
        self._minimum: FloatArray | None = None
        self._span: FloatArray | None = None
        self._random_seed = random_seed

    def _normalize(self, x: FloatArray, fitting: bool) -> FloatArray:
        if fitting:
            self._minimum = np.min(x, axis=0)
            self._span = np.max(x, axis=0) - self._minimum
            self._span[self._span <= np.finfo(float).eps] = 1.0
        if self._minimum is None or self._span is None:
            raise RuntimeError("Objective emulator normalization state is missing")
        return (x - self._minimum) / self._span

    def fit(self, designs: FloatArray, means: FloatArray, variances: FloatArray) -> None:
        x = self._normalize(designs, fitting=True)
        dimension = x.shape[1]
        kernel = ConstantKernel(1.0, (1.0e-4, 1.0e4)) * RBF(
            np.full(dimension, 0.25),
            (1.0e-3, 1.0e3),
        )
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=np.maximum(variances, 1.0e-12),
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=self._random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self._model.fit(x, means)

    def predict(self, designs: FloatArray, *, return_std: bool = False) -> Any:
        if self._model is None:
            raise RuntimeError("Objective emulator has not been fitted")
        x = self._normalize(designs, fitting=False)
        return self._model.predict(x, return_std=return_std)


@dataclass
class PilotStudyResult:
    """Data, diagnostics, and fitted emulators from an active pilot study."""

    designs: FloatArray
    gammas: FloatArray
    log_standard_deviations: FloatArray
    gamma_noise_variances: FloatArray
    log_standard_deviation_noise_variances: FloatArray
    local_covariances: FloatArray
    model_means: FloatArray
    model_mean_variances: FloatArray
    sample_counts: NDArray[np.int64]
    initial_design_count: int
    records: list[PilotIterationRecord]
    config: PilotStudyConfig
    emulator: CovarianceEmulator

    @property
    def high_fidelity_means(self) -> FloatArray:
        return self.model_means[:, 0]

    @property
    def high_fidelity_mean_variances(self) -> FloatArray:
        return self.model_mean_variances[:, 0]

    @property
    def flat_covariance(self) -> FloatArray:
        """Domain-averaged covariance used by the flat baseline."""
        return np.mean(self.local_covariances, axis=0)

    def save(self, path: str | Path) -> Path:
        """Save portable arrays; the emulator is refitted on load."""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "initial_design_count": self.initial_design_count,
            "config": {
                **asdict(self.config),
                "gp_settings": asdict(self.config.gp_settings),
            },
            "records": [
                {
                    "iteration": record.iteration,
                    "acquisition": record.acquisition,
                    "selected_design": record.selected_design.tolist(),
                    "acquisition_value": record.acquisition_value,
                    "predicted_objective": record.predicted_objective,
                    "covariance_uncertainty": record.covariance_uncertainty,
                }
                for record in self.records
            ],
        }
        np.savez_compressed(
            destination,
            designs=self.designs,
            gammas=self.gammas,
            log_standard_deviations=self.log_standard_deviations,
            gamma_noise_variances=self.gamma_noise_variances,
            log_standard_deviation_noise_variances=self.log_standard_deviation_noise_variances,
            local_covariances=self.local_covariances,
            model_means=self.model_means,
            model_mean_variances=self.model_mean_variances,
            sample_counts=self.sample_counts,
            metadata=np.array(json.dumps(metadata)),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PilotStudyResult":
        """Load saved arrays and refit the covariance emulator."""
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            config = PilotStudyConfig.from_mapping(metadata["config"])
            emulator = CovarianceEmulator(config.gp_settings).fit(
                archive["designs"],
                archive["gammas"],
                archive["log_standard_deviations"],
                archive["gamma_noise_variances"],
                archive["log_standard_deviation_noise_variances"],
            )
            records = [
                PilotIterationRecord(
                    iteration=item["iteration"],
                    acquisition=item["acquisition"],
                    selected_design=np.asarray(item["selected_design"], dtype=float),
                    acquisition_value=item["acquisition_value"],
                    predicted_objective=item["predicted_objective"],
                    covariance_uncertainty=item["covariance_uncertainty"],
                )
                for item in metadata["records"]
            ]
            return cls(
                designs=archive["designs"].copy(),
                gammas=archive["gammas"].copy(),
                log_standard_deviations=archive["log_standard_deviations"].copy(),
                gamma_noise_variances=archive["gamma_noise_variances"].copy(),
                log_standard_deviation_noise_variances=archive[
                    "log_standard_deviation_noise_variances"
                ].copy(),
                local_covariances=archive["local_covariances"].copy(),
                model_means=archive["model_means"].copy(),
                model_mean_variances=archive["model_mean_variances"].copy(),
                sample_counts=archive["sample_counts"].copy(),
                initial_design_count=int(metadata["initial_design_count"]),
                records=records,
                config=config,
                emulator=emulator,
            )


def _validate_pilot_sample_counts(config: PilotStudyConfig, n_models: int) -> None:
    minimum_full_rank_samples = n_models + 1
    if (
        config.n_initial_samples < minimum_full_rank_samples
        or config.n_samples_per_design < minimum_full_rank_samples
    ):
        raise ValueError(
            "The Wishart observation-noise approximation requires at least "
            "n_models + 1 samples in every pilot batch"
        )


def _append_observation(
    observation: CovarianceObservation,
    lists: dict[str, list[FloatArray | int]],
) -> None:
    lists["gammas"].append(observation.gamma)
    lists["log_sds"].append(observation.log_standard_deviations)
    lists["gamma_noise"].append(observation.gamma_noise_variances)
    lists["log_sd_noise"].append(observation.log_standard_deviation_noise_variances)
    lists["covariances"].append(observation.covariance)
    lists["means"].append(observation.model_means)
    lists["mean_variances"].append(observation.model_mean_variances)
    lists["sample_counts"].append(observation.n_samples)


def run_pilot_study(
    *,
    models: ModelSequence,
    sample_random_inputs: RandomInputSampler,
    candidate_designs: ArrayLike,
    initial_designs: ArrayLike,
    config: PilotStudyConfig,
) -> PilotStudyResult:
    """Run the offline active pilot-sampling stage.

    Every local batch evaluates all fidelities on common random inputs. The
    covariance and high-fidelity sample mean are both retained, so the pilot data
    can initialize a downstream objective surrogate without additional model runs.
    """
    candidates = as_2d_designs(np.asarray(candidate_designs, dtype=float))
    initial = as_2d_designs(np.asarray(initial_designs, dtype=float))
    if initial.shape[1] != candidates.shape[1]:
        raise ValueError("Initial and candidate designs must have equal dimension")
    if len(models) < 2:
        raise ValueError("Covariance emulation requires at least two model fidelities")
    _validate_pilot_sample_counts(config, len(models))

    rng = np.random.default_rng(config.random_seed)
    lists: dict[str, list[Any]] = {
        "designs": [],
        "gammas": [],
        "log_sds": [],
        "gamma_noise": [],
        "log_sd_noise": [],
        "covariances": [],
        "means": [],
        "mean_variances": [],
        "sample_counts": [],
    }

    common_initial_inputs: FloatArray | None = None
    if config.common_random_numbers_across_initial_designs:
        common_initial_inputs = sample_random_inputs(config.n_initial_samples, rng)

    for design in initial:
        random_inputs = (
            common_initial_inputs
            if common_initial_inputs is not None
            else sample_random_inputs(config.n_initial_samples, rng)
        )
        outputs = evaluate_ensemble(models, design, random_inputs)
        observation = observation_from_outputs(
            outputs,
            wishart_draws=config.wishart_draws,
            rng=rng,
        )
        lists["designs"].append(design.copy())
        _append_observation(observation, lists)

    emulator = CovarianceEmulator(config.gp_settings)
    objective_emulator = _ObjectiveEmulator(config.random_seed + 10_000)
    records: list[PilotIterationRecord] = []

    for iteration in range(config.n_active_designs):
        designs_array = np.stack(lists["designs"], axis=0)
        emulator.fit(
            designs_array,
            np.stack(lists["gammas"], axis=0),
            np.stack(lists["log_sds"], axis=0),
            np.stack(lists["gamma_noise"], axis=0),
            np.stack(lists["log_sd_noise"], axis=0),
        )
        means_array = np.stack(lists["means"], axis=0)
        mean_variances_array = np.stack(lists["mean_variances"], axis=0)
        objective_emulator.fit(
            designs_array,
            means_array[:, 0],
            mean_variances_array[:, 0],
        )

        transformed = emulator.predict_transformed(candidates)
        uncertainty = covariance_uncertainty(
            transformed.gamma_standard_deviation,
            transformed.log_standard_deviation_standard_deviation,
        )
        objective_prediction = np.asarray(objective_emulator.predict(candidates), dtype=float)
        switch_iteration = int(
            np.floor(config.n_active_designs * config.acquisition_switch_fraction)
        )
        if iteration < switch_iteration:
            acquisition_name = "uncertainty"
            acquisition = uncertainty.copy()
        else:
            acquisition_name = "mean_weighted_uncertainty"
            acquisition = mean_weighted_covariance_uncertainty(
                uncertainty,
                objective_prediction,
                weight_mode=config.objective_weight_mode,
            )

        if config.candidate_exclusion_radius > 0.0:
            for observed_design in designs_array:
                distances = np.linalg.norm(candidates - observed_design, axis=1)
                acquisition[distances <= config.candidate_exclusion_radius] = -np.inf
        if not np.any(np.isfinite(acquisition)):
            raise RuntimeError("No eligible candidate designs remain")

        selected_index = int(np.nanargmax(acquisition))
        selected_design = candidates[selected_index].copy()
        records.append(
            PilotIterationRecord(
                iteration=iteration,
                acquisition=acquisition_name,
                selected_design=selected_design,
                acquisition_value=float(acquisition[selected_index]),
                predicted_objective=float(objective_prediction[selected_index]),
                covariance_uncertainty=float(uncertainty[selected_index]),
            )
        )

        random_inputs = sample_random_inputs(config.n_samples_per_design, rng)
        outputs = evaluate_ensemble(models, selected_design, random_inputs)
        observation = observation_from_outputs(
            outputs,
            wishart_draws=config.wishart_draws,
            rng=rng,
        )
        lists["designs"].append(selected_design)
        _append_observation(observation, lists)

    designs_array = np.stack(lists["designs"], axis=0)
    emulator.fit(
        designs_array,
        np.stack(lists["gammas"], axis=0),
        np.stack(lists["log_sds"], axis=0),
        np.stack(lists["gamma_noise"], axis=0),
        np.stack(lists["log_sd_noise"], axis=0),
    )

    return PilotStudyResult(
        designs=designs_array,
        gammas=np.stack(lists["gammas"], axis=0),
        log_standard_deviations=np.stack(lists["log_sds"], axis=0),
        gamma_noise_variances=np.stack(lists["gamma_noise"], axis=0),
        log_standard_deviation_noise_variances=np.stack(
            lists["log_sd_noise"], axis=0
        ),
        local_covariances=np.stack(lists["covariances"], axis=0),
        model_means=np.stack(lists["means"], axis=0),
        model_mean_variances=np.stack(lists["mean_variances"], axis=0),
        sample_counts=np.asarray(lists["sample_counts"], dtype=np.int64),
        initial_design_count=initial.shape[0],
        records=records,
        config=config,
        emulator=emulator,
    )


def run_pilot_at_designs(
    *,
    models: ModelSequence,
    sample_random_inputs: RandomInputSampler,
    designs: ArrayLike,
    initial_design_count: int,
    config: PilotStudyConfig,
) -> PilotStudyResult:
    """Collect covariance observations at a prescribed set of designs.

    This fixed-design companion to :func:`run_pilot_study` is useful for
    space-filling and domain-averaged baselines.  The first
    ``initial_design_count`` designs use ``config.n_initial_samples``; all
    remaining designs use ``config.n_samples_per_design``.  No acquisition
    records are created.
    """
    fixed_designs = as_2d_designs(np.asarray(designs, dtype=float))
    if len(models) < 2:
        raise ValueError("Covariance emulation requires at least two model fidelities")
    _validate_pilot_sample_counts(config, len(models))
    if not 1 <= initial_design_count <= fixed_designs.shape[0]:
        raise ValueError("initial_design_count must select at least one supplied design")

    rng = np.random.default_rng(config.random_seed)
    lists: dict[str, list[Any]] = {
        "designs": [],
        "gammas": [],
        "log_sds": [],
        "gamma_noise": [],
        "log_sd_noise": [],
        "covariances": [],
        "means": [],
        "mean_variances": [],
        "sample_counts": [],
    }
    common_initial_inputs: FloatArray | None = None
    if config.common_random_numbers_across_initial_designs:
        common_initial_inputs = sample_random_inputs(config.n_initial_samples, rng)

    for index, design in enumerate(fixed_designs):
        n_samples = (
            config.n_initial_samples
            if index < initial_design_count
            else config.n_samples_per_design
        )
        random_inputs = (
            common_initial_inputs
            if index < initial_design_count and common_initial_inputs is not None
            else sample_random_inputs(n_samples, rng)
        )
        outputs = evaluate_ensemble(models, design, random_inputs)
        observation = observation_from_outputs(
            outputs,
            wishart_draws=config.wishart_draws,
            rng=rng,
        )
        lists["designs"].append(design.copy())
        _append_observation(observation, lists)

    designs_array = np.stack(lists["designs"], axis=0)
    emulator = CovarianceEmulator(config.gp_settings).fit(
        designs_array,
        np.stack(lists["gammas"], axis=0),
        np.stack(lists["log_sds"], axis=0),
        np.stack(lists["gamma_noise"], axis=0),
        np.stack(lists["log_sd_noise"], axis=0),
    )
    return PilotStudyResult(
        designs=designs_array,
        gammas=np.stack(lists["gammas"], axis=0),
        log_standard_deviations=np.stack(lists["log_sds"], axis=0),
        gamma_noise_variances=np.stack(lists["gamma_noise"], axis=0),
        log_standard_deviation_noise_variances=np.stack(
            lists["log_sd_noise"], axis=0
        ),
        local_covariances=np.stack(lists["covariances"], axis=0),
        model_means=np.stack(lists["means"], axis=0),
        model_mean_variances=np.stack(lists["mean_variances"], axis=0),
        sample_counts=np.asarray(lists["sample_counts"], dtype=np.int64),
        initial_design_count=initial_design_count,
        records=[],
        config=config,
        emulator=emulator,
    )
