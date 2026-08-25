"""Load and evaluate the supplied MLP/FNO convection–diffusion surrogate ensemble."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from adaptive_covariance.oed.physics import normalized_velocity_fields, source_fields
from adaptive_covariance.oed.scaling import StandardizedYeoJohnsonTransform

FloatArray = NDArray[np.float64]
Fidelity = Literal[0, 1, 2]


@dataclass(frozen=True)
class OEDModelManifest:
    high_checkpoint: str
    medium_checkpoint: str
    low_checkpoint: str
    mlp_output_transform: str
    mlp_bounds: str
    grid_spacing: float = 0.1
    medium_time_step: float = 0.05
    low_time_step: float = 0.1
    final_time: float = 0.6
    model_costs_seconds: tuple[float, float, float] = (31.48, 2.039, 0.65)

    @classmethod
    def load(cls, asset_root: str | Path) -> "OEDModelManifest":
        path = Path(asset_root) / "manifest.json"
        with path.open("r", encoding="utf-8") as stream:
            return cls(**json.load(stream))


class OEDSurrogateEnsemble:
    """One high-fidelity proxy MLP and two autoregressive FNO surrogates."""

    def __init__(
        self,
        *,
        asset_root: Path,
        manifest: OEDModelManifest,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "OED surrogate inference requires PyTorch. Install with `.[oed]`."
            ) from error
        from adaptive_covariance.oed.networks import (
            BoundedMLP,
            LegacyFNO2dFourLayer,
            LegacyFNOLowFidelity,
        )

        self.torch = torch
        self.device = torch.device(device)
        self.asset_root = asset_root
        self.manifest = manifest
        spacing = manifest.grid_spacing
        self.grid_1d = np.linspace(-1.0, 1.0, int(round(2.0 / spacing)) + 1)
        self.x_grid, self.y_grid = np.meshgrid(
            self.grid_1d, self.grid_1d, indexing="ij"
        )
        self.nx = self.grid_1d.size
        self.ny = self.grid_1d.size
        self.medium_times = np.arange(
            manifest.medium_time_step,
            manifest.final_time + 0.5 * manifest.medium_time_step,
            manifest.medium_time_step,
        )
        self.low_times = np.arange(
            manifest.low_time_step,
            manifest.final_time + 0.5 * manifest.low_time_step,
            manifest.low_time_step,
        )
        self.costs = np.asarray(manifest.model_costs_seconds, dtype=float)

        self.mlp_output_transform = StandardizedYeoJohnsonTransform.load(
            asset_root / manifest.mlp_output_transform
        )
        bounds = np.load(asset_root / manifest.mlp_bounds, allow_pickle=False)

        self.high_model = BoundedMLP(
            [8, 48, 64, 128, 48, 10, 1],
            activation=nn.GELU(),
            bounds=torch.as_tensor(bounds, dtype=torch.float32, device=self.device),
        ).to(self.device)
        self.medium_model = LegacyFNO2dFourLayer(
            num_channels=1,
            modes_1=12,
            modes_2=12,
            width=20,
            initial_step=1,
        ).to(self.device)
        self.low_model = LegacyFNOLowFidelity(
            num_channels=1,
            modes_1=12,
            modes_2=12,
            width=20,
            num_layers=2,
            initial_step=1,
        ).to(self.device)

        self._load_state(self.high_model, asset_root / manifest.high_checkpoint)
        self._load_state(self.medium_model, asset_root / manifest.medium_checkpoint)
        self._load_state(self.low_model, asset_root / manifest.low_checkpoint)
        self.high_model.eval()
        self.medium_model.eval()
        self.low_model.eval()

        grid = np.stack((self.x_grid, self.y_grid), axis=-1)
        self._grid_tensor = torch.as_tensor(grid, dtype=torch.float32, device=self.device)

    @classmethod
    def from_assets(
        cls,
        asset_root: str | Path,
        *,
        device: str = "cpu",
    ) -> "OEDSurrogateEnsemble":
        root = Path(asset_root).expanduser().resolve()
        return cls(asset_root=root, manifest=OEDModelManifest.load(root), device=device)

    def _load_state(self, model: Any, checkpoint_path: Path) -> None:
        try:
            checkpoint = self.torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:  # PyTorch versions before weights_only
            checkpoint = self.torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)

    @staticmethod
    def _validate_theta(parameters: ArrayLike) -> FloatArray:
        theta = np.asarray(parameters, dtype=float)
        if theta.ndim == 1:
            theta = theta[None, :]
        if theta.ndim != 2 or theta.shape[1] != 5:
            raise ValueError(f"Expected theta shape (n, 5), got {theta.shape}")
        return theta

    def predict_high(
        self,
        parameters: ArrayLike,
        sensor_locations: ArrayLike,
        times: ArrayLike,
        *,
        batch_size: int = 100_000,
    ) -> FloatArray:
        """Evaluate the pointwise high-fidelity proxy.

        Inputs may be one sensor/time shared by all parameter samples or arrays
        with one row/value per parameter sample.
        """
        theta = self._validate_theta(parameters)
        sensors = np.asarray(sensor_locations, dtype=float)
        if sensors.ndim == 1:
            sensors = np.repeat(sensors.reshape(1, 2), theta.shape[0], axis=0)
        if sensors.shape != (theta.shape[0], 2):
            raise ValueError("sensor_locations must have shape (2,) or (n_samples, 2)")
        time_values = np.asarray(times, dtype=float).reshape(-1)
        if time_values.size == 1:
            time_values = np.full(theta.shape[0], time_values.item())
        if time_values.size != theta.shape[0]:
            raise ValueError("times must be scalar or have one value per parameter sample")

        inputs = np.column_stack((theta, sensors, time_values))
        # The fitted MLP input transformer in the collaborative archive was identity.
        scaled = inputs
        predictions: list[FloatArray] = []
        with self.torch.no_grad():
            for start in range(0, theta.shape[0], batch_size):
                stop = min(start + batch_size, theta.shape[0])
                tensor = self.torch.as_tensor(
                    scaled[start:stop], dtype=self.torch.float32, device=self.device
                )
                predictions.append(
                    self.high_model(tensor).detach().cpu().numpy()
                )
        stacked = np.vstack(predictions)
        return self.mlp_output_transform.inverse_transform(stacked).reshape(-1)

    def predict_high_fields(
        self,
        parameters: ArrayLike,
        time: float,
        *,
        batch_size: int = 100_000,
    ) -> FloatArray:
        theta = self._validate_theta(parameters)
        points = np.column_stack((self.x_grid.ravel(), self.y_grid.ravel()))
        outputs = np.empty((theta.shape[0], self.nx * self.ny), dtype=float)
        for index, row in enumerate(theta):
            repeated_theta = np.repeat(row[None, :], points.shape[0], axis=0)
            outputs[index] = self.predict_high(
                repeated_theta,
                points,
                np.full(points.shape[0], time),
                batch_size=batch_size,
            )
        return outputs.reshape(theta.shape[0], self.nx, self.ny)

    def _rollout(
        self,
        parameters: FloatArray,
        fidelity: Literal[1, 2],
        *,
        batch_size: int,
    ) -> tuple[FloatArray, FloatArray]:
        theta = self._validate_theta(parameters)
        if fidelity == 1:
            model = self.medium_model
            times = self.medium_times
        else:
            model = self.low_model
            times = self.low_times

        all_fields: list[FloatArray] = []
        for start in range(0, theta.shape[0], batch_size):
            stop = min(start + batch_size, theta.shape[0])
            batch = theta[start:stop]
            source = source_fields(self.x_grid, self.y_grid, batch)
            # Both fitted FNO source transformers in the collaborative archive were identity.
            source_scaled = source
            velocity = normalized_velocity_fields(
                self.x_grid, self.y_grid, batch[:, 4]
            )
            source_tensor = self.torch.as_tensor(
                source_scaled, dtype=self.torch.float32, device=self.device
            )
            velocity_tensor = self.torch.as_tensor(
                velocity, dtype=self.torch.float32, device=self.device
            )
            grid_tensor = self._grid_tensor.unsqueeze(0).repeat(batch.shape[0], 1, 1, 1)
            features = self.torch.cat(
                (source_tensor.unsqueeze(-1), velocity_tensor, grid_tensor), dim=-1
            )
            state = self.torch.zeros(
                (batch.shape[0], self.nx, self.ny, 1, 1),
                dtype=self.torch.float32,
                device=self.device,
            )
            predictions = []
            with self.torch.no_grad():
                for _ in times:
                    flattened_state = state.reshape(batch.shape[0], self.nx, self.ny, -1)
                    next_state = model(flattened_state, features)
                    predictions.append(next_state)
                    state = self.torch.cat((state[..., 1:, :], next_state), dim=-2)
            stacked = self.torch.cat(predictions, dim=-2).squeeze(-1)
            # [batch, nx, ny, time] -> [batch, time, nx, ny]
            all_fields.append(stacked.permute(0, 3, 1, 2).cpu().numpy())
        return np.concatenate(all_fields, axis=0), times.copy()

    def predict_fno_fields(
        self,
        parameters: ArrayLike,
        time: float,
        fidelity: Literal[1, 2],
        *,
        batch_size: int = 32,
    ) -> FloatArray:
        theta = self._validate_theta(parameters)
        rollout, times = self._rollout(theta, fidelity, batch_size=batch_size)
        spline = CubicSpline(times, rollout, axis=1)
        return np.asarray(spline(float(time)), dtype=float)

    def predict_fno_sensor(
        self,
        parameters: ArrayLike,
        sensor_location: ArrayLike,
        time: float,
        fidelity: Literal[1, 2],
        *,
        batch_size: int = 32,
    ) -> FloatArray:
        theta = self._validate_theta(parameters)
        rollout, times = self._rollout(theta, fidelity, batch_size=batch_size)
        sensor = tuple(np.asarray(sensor_location, dtype=float).reshape(2))
        sensor_values = np.empty((theta.shape[0], times.size), dtype=float)
        for sample in range(theta.shape[0]):
            for time_index in range(times.size):
                interpolator = RegularGridInterpolator(
                    (self.grid_1d, self.grid_1d),
                    rollout[sample, time_index],
                    bounds_error=False,
                    fill_value=None,
                )
                sensor_values[sample, time_index] = float(interpolator(sensor))
        spline = CubicSpline(times, sensor_values, axis=1)
        return np.asarray(spline(float(time)), dtype=float).reshape(-1)

    def predict_at_sensor(
        self,
        fidelity: Fidelity,
        parameters: ArrayLike,
        sensor_location: ArrayLike,
        time: float,
        *,
        batch_size: int = 32,
    ) -> FloatArray:
        if fidelity == 0:
            return self.predict_high(parameters, sensor_location, np.array([time]))
        return self.predict_fno_sensor(
            parameters,
            sensor_location,
            time,
            fidelity,
            batch_size=batch_size,
        )

    def predict_fields(
        self,
        fidelity: Fidelity,
        parameters: ArrayLike,
        time: float,
        *,
        batch_size: int = 32,
    ) -> FloatArray:
        if fidelity == 0:
            return self.predict_high_fields(parameters, time)
        return self.predict_fno_fields(
            parameters,
            time,
            fidelity,
            batch_size=batch_size,
        )
