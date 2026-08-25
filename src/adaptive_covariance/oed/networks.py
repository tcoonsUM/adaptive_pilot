"""Neural architectures used by the supplied convection–diffusion checkpoints.

The FNO layers are adapted from the original Fourier Neural Operator implementation
and PDEBench.  This module contains architecture definitions only; model loading is
explicit in :mod:`adaptive_covariance.oed.ensemble`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as functional


class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_1: int,
        modes_2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes_1
        self.modes2 = modes_2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                modes_1,
                modes_2,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                modes_1,
                modes_2,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def _multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        coefficients = torch.fft.rfft2(inputs)
        output = torch.zeros(
            batch_size,
            self.out_channels,
            inputs.size(-2),
            inputs.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=inputs.device,
        )
        output[:, :, : self.modes1, : self.modes2] = self._multiply(
            coefficients[:, :, : self.modes1, : self.modes2], self.weights1
        )
        output[:, :, -self.modes1 :, : self.modes2] = self._multiply(
            coefficients[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(output, s=(inputs.size(-2), inputs.size(-1)))


class FNO2d(nn.Module):
    """Autoregressive 2-D FNO with a configurable number of spectral layers."""

    def __init__(
        self,
        *,
        num_channels: int = 1,
        modes_1: int = 12,
        modes_2: int = 12,
        width: int = 20,
        num_layers: int = 4,
        initial_step: int = 1,
    ) -> None:
        super().__init__()
        self.padding = 2
        self.num_layers = num_layers
        self.lift = nn.Linear(initial_step * num_channels + 5, width)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes_1, modes_2) for _ in range(num_layers)]
        )
        self.local_layers = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(num_layers)]
        )
        self.project_1 = nn.Linear(width, 128)
        self.project_2 = nn.Linear(128, num_channels)

    def forward(self, state: torch.Tensor, grid_features: torch.Tensor) -> torch.Tensor:
        values = torch.cat((state, grid_features), dim=-1)
        values = self.lift(values)
        values = values.permute(0, 3, 1, 2)
        values = functional.pad(values, [0, self.padding, 0, self.padding])
        for spectral, local in zip(self.spectral_layers, self.local_layers, strict=True):
            values = spectral(values) + local(values)
            values = functional.gelu(values)
        values = values[..., : -self.padding, : -self.padding]
        values = values.permute(0, 2, 3, 1)
        values = functional.gelu(self.project_1(values))
        values = self.project_2(values)
        return values.unsqueeze(-2)


class LegacyFNO2dFourLayer(nn.Module):
    """Four-layer state-dict-compatible architecture used by the medium checkpoint."""

    def __init__(
        self,
        *,
        num_channels: int = 1,
        modes_1: int = 12,
        modes_2: int = 12,
        width: int = 20,
        initial_step: int = 1,
    ) -> None:
        super().__init__()
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 5, width)
        self.conv0 = SpectralConv2d(width, width, modes_1, modes_2)
        self.conv1 = SpectralConv2d(width, width, modes_1, modes_2)
        self.conv2 = SpectralConv2d(width, width, modes_1, modes_2)
        self.conv3 = SpectralConv2d(width, width, modes_1, modes_2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, state: torch.Tensor, grid_features: torch.Tensor) -> torch.Tensor:
        values = torch.cat((state, grid_features), dim=-1)
        values = self.fc0(values).permute(0, 3, 1, 2)
        values = functional.pad(values, [0, self.padding, 0, self.padding])
        values = functional.gelu(self.conv0(values) + self.w0(values))
        values = functional.gelu(self.conv1(values) + self.w1(values))
        values = functional.gelu(self.conv2(values) + self.w2(values))
        values = self.conv3(values) + self.w3(values)
        values = values[..., : -self.padding, : -self.padding].permute(0, 2, 3, 1)
        values = functional.gelu(self.fc1(values))
        values = self.fc2(values)
        return values.unsqueeze(-2)


class LegacyFNOLowFidelity(nn.Module):
    """State-dict-compatible configurable-layer low-fidelity FNO."""

    def __init__(
        self,
        *,
        num_channels: int = 1,
        modes_1: int = 12,
        modes_2: int = 12,
        width: int = 20,
        num_layers: int = 2,
        initial_step: int = 1,
    ) -> None:
        super().__init__()
        self.padding = 2
        self.num_layers = num_layers
        self.fc0 = nn.Linear(initial_step * num_channels + 5, width)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes_1, modes_2) for _ in range(num_layers)]
        )
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(num_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, state: torch.Tensor, grid_features: torch.Tensor) -> torch.Tensor:
        values = torch.cat((state, grid_features), dim=-1)
        values = self.fc0(values).permute(0, 3, 1, 2)
        values = functional.pad(values, [0, self.padding, 0, self.padding])
        for spectral, local in zip(self.spectral_layers, self.conv_layers, strict=True):
            values = functional.gelu(spectral(values) + local(values))
        values = values[..., : -self.padding, : -self.padding].permute(0, 2, 3, 1)
        values = functional.gelu(self.fc1(values))
        values = self.fc2(values)
        return values.unsqueeze(-2)


class BoundedMLP(nn.Module):
    """MLP architecture used by the high-fidelity proxy checkpoint."""

    def __init__(
        self,
        dimensions: Sequence[int],
        activation: nn.Module,
        bounds: torch.Tensor,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(activation)
        self.net = nn.Sequential(*layers)
        self.bounds = bounds
        self.has_infinite_bound = bool(torch.isinf(bounds).any().item())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.net(inputs)
        if self.has_infinite_bound:
            return torch.minimum(torch.maximum(values, self.bounds[0]), self.bounds[1])
        return torch.sigmoid(values) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
