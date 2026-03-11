"""PyTorch ephemeris emulator model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Model hyper-parameters."""

    num_bodies: int
    hidden_dim: int = 256
    num_layers: int = 4
    fourier_features: int = 16
    min_frequency: float = 1.0
    max_frequency: float = 8.0
    frequency_spacing: str = "linear"
    head_layers: int = 1
    head_hidden_dim: int = 128
    dropout: float = 0.0

    def to_kwargs(self) -> dict:
        return asdict(self)


class EmulatorModel(nn.Module):
    """
    MLP with Fourier time features and one head per body.

    Architecture stays backward compatible with historical checkpoints:
    - shared backbone in ``self.backbone`` (nn.Sequential)
    - per-body heads in ``self.heads`` (ModuleList)
    """

    def __init__(
        self,
        num_bodies: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        fourier_features: int = 16,
        min_frequency: float = 1.0,
        max_frequency: float = 8.0,
        frequency_spacing: str = "linear",
        head_layers: int = 1,
        head_hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_bodies <= 0:
            raise ValueError("num_bodies must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if fourier_features < 1:
            raise ValueError("fourier_features must be >= 1")
        if min_frequency <= 0.0 or max_frequency <= 0.0:
            raise ValueError("frequencies must be positive")
        if min_frequency > max_frequency:
            raise ValueError("min_frequency must be <= max_frequency")
        if frequency_spacing not in {"linear", "log"}:
            raise ValueError("frequency_spacing must be 'linear' or 'log'")
        if head_layers < 1:
            raise ValueError("head_layers must be >= 1")
        if head_hidden_dim < 8:
            raise ValueError("head_hidden_dim must be >= 8")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_bodies = int(num_bodies)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.fourier_features = int(fourier_features)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.frequency_spacing = str(frequency_spacing)
        self.head_layers = int(head_layers)
        self.head_hidden_dim = int(head_hidden_dim)
        self.dropout = float(dropout)

        if self.frequency_spacing == "log":
            if self.fourier_features == 1:
                base = torch.tensor([self.max_frequency], dtype=torch.float32)
            else:
                base = torch.logspace(
                    start=torch.log10(torch.tensor(self.min_frequency)).item(),
                    end=torch.log10(torch.tensor(self.max_frequency)).item(),
                    steps=self.fourier_features,
                )
        else:
            base = torch.linspace(self.min_frequency, self.max_frequency, steps=self.fourier_features)

        frequencies = base * (2.0 * torch.pi)
        self.register_buffer("frequencies", frequencies, persistent=False)

        input_dim = 1 + 2 * self.fourier_features
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [
                self._make_head(
                    in_dim=self.hidden_dim,
                    out_dim=6,
                    num_layers=self.head_layers,
                    hidden_dim=self.head_hidden_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_bodies)
            ]
        )

    @staticmethod
    def _make_head(
        in_dim: int,
        out_dim: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
    ) -> nn.Module:
        if num_layers == 1:
            return nn.Linear(in_dim, out_dim)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def model_kwargs(self) -> dict:
        return {
            "num_bodies": self.num_bodies,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "fourier_features": self.fourier_features,
            "min_frequency": self.min_frequency,
            "max_frequency": self.max_frequency,
            "frequency_spacing": self.frequency_spacing,
            "head_layers": self.head_layers,
            "head_hidden_dim": self.head_hidden_dim,
            "dropout": self.dropout,
        }

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """Encode normalized time scalar(s) with Fourier features."""
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError("t must have shape [N] or [N,1]")

        phases = t * self.frequencies.unsqueeze(0)
        return torch.cat((t, torch.sin(phases), torch.cos(phases)), dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass returning normalized states with shape [N, B, 6]."""
        features = self.encode_time(t)
        shared = self.backbone(features)
        body_outputs = [head(shared) for head in self.heads]
        return torch.stack(body_outputs, dim=1)
