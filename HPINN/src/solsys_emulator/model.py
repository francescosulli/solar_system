"""PyTorch ephemeris emulator model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Model hyper-parameters."""

    num_bodies: int
    state_mode: str = "full"
    backbone_type: str = "plain"
    hidden_dim: int = 256
    num_layers: int = 4
    fourier_features: int = 16
    min_frequency: float = 1.0
    max_frequency: float = 8.0
    frequency_spacing: str = "linear"
    head_layers: int = 1
    head_hidden_dim: int = 128
    body_embedding_dim: int = 0
    interaction_layers: int = 0
    interaction_hidden_dim: int = 128
    hybrid_correction: bool = False
    correction_layers: int = 2
    correction_hidden_dim: int = 128
    correction_init_scale: float = 0.02
    use_layer_norm: bool = False
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

    class ResidualBlock(nn.Module):
        """Simple residual MLP block for deeper backbones."""

        def __init__(self, hidden_dim: int, dropout: float = 0.0, use_layer_norm: bool = False) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.act = nn.SiLU()
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            y = self.norm(x)
            y = self.fc1(y)
            y = self.act(y)
            y = self.dropout(y)
            y = self.fc2(y)
            return residual + y

    class InteractionBlock(nn.Module):
        """Lightweight interaction-network block over body features."""

        def __init__(
            self,
            feature_dim: int,
            hidden_dim: int,
            dropout: float = 0.0,
            use_layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.node_norm = nn.LayerNorm(feature_dim) if use_layer_norm else nn.Identity()
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, feature_dim),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(2 * feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, feature_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 3:
                raise ValueError("InteractionBlock expects input [N, B, F]")
            n_bodies = int(x.shape[1])
            if n_bodies < 2:
                return x

            x_norm = self.node_norm(x)
            x_i = x_norm.unsqueeze(2).expand(-1, -1, n_bodies, -1)
            x_j = x_norm.unsqueeze(1).expand(-1, n_bodies, -1, -1)
            pair_features = torch.cat((x_i, x_j), dim=-1)
            messages = self.edge_mlp(pair_features)

            eye = torch.eye(n_bodies, dtype=torch.bool, device=x.device).view(1, n_bodies, n_bodies, 1)
            messages = messages.masked_fill(eye, 0.0)
            aggregated = messages.sum(dim=2) / float(max(1, n_bodies - 1))
            update = self.node_mlp(torch.cat((x_norm, aggregated), dim=-1))
            return x + update

    def __init__(
        self,
        num_bodies: int,
        state_mode: str = "full",
        backbone_type: str = "plain",
        hidden_dim: int = 256,
        num_layers: int = 4,
        fourier_features: int = 16,
        min_frequency: float = 1.0,
        max_frequency: float = 8.0,
        frequency_spacing: str = "linear",
        head_layers: int = 1,
        head_hidden_dim: int = 128,
        body_embedding_dim: int = 0,
        interaction_layers: int = 0,
        interaction_hidden_dim: int = 128,
        hybrid_correction: bool = False,
        correction_layers: int = 2,
        correction_hidden_dim: int = 128,
        correction_init_scale: float = 0.02,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_bodies <= 0:
            raise ValueError("num_bodies must be positive")
        if state_mode not in {"full", "position_only"}:
            raise ValueError("state_mode must be 'full' or 'position_only'")
        if backbone_type not in {"plain", "residual"}:
            raise ValueError("backbone_type must be 'plain' or 'residual'")
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
        if body_embedding_dim < 0:
            raise ValueError("body_embedding_dim must be >= 0")
        if interaction_layers < 0:
            raise ValueError("interaction_layers must be >= 0")
        if interaction_hidden_dim < 8:
            raise ValueError("interaction_hidden_dim must be >= 8")
        if correction_layers < 1:
            raise ValueError("correction_layers must be >= 1")
        if correction_hidden_dim < 8:
            raise ValueError("correction_hidden_dim must be >= 8")
        if correction_init_scale <= 0.0:
            raise ValueError("correction_init_scale must be > 0")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if hybrid_correction and state_mode != "position_only":
            raise ValueError("hybrid_correction currently requires state_mode='position_only'")

        self.num_bodies = int(num_bodies)
        self.state_mode = str(state_mode)
        self.backbone_type = str(backbone_type)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.fourier_features = int(fourier_features)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.frequency_spacing = str(frequency_spacing)
        self.head_layers = int(head_layers)
        self.head_hidden_dim = int(head_hidden_dim)
        self.body_embedding_dim = int(body_embedding_dim)
        self.interaction_layers = int(interaction_layers)
        self.interaction_hidden_dim = int(interaction_hidden_dim)
        self.hybrid_correction = bool(hybrid_correction)
        self.correction_layers = int(correction_layers)
        self.correction_hidden_dim = int(correction_hidden_dim)
        self.correction_init_scale = float(correction_init_scale)
        self.use_layer_norm = bool(use_layer_norm)
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
        if self.backbone_type == "plain":
            layers: list[nn.Module] = []
            in_dim = input_dim
            for _ in range(self.num_layers):
                layers.append(nn.Linear(in_dim, self.hidden_dim))
                layers.append(nn.SiLU())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
                in_dim = self.hidden_dim
            self.backbone = nn.Sequential(*layers)
        else:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                *[
                    self.ResidualBlock(
                        hidden_dim=self.hidden_dim,
                        dropout=self.dropout,
                        use_layer_norm=self.use_layer_norm,
                    )
                    for _ in range(self.num_layers)
                ],
            )
        if self.body_embedding_dim > 0:
            self.body_embeddings = nn.Embedding(self.num_bodies, self.body_embedding_dim)
            nn.init.normal_(self.body_embeddings.weight, mean=0.0, std=0.02)
        else:
            self.body_embeddings = None
        interaction_feature_dim = self.hidden_dim + self.body_embedding_dim
        if self.interaction_layers > 0:
            self.interaction_blocks = nn.ModuleList(
                [
                    self.InteractionBlock(
                        feature_dim=interaction_feature_dim,
                        hidden_dim=self.interaction_hidden_dim,
                        dropout=self.dropout,
                        use_layer_norm=self.use_layer_norm,
                    )
                    for _ in range(self.interaction_layers)
                ]
            )
        else:
            self.interaction_blocks = nn.ModuleList()
        out_dim = 6 if self.state_mode == "full" else 3
        self.heads = nn.ModuleList(
            [
                self._make_head(
                    in_dim=interaction_feature_dim,
                    out_dim=out_dim,
                    num_layers=self.head_layers,
                    hidden_dim=self.head_hidden_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_bodies)
            ]
        )
        if self.hybrid_correction:
            self.correction_heads = nn.ModuleList(
                [
                    self._make_head(
                        in_dim=interaction_feature_dim,
                        out_dim=3,
                        num_layers=self.correction_layers,
                        hidden_dim=self.correction_hidden_dim,
                        dropout=self.dropout,
                    )
                    for _ in range(self.num_bodies)
                ]
            )
            init_unconstrained = torch.log(torch.expm1(torch.tensor(self.correction_init_scale, dtype=torch.float32)))
            self.correction_gain_unconstrained = nn.Parameter(
                torch.full((self.num_bodies, 1), float(init_unconstrained.item()))
            )
        else:
            self.correction_heads = nn.ModuleList()
            self.register_parameter("correction_gain_unconstrained", None)

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
            "state_mode": self.state_mode,
            "backbone_type": self.backbone_type,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "fourier_features": self.fourier_features,
            "min_frequency": self.min_frequency,
            "max_frequency": self.max_frequency,
            "frequency_spacing": self.frequency_spacing,
            "head_layers": self.head_layers,
            "head_hidden_dim": self.head_hidden_dim,
            "body_embedding_dim": self.body_embedding_dim,
            "interaction_layers": self.interaction_layers,
            "interaction_hidden_dim": self.interaction_hidden_dim,
            "hybrid_correction": self.hybrid_correction,
            "correction_layers": self.correction_layers,
            "correction_hidden_dim": self.correction_hidden_dim,
            "correction_init_scale": self.correction_init_scale,
            "use_layer_norm": self.use_layer_norm,
            "dropout": self.dropout,
        }

    def correction_gains(self) -> torch.Tensor:
        if not self.hybrid_correction:
            raise ValueError("correction_gains is available only when hybrid_correction=True")
        return torch.nn.functional.softplus(self.correction_gain_unconstrained)

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """Encode normalized time scalar(s) with Fourier features."""
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError("t must have shape [N] or [N,1]")

        phases = t * self.frequencies.unsqueeze(0)
        return torch.cat((t, torch.sin(phases), torch.cos(phases)), dim=-1)

    def forward_components(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with decomposition.

        Returns:
        - final output
        - base output
        - correction output (zeros if disabled)
        """
        features = self.encode_time(t)
        shared = self.backbone(features)
        body_inputs = []
        for body_idx in range(self.num_bodies):
            if self.body_embeddings is not None:
                emb = self.body_embeddings.weight[body_idx].unsqueeze(0).expand(shared.shape[0], -1)
                head_input = torch.cat((shared, emb), dim=-1)
            else:
                head_input = shared
            body_inputs.append(head_input)
        body_features = torch.stack(body_inputs, dim=1)
        for block in self.interaction_blocks:
            body_features = block(body_features)

        body_outputs = []
        for body_idx, head in enumerate(self.heads):
            body_outputs.append(head(body_features[:, body_idx, :]))
        base_output = torch.stack(body_outputs, dim=1)
        if not self.hybrid_correction:
            correction = torch.zeros_like(base_output)
            return base_output, base_output, correction

        raw_corrections = []
        for body_idx, head in enumerate(self.correction_heads):
            raw_corrections.append(head(body_features[:, body_idx, :]))
        raw_correction = torch.stack(raw_corrections, dim=1)
        gains = self.correction_gains().unsqueeze(0)
        correction = raw_correction * gains
        final_output = base_output + correction
        return final_output, base_output, correction

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns:
        - state_mode='full': normalized states [N, B, 6]
        - state_mode='position_only': normalized positions [N, B, 3]
        """
        final_output, _, _ = self.forward_components(t)
        return final_output
