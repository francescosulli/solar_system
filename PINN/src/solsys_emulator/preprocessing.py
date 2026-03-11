"""Normalization helpers for state tensors [T, B, 6]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StateScaler:
    """Feature-wise standard scaler (default: per-body)."""

    mean: np.ndarray
    std: np.ndarray
    mode: str = "per_body_feature"

    def transform(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        return (states - self.mean) / self.std

    def inverse_transform(self, norm_states: np.ndarray) -> np.ndarray:
        norm_states = np.asarray(norm_states, dtype=float)
        return norm_states * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StateScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=float),
            std=np.asarray(payload["std"], dtype=float),
            mode=str(payload.get("mode", "global_feature")),
        )


def fit_scaler(dataset_or_states: Any) -> StateScaler:
    """Fit feature scaler from raw states or dataset dict."""
    if isinstance(dataset_or_states, dict):
        states = np.asarray(dataset_or_states["states"], dtype=float)
    else:
        states = np.asarray(dataset_or_states, dtype=float)

    if states.ndim != 3 or states.shape[-1] != 6:
        raise ValueError("Expected states shape [T, B, 6]")

    # Per-body normalization improves multi-body balance (inner vs outer planets).
    mean = states.mean(axis=0, keepdims=True)
    std = states.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return StateScaler(mean=mean, std=std, mode="per_body_feature")


def round_trip_max_error(states: np.ndarray, scaler: StateScaler) -> float:
    """Compute max absolute error for inverse(transform(x))."""
    restored = scaler.inverse_transform(scaler.transform(states))
    return float(np.max(np.abs(np.asarray(states, dtype=float) - restored)))
