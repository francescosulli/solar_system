"""Deterministic gravity-field computations from mu and body states."""

from __future__ import annotations

from typing import Mapping, Sequence

import astropy.units as u
import numpy as np


def _to_xyz(vector: object) -> np.ndarray:
    if hasattr(vector, "to_value"):
        return np.asarray(vector.to_value(u.km), dtype=float)
    return np.asarray(vector, dtype=float)


def _extract_position(entry: object) -> np.ndarray:
    if isinstance(entry, dict):
        if "r" in entry:
            return _to_xyz(entry["r"])
        if "position" in entry:
            return _to_xyz(entry["position"])
        raise KeyError("State dict entries must contain key 'r' or 'position'")
    return _to_xyz(entry)


def _mu_to_float(mu_value: object) -> float:
    if hasattr(mu_value, "to_value"):
        return float(mu_value.to_value(u.km**3 / u.s**2))
    return float(mu_value)


def _prepare_inputs(
    point_r: object,
    states_at_t: object,
    mu_values: object,
    body_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = _to_xyz(point_r).reshape(3)

    if isinstance(states_at_t, Mapping):
        keys = list(body_order) if body_order is not None else list(states_at_t.keys())
        positions = np.vstack([_extract_position(states_at_t[body]) for body in keys]).astype(float)
    else:
        raw = np.asarray(states_at_t, dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 3:
            raise ValueError("states_at_t must be dict or array with shape [B,3] or [B,6]")
        positions = raw[:, :3]
        keys = list(body_order) if body_order is not None else [f"body_{i}" for i in range(len(positions))]

    if isinstance(mu_values, Mapping):
        mu = np.asarray([_mu_to_float(mu_values[k]) for k in keys], dtype=float)
    else:
        mu = np.asarray(mu_values, dtype=float).reshape(-1)
        if mu.shape[0] != positions.shape[0]:
            raise ValueError("mu_values length must match number of bodies")

    return point, positions, mu


def acceleration(
    point_r: object,
    states_at_t: object,
    mu_values: object,
    epsilon: float = 0.0,
    body_order: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute total gravitational acceleration vector in km/s^2."""
    point, positions, mu = _prepare_inputs(point_r, states_at_t, mu_values, body_order=body_order)
    delta = positions - point[None, :]
    dist_sq = np.sum(delta * delta, axis=1) + float(epsilon) ** 2
    if np.any(dist_sq <= 0):
        raise ValueError("Encountered zero distance; use epsilon > 0 if needed")

    inv_dist3 = np.power(dist_sq, -1.5)
    weighted = mu[:, None] * delta * inv_dist3[:, None]
    return weighted.sum(axis=0)


def potential(
    point_r: object,
    states_at_t: object,
    mu_values: object,
    epsilon: float = 0.0,
    body_order: Sequence[str] | None = None,
) -> float:
    """Compute total gravitational potential in km^2/s^2."""
    point, positions, mu = _prepare_inputs(point_r, states_at_t, mu_values, body_order=body_order)
    delta = positions - point[None, :]
    dist_sq = np.sum(delta * delta, axis=1) + float(epsilon) ** 2
    if np.any(dist_sq <= 0):
        raise ValueError("Encountered zero distance; use epsilon > 0 if needed")

    dist = np.sqrt(dist_sq)
    return float(-np.sum(mu / dist))
