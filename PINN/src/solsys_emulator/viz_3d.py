"""Interactive 3D visualization helpers (Plotly)."""

from __future__ import annotations

from typing import Mapping

import astropy.units as u
import numpy as np
import plotly.graph_objects as go


def _to_km(position: object) -> np.ndarray:
    if hasattr(position, "to_value"):
        return np.asarray(position.to_value(u.km), dtype=float)
    return np.asarray(position, dtype=float)


def _state_position(entry: object) -> np.ndarray:
    if isinstance(entry, dict):
        if "r" in entry:
            return _to_km(entry["r"])
        if "position" in entry:
            return _to_km(entry["position"])
        raise KeyError("State entries must include 'r' or 'position'")
    return _to_km(entry)


def _body_colors(body_names: list[str]) -> dict[str, str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return {name: palette[idx % len(palette)] for idx, name in enumerate(body_names)}


def plot_scene(
    states_at_t: Mapping[str, object] | np.ndarray,
    trajectories: Mapping[str, np.ndarray] | None = None,
    reference_trajectories: Mapping[str, np.ndarray] | None = None,
    reference_label: str = "DE440",
    frame_label: str = "barycentric icrs",
    units_label: str = "km",
) -> go.Figure:
    """Build a Plotly 3D scene with trajectories and body markers."""
    figure = go.Figure()

    names_for_colors: list[str] = []
    if isinstance(states_at_t, Mapping):
        names_for_colors.extend([str(k) for k in states_at_t.keys()])
    if trajectories:
        names_for_colors.extend([str(k) for k in trajectories.keys()])
    if reference_trajectories:
        names_for_colors.extend([str(k) for k in reference_trajectories.keys()])
    color_map = _body_colors(sorted(set(names_for_colors)))

    if trajectories:
        for body, path in trajectories.items():
            path_arr = np.asarray(path, dtype=float)
            if path_arr.size == 0:
                continue
            figure.add_trace(
                go.Scatter3d(
                    x=path_arr[:, 0],
                    y=path_arr[:, 1],
                    z=path_arr[:, 2],
                    mode="lines",
                    name=f"{body} PINN",
                    line={"width": 4, "color": color_map.get(body)},
                )
            )

    if reference_trajectories:
        for body, path in reference_trajectories.items():
            path_arr = np.asarray(path, dtype=float)
            if path_arr.size == 0:
                continue
            figure.add_trace(
                go.Scatter3d(
                    x=path_arr[:, 0],
                    y=path_arr[:, 1],
                    z=path_arr[:, 2],
                    mode="lines",
                    name=f"{body} {reference_label}",
                    line={"width": 2, "dash": "dot", "color": color_map.get(body)},
                    opacity=0.8,
                )
            )

    if isinstance(states_at_t, Mapping):
        for body, state in states_at_t.items():
            pos = _state_position(state)
            figure.add_trace(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="markers+text",
                    text=[body],
                    textposition="top center",
                    marker={"size": 5, "color": color_map.get(body)},
                    name=body,
                )
            )
    else:
        states_arr = np.asarray(states_at_t, dtype=float)
        if states_arr.ndim != 2 or states_arr.shape[1] < 3:
            raise ValueError("states_at_t must be dict or array shape [B,3] / [B,6]")
        for idx, pos in enumerate(states_arr[:, :3]):
            figure.add_trace(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="markers+text",
                    text=[f"body_{idx}"],
                    textposition="top center",
                    marker={"size": 5},
                    name=f"body_{idx}",
                )
            )

    figure.update_layout(
        title=f"Solar System Scene ({frame_label})",
        scene={
            "xaxis_title": f"x [{units_label}]",
            "yaxis_title": f"y [{units_label}]",
            "zaxis_title": f"z [{units_label}]",
            "aspectmode": "data",
            "xaxis": {"showgrid": True, "zeroline": True},
            "yaxis": {"showgrid": True, "zeroline": True},
            "zaxis": {"showgrid": True, "zeroline": True},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return figure
