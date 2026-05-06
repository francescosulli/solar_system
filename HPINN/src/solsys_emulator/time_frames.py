"""Time parsing and model-time conversions."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence, Union

import astropy.units as u
import numpy as np
from astropy.time import Time

from .config import FRAME_INTERNAL, FRAME_ORIGIN, MODEL_EPOCH_ISO, TIME_SCALE

FRAME_CONVENTION = f"{FRAME_ORIGIN} {FRAME_INTERNAL}"
MODEL_EPOCH = Time(MODEL_EPOCH_ISO, scale=TIME_SCALE)

TimeInput = Union[str, float, int, datetime, Time, Sequence[object], np.ndarray]


def parse_time(value: TimeInput) -> Time:
    """Parse heterogeneous user input to a Time object in project timescale."""
    if isinstance(value, Time):
        parsed = value
    elif isinstance(value, datetime):
        parsed = Time(value, scale="utc")
    elif isinstance(value, (int, float, np.integer, np.floating)):
        parsed = Time(float(value), format="jd", scale=TIME_SCALE)
    elif isinstance(value, str):
        try:
            parsed = Time(value, scale=TIME_SCALE)
        except ValueError:
            parsed = Time(value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            parsed = Time(array.astype(float), format="jd", scale=TIME_SCALE)
        else:
            parsed = Time(array.tolist(), scale=TIME_SCALE)
    else:
        raise TypeError(f"Unsupported time input type: {type(value)!r}")

    return getattr(parsed, TIME_SCALE)


def to_model_time(time_value: TimeInput) -> np.ndarray:
    """Convert time input into seconds from MODEL_EPOCH."""
    t = parse_time(time_value)
    delta_seconds = (t - MODEL_EPOCH).to_value(u.s)
    return np.asarray(delta_seconds, dtype=float)


def from_model_time(seconds_from_epoch: Union[float, Sequence[float], np.ndarray]) -> Time:
    """Convert model-time seconds back to astropy Time in project scale."""
    seconds = np.asarray(seconds_from_epoch, dtype=float)
    return MODEL_EPOCH + seconds * u.s


def build_time_grid(
    start_time: TimeInput,
    end_time: TimeInput,
    step_seconds: float,
) -> tuple[np.ndarray, Time]:
    """Build an inclusive time grid in seconds from epoch and Time array."""
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")

    t0 = float(to_model_time(start_time))
    t1 = float(to_model_time(end_time))
    if t1 <= t0:
        raise ValueError("end_time must be greater than start_time")

    grid = np.arange(t0, t1 + 0.5 * step_seconds, step_seconds, dtype=float)
    return grid, from_model_time(grid)
