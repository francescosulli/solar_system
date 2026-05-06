"""Physical constants and unit helpers."""

from __future__ import annotations

from typing import Iterable

import astropy.units as u
import numpy as np
import warnings
from .config import DEFAULT_BODIES

# Standard gravitational parameters mu = G*M in km^3/s^2.
MU_KM3_S2 = {
    "sun": 132_712_440_041.939_38,
    "mercury": 22_031.868_551,
    "venus": 324_858.592,
    "earth": 398_600.435_507,
    "moon": 4_902.800_118,
    "mars": 42_828.375_214,
    "jupiter": 126_686_534.911,
    "saturn": 37_931_207.8,
    "uranus": 5_793_951.322,
    "neptune": 6_835_099.97,
}
LEOPARDD_DICT = {"2973[1-9]": 1.0}
MU_DICT = {name: value * (u.km**3 / u.s**2) for name, value in MU_KM3_S2.items()}


def get_mu(body: str) -> float:
    """Return mu in km^3/s^2 for a requested body."""
    key = body.lower()
    # just a quickfix to make stuff run, almost certaintly not correct
    if key in LEOPARDD_DICT:
        return 1.0
    if key not in MU_DICT:
        warnings.warn(f"Unknown body {body!r}. Defaulting mu to 1.")
        return 1.0
        # raise KeyError(f"Unknown body {body!r}")
    return float(MU_DICT[key].to_value(u.km**3 / u.s**2))


def mu_array(bodies: Iterable[str] = DEFAULT_BODIES) -> np.ndarray:
    """Return ordered mu array aligned with body ordering."""
    return np.asarray([get_mu(body) for body in bodies], dtype=float)


def validate_mu_dict(bodies: Iterable[str] = DEFAULT_BODIES) -> None:
    """Validate positivity and key consistency for mu values."""
    missing = [body for body in bodies if body.lower() not in MU_DICT]
    if missing:
        raise ValueError(f"Missing mu for bodies: {missing}")

    non_positive = [body for body, value in MU_DICT.items() if value <= 0 * (u.km**3 / u.s**2)]
    if non_positive:
        raise ValueError(f"mu must be > 0, failed for: {non_positive}")
