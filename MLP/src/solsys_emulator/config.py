"""Project configuration for the ephemeris emulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT.parent / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

FRAME_INTERNAL = "icrs"
FRAME_ORIGIN = "barycentric"
TIME_SCALE = "tdb"
MODEL_EPOCH_ISO = "2000-01-01T12:00:00"

DEFAULT_BODIES = (
    "sun",
    "mercury",
    "venus",
    "earth",
    "moon",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
)

DEFAULT_DE440_URL = "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp"
DEFAULT_KERNEL_PATH = DATA_DIR / "de440.bsp"
DEFAULT_DATASET_PATH = DATA_DIR / "dataset_demo.npz"
DEFAULT_CHECKPOINT_PATH = ARTIFACTS_DIR / "emulator_demo.pt"

SOFTENING_EPSILON_KM = 0.0


@dataclass(frozen=True)
class UnitSystem:
    """Canonical internal units used by the project."""

    length: str = "km"
    time: str = "s"
    velocity: str = "km/s"
    mu: str = "km^3/s^2"


UNIT_SYSTEM = UnitSystem()


def ensure_project_dirs() -> None:
    """Create local cache/output directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
