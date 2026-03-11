import numpy as np

from solsys_emulator.config import DEFAULT_BODIES
from solsys_emulator.de440_dataset import build_dataset, find_local_kernel
from solsys_emulator.inference import EphemerisEmulator
from solsys_emulator.model import EmulatorModel
from solsys_emulator.preprocessing import fit_scaler


def test_predict_state_and_trajectory_shapes():
    kernel = find_local_kernel()
    assert kernel is not None, "DE440/DE441 kernel required in data/ for strict dataset mode."
    dataset = build_dataset(
        start_time="2000-01-01T12:00:00",
        end_time="2000-01-08T12:00:00",
        step=86400.0,
        bodies=DEFAULT_BODIES,
        kernel_path=kernel,
    )

    scaler = fit_scaler(dataset["states"])
    model = EmulatorModel(
        num_bodies=len(DEFAULT_BODIES),
        hidden_dim=64,
        num_layers=2,
        fourier_features=8,
        max_frequency=4.0,
    )
    emulator = EphemerisEmulator(
        model=model,
        scaler=scaler,
        bodies=DEFAULT_BODIES,
        time_mean=float(np.mean(dataset["times_seconds"])),
        time_std=float(np.std(dataset["times_seconds"]) + 1e-12),
    )

    state = emulator.predict_state("2000-01-04T12:00:00")
    assert set(state.keys()) == set(DEFAULT_BODIES)
    earth = state["earth"]
    assert earth["r"].shape == (3,)
    assert earth["v"].shape == (3,)
    assert earth["units"]["r"] == "km"
    assert earth["units"]["v"] == "km/s"
    assert np.isfinite(earth["r"].value).all()
    assert np.isfinite(earth["v"].value).all()

    traj = emulator.predict_trajectory("earth", "2000-01-01T12:00:00", "2000-01-05T12:00:00", 86400.0)
    assert traj.ndim == 2
    assert traj.shape[1] == 3
    assert np.isfinite(traj).all()


def test_position_only_model_inference_path():
    kernel = find_local_kernel()
    assert kernel is not None, "DE440/DE441 kernel required in data/ for strict dataset mode."
    dataset = build_dataset(
        start_time="2000-01-01T12:00:00",
        end_time="2000-01-05T12:00:00",
        step=86400.0,
        bodies=DEFAULT_BODIES,
        kernel_path=kernel,
    )

    scaler = fit_scaler(dataset["states"])
    model = EmulatorModel(
        num_bodies=len(DEFAULT_BODIES),
        state_mode="position_only",
        hidden_dim=64,
        num_layers=2,
        fourier_features=8,
        max_frequency=4.0,
    )
    emulator = EphemerisEmulator(
        model=model,
        scaler=scaler,
        bodies=DEFAULT_BODIES,
        time_mean=float(np.mean(dataset["times_seconds"])),
        time_std=float(np.std(dataset["times_seconds"]) + 1e-12),
    )

    state = emulator.predict_state("2000-01-03T12:00:00")
    earth = state["earth"]
    assert earth["r"].shape == (3,)
    assert earth["v"].shape == (3,)
    assert np.isfinite(earth["r"].value).all()
    assert np.isfinite(earth["v"].value).all()
