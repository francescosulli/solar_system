import numpy as np

from solsys_emulator.config import DEFAULT_BODIES
from solsys_emulator.de440_dataset import build_dataset, find_local_kernel, load_dataset, save_dataset


def test_dataset_shapes_and_metadata(tmp_path):
    kernel = find_local_kernel()
    assert kernel is not None, "DE440/DE441 kernel required in data/ for strict dataset mode."
    dataset = build_dataset(
        start_time="2000-01-01T12:00:00",
        end_time="2000-01-10T12:00:00",
        step=86400.0,
        bodies=DEFAULT_BODIES,
        kernel_path=kernel,
    )

    states = dataset["states"]
    assert states.ndim == 3
    assert states.shape[1] == len(DEFAULT_BODIES)
    assert states.shape[2] == 6
    assert states.shape[0] == len(dataset["times_seconds"])
    assert np.isfinite(states).all()
    assert "frame" in dataset["metadata"]
    assert "timescale" in dataset["metadata"]
    assert "units" in dataset["metadata"]

    out_path = tmp_path / "dataset_test.npz"
    save_dataset(out_path, dataset)
    loaded = load_dataset(out_path)
    assert loaded["states"].shape == states.shape
    assert loaded["metadata"]["frame"] == dataset["metadata"]["frame"]
