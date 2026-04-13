"""Dataset generation utilities from DE440/DE441 kernels (strict mode)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import astropy.units as u
import numpy as np
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time

from .config import (
    DEFAULT_BODIES,
    DEFAULT_DE440_URL,
    DEFAULT_KERNEL_PATH,
    FRAME_INTERNAL,
    FRAME_ORIGIN,
    MODEL_EPOCH_ISO,
    TIME_SCALE,
    ensure_project_dirs,
)
from .time_frames import TimeInput, build_time_grid, parse_time

import numpy as np
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class BodyGroupedDistributedSampler(Sampler):
    """Distributes flattened (timestep, body) data by assigning complete bodies to each GPU.
    
    Assumes data is flattened where: flat_index = timestep * n_bodies + body_index
    Each GPU processes ALL timesteps for its assigned bodies.
    
    Args:
        dataset_length: Total number of samples in the dataset
        n_bodies: Total number of bodies in the system
        body_names: Optional list of body names for logging
        num_replicas: Number of processes (GPUs)
        rank: Rank of current process
        shuffle: Whether to shuffle samples (shuffles within assigned bodies)
        seed: Random seed for shuffling
    """
    
    def __init__(self, dataset_length, n_bodies, body_names=None,
                 num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        self.dataset_length = dataset_length
        self.n_bodies = n_bodies
        self.body_names = body_names if body_names is not None else [f"body_{i}" for i in range(n_bodies)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        
        # Distribute bodies across GPUs
        bodies_per_rank = n_bodies // num_replicas
        remainder = n_bodies % num_replicas
        
        # Give extra bodies to first 'remainder' ranks
        if rank < remainder:
            start_body = rank * (bodies_per_rank + 1)
            end_body = start_body + bodies_per_rank + 1
        else:
            start_body = remainder * (bodies_per_rank + 1) + (rank - remainder) * bodies_per_rank
            end_body = start_body + bodies_per_rank
        
        self.my_body_indices = list(range(start_body, end_body))
        self.my_body_names = [self.body_names[i] for i in self.my_body_indices]
        
        # Collect all indices belonging to my assigned bodies
        # Since flat_index = timestep * n_bodies + body_index,
        # we want all indices where (index % n_bodies) is in my_body_indices
        self.indices = [i for i in range(dataset_length) if i % n_bodies in self.my_body_indices]
        self.num_samples = len(self.indices)
        
        print(f"[Rank {rank}/{num_replicas}] Assigned bodies: {self.my_body_names}")
        print(f"[Rank {rank}/{num_replicas}] Processing {self.num_samples:,} samples (out of {dataset_length:,} total)")
    
    def __iter__(self):
        if self.shuffle:
            # Shuffle samples within this rank's assigned bodies
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.indices), generator=g).tolist()
            indices = [self.indices[i] for i in perm]
        else:
            indices = self.indices.copy()
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        """Set epoch for shuffling (call at start of each epoch)."""
        self.epoch = epoch
# class BodyGroupedDistributedSampler(Sampler):
#     """Distributes data by grouping complete bodies to each GPU."""
    
#     def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
#         if num_replicas is None:
#             num_replicas = dist.get_world_size()
#         if rank is None:
#             rank = dist.get_rank()
            
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.shuffle = shuffle
#         self.seed = seed
        
#         body_states = dataset["states"][:, i, :]
#         all_indices = np.arange(dataset["states"].shape[0])
        
#         print(f" {b} (column {i}):")
#         print(f"   states shape: {body_states.shape}")
#         print(f"   indices: 0 to {len(all_indices)-1}")   

#         # Group indices by body
#         self.body_to_indices = {}
#         for idx, body in enumerate(dataset["bodies"]):
#             if body not in self.body_to_indices:
#                 self.body_to_indices[body] = []
#             self.body_to_indices[body].append(idx)
        
#         # Assign bodies to ranks (split bodies evenly across GPUs)
#         all_bodies = sorted(self.body_to_indices.keys())
#         bodies_per_rank = len(all_bodies) // num_replicas
#         start_body = rank * bodies_per_rank
#         end_body = start_body + bodies_per_rank if rank < num_replicas - 1 else len(all_bodies)
        
#         self.my_bodies = all_bodies[start_body:end_body]
        
#         # Collect all indices for this rank's bodies
#         self.indices = []
#         for body in self.my_bodies:
#             self.indices.extend(self.body_to_indices[body])
        
#         self.num_samples = len(self.indices)
        
#         print(f"Rank {rank}: assigned bodies {self.my_bodies}")
#         print(f"Rank {rank}: {self.num_samples} total samples")
    
#     def __iter__(self):
#         if self.shuffle:
#             # Shuffle within each body group, then concatenate
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             indices = []
#             for body in self.my_bodies:
#                 body_indices = self.body_to_indices[body].copy()
#                 perm = torch.randperm(len(body_indices), generator=g).tolist()
#                 indices.extend([body_indices[i] for i in perm])
#         else:
#             indices = self.indices.copy()
        
#         return iter(indices)
    
#     def __len__(self):
#         return self.num_samples
    
#     def set_epoch(self, epoch):
#         self.epoch = epoch

# Usage:
# sampler = BodyGroupedDistributedSampler(dataset, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

def _normalize_body_names(bodies: Sequence[str]) -> list[str]:
    return [body.lower() for body in bodies]


def _as_time_array(time_input: TimeInput | Time) -> Time:
    times = parse_time(time_input)
    if times.isscalar:
        return Time([times.jd], format="jd", scale=times.scale)
    return times


def download_kernel(
    url: str = DEFAULT_DE440_URL,
    target_path: str | Path = DEFAULT_KERNEL_PATH,
    overwrite: bool = False,
) -> Path:
    """Download a BSP kernel into local data cache."""
    ensure_project_dirs()
    target = Path(target_path)
    if target.exists() and not overwrite:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    return target


def load_kernel(path: str | Path) -> dict[str, str]:
    """Validate and return kernel handle metadata."""
    kernel_path = Path(path)
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    # Validation step: if jplephem is installed this guarantees .bsp readability.
    try:
        from jplephem.spk import SPK  # type: ignore

        handle = SPK.open(str(kernel_path))
        handle.close()
    except ImportError:
        # jplephem is optional at runtime if astropy can still use the path.
        pass
    except Exception as exc:  # pragma: no cover - rare malformed kernel path.
        raise RuntimeError(f"Unable to open kernel with jplephem: {kernel_path}") from exc

    return {"path": str(kernel_path), "name": kernel_path.name}


def _sample_with_ephemeris(times: Time, bodies: Sequence[str], ephemeris: str) -> np.ndarray:
    states = np.zeros((len(times), len(bodies), 6), dtype=float)
    with solar_system_ephemeris.set(ephemeris):
        for body_idx, body in enumerate(bodies):
            pos, vel = get_body_barycentric_posvel(body, times)
            states[:, body_idx, :3] = pos.xyz.to_value(u.km).T
            states[:, body_idx, 3:] = vel.xyz.to_value(u.km / u.s).T
    return states


def sample_states(
    times: TimeInput | Time,
    bodies: Sequence[str] = DEFAULT_BODIES,
    kernel: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Sample barycentric states [T, B, 6] in km and km/s.

    Returns (states, metadata).
    """
    if kernel is None:
        raise ValueError("Kernel handle is required. Load DE440/DE441 with load_kernel(...).")

    body_list = _normalize_body_names(bodies)
    time_array = _as_time_array(times)
    ephemeris_name = kernel["path"]
    try:
        sampled = _sample_with_ephemeris(time_array, body_list, ephemeris_name)
    except Exception as exc:
        raise RuntimeError(f"Unable to sample states from kernel: {ephemeris_name}") from exc
    return sampled, {"source": ephemeris_name}


def build_dataset(
    start_time: TimeInput,
    end_time: TimeInput,
    step: float | u.Quantity,
    bodies: Sequence[str] = DEFAULT_BODIES,
    kernel_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create dataset dict with states and metadata."""
    if isinstance(step, u.Quantity):
        step_seconds = float(step.to_value(u.s))
    else:
        step_seconds = float(step)

    time_seconds, time_array = build_time_grid(start_time, end_time, step_seconds)

    if kernel_path is None:
        auto_kernel = find_local_kernel()
        if auto_kernel is None:
            raise FileNotFoundError(
                "No DE440/DE441 kernel found in ./data. "
                "Place de440.bsp or de441.bsp in data/ or pass kernel_path explicitly."
            )
        kernel_path = auto_kernel
    kernel = load_kernel(kernel_path)

    states, sample_meta = sample_states(
        time_array,
        bodies=bodies,
        kernel=kernel,
    )

    body_list = _normalize_body_names(bodies)
    metadata = {
        "frame": FRAME_INTERNAL,
        "origin": FRAME_ORIGIN,
        "timescale": TIME_SCALE,
        "units": {"r": "km", "v": "km/s", "mu": "km^3/s^2"},
        "epoch": MODEL_EPOCH_ISO,
        "bodies": body_list,
        "kernel_path": str(kernel_path),
        "sample_source": sample_meta["source"],
    }
    return {
        "times_seconds": time_seconds,
        "times_iso": np.asarray(time_array.isot, dtype="U32"),
        "states": states,
        "bodies": np.asarray(body_list, dtype="U16"),
        "metadata": metadata,
    }


def save_dataset(path: str | Path, dataset: dict[str, Any]) -> Path:
    """Save dataset to compressed NPZ with JSON metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        times_seconds=np.asarray(dataset["times_seconds"], dtype=float),
        times_iso=np.asarray(dataset["times_iso"]),
        states=np.asarray(dataset["states"], dtype=float),
        bodies=np.asarray(dataset["bodies"]),
        metadata_json=json.dumps(dataset["metadata"]),
    )
    return target


def load_dataset(path: str | Path, debug=True) -> dict[str, Any]:
    """Load dataset from NPZ while restoring metadata."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        raw_metadata = payload["metadata_json"].item()
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        metadata = json.loads(raw_metadata)
        dataset = {
            "times_seconds": payload["times_seconds"],
            "times_iso": payload["times_iso"],
            "states": payload["states"],
            "bodies": payload["bodies"],
            "metadata": metadata,
        }
        
        if debug:
            print("Debugging de440 dataset.")
            print(f"\nDataset structure:")
            for key in dataset:
                if key != 'metadata':
                    print(f"  {key}: shape={dataset[key].shape}, dtype={dataset[key].dtype}")
                else:
                    print(f"  {key}: {type(dataset[key])}")
            
            # Create a DataFrame-like view
            print(f"\nDataset preview (first 5 rows):")
            n_preview = min(5, len(dataset['times_seconds']))
            
            # Determine state columns (assuming states is 2D: [n_times, n_dims])
            if dataset['states'].ndim > 1:
                n_state_cols = dataset['states'].shape[1]
                state_labels = [f"state_{i}" for i in range(n_state_cols)]
            else:
                state_labels = ["state"]
            
            # Build preview
            for i in range(n_preview):
                print(f"\n[{i}]")
                print(f"  time_seconds: {dataset['times_seconds'][i]}")
                print(f"  time_iso: {dataset['times_iso'][i]}")
                if dataset['states'].ndim > 1:
                    print(f"  states: {dataset['states'][i]}")
                else:
                    print(f"  states: {dataset['states'][i]}")
                if len(dataset['bodies']) > i:
                    print(f"  body: {dataset['bodies'][i]}")

            bodies = dataset["bodies"]

            print("Solar system bodies:", bodies)

            for i, b in enumerate(bodies):
                body_states = dataset["states"][:, i, :]
                all_indices = np.arange(dataset["states"].shape[0])

                print(f" {b} (column {i}):")
                print(f"   states shape: {body_states.shape}")
                print(f"   indices: 0 to {len(all_indices)-1}")

            print(f"\nMetadata keys: {list(dataset['metadata'].keys())}")

        return dataset


def find_local_kernel(
    candidate_paths: Sequence[str | Path] | None = None,
) -> Path | None:
    """Find a local BSP kernel candidate in ./data."""
    candidates = list(candidate_paths or [])
    candidates.extend(
        [
            DEFAULT_KERNEL_PATH,
            DEFAULT_KERNEL_PATH.with_name("de441.bsp"),
            DEFAULT_KERNEL_PATH.with_name("de440s.bsp"),
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None
