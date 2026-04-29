import argparse
from pathlib import Path

import numpy as np
import torch


def inspect_npz(path):
    data = np.load(path, allow_pickle=True)
    print(f"\n[NPZ] {path}")
    print("=" * 60)

    for key in data.files:
        value = data[key]
        print(f"{key}:")
        print(f"  type  : {type(value)}")
        if hasattr(value, "shape"):
            print(f"  shape : {value.shape}")
        if hasattr(value, "dtype"):
            print(f"  dtype : {value.dtype}")
        print(f"  sample: {str(value)[:200]}")
        print("-" * 60)


def inspect_pt(path):
    data = torch.load(path, map_location="cpu")

    print(f"\n[PT] {path}")
    print("=" * 60)

    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}:")
            print(f"  type  : {type(value)}")

            if hasattr(value, "shape"):
                print(f"  shape : {tuple(value.shape)}")

            if isinstance(value, torch.Tensor):
                print(f"  dtype : {value.dtype}")
                print(f"  device: {value.device}")
                print(f"  sample: {value.flatten()[:5]}")

            else:
                print(f"  value : {str(value)[:200]}")

            print("-" * 60)
    else:
        print(f"Top-level object: {type(data)}")
        print(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to .npz or .pt file")
    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npz":
        inspect_npz(path)
    elif path.suffix == ".pt":
        inspect_pt(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


if __name__ == "__main__":
    main()
