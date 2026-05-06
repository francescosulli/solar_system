#!/usr/bin/env bash
set -euo pipefail

DEFAULT_UV_BIN="$HOME/dlprojenv/bin/uv"
UV_BIN="$DEFAULT_UV_BIN"
VENV_DIR=".venv"

usage() {
  cat <<EOF
Usage:
  $0 [UV_BIN]

Description:
  Rebuilds the Python virtual environment using uv with all the right cuda dependencies.

Arguments:
  UV_BIN      Optional path to uv.
              Default: $DEFAULT_UV_BIN

Options:
  -h, --help  Show this help message.

Examples:
  $0
  $0 /path/to/uv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  echo "Error: too many arguments."
  echo
  usage
  exit 1
fi

if [[ $# -eq 1 ]]; then
  UV_BIN="$1"
fi

if [[ ! -x "$UV_BIN" ]]; then
  echo "Error: uv binary not found or not executable:"
  echo "  $UV_BIN"
  exit 1
fi

echo "Using uv binary: $UV_BIN"

echo "=== 1. Remove old environment ==="
rm -rf "$VENV_DIR"

echo "=== 2. Create fresh environment ==="
"$UV_BIN" venv "$VENV_DIR"

echo "=== 3. Upgrade base tooling ==="
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --upgrade pip setuptools wheel

echo "=== 4. Install PyTorch pinned to CUDA 12.1 ==="
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --torch-backend=cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

echo "=== 5. Install common project deps ==="
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  numpy matplotlib tqdm tensorboard wandb astropy

echo "=== 6. Writing uv lock file ==="
"$UV_BIN" lock

echo "=== 7. Verify torch + CUDA ==="
"$VENV_DIR/bin/python" - <<'PY'
import torch

print("=" * 60)
print("TORCH CHECK")
print("=" * 60)
print("torch version    :", torch.__version__)
print("torch cuda build :", torch.version.cuda)
print("cuda available   :", torch.cuda.is_available())
print("device count     :", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    x = torch.randn(1024, 1024, device="cuda")
    print("test tensor device:", x.device)
else:
    print("CUDA is still unavailable.")

print("=" * 60)
PY

echo "Environment rebuild completed successfully."
