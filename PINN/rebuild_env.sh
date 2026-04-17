#!/usr/bin/env bash
set -euo pipefail

UV_BIN="$HOME/dlprojenv/bin/uv"
VENV_DIR=".venv"

echo "=== 1. Remove old environment ==="
rm -rf "$VENV_DIR"

echo "=== 2. Create fresh environment ==="
"$UV_BIN" venv "$VENV_DIR"

echo "=== 3. Upgrade base tooling ==="
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" --upgrade pip setuptools wheel

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
