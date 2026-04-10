#!/usr/bin/env python3
"""
Setup verification script for PINN multi-GPU training.
Run this to check your environment before training.
"""

import sys
import subprocess
from pathlib import Path


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_python():
    """Check Python version."""
    print_section("PYTHON VERSION")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠ WARNING: Python 3.8+ recommended")
        return False
    else:
        print("✓ OK")
        return True


def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print_section("PYTORCH")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            # List GPUs
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            
            print("✓ OK - CUDA ready")
            return True, torch.cuda.device_count()
        else:
            print("⚠ WARNING: CUDA not available (CPU-only mode)")
            return True, 0
    
    except ImportError:
        print("✗ ERROR: PyTorch not installed")
        print("  Install with: pip install torch")
        return False, 0


def check_distributed():
    """Check if distributed training is available."""
    print_section("DISTRIBUTED TRAINING")
    
    try:
        import torch
        import torch.distributed as dist
        
        print(f"torch.distributed available: {dist.is_available()}")
        
        if dist.is_available():
            backends = []
            if dist.is_nccl_available():
                backends.append("nccl")
            if dist.is_gloo_available():
                backends.append("gloo")
            if dist.is_mpi_available():
                backends.append("mpi")
            
            print(f"Available backends: {', '.join(backends) if backends else 'none'}")
            
            if "nccl" in backends:
                print("✓ OK - NCCL backend available (recommended for multi-GPU)")
                return True
            else:
                print("⚠ WARNING: NCCL backend not available")
                return True
        else:
            print("✗ ERROR: torch.distributed not available")
            return False
    
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def check_dependencies():
    """Check optional dependencies."""
    print_section("OPTIONAL DEPENDENCIES")
    
    deps = {
        "numpy": "Required",
        "matplotlib": "Required for plotting",
        "tqdm": "Progress bars",
        "tensorboard": "TensorBoard logging",
        "wandb": "Weights & Biases logging",
    }
    
    all_ok = True
    for package, description in deps.items():
        try:
            __import__(package)
            print(f"✓ {package:15s} - {description}")
        except ImportError:
            if package in ["numpy", "matplotlib"]:
                print(f"✗ {package:15s} - {description} (REQUIRED)")
                all_ok = False
            else:
                print(f"⚠ {package:15s} - {description} (optional)")
    
    return all_ok


def check_files():
    """Check if required files exist."""
    print_section("PROJECT FILES")
    
    required = [
        ("train.py", "Unified training script"),
        ("train_ddp.py", "DDP training script"),
        ("src/solsys_emulator/train.py", "Training module"),
        ("src/solsys_emulator/model.py", "Model module"),
        ("data/dataset_demo.npz", "Demo dataset"),
    ]
    
    optional = [
        ("DDP_IMPLEMENTATION.md", "DDP guide"),
        ("LOGGING_GUIDE.md", "Logging guide"),
        ("QUICK_REFERENCE.md", "Quick reference"),
        ("minimal_ddp_example.py", "Minimal DDP example"),
    ]
    
    all_ok = True
    
    print("\nRequired files:")
    for filepath, description in required:
        exists = Path(filepath).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {filepath:40s} - {description}")
        if not exists:
            all_ok = False
    
    print("\nDocumentation:")
    for filepath, description in optional:
        exists = Path(filepath).exists()
        status = "✓" if exists else "⚠"
        print(f"{status} {filepath:40s} - {description}")
    
    return all_ok


def check_gpu_memory():
    """Check GPU memory availability."""
    print_section("GPU MEMORY")
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    idx, name, total, free = parts[:4]
                    total_gb = float(total) / 1024
                    free_gb = float(free) / 1024
                    print(f"GPU {idx}: {name}")
                    print(f"  Total: {total_gb:.1f} GB")
                    print(f"  Free:  {free_gb:.1f} GB")
                    
                    if free_gb < 20:
                        print(f"  ⚠ WARNING: Low free memory")
            
            print("✓ OK")
            return True
        else:
            print("⚠ WARNING: nvidia-smi not available")
            return True
    
    except Exception as e:
        print(f"⚠ WARNING: Could not query GPU memory: {e}")
        return True


def provide_recommendations(gpu_count):
    """Provide training recommendations based on setup."""
    print_section("RECOMMENDATIONS")
    
    if gpu_count == 0:
        print("No GPUs detected:")
        print("  • Use CPU training for testing: python train.py --device cpu --epochs 10")
        print("  • For production, use a machine with GPU(s)")
    
    elif gpu_count == 1:
        print("Single GPU setup:")
        print("  • Basic training:")
        print("    python train.py --training-mode unified --device cuda")
        print()
        print("  • If you encounter OOM errors:")
        print("    python train.py --training-mode multi --device cuda")
        print("    (Multi-stage training uses less memory per stage)")
    
    elif gpu_count >= 2:
        print(f"Multi-GPU setup ({gpu_count} GPUs):")
        print("  • First, implement DDP modifications:")
        print("    1. Read DDP_PATCH.md")
        print("    2. Modify src/solsys_emulator/train.py")
        print("    3. Test with minimal example:")
        print(f"       torchrun --nproc_per_node={min(2, gpu_count)} minimal_ddp_example.py")
        print()
        print("  • Then use DDP training:")
        print(f"    torchrun --nproc_per_node={gpu_count} train_ddp.py \\")
        print("      --training-mode unified \\")
        print("      --batch-size 768 \\")
        print("      --epochs 1400")
        print()
        print(f"  • Expected speedup: ~{gpu_count * 0.85:.1f}x")
        print(f"  • Memory per GPU: ~50% of single-GPU mode")


def main():
    """Run all checks."""
    print("=" * 80)
    print("PINN TRAINING ENVIRONMENT VERIFICATION")
    print("=" * 80)
    
    results = {}
    
    # Run checks
    results['python'] = check_python()
    results['pytorch'], gpu_count = check_pytorch()
    results['distributed'] = check_distributed()
    results['dependencies'] = check_dependencies()
    results['files'] = check_files()
    results['gpu_memory'] = check_gpu_memory()
    
    # Summary
    print_section("SUMMARY")
    
    all_ok = all(results.values())
    
    for check, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{check.upper():20s} {status_str}")
    
    print()
    
    if all_ok:
        print("✓ All checks passed!")
        print()
        provide_recommendations(gpu_count)
    else:
        print("✗ Some checks failed. Please address the issues above.")
        print()
        print("Common fixes:")
        print("  • Install PyTorch: pip install torch")
        print("  • Install dependencies: pip install numpy matplotlib tqdm")
        print("  • Check CUDA installation: nvidia-smi")
    
    print()
    print("=" * 80)
    print("Next steps:")
    print("  1. Read README.md for overview")
    print("  2. Try a test run: python train.py --device cuda --epochs 10")
    print("  3. Read DDP_IMPLEMENTATION.md for multi-GPU setup")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
