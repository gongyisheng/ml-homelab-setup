"""Torch / CUDA environment diagnostic.

Reports torch, CUDA, cuDNN, NCCL versions, per-GPU name + compute capability, plus the
driver version (nvidia-smi) and CUDA toolkit version (nvcc).

    python3 check_env.py
"""
import re
import shutil
import subprocess

import torch


def run(cmd):
    """Run a command, return stripped stdout or None if it's missing/fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def fmt(v):
    if v is None:
        return "None"
    if isinstance(v, (tuple, list)):
        return ".".join(map(str, v))
    return str(v)


print("=== Torch / CUDA stack ===")
print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")
print(f"torch cuda     : {fmt(getattr(torch.version, 'cuda', None))}")
print(f"cudnn enabled  : {torch.backends.cudnn.enabled}")
print(f"cudnn version  : {torch.backends.cudnn.version()}")

nccl_ver = None
if torch.cuda.is_available():
    try:
        nccl_ver = torch.cuda.nccl.version()
    except Exception as e:
        nccl_ver = f"unavailable ({type(e).__name__}: {e})"
print(f"nccl version   : {fmt(nccl_ver)}")

if torch.cuda.is_available():
    print(f"device count   : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        cc = fmt(torch.cuda.get_device_capability(i))
        print(f"  gpu {i}        : {torch.cuda.get_device_name(i)} (sm_{cc.replace('.', '')})")
driver = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
print(f"driver version : {fmt(driver.splitlines()[0] if driver else None)}")

nvcc = run(["nvcc", "--version"])
nvcc_rel = re.search(r"release ([\d.]+)", nvcc).group(1) if nvcc and "release" in nvcc else None
print(f"nvcc cuda      : {fmt(nvcc_rel)}")
