"""Torch / CUDA environment diagnostic.

Reports torch, CUDA, cuDNN, NCCL versions, per-GPU name + compute capability, plus the
driver version (nvidia-smi) and CUDA toolkit version (nvcc). Output is printed and saved
to a txt file.

    python3 check_env.py                 # saves to env.txt next to this script
    python3 check_env.py /tmp/env.txt    # custom path
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

_lines = []


def out(s):
    print(s)
    _lines.append(s)


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


out("=== Torch / CUDA stack ===")
out(f"torch          : {torch.__version__}")
out(f"cuda available : {torch.cuda.is_available()}")
out(f"torch cuda     : {fmt(getattr(torch.version, 'cuda', None))}")
out(f"cudnn enabled  : {torch.backends.cudnn.enabled}")
out(f"cudnn version  : {torch.backends.cudnn.version()}")

nccl_ver = None
if torch.cuda.is_available():
    try:
        nccl_ver = torch.cuda.nccl.version()
    except Exception as e:
        nccl_ver = f"unavailable ({type(e).__name__}: {e})"
out(f"nccl version   : {fmt(nccl_ver)}")

if torch.cuda.is_available():
    out(f"device count   : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        cc = fmt(torch.cuda.get_device_capability(i))
        out(f"  gpu {i}        : {torch.cuda.get_device_name(i)} (sm_{cc.replace('.', '')})")

driver = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
out(f"driver version : {fmt(driver.splitlines()[0] if driver else None)}")

nvcc = run(["nvcc", "--version"])
nvcc_rel = re.search(r"release ([\d.]+)", nvcc).group(1) if nvcc and "release" in nvcc else None
out(f"nvcc cuda      : {fmt(nvcc_rel)}")

out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "env.txt"
out_path.write_text("\n".join(_lines) + "\n")
print(f"\nsaved -> {out_path}")
