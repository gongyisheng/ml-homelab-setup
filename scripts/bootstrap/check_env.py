"""Torch / CUDA environment diagnostic.

Reports torch, CUDA, cuDNN, NCCL versions and per-GPU name + compute capability.
Driver/runtime are deferred to nvidia-smi / nvcc.

    python3 check_env.py
"""
import torch


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
    print("driver/runtime : n/a (use nvidia-smi / nvcc for exact)")
