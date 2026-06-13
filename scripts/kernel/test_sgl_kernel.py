"""sgl-kernel attention: import + correctness vs SDPA.

sgl-kernel packages FlashAttention-style kernels for SGLang. The entry point has moved
across versions, so this tries the known ones and skips if none are importable.

    python3 test_sgl_kernel.py
"""
import sys

import torch

from _attn_common import make_qkv, sdpa_reference, report


def _load_flash_attn_func():
    try:
        from sgl_kernel.flash_attn import flash_attn_func
        return flash_attn_func
    except Exception:
        pass
    try:
        from sgl_kernel import flash_attn_func
        return flash_attn_func
    except Exception as e:
        print(f"SKIP sgl_kernel: {e}")
        print("install: pip install sgl-kernel  (sm_120 needs a Blackwell-built wheel)")
        sys.exit(0)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    flash_attn_func = _load_flash_attn_func()
    device = "cuda"
    q, k, v = make_qkv(seq=512, heads=8, dim=64, dtype=torch.float16, device=device)
    # sgl-kernel's flash_attn_func follows the (B, S, H, D) FlashAttention layout.
    out = flash_attn_func(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=True)
    out = out[0] if isinstance(out, (tuple, list)) else out
    ref = sdpa_reference(q, k, v, causal=True)
    ok = report("sgl_kernel", out.squeeze(0), ref)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
