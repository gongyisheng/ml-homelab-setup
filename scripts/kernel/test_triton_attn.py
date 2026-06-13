"""Triton flash-attention (vendored): correctness vs SDPA.

    python3 test_triton_attn.py
"""
import sys

import torch

from _attn_common import make_qkv, sdpa_reference, report

try:
    from _triton_attn import triton_attn_func
except Exception as e:
    print(f"SKIP triton_attn: {e}")
    sys.exit(0)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    device = "cuda"
    q, k, v = make_qkv(seq=512, heads=8, dim=64, dtype=torch.float16, device=device)
    # triton_attn_func wants (B, H, S, D); inputs are (S, H, D).
    qb, kb, vb = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    out = triton_attn_func(qb, kb, vb, causal=True).squeeze(0).transpose(0, 1).contiguous()
    ref = sdpa_reference(q, k, v, causal=True)
    ok = report("triton_attn", out, ref)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
