"""xFormers memory_efficient_attention: import + correctness vs SDPA.

    python3 test_xformers.py
"""
import sys

import torch

from _attn_common import make_qkv, sdpa_reference, report

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import LowerTriangularMask
except Exception as e:
    print(f"SKIP xformers: {e}")
    print("install: pip install xformers  (no cu130/torch2.12 wheel -> needs a source build)")
    sys.exit(0)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    device = "cuda"
    q, k, v = make_qkv(seq=512, heads=8, dim=64, dtype=torch.float16, device=device)
    # memory_efficient_attention wants (B, M, H, K); inputs are (S, H, D).
    try:
        out = memory_efficient_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            attn_bias=LowerTriangularMask()).squeeze(0)
    except Exception as e:
        print(f"SKIP xformers (runtime, no sm_120 CUDA ext): {e}")
        sys.exit(0)
    ref = sdpa_reference(q, k, v, causal=True)
    ok = report("xformers", out, ref)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
