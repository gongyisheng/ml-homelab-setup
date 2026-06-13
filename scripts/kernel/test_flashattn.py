"""FlashAttention: import + correctness vs SDPA.

    python3 test_flashattn.py
"""
import sys

import torch

from _attn_common import make_qkv, sdpa_reference, report

try:
    from flash_attn import flash_attn_func
except Exception as e:
    print(f"SKIP flash_attn: {e}")
    print("install (sm_120 needs a Blackwell-built wheel): pip install flash-attn --no-build-isolation")
    sys.exit(0)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    device = "cuda"
    q, k, v = make_qkv(seq=512, heads=8, dim=64, dtype=torch.float16, device=device)
    # flash_attn_func wants (B, S, H, D).
    out = flash_attn_func(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=True).squeeze(0)
    ref = sdpa_reference(q, k, v, causal=True)
    ok = report("flash_attn", out, ref)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
