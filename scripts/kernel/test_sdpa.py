"""PyTorch SDPA smoke + correctness vs a naive math attention. SDPA is the baseline the
other backends are compared against, so this checks SDPA itself against first principles.

    python3 test_sdpa.py
"""
import sys

import torch
import torch.nn.functional as F

from _attn_common import make_qkv, sdpa_reference


def naive_attention(q, k, v, causal=True):
    """Reference softmax attention in fp32. q,k,v: (S, H, D)."""
    q, k, v = (t.transpose(0, 1).float() for t in (q, k, v))  # (H, S, D)
    scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)   # (H, S, S)
    if causal:
        s = scores.size(-1)
        mask = torch.triu(torch.ones(s, s, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    out = F.softmax(scores, dim=-1) @ v
    return out.transpose(0, 1)  # (S, H, D)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    device = "cuda"
    q, k, v = make_qkv(seq=512, heads=8, dim=64, dtype=torch.float32, device=device)
    out = sdpa_reference(q, k, v, causal=True)
    ref = naive_attention(q, k, v, causal=True)
    maxd = (out - ref).abs().max().item()
    ok = maxd < 1e-4
    print(f"[sdpa] max abs diff vs naive: {maxd:.6f} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
