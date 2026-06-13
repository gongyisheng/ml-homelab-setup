"""FlashInfer single-prefill attention: import + correctness vs SDPA.

    python3 test_flashinfer.py
"""
import sys

import torch

from _attn_common import make_qkv, sdpa_reference, report

try:
    import flashinfer
except Exception as e:
    print(f"SKIP flashinfer: {e}")
    print("install: pip install flashinfer-python")
    sys.exit(0)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    device = "cuda"
    q, k, v = make_qkv(seq=512, heads=8, dim=64, dtype=torch.float16, device=device)
    out = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    ref = sdpa_reference(q, k, v, causal=True)
    ok = report("flashinfer", out, ref)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
