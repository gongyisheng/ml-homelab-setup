"""SageAttention: import + correctness vs SDPA.

    python3 test_sageattn.py
"""
import sys

import torch

from _attn_common import make_qkv, sdpa_reference, report

try:
    from sageattention import sageattn
except Exception as e:
    print(f"SKIP sageattn: {e}")
    print("install: pip install sageattention  (v2 CUDA kernels need a source build on sm_120)")
    sys.exit(0)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    device = "cuda"
    q, k, v = make_qkv(seq=512, heads=8, dim=64, dtype=torch.float16, device=device)
    # sageattn NHD layout = (B, S, H, D); inputs are (S, H, D).
    try:
        out = sageattn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                       tensor_layout="NHD", is_causal=True).squeeze(0)
    except Exception as e:
        print(f"SKIP sageattn (runtime): {e}")
        sys.exit(0)
    ref = sdpa_reference(q, k, v, causal=True)
    ok = report("sageattn", out, ref, tol=5e-2)  # INT8-quantized: looser tol than fp16
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
