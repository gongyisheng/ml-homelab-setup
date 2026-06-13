"""Benchmark available attention backends (SDPA, FlashAttention, FlashInfer)
across sequence lengths. Reports forward latency (ms); missing backends are skipped.

    python3 bench_attention.py
    python3 bench_attention.py --heads 32 --dim 128 --seqlens 1024,2048,4096,8192
"""
import argparse
import sys

import torch

from _attn_common import make_qkv, sdpa_reference


def build_backends():
    backends = {"sdpa": lambda q, k, v: sdpa_reference(q, k, v, causal=True)}
    try:
        import flashinfer
        backends["flashinfer"] = lambda q, k, v: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    except Exception:
        pass
    try:
        from flash_attn import flash_attn_func
        backends["flash_attn"] = lambda q, k, v: flash_attn_func(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=True)
    except Exception:
        pass
    return backends


def time_ms(fn, q, k, v, warmup=5, iters=20):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(q, k, v)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seqlens", default="512,1024,2048,4096")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    device = "cuda"
    seqlens = [int(s) for s in args.seqlens.split(",")]
    backends = build_backends()
    print(f"GPU: {torch.cuda.get_device_name(0)}  heads={args.heads} dim={args.dim} dtype=fp16")
    print(f"backends: {', '.join(backends)}\n")

    header = f"{'seqlen':>8}" + "".join(f"{name:>14}" for name in backends)
    print(header + "   (ms)")
    for seq in seqlens:
        q, k, v = make_qkv(seq, args.heads, args.dim, torch.float16, device)
        cells = []
        for name, fn in backends.items():
            try:
                cells.append(f"{time_ms(fn, q, k, v):>14.3f}")
            except Exception:
                cells.append(f"{'ERR':>14}")
        print(f"{seq:>8}" + "".join(cells))


if __name__ == "__main__":
    main()
