"""Benchmark Marlin INT4 GEMM (gptqmodel) vs fp16 torch.nn.functional.linear.

Synthesizes a symmetric INT4 (uint4b8, group 128) weight, repacks to Marlin layout,
and times the Marlin kernel against an fp16 baseline across LLM-shaped (M, K, N).
Marlin kernels JIT-compile on first use. SKIP if gptqmodel/Marlin runtime is absent.

    python3 bench_marlin.py
"""
import sys

import torch

try:
    from gptqmodel.utils.marlin import (
        apply_gptq_marlin_linear, gptq_marlin_repack, marlin_make_workspace_new,
        marlin_permute_scales, marlin_runtime_available, marlin_runtime_error,
    )
    from gptqmodel.utils.marlin_scalar_type import scalar_types
except Exception as e:
    print(f"SKIP marlin: {e}")
    print("install: pip install gptqmodel")
    sys.exit(0)

GROUP_SIZE = 128
NUM_BITS = 4


def make_marlin_int4(size_k, size_n, device, dtype=torch.float16):
    """Symmetric INT4 quant of a random weight, repacked to Marlin layout."""
    w = torch.randn(size_k, size_n, device=device, dtype=dtype)
    ng = size_k // GROUP_SIZE
    wg = w.reshape(ng, GROUP_SIZE, size_n)
    scale = (wg.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-8)
    q = torch.clamp((wg / scale).round() + 8, 0, 15).reshape(size_k, size_n).to(torch.int32)
    scale = scale.reshape(ng, size_n).to(dtype)

    # GPTQ pack: 8 consecutive k-rows -> one int32.
    q_packed = q.reshape(size_k // 8, 8, size_n)
    gptq_w = sum(q_packed[:, j, :] << (4 * j) for j in range(8)).to(torch.int32).contiguous()

    perm = torch.empty(0, dtype=torch.int, device=device)
    marlin_w = gptq_marlin_repack(gptq_w, perm, size_k, size_n, NUM_BITS)
    marlin_s = marlin_permute_scales(scale, size_k, size_n, GROUP_SIZE)
    w_ref = ((q.reshape(ng, GROUP_SIZE, size_n) - 8) * scale.reshape(ng, 1, size_n)
             ).reshape(size_k, size_n)  # dequantized fp16 reference
    return marlin_w, marlin_s, w_ref


def time_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); sys.exit(0)
    if not marlin_runtime_available(torch.float16):
        print(f"SKIP marlin runtime: {marlin_runtime_error(torch.float16)}")
        sys.exit(0)

    device = "cuda"
    empty = torch.empty(0, dtype=torch.int, device=device)
    workspace = marlin_make_workspace_new(torch.device(device))
    print(f"GPU: {torch.cuda.get_device_name(0)}  INT4 group={GROUP_SIZE} dtype=fp16")
    # (K, N) from typical LLM projections; M = batch tokens.
    shapes = [(4096, 4096), (4096, 11008), (11008, 4096)]
    Ms = [16, 64, 256]
    print(f"\n{'K x N':>14}{'M':>6}{'fp16':>12}{'marlin':>12}{'speedup':>10}   (ms)")
    for size_k, size_n in shapes:
        marlin_w, marlin_s, w_ref = make_marlin_int4(size_k, size_n, device)
        marlin = lambda x: apply_gptq_marlin_linear(
            x, marlin_w, marlin_s, empty, empty, empty, workspace,
            scalar_types.uint4b8, size_n, size_k, is_k_full=True)
        # correctness sanity vs dequantized reference (marlin computes x @ B, B is (K,N))
        xs = torch.randn(64, size_k, device=device, dtype=torch.float16)
        ref = xs.float() @ w_ref.float()
        rel = (marlin(xs).float() - ref).abs().max().item() / ref.abs().max().item()
        for M in Ms:
            x = torch.randn(M, size_k, device=device, dtype=torch.float16)
            fp16 = lambda: torch.matmul(x, w_ref)
            marlin_fn = lambda: marlin(x)
            try:
                t_fp16, t_marlin = time_ms(fp16), time_ms(marlin_fn)
                spd = f"{t_fp16 / t_marlin:>9.2f}x"
                note = f"  (rel.err {rel:.1e})" if M == Ms[0] else ""
                print(f"{size_k}x{size_n:<7}{M:>6}{t_fp16:>12.4f}{t_marlin:>12.4f}{spd:>10}{note}")
            except Exception as e:
                print(f"{size_k}x{size_n:<7}{M:>6}{'ERR':>12}  {type(e).__name__}: {str(e)[:50]}")


if __name__ == "__main__":
    main()
