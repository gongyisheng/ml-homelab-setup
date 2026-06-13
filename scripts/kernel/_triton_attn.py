"""Forward-only Triton flash-attention (causal), adapted from the official Triton
`06-fused-attention` tutorial (Apache-2.0). Exposes `triton_attn_func(q, k, v, causal)`
for (B, H, S, D) fp16 tensors. Benchmark/parity use only — no backward pass.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vn, stride_vk,
              stride_oz, stride_oh, stride_om, stride_ok,
              H, N_CTX,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    o_base = Out + off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qf = q.to(tl.float32) * sm_scale

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        qk = tl.dot(qf, tl.trans(k.to(tl.float32)))
        mask = offs_n[None, :] < N_CTX
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32))
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def triton_attn_func(q, k, v, causal=True):
    """q, k, v: (B, H, S, D) fp16. Returns (B, H, S, D)."""
    B, H, S, D = q.shape
    assert D in (16, 32, 64, 128), f"HEAD_DIM {D} unsupported"
    o = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_M = BLOCK_N = 32 if D >= 128 else 64  # cap shared memory for large head dims
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _attn_fwd[grid](
        q, k, v, sm_scale, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o
