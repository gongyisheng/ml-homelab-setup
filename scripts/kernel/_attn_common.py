"""Shared helpers for attention backend tests: fixed inputs, SDPA reference, comparison.

All backends use the (seq, heads, dim) layout (what FlashAttention / FlashInfer expect);
the SDPA reference transposes to (batch, heads, seq, dim) internally.
"""
import torch
import torch.nn.functional as F


def make_qkv(seq, heads, dim, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (seq, heads, dim)
    mk = lambda: torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    return mk(), mk(), mk()


def sdpa_reference(q, k, v, causal=True):
    """q,k,v: (S, H, D) -> attention output (S, H, D) via torch SDPA."""
    qb, kb, vb = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # (1, H, S, D)
    o = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal)
    return o.squeeze(0).transpose(0, 1).contiguous()


def report(name, out, ref, tol=2e-2):
    maxd = (out.float() - ref.float()).abs().max().item()
    ok = maxd < tol
    print(f"[{name}] max abs diff vs SDPA: {maxd:.4f} (tol {tol}) -> {'PASS' if ok else 'FAIL'}")
    return ok
