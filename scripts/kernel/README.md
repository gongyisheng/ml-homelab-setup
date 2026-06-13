# kernel

Attention/kernel backends: import + correctness smoke tests, then benchmarks. Backends:
FlashAttention, FlashInfer, PyTorch SDPA, sgl-kernel.

## Scripts

| Script               | Purpose                                                       |
|----------------------|---------------------------------------------------------------|
| `test_sdpa.py`       | PyTorch SDPA reference (the correctness baseline)             |
| `test_flashattn.py`  | FlashAttention import + correctness vs SDPA                   |
| `test_flashinfer.py` | FlashInfer import + correctness vs SDPA                       |
| `test_sgl_kernel.py` | sgl-kernel import + correctness vs SDPA                       |
| `bench_attention.py` | compare backends across seqlen / heads / dtype (latency)      |

> Blackwell sm_120 wheel availability is the key caveat — see install notes below.

## Install notes

<!-- TODO: per-backend install (sm_120 / CUDA 13.0 wheels) -->

## Verify (this box)

Run on the free GPU: `CUDA_VISIBLE_DEVICES=1 python3 test_sdpa.py`

<!-- TODO: fill scripts -->
