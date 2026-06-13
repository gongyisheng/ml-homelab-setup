# kernel

Attention/kernel backends: import + correctness smoke tests (vs SDPA), then benchmarks.
Backends: PyTorch SDPA, FlashAttention, FlashInfer, sgl-kernel.

## Scripts

| Script               | Purpose                                                       |
|----------------------|---------------------------------------------------------------|
| `_attn_common.py`    | shared inputs, SDPA reference, comparison helper             |
| `test_sdpa.py`       | SDPA vs naive fp32 attention (the correctness baseline)      |
| `test_flashattn.py`  | FlashAttention import + correctness vs SDPA                  |
| `test_flashinfer.py` | FlashInfer import + correctness vs SDPA                      |
| `test_sgl_kernel.py` | sgl-kernel import + correctness vs SDPA                      |
| `bench_attention.py` | latency across seqlen / heads / dim for available backends   |

Each test prints `PASS` / `FAIL`, or `SKIP` with an install hint if the backend is absent.

## Run

Needs the `kernel` extra: `uv sync --extra kernel` (installs FlashInfer + ninja). `uv run`
puts ninja on PATH so FlashInfer's JIT works.

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/test_sdpa.py
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/test_flashinfer.py
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/bench_attention.py --seqlens 512,1024,2048,4096
```

## Install notes (sm_120 / CUDA 13.0)

Blackwell sm_120 needs Blackwell-built wheels — generic PyPI wheels may not include the
`sm_120` arch and will fail at load or fall back to slow paths.

- **FlashInfer**: `pip install flashinfer-python`. JIT-compiles kernels on first use, so
  **`ninja` must be on `PATH`** (it ships in the venv's `bin/`; activate the venv or add
  `.venv/bin` to `PATH`, otherwise you get `FileNotFoundError: 'ninja'`).
- **FlashAttention**: `pip install flash-attn --no-build-isolation` (needs a wheel/build
  with sm_120; may require building from source against CUDA 13.0).
- **sgl-kernel**: `pip install sgl-kernel`. Entry point varies by version; the test probes
  the known ones.

## Verification status (RTX 5060 Ti, torch 2.11+cu130)

- `test_sdpa`: PASS (diff ~1e-6 vs naive).
- `test_flashinfer`: PASS (diff ~1e-3 vs SDPA).
- `bench_attention`: runs SDPA + FlashInfer; FlashInfer overtakes SDPA at longer seqlens.
- `test_flashattn`, `test_sgl_kernel`: SKIP (not installed) — install per above to enable.
