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

Needs the `kernel` extra: `uv sync --extra kernel` (installs FlashInfer).

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/test_sdpa.py
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/test_flashinfer.py
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/bench_attention.py --seqlens 512,1024,2048,4096
```

## Install notes (sm_120 / CUDA 13.0)

Blackwell sm_120 needs Blackwell-built wheels — generic PyPI wheels may not include the
`sm_120` arch and will fail at load or fall back to slow paths.

- **FlashInfer**: `pip install flashinfer-python==0.6.12` (the version verified on sm_120
  here). JIT-compiles kernels on first use.
- **FlashAttention**: FA3/FA4 are **not supported on sm_120** — a hardware mismatch, not a
  missing build target. "Blackwell" covers two different tensor-core designs, and sm_120
  has neither's key feature:
  - **FA3** needs **WGMMA** + **TMA** (Hopper/SM90, H100) — async matmul + copy engine.
  - **FA4** needs **tcgen05** + the **TMEM** tensor-memory block (data-center Blackwell/
    SM100, B200).
  - **sm_120** (consumer/workstation Blackwell, RTX 5090 / RTX PRO 6000) has neither — just
    an extended Ampere-era `mma.sync` (HMMA) with new FP4/FP6 datatypes. `ptxas` rejects the
    instructions outright (`wgmma.fence` / `tcgen05.*` "not supported on .target sm_120"),
    so it falls back to **FA2**.

  `flash-attn` (FA2) is the **general, widely-supported** kernel (Ampere onward), 
  but ships **no prebuilt sm_120 wheel** — so it builds from source against CUDA 13.0, 
  **slow and RAM-heavy** (can take hours). Cap parallelism and target only sm_120 so 
  it doesn't OOM or build every arch:
  ```bash
  MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="12.0" \
    uv pip install flash-attn==2.8.3.post1 --no-build-isolation
  ```
  Peak RAM ≈ `MAX_JOBS` × 25 GB — raise `MAX_JOBS` only if you have the headroom. Needs
  `ninja` (in the `kernel` extra) or the build is serial and far slower. Run under
  `tmux`/`nohup` so a dropped SSH session doesn't kill the multi-hour build.
- **sgl-kernel**: `pip install sgl-kernel`. Entry point varies by version; the test probes
  the known ones.

## Verification status (RTX 5060 Ti, torch 2.11+cu130)

- `test_sdpa`: PASS (diff ~1e-6 vs naive).
- `test_flashinfer`: PASS (diff ~1e-3 vs SDPA).
- `test_flashattn`, `test_sgl_kernel`: SKIP (not installed) — install per above to enable.
