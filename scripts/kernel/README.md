# kernel

Attention/kernel backends: import + correctness smoke tests (vs SDPA), then benchmarks.
Backends: PyTorch SDPA, FlashAttention, FlashInfer.

## Scripts

| Script               | Purpose                                                       |
|----------------------|---------------------------------------------------------------|
| `_attn_common.py`    | shared inputs, SDPA reference, comparison helper             |
| `test_sdpa.py`       | SDPA vs naive fp32 attention (the correctness baseline)      |
| `test_flashattn.py`  | FlashAttention import + correctness vs SDPA                  |
| `test_flashinfer.py` | FlashInfer import + correctness vs SDPA                      |
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
  **slow and RAM-heavy** (~4.5 h at `MAX_JOBS=1`). Cap parallelism and target only sm_120 
  so it doesn't OOM or build every arch:
  ```bash
  MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="12.0" \
    uv pip install flash-attn==2.8.3.post1 --no-build-isolation
  ```
  Peak RAM ≈ `MAX_JOBS` × 25 GB — raise `MAX_JOBS` only if you have the headroom. Needs
  `ninja` (in the `kernel` extra) or the build is serial and far slower. Run under
  `tmux`/`nohup` so a dropped SSH session doesn't kill the multi-hour build.

## Verification status

**RTX PRO 6000 Blackwell (sm_120), torch 2.12+cu130**
- `test_sdpa`, `test_flashinfer`: PASS.
- `test_flashattn`: PASS (FA2, max abs diff 1e-4 vs SDPA). Source build took ~3h43m at
  `MAX_JOBS=1`.

### Benchmark (heads=32, dim=128, fp16, causal — fwd latency ms)

| seqlen | sdpa  | flashinfer | flash_attn (FA2) |
|-------:|------:|-----------:|-----------------:|
| 512    | 0.063 | 0.125      | 0.122 |
| 1024   | 0.059 | 0.062      | 0.122 |
| 2048   | 0.140 | 0.153      | 0.140 |
| 4096   | 0.460 | 0.469      | 0.454 |
| 8192   | 1.629 | 1.710      | 2.250 |

**SDPA is good enough** — it wins or ties at every seqlen on Blackwell, so just use it.
FA2/FlashInfer are for parity/compat testing, not a speedup here.
