# kernel

Attention/kernel backends: import + correctness smoke tests (vs SDPA), then benchmarks.
Attention: PyTorch SDPA, FlashAttention, FlashInfer, Triton, SageAttention, xFormers.
GEMM: Marlin INT4 (gptqmodel) vs fp16, in a separate benchmark.

## Scripts

| Script                | Purpose                                                      |
|-----------------------|--------------------------------------------------------------|
| `_attn_common.py`     | shared inputs, SDPA reference, comparison helper             |
| `_triton_attn.py`     | vendored forward-only Triton flash-attention kernel          |
| `test_sdpa.py`        | SDPA vs naive fp32 attention (the correctness baseline)      |
| `test_flashattn.py`   | FlashAttention import + correctness vs SDPA                  |
| `test_flashinfer.py`  | FlashInfer import + correctness vs SDPA                      |
| `test_triton_attn.py` | Triton flash-attention correctness vs SDPA                   |
| `test_sageattn.py`    | SageAttention (INT8) correctness vs SDPA                     |
| `test_xformers.py`    | xFormers `memory_efficient_attention` correctness vs SDPA    |
| `bench_attention.py`  | attention latency across seqlen for available backends       |
| `bench_marlin.py`     | Marlin INT4 GEMM vs fp16 latency across LLM shapes           |

Each test prints `PASS` / `FAIL`, or `SKIP` with an install hint if the backend is absent.

## Run

Needs the `kernel` extra: `uv sync --extra kernel` (installs FlashInfer + SageAttention).

> ⚠️ `uv sync` **prunes** packages not in `pyproject.toml` — it removes manually-installed
> flash-attn / xformers / gptqmodel. Reinstall them after syncing (flash-attn restores
> instantly from uv's build cache, no rebuild).

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/test_sdpa.py
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/test_flashinfer.py
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/bench_attention.py --seqlens 512,1024,2048,4096
CUDA_VISIBLE_DEVICES=1 uv run python scripts/kernel/bench_marlin.py   # needs gptqmodel
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
- **DeepGEMM**: **not supported on sm_120** — same hardware reason. It only ships SM90
  (WGMMA) and SM100 (`tcgen05`/TMEM) kernels; consumer Blackwell has neither, so kernels
  fail to JIT-compile (`tcgen05 not supported on .target sm_120`). Unlike FA2 there's **no
  source-build workaround** — sm_120-native kernels don't exist yet (tracked in DeepGEMM
  issues [#236](https://github.com/deepseek-ai/DeepGEMM/issues/236),
  [#317](https://github.com/deepseek-ai/DeepGEMM/issues/317)). Skip it on this GPU.
- **Triton**: bundled with torch; the vendored `_triton_attn.py` kernel JIT-compiles and
  runs on sm_120 (Triton treats it as sm_80). No install needed.
- **SageAttention**: `pip install sageattention` gives **v1** (INT8, Triton) — runs on
  sm_120. The fast **v2** CUDA kernels do support sm_120 but have no matching
  torch2.12+cu130 wheel, so they'd need a source build (with a known header patch) — skipped.
- **xFormers**: `pip install xformers` resolves a wheel, but it has **no sm_120 CUDA
  extension** (built for cu128/torch2.10, and its cutlass path caps at capability ≤ 9.0), so
  `memory_efficient_attention` raises at runtime and the bench self-excludes it. A source
  build (`TORCH_CUDA_ARCH_LIST=12.0`) is needed for real sm_120 support — skipped.
- **Marlin (GEMM)**: `pip install gptqmodel`. The Marlin INT4 kernels **JIT-compile and run
  on sm_120** (Ampere+ `mma.sync` path) — they work, unlike DeepGEMM.

## Verification status

**RTX PRO 6000 Blackwell (sm_120), torch 2.12+cu130**
- `test_sdpa`, `test_flashinfer`, `test_triton_attn`: PASS.
- `test_flashattn`: PASS (FA2, max abs diff 1e-4 vs SDPA). Source build took ~3h43m at
  `MAX_JOBS=1`.
- `test_sageattn`: PASS (INT8 v1, diff 0.035 vs SDPA — looser tol, it's quantized).
- `test_xformers`: SKIP (no sm_120 CUDA extension in the wheel).

### Attention benchmark (heads=32, dim=128, fp16, causal — fwd latency ms)

| seqlen | sdpa  | flashinfer | flash_attn | triton | sageattn (INT8) |
|-------:|------:|-----------:|-----------:|-------:|----------------:|
| 512    | 0.065 | 0.046      | 0.125      | 0.074  | 0.216 |
| 1024   | 0.056 | 0.062      | 0.122      | 0.094  | 0.215 |
| 2048   | 0.140 | 0.153      | 0.140      | 0.311  | 0.225 |
| 4096   | 0.460 | 0.476      | 0.453      | 1.137  | 0.443 |
| 8192   | 1.628 | 1.724      | 2.244      | 5.924  | 2.074 |

**SDPA is good enough** — it wins or ties almost everywhere on Blackwell, so just use it.
FlashInfer is fastest at short seqlens (JIT-tuned); FA2 ties SDPA mid-range; the vendored
Triton kernel is a correctness reference (unoptimized, scales poorly); SageAttention v1
(Triton INT8) only reaches parity at 4096. None beats SDPA across the board — the libs are
for parity/compat, not a speedup here.

### Marlin INT4 GEMM benchmark (group=128, fp16 act — fwd latency ms)

| K × N       | M   | fp16   | marlin | speedup |
|-------------|----:|-------:|-------:|--------:|
| 4096×4096   | 16  | 0.024  | 0.061  | 0.39×  |
| 4096×4096   | 256 | 0.033  | 0.069  | 0.48×  |
| 4096×11008  | 256 | 0.062  | 0.067  | 0.93×  |
| 11008×4096  | 256 | 0.074  | 0.069  | 1.08×  |

Correctness vs dequantized ref: rel.err ~4e-4. **Marlin runs correctly on sm_120 but isn't
a latency win here** — Blackwell's fp16 tensor cores + bandwidth make cuBLAS fast, so INT4
dequant overhead only pays off (~1×) at larger batch on big shapes. Its real value on this
GPU is the **4× VRAM reduction**, not speed.
