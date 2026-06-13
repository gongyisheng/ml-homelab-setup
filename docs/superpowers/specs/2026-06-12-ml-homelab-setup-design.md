# ml-homelab-setup — Design

LLM/ML research homelab setup scripts for x86 + NVIDIA GPU. Ports and consolidates
prior art (ml/nv runbook, pretrain/scripts GPU utils, miles multi-node launcher) into
one repo, parameterized across single-GPU, single-node-multi-GPU, and multi-node.

## Hardware fleet

| Machine          | GPU(s)        | Arch     | sm    | Notes                          |
|------------------|---------------|----------|-------|--------------------------------|
| workstation      | RTX 6000 Pro  | Blackwell| sm_120| CUDA 13.0+ required            |
| workstation      | RTX 5090      | Blackwell| sm_120| CUDA 13.0+ required            |
| **current box**  | 2× RTX 5060ti | Blackwell| sm_120| CUDA 13.0+; multi-GPU dev box  |
| server           | L40           | Ada      | sm_89 | CUDA 12.x                      |

Scripts cover three topologies: **single-node single-GPU**, **single-node multi-GPU**
(TP), and **multi-node** (head-node IP + node rank, miles-style).

## Repo structure

```
ml-homelab-setup/
├── README.md          # hardware inventory table + module index + quickstart
└── scripts/
    ├── init/          # driver, CUDA, cuDNN, NCCL, docker installs + version checks
    ├── gpu/           # fans, power, idle alert + cron setup
    ├── inference/     # sglang + vllm: single, multi-gpu, multi-node + bench
    ├── train/         # single-gpu, multi-gpu, multi-node launchers
    └── kernel/        # flashattn, flashinfer, sdpa, sgl-kernel tests + bench
```

Conventions:
- Each module owns a `README.md` runbook plus runnable scripts (bash + python),
  matching the existing ml/nv + pretrain style.
- Shared helpers live where first used (e.g. `send_email.py` under `scripts/gpu/`);
  no separate top-level common dir.
- The top README hardware table is the source of truth for sm/CUDA build flags used
  across kernel and inference modules.

## Module scope

### scripts/init/
Port the ml/nv runbook into runnable scripts.

- `install_driver.sh` — NVIDIA network repo, `nvidia-open` kernel module, reboot note.
- `install_cuda.sh` — install CUDA (13.0+ for sm_120 Blackwell; 12.x for L40),
  set `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH`, support version switching via
  `/usr/local/cuda` symlink.
- `install_cudnn_nccl.sh` — cuDNN + NCCL install.
- `install_docker.sh` — docker engine + nvidia-container-toolkit.
- `check_version.py` — report driver, CUDA (`nvcc`), cuDNN, NCCL, and torch versions in
  one place; flag mismatches against the target.
- `run_nccl_test.sh` — docker NCCL multi-GPU sanity check with the known-good flags
  (`--ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864`),
  documenting the `Cuda failure 304` fix.
- `README.md` — the install runbook + version-check + nvidia-smi diagnostics
  (`watch nvidia-smi`, `nvidia-smi topo -m`, `dmon`).

### scripts/gpu/
Port pretrain/scripts GPU utilities.

- `gpu_fans.py` — fan speed monitor/control.
- `gpu_power.py` — power draw monitor / power-cap setter.
- `gpu_idle_alert.sh` — detect idle GPU(s), trigger alert via `send_email.py`.
- `send_email.py` — shared email notification helper.
- `.env.example` — config: email creds, idle threshold, poll interval, power cap.
- `setup_crontab.sh` — install idle-alert (and monitors) as cron jobs.
- `README.md` — usage + how to wire alerts via cron, referencing `.env`.

### scripts/inference/
SGLang and vLLM side by side, one set of launchers per framework per topology.

```
inference/
├── sglang/  serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh
├── vllm/    serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh
├── bench/   bench_serving.sh (throughput/latency)  compare.py (sglang vs vllm)
└── README.md
```

- `serve_multi_gpu.sh` — tensor parallel across local GPUs.
- `serve_multi_node.sh` — head-node init addr + nnodes/node-rank.
- `bench/` — drive each server, record throughput / TTFT / latency, compare frameworks.
- `README.md` — launch flags per topology, how to point at a model path.

### scripts/train/
miles-style launchers.

- `run_single_gpu.sh` — single GPU.
- `run_multi_gpu.sh` — `torchrun` single node, all local GPUs.
- `run_multi_node.sh` — env-var wrapper (`HEAD_NODE_IP`, `GPUS_PER_NODE`, node rank),
  invoking the underlying training entry per node.
- A small reference training entry (LoRA/SFT on a small model) so launchers run end-to-end.
- `README.md` — rendezvous/topology config, how to scale node count.

### scripts/kernel/
Attention/kernel backends: import + correctness smoke tests, then benchmark.

- `test_flashattn.py`, `test_flashinfer.py`, `test_sdpa.py`, `test_sgl_kernel.py` —
  import + small correctness check (compare output vs sdpa reference).
- `bench_attention.py` — compare backends across seqlen / heads / dtype; report latency.
- `README.md` — per-backend install notes (Blackwell sm_120 wheel availability is the
  key caveat) + how to read the benchmark output.

## Build sequence

1. **Skeleton pass** — create `scripts/{init,gpu,inference,train,kernel}/`, a README stub
   (outline + headers) in each, and the top-level README with the hardware table + module
   index. Commit.
2. **Fill modules in order**: init → gpu → inference → train → kernel. Each fill ports/writes
   the scripts and completes the module README.

## Verification

On the current 2× RTX 5060ti box I can run and verify:
- **gpu/** — nvidia-smi-based utils.
- **kernel/** — import + correctness smoke tests, benchmark.
- **inference/** — single-GPU and single-node-multi-GPU (2 GPUs) launch + bench.

Written but **not locally testable** (marked as such in their READMEs):
- multi-node inference and train (no second node available here).
- scripts targeting the other machines (6000 Pro, 5090, L40).

`init/` scripts are destructive/system-level (driver, CUDA) — verified by review and
`check_version.py`, not by re-running installs on the dev box.

## Out of scope

- OS provisioning / bare-metal imaging.
- Orchestration (k8s, slurm) — launchers are plain ssh/torchrun/env-var based.
- Model training research itself — train/ provides launchers + a reference entry only.
