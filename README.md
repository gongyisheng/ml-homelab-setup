# ml-homelab-setup

LLM/ML research homelab setup scripts, focus on x86 + NVIDIA GPU.

Consolidates setup, monitoring, inference, training, and kernel-benchmark scripts for a
heterogeneous NVIDIA fleet, parameterized across single-GPU, single-node-multi-GPU, and
multi-node topologies.

## Hardware fleet

| Machine         | GPU(s)        | Arch      | sm     | CUDA      |
|-----------------|---------------|-----------|--------|-----------|
| workstation     | RTX 6000 Pro  | Blackwell | sm_120 | 13.0+     |
| workstation     | RTX 5090      | Blackwell | sm_120 | 13.0+     |
| dev box         | 2× RTX 5060 Ti| Blackwell | sm_120 | 13.0+     |
| server          | L40           | Ada       | sm_89  | 12.x      |

sm_120 (Blackwell) requires **CUDA 13.0+**. This drives build flags and wheel selection
across the kernel and inference modules.

## Modules

| Module                | What it covers                                                    |
|-----------------------|-------------------------------------------------------------------|
| `scripts/bootstrap/`  | NVIDIA driver, CUDA, cuDNN, NCCL, Docker + NVIDIA runtime, checks  |
| `scripts/gpu/`        | env check, fan/power monitor, idle alert, cron setup              |
| `scripts/monitoring/` | Prometheus + node/GPU exporters + Grafana + Cloudflare Tunnel     |
| `scripts/inference/`  | SGLang & vLLM in Docker: single/multi-GPU/multi-node + benchmarks  |
| `scripts/train/`      | single-GPU, multi-GPU (torchrun), multi-node launchers            |
| `scripts/kernel/`     | flashattn / flashinfer / sdpa / sgl-kernel tests + benchmarks     |

## Quickstart

1. `scripts/bootstrap/` — install driver/CUDA/Docker on a fresh box.
2. `scripts/gpu/check_env.py` — confirm the torch/CUDA environment.
3. Pick a module and follow its `README.md`.

See each module's `README.md` for details. Design: `docs/superpowers/specs/`.
