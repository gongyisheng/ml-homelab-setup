# ml-homelab-setup

LLM/ML research homelab setup scripts, focus on x86 + NVIDIA GPU.

Currently I have following dedicate hardwares: 
1 RTX 6000 Pro Blackwell, 1 RTX 5090 (rent from my friend), 1 2X RTX 5060ti, 1 A10G (company), 1 L40S (company)

and some shared instances:
1 4XL40S (company, but shared, cannot use for a long time), and some 8XH200/B200 provided by sglang community shared by all developers.

I run [pretrain](https://github.com/gongyisheng/pretrain) experiments heavily on RTX 6000 pro, RTX 5090 and L40S. and 2X 5060ti usually for ci, kernel dev and nccl tests. I host 3 instances at my home and they are connected with ethernet switch.

Sometimes I also do distributed training (eg, multi node RL). I had a lot of fun hosting the homelab, some funny stories like:
- [electricity bills](https://x.com/Orange41324306/status/2065662526155788380)
- [electricity outage](https://x.com/Orange41324306/status/2065366056127107575)

## Hardware fleet

| GPU(s)        | Arch      | sm     | CUDA      |
|---------------|-----------|--------|-----------|
| RTX 6000 Pro  | Blackwell | sm_120 | 13.0+     |
| RTX 5090      | Blackwell | sm_120 | 13.0+     |
| 2× RTX 5060 Ti| Blackwell | sm_120 | 13.0+     |
| A10G          | Ampere    | sm_86  | 12.x      |
| L40S          | Ada       | sm_89  | 12.x      |

sm_120 (Blackwell) requires **CUDA 13.0+**. This drives build flags and wheel selection
across the kernel and inference modules.

## Modules

| Module                | What it covers                                                    |
|-----------------------|-------------------------------------------------------------------|
| `scripts/bootstrap/`  | NVIDIA driver, CUDA, cuDNN, NCCL, Docker + runtime, env/NCCL checks|
| `scripts/gpu/`        | fan/power monitor, idle alert, cron setup                        |
| `scripts/monitoring/` | Prometheus + node/GPU exporters + Grafana + Cloudflare Tunnel     |
| `scripts/inference/`  | SGLang & vLLM in Docker: single/multi-GPU/multi-node + benchmarks  |
| `scripts/train/`      | single-GPU, multi-GPU (torchrun), multi-node launchers            |
| `scripts/kernel/`     | flashattn / flashinfer / sdpa / sgl-kernel tests + benchmarks     |

## Python environment

Managed with [uv](https://docs.astral.sh/uv/). torch is pinned to the CUDA 13.0 build
(sm_120 Blackwell; also runs the L40).

```bash
uv sync                 # base deps (torch, nvidia-ml-py)
uv sync --extra kernel  # + FlashInfer & ninja for the kernel benchmarks
uv run python scripts/bootstrap/check_env.py
```

flash-attn and sgl-kernel need Blackwell-built wheels and are installed manually — see
`scripts/kernel/README.md`.

## Quickstart

1. `scripts/bootstrap/` — install driver/CUDA/Docker on a fresh box.
2. `uv sync` then `uv run python scripts/bootstrap/check_env.py` — confirm the torch/CUDA environment.
3. Pick a module and follow its `README.md`.

See each module's `README.md` for details.
