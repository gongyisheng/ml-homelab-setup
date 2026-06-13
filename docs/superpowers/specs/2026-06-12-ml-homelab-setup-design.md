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
    ├── bootstrap/     # driver, CUDA, cuDNN, NCCL, docker installs + version checks
    ├── gpu/           # env check, fans, power, idle alert + cron setup
    ├── monitoring/    # prometheus + node_exporter + gpu exporter (docker compose)
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

### scripts/bootstrap/
Port the ml/nv runbook into runnable scripts.

- `install_driver.sh` — NVIDIA network repo, `nvidia-open` kernel module, reboot note.
- `install_cuda.sh` — install CUDA (13.0+ for sm_120 Blackwell; 12.x for L40),
  set `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH`, support version switching via
  `/usr/local/cuda` symlink.
- `install_cudnn_nccl.sh` — cuDNN + NCCL install.
- `install_docker.sh` — docker engine, then nvidia-container-toolkit: install the toolkit,
  register the NVIDIA runtime with Docker (`nvidia-ctk runtime configure --runtime=docker`),
  restart the daemon, and smoke-test with `docker run --gpus all … nvidia-smi`.
- Version check after install reuses `scripts/gpu/check_env.py` (no duplication).
- `run_nccl_test.sh` — docker NCCL multi-GPU sanity check with the known-good flags
  (`--ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864`),
  documenting the `Cuda failure 304` fix.
- `README.md` — the install runbook + version-check + nvidia-smi diagnostics
  (`watch nvidia-smi`, `nvidia-smi topo -m`, `dmon`).

### scripts/gpu/
Port pretrain/scripts GPU utilities + the ml/nv env diagnostic.

- `check_env.py` — torch/CUDA env diagnostic (ported from ml/nv `check_version.py`):
  CUDA available, torch version + build CUDA, cuDNN, NCCL (guarded), per-GPU name +
  compute capability; defers driver/runtime to `nvidia-smi`/`nvcc`. Also called by
  `bootstrap/` after install.
- `gpu_fans.py` — fan speed monitor/control.
- `gpu_power.py` — power draw monitor / power-cap setter.
- `gpu_idle_alert.sh` — detect idle GPU(s), trigger alert via `send_email.py`.
- `send_email.py` — shared email notification helper.
- `.env.example` — config: email creds, idle threshold, poll interval, power cap.
- `setup_crontab.sh` — install idle-alert (and monitors) as cron jobs.
- `README.md` — usage + how to wire alerts via cron, referencing `.env`.

### scripts/monitoring/
Prometheus metrics stack, run via Docker Compose. Configs below are user-provided and are
the source of truth — reproduce them verbatim, do not invent values. The user runs each
component as its own compose file (host networking); keep that layout.

- `prometheus/docker-compose.yml` + `prometheus/prometheus.yml` — Prometheus server (9090).
- `node-exporter/docker-compose.yml` — host CPU/mem/disk/net (9100).
- `nvidia-smi-exporter/docker-compose.yml` — GPU metrics via `utkuozdemir/nvidia_gpu_exporter`
  (9835), needs the NVIDIA docker runtime from `bootstrap/`.
- `grafana/docker-compose.yml` — Grafana dashboards (3000), Prometheus as datasource.
- Cloudflare Tunnel — no separate folder; documented in `README.md` only. Exposes
  Grafana/exporters without inbound ports. Tunnel token/ingress is a credential the user
  provides; do not invent.
- `README.md` — bring-up (`docker compose up -d` per dir), exporter endpoints, example
  PromQL (GPU util/power/temp, host load), Grafana datasource setup, and the cloudflared
  tunnel setup + ingress map.

Reference config provided by user (node-exporter; smartctl textfile collector removed):

```yaml
version: '3.8'

services:
  node-exporter:
    image: quay.io/prometheus/node-exporter:latest
    container_name: node-exporter
    restart: always
    network_mode: "host"
    pid: "host"
    volumes:
      - "/:/host:ro,rslave"
    command:
      - --path.rootfs=/host
```

`prometheus.yml` (trimmed to the exporters in scope — node-exporter + GPU exporter; other
jobs from the user's full config dropped):

```yaml
global:
  scrape_interval: 10s
  external_labels:
    monitor: 'node'
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
  - job_name: 'nvidia_smi_exporter'
    static_configs:
      - targets: ['localhost:9835']
```

Prometheus `docker-compose.yml` (user-provided):

```yaml
version: '3'
services:
  prometheus:
    container_name: prometheus
    network_mode: host
    pid: host
    user: "0"
    ports:
      - 9090:9090
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - /var/log/prometheus:/prometheus
    restart: always
    image: prom/prometheus
```

GPU exporter `docker-compose.yml` (user-provided; container_name set to
`nvidia_smi_exporter` per user instruction):

```yaml
services:
  nvidia_smi_exporter:
    image: utkuozdemir/nvidia_gpu_exporter:1.4.1
    container_name: nvidia_smi_exporter
    restart: always
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=utility
    ports:
      - "9835:9835"
```

Grafana `docker-compose.yml` (user-provided):

```yaml
version: '3.8'

services:
  grafana:
    image: grafana/grafana
    container_name: grafana
    user: "0"
    ports:
      - "3000:3000"
    volumes:
      - "/home/yisheng/Documents/monitoring/grafana-storage:/var/lib/grafana"
    restart: always
```

Cloudflare Tunnel fronts Grafana/exporters so they're reachable remotely without inbound
port forwarding. It is documented in the monitoring `README.md` (no separate folder). The
tunnel token and ingress config are credentials — the user will paste them, kept as the
source of truth.

(Further monitoring configs with credentials will be pasted by the user and kept here as
the source of truth — do not invent credential values.)

### scripts/inference/
SGLang and vLLM side by side, **run inside their official Docker images** (not bare-metal
installs) — avoids polluting the host and pins the sm_120/CUDA 13.0 toolchain to the image.

```
inference/
├── sglang/  serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh
├── vllm/    serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh
├── bench/   bench_serving.sh (throughput/latency)  compare.py (sglang vs vllm)
└── README.md
```

- Each `serve_*.sh` is a `docker run` wrapper around the framework's official image
  (`lmsysorg/sglang`, `vllm/vllm-openai`), with the NCCL flags from `bootstrap/`
  (`--ipc=host --ulimit memlock=-1 …`), `--gpus`, model/HF-cache volume mounts, and the
  OpenAI-compatible port published.
- `serve_single_gpu.sh` — pin one GPU via `--gpus '"device=N"'`.
- `serve_multi_gpu.sh` — tensor parallel across local GPUs (`--tp N`).
- `serve_multi_node.sh` — multi-node: head-node init addr + nnodes/node-rank, container
  networked with `--network host`.
- `bench/` — drive each server over its OpenAI endpoint, record throughput / TTFT /
  latency, compare frameworks. Can run on host or in image.
- `README.md` — image tags, `docker run` flags per topology, how to mount a model path.

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

1. **Skeleton pass** — create `scripts/{bootstrap,gpu,monitoring,inference,train,kernel}/`, a README stub
   (outline + headers) in each, and the top-level README with the hardware table + module
   index. Commit.
2. **Fill modules in order**: gpu → bootstrap → monitoring → inference → train → kernel
   (gpu first since `bootstrap/` reuses `gpu/check_env.py`). Each fill ports/writes
   the scripts and completes the module README.

## Verification

On the current 2× RTX 5060ti box I can run and verify:
- **gpu/** — nvidia-smi-based utils.
- **monitoring/** — full compose stack (prometheus + exporters) on this box.
- **kernel/** — import + correctness smoke tests, benchmark.
- **inference/** — single-GPU and single-node-multi-GPU (2 GPUs) launch + bench.

Written but **not locally testable** (marked as such in their READMEs):
- multi-node inference and train (no second node available here).
- scripts targeting the other machines (6000 Pro, 5090, L40).

`bootstrap/` scripts are destructive/system-level (driver, CUDA) — verified by review
and `gpu/check_env.py`, not by re-running installs on the dev box.

## Out of scope

- OS provisioning / bare-metal imaging.
- Orchestration (k8s, slurm) — launchers are plain ssh/torchrun/env-var based.
- Model training research itself — train/ provides launchers + a reference entry only.
