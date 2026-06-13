# bootstrap

Base setup for a fresh x86 + NVIDIA box: driver, CUDA, cuDNN, NCCL, Docker (+ NVIDIA
runtime), and version checks. Targets Ubuntu 24.04.

> sm_120 (Blackwell: RTX 6000 Pro, 5090, 5060 Ti) requires **CUDA 13.0+**. L40 (sm_89) uses CUDA 12.x.

## Order

```bash
bash install_driver.sh              # nvidia-open + cuda-keyring repo, then: sudo reboot
bash install_cuda.sh                # CUDA 13.0 -> /usr/local/cuda, env in ~/.bashrc
bash install_cudnn_nccl.sh          # cuDNN + NCCL
bash install_docker.sh              # docker engine
bash install_cuda_container_kit.sh  # nvidia-container-toolkit + runtime + smoke test
```

Each takes env overrides (`CUDA_VERSION`, `CUDA_MAJOR`, `CUDA_DISTRO`, `CUDA_IMAGE`).

`check_env.py` reports the torch/CUDA/cuDNN/NCCL stack, per-GPU compute capability, driver
and nvcc versions, and saves them to `env.txt`; `run_nccl_test.sh` + `test_nccl.py`
validate multi-GPU NCCL.

## Verify

```bash
nvidia-smi                              # driver + GPU health
nvcc -V                                  # CUDA toolkit version
dpkg -l | grep -E 'cudnn|nccl'           # cuDNN / NCCL
uv run python check_env.py               # torch/CUDA/cuDNN/NCCL + per-GPU sm
uv run bash run_nccl_test.sh              # PyTorch + NCCL multi-GPU all-reduce
```

Example `check_env.py` output (2× RTX 5060 Ti dev box):

```
=== Torch / CUDA stack ===
torch          : 2.12.0+cu130
cuda available : True
torch cuda     : 13.0
cudnn enabled  : True
cudnn version  : 92000
nccl version   : 2.29.7
device count   : 2
  gpu 0        : NVIDIA GeForce RTX 5060 Ti (sm_120)
  gpu 1        : NVIDIA GeForce RTX 5060 Ti (sm_120)
driver version : 580.126.20
nvcc cuda      : 13.0
```

Example `run_nccl_test.sh` output (2× RTX 5060 Ti, all-reduce sum = 1+2 = 3):

```
Running NCCL all-reduce on 2 GPU(s).
[rank 0/2] local 1 -> all_reduce sum 3
[rank 1/2] local 2 -> all_reduce sum 3

all_reduce sum = 3 (expected 3) -> PASS
```


## Docker + multi-GPU NCCL

Known issue: `Cuda failure 304 'OS call failed or operation not supported on this OS'`
on docker + multi-GPU. Fix by adding these flags to the container:

```
--ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864
```

- `ipc=host` — container shares host shared memory / IPC
- `seccomp=unconfined` — disable syscall filtering
- `memlock=-1` — GPU ops pin host memory; default 64 KB is too low
- `stack=67108864` — raise per-thread stack from the 8 MB default

`run_nccl_test.sh` runs the host check; inside a container, add the flags above.


## Multi-node NCCL

Two ways to run it. Both use c10d rendezvous on the head; the expected sum scales to
`1+2+...+(NNODES*GPUS_PER_NODE)`, and nodes must reach `HEAD_NODE_IP:RDZV_PORT`
(default 29500). In containers add `--network host`.

Example below: two nodes, **10.0.0.243** (head) and **10.0.0.244** (worker), 1 GPU each.

### Case 1: per node

Log into each instance and run `run_nccl_test.sh` with `HEAD_NODE_IP` set and a distinct
`NODE_RANK`:

```bash
# on 10.0.0.243 (head):
HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=0 GPUS_PER_NODE=1 bash run_nccl_test.sh
# on 10.0.0.244 (worker):
HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=1 GPUS_PER_NODE=1 bash run_nccl_test.sh
```

### Case 2: control instance

From one box with SSH access to both nodes, run `run_nccl_test_multinode.sh` once and it
SSHes into every node for you:

```bash
# from any box with SSH access to both nodes:
NODES="10.0.0.243 10.0.0.244" GPUS_PER_NODE=1 bash run_nccl_test_multinode.sh
NODES="10.0.0.243 10.0.0.244" DRY_RUN=1 bash run_nccl_test_multinode.sh   # print commands without running
```

- First node in `NODES` = head; each node gets the correct `NODE_RANK` and the shared `HEAD_NODE_IP`.
- By default it rsyncs this repo to `/tmp/ml-homelab-setup` on every node (no pre-clone
  needed), then runs `uv run bash run_nccl_test.sh` there — output is streamed prefixed
  `[node N]`, and it exits non-zero if any node fails.
- `SSH_USER`, `REPO_DIR`, `RDZV_PORT` are overridable. `SYNC=0` skips the copy (repo already
  at `REPO_DIR`); `DRY_RUN=1` prints the commands without executing.
- Requirements: the control instance has passwordless SSH + rsync to every node, uv installed
  on each, and all nodes can reach the head's `RDZV_PORT` (29500).

## nvidia-smi diagnostics

| Command                  | Use                                                        |
|--------------------------|------------------------------------------------------------|
| `watch nvidia-smi`       | live GPU health (util, mem, power, temp)                   |
| `nvidia-smi topo -m`     | multi-GPU connectivity (NVLink / PCIe / SYS)              |
| `nvidia-smi dmon`        | per-device monitor (sm/mem/pwr/temp/clocks)               |
| `nvidia-smi dmon -s t`   | PCIe bandwidth (rxpci host→gpu, txpci gpu→host)           |

Bottleneck reading from `dmon`:
- **memory-bound**: mem high, sm high, pwr below cap
- **compute-bound**: mem low, sm high, pwr at cap
- **cpu-bound**: mem low, sm < 50%, pwr below cap

PCIe x16 per-direction bandwidth: Gen3 ~16 GB/s, Gen4 ~32 GB/s, Gen5 ~64 GB/s.
