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

`check_env.py` reports the torch/CUDA/cuDNN/NCCL stack and per-GPU compute capability;
`run_nccl_test.sh` + `test_nccl.py` validate multi-GPU NCCL.

## Verify

```bash
nvidia-smi                              # driver + GPU health
nvcc -V                                  # CUDA toolkit version
dpkg -l | grep -E 'cudnn|nccl'           # cuDNN / NCCL
uv run python check_env.py               # torch/CUDA/cuDNN/NCCL + per-GPU sm
uv run bash run_nccl_test.sh              # PyTorch + NCCL multi-GPU all-reduce
```

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

## Docker + NCCL multi-GPU

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
