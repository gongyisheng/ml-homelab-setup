# bootstrap

Base setup for a fresh x86 + NVIDIA box: driver, CUDA, cuDNN, NCCL, Docker (+ NVIDIA
runtime), and version checks.

> sm_120 (Blackwell: RTX 6000 Pro, 5090, 5060 Ti) requires **CUDA 13.0+**. L40 (sm_89) uses CUDA 12.x.

## Scripts

| Script                  | Purpose                                                         |
|-------------------------|-----------------------------------------------------------------|
| `install_driver.sh`     | NVIDIA network repo + `nvidia-open` kernel module (reboot after) |
| `install_cuda.sh`       | CUDA toolkit install + env vars + `/usr/local/cuda` switch       |
| `install_cudnn_nccl.sh` | cuDNN + NCCL install                                             |
| `install_docker.sh`     | Docker engine + nvidia-container-toolkit + runtime configure     |
| `run_nccl_test.sh`      | Multi-GPU NCCL sanity check in Docker (known-good flags)          |

## Verify

```bash
python3 ../gpu/check_env.py     # torch/CUDA/cuDNN/NCCL + per-GPU compute capability
nvcc -V                          # CUDA toolkit version
nvidia-smi                       # driver + GPU health
```

<!-- TODO: fill scripts -->
