#!/bin/bash
# PyTorch + NCCL + multi-GPU all-reduce sanity check (single node).
#   bash run_nccl_test.sh                          # all visible GPUs
#   CUDA_VISIBLE_DEVICES=1 bash run_nccl_test.sh    # single GPU
#   NPROC=2 bash run_nccl_test.sh                   # force GPU count
#
# Inside Docker, multi-GPU NCCL needs these flags (fixes `Cuda failure 304`):
#   --ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${NPROC:-}" ]; then
    nproc="$NPROC"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nproc=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    nproc=$(nvidia-smi -L | wc -l)
fi

echo "Running NCCL all-reduce on $nproc GPU(s)."
torchrun --standalone --nnodes=1 --nproc_per_node="$nproc" "$SCRIPT_DIR/test_nccl.py"
