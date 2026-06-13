#!/bin/bash
# Single-node multi-GPU training via torchrun (DDP).
#   bash run_multi_gpu.sh --steps 100              # all GPUs
#   NPROC=2 bash run_multi_gpu.sh                   # force GPU count
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash run_multi_gpu.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$DIR/train.py}"
if [ -n "${NPROC:-}" ]; then
    nproc="$NPROC"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nproc=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    nproc=$(nvidia-smi -L | wc -l)
fi

exec torchrun --standalone --nnodes=1 --nproc_per_node="$nproc" "$TRAIN_SCRIPT" "$@"
