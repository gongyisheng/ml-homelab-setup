#!/bin/bash
# PyTorch + NCCL all-reduce sanity check. Single node by default; multi-node when
# HEAD_NODE_IP is set (run on every node with a distinct NODE_RANK).
#
# Single node:
#   bash run_nccl_test.sh                           # all visible GPUs
#   CUDA_VISIBLE_DEVICES=1 bash run_nccl_test.sh    # single GPU
#   NPROC=2 bash run_nccl_test.sh                   # force GPU count
#
# Multi node (2 nodes, 2 GPUs each):
#   # on the head node:
#   HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=0 GPUS_PER_NODE=2 bash run_nccl_test.sh
#   # on the worker node:
#   HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=1 GPUS_PER_NODE=2 bash run_nccl_test.sh
#
# Inside Docker, multi-GPU NCCL needs these flags (fixes `Cuda failure 304`):
#   --ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864
# Multi-node containers also need --network host so ranks can reach HEAD_NODE_IP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

count_gpus() {
    if [ -n "${NPROC:-}" ]; then
        echo "$NPROC"
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .
    else
        nvidia-smi -L | wc -l
    fi
}

if [ -n "${HEAD_NODE_IP:-}" ]; then
    # --- multi-node ---
    NNODES="${NNODES:-2}"
    NODE_RANK="${NODE_RANK:-0}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-$(count_gpus)}"
    RDZV_PORT="${RDZV_PORT:-29500}"
    echo "NCCL all-reduce: ${NNODES} nodes x ${GPUS_PER_NODE} GPU (this is node ${NODE_RANK}), head ${HEAD_NODE_IP}."
    # Keep NCCL off virtual interfaces (docker0, br-*, veth, lo); it otherwise may
    # pick a docker bridge (e.g. 172.18.0.1) that isn't routable between nodes.
    # Override NCCL_SOCKET_IFNAME if your real NIC doesn't match the default.
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,br-,veth,lo,virbr0}"
    # Static rendezvous: node-rank 0 is unconditionally the master and hosts the
    # store. Avoids the c10d backend's host election, which fails when the head's
    # hostname resolves to 127.0.1.1 (Ubuntu default) instead of HEAD_NODE_IP.
    exec torchrun \
        --nnodes="$NNODES" \
        --node-rank="$NODE_RANK" \
        --nproc_per_node="$GPUS_PER_NODE" \
        --master-addr="$HEAD_NODE_IP" \
        --master-port="$RDZV_PORT" \
        "$SCRIPT_DIR/test_nccl.py"
else
    # --- single node ---
    nproc=$(count_gpus)
    echo "NCCL all-reduce on $nproc GPU(s) (single node)."
    exec torchrun --standalone --nnodes=1 --nproc_per_node="$nproc" "$SCRIPT_DIR/test_nccl.py"
fi
