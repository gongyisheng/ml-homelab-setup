#!/bin/bash
# Multi-node training via torchrun with c10d rendezvous (miles-style env wrapper).
# Run on every node with the same HEAD_NODE_IP and a distinct NODE_RANK.
#
#   # head (node 0):
#   HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=0 GPUS_PER_NODE=2 bash run_multi_node.sh --steps 200
#   # worker (node 1):
#   HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=1 GPUS_PER_NODE=2 bash run_multi_node.sh --steps 200
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$DIR/train.py}"
: "${HEAD_NODE_IP:?set HEAD_NODE_IP}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
RDZV_PORT="${RDZV_PORT:-29500}"

exec torchrun \
    --nnodes="$NNODES" \
    --node-rank="$NODE_RANK" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${HEAD_NODE_IP}:${RDZV_PORT}" \
    "$TRAIN_SCRIPT" "$@"
