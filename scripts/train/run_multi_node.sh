#!/bin/bash
# Multi-node training via torchrun, static rendezvous (node-rank 0 is the master).
# Run on every node with the same HEAD_NODE_IP and a distinct NODE_RANK.
#
#   # head (node 0):
#   HEAD_NODE_IP=10.0.0.244 NNODES=2 NODE_RANK=0 GPUS_PER_NODE=1 bash run_multi_node.sh --steps 200
#   # worker (node 1):
#   HEAD_NODE_IP=10.0.0.244 NNODES=2 NODE_RANK=1 GPUS_PER_NODE=1 bash run_multi_node.sh --steps 200
#
# Static (not c10d) rendezvous on purpose: c10d elects the store host by matching the
# endpoint against the machine's resolvable addresses, which fails when the hostname maps
# to 127.0.1.1 (stock Ubuntu /etc/hosts) -- no node becomes host and every rank hangs as a
# client. Static makes node-rank 0 the master unconditionally.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$DIR/train.py}"
: "${HEAD_NODE_IP:?set HEAD_NODE_IP}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
RDZV_PORT="${RDZV_PORT:-29500}"

# Keep NCCL off docker/virtual bridges: br-*/docker0 can carry the same 172.x subnet on
# every node, which is unroutable cross-host and hangs collectives. Exclusion syntax (^)
# is used so it works regardless of the LAN NIC name (enp7s0 vs enp3s0 across the fleet).
# Override by exporting NCCL_SOCKET_IFNAME yourself.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker,br-,lo,veth,virbr}"

exec torchrun \
    --nnodes="$NNODES" \
    --node-rank="$NODE_RANK" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master-addr="$HEAD_NODE_IP" \
    --master-port="$RDZV_PORT" \
    "$TRAIN_SCRIPT" "$@"
