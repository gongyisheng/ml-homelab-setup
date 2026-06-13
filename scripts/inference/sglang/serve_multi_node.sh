#!/bin/bash
# SGLang across multiple nodes. TP is the TOTAL GPU count across all nodes.
# Run on every node with the same HEAD_NODE_IP and a distinct NODE_RANK.
#
#   # head (node 0):
#   HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=0 TP=4 bash serve_multi_node.sh
#   # worker (node 1):
#   HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=1 TP=4 bash serve_multi_node.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../_common.sh"

: "${HEAD_NODE_IP:?set HEAD_NODE_IP}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
TP="${TP:-4}"
PORT="${PORT:-30000}"
DIST_PORT="${DIST_PORT:-5000}"
GPUS="${GPUS:-all}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"

# --network host so ranks can reach HEAD_NODE_IP:DIST_PORT across machines.
exec $DOCKER run --rm --gpus "${GPUS}" \
    --network host \
    "${DOCKER_COMMON[@]}" \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --tp "${TP}" \
        --nnodes "${NNODES}" \
        --node-rank "${NODE_RANK}" \
        --dist-init-addr "${HEAD_NODE_IP}:${DIST_PORT}" \
        --host 0.0.0.0 --port "${PORT}"
