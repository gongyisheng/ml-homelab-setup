#!/bin/bash
# SGLang across multiple nodes (tensor parallel). TP is the TOTAL GPU count across all nodes.
# Run on EVERY node with the same HEAD_NODE_IP and a distinct NODE_RANK.
#
#   # head (rank 0), pc2:
#   HEAD_NODE_IP=10.0.0.101 NNODES=2 NODE_RANK=0 TP=2 GPUS=device=0 bash serve_multi_node.sh
#   # worker (rank 1), pc3:
#   HEAD_NODE_IP=10.0.0.101 NNODES=2 NODE_RANK=1 TP=2 GPUS=device=1 bash serve_multi_node.sh
#
# Homelab notes (heterogeneous GPUs / 1GbE interconnect):
#  - IFACE auto-detected and pinned so NCCL+Gloo use the LAN interface, not the
#    hostname's 127.0.1.1 (Ubuntu /etc/hosts) which makes cross-node connect fail.
#  - SGLang aborts when GPU memory is unbalanced across TP ranks (tensor parallel shards
#    every layer and assumes symmetric ranks). SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
#    relaxes that to a warning; the cluster is then gated by the SMALLEST GPU.
#  - MEM_FRACTION / CONTEXT_LEN are sized for the smallest GPU.
#  - Cross-node CUDA-graph capture HANGS over a slow link (it captures NCCL collectives
#    across nodes). DISABLE_CUDA_GRAPH=1 (default) runs eager; set 0 on a fast interconnect.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../_common.sh"

: "${HEAD_NODE_IP:?set HEAD_NODE_IP}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
TP="${TP:-2}"
PORT="${PORT:-30000}"
DIST_PORT="${DIST_PORT:-5000}"
GPUS="${GPUS:-device=0}"
MEM_FRACTION="${MEM_FRACTION:-0.8}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
CNAME="${CNAME:-sglang-node}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"

HOST_IP="${HOST_IP:-$(ip -4 route get "$HEAD_NODE_IP" 2>/dev/null | grep -oP 'src \K\S+')}"
IFACE="${IFACE:-$(ip -o -4 addr show | awk -v ip="$HOST_IP" '$4 ~ "^"ip"/" {print $2; exit}')}"
: "${IFACE:?could not auto-detect IFACE; set it}"

GRAPH_FLAGS=()
[ "$DISABLE_CUDA_GRAPH" = 1 ] && GRAPH_FLAGS=(--disable-cuda-graph --disable-piecewise-cuda-graph)

$DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
# Detached (-d) so this script exits and the cluster wrapper can launch the next rank;
# --network host so ranks reach HEAD_NODE_IP:DIST_PORT across machines. Logs: docker logs $CNAME.
$DOCKER run -d --gpus "${GPUS}" --network host --name "$CNAME" \
    -e NCCL_SOCKET_IFNAME="${IFACE}" -e GLOO_SOCKET_IFNAME="${IFACE}" \
    -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
    "${DOCKER_COMMON[@]}" \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --tp "${TP}" \
        --nnodes "${NNODES}" --node-rank "${NODE_RANK}" \
        --dist-init-addr "${HEAD_NODE_IP}:${DIST_PORT}" \
        --mem-fraction-static "${MEM_FRACTION}" \
        --context-length "${CONTEXT_LEN}" \
        "${GRAPH_FLAGS[@]}" \
        --host 0.0.0.0 --port "${PORT}" >/dev/null
echo "SGLang rank ${NODE_RANK}/${NNODES} launched (detached) in ${CNAME}."
