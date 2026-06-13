#!/bin/bash
# vLLM across multiple nodes via a Ray cluster (vLLM's multi-node mechanism).
# Run on every node with the same HEAD_NODE_IP and the right ROLE.
# TP = GPUs per node, PP = number of nodes  (TP * PP = total GPUs).
#
#   # head (node 0):
#   ROLE=head   HEAD_NODE_IP=10.0.0.243 TP=2 PP=2 bash serve_multi_node.sh
#   # worker (node 1):
#   ROLE=worker HEAD_NODE_IP=10.0.0.243 bash serve_multi_node.sh
#
# NOTE: written for the multi-node topology; untested in this homelab (single box).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../_common.sh"

ROLE="${ROLE:?set ROLE=head|worker}"
: "${HEAD_NODE_IP:?set HEAD_NODE_IP}"
TP="${TP:-2}"
PP="${PP:-2}"
PORT="${PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"
GPUS="${GPUS:-all}"
CNAME="${CNAME:-vllm-node}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

if [ "$ROLE" = head ]; then
    RAY_ARGS="--head --port=${RAY_PORT}"
else
    RAY_ARGS="--address=${HEAD_NODE_IP}:${RAY_PORT}"
fi

# Long-lived container on each node that joins the Ray cluster.
$DOCKER run -d --rm --gpus "${GPUS}" --network host --name "$CNAME" \
    "${DOCKER_COMMON[@]}" \
    --entrypoint /bin/bash "${IMAGE}" \
    -c "ray start ${RAY_ARGS} --block"

# The head node drives serving once the cluster is formed.
if [ "$ROLE" = head ]; then
    echo "Ray head up. Give workers a moment to join, then serving on :${PORT}..."
    sleep 10
    exec $DOCKER exec "$CNAME" \
        vllm serve "${MODEL}" \
            --tensor-parallel-size "${TP}" \
            --pipeline-parallel-size "${PP}" \
            --host 0.0.0.0 --port "${PORT}"
fi
