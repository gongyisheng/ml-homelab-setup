#!/bin/bash
# vLLM across multiple nodes via a Ray cluster (vLLM's multi-node mechanism).
# Run on EVERY node with the same HEAD_NODE_IP; the head also drives serving.
# TP = GPUs per node, PP = number of nodes  (TP * PP = total GPUs).
#
#   # head (node 0), pc2:
#   ROLE=head   HEAD_NODE_IP=10.0.0.101 TP=1 PP=2 GPUS=device=0 bash serve_multi_node.sh
#   # worker (node 1), pc3:
#   ROLE=worker HEAD_NODE_IP=10.0.0.101 GPUS=device=1 bash serve_multi_node.sh
#
# Homelab notes (learned the hard way on a heterogeneous, 1GbE pair):
#  - The stock vllm/vllm-openai image ships WITHOUT Ray; we pip-install it at start.
#  - vLLM needs --distributed-executor-backend ray (it will NOT auto-detect the cluster).
#  - HOST_IP / IFACE are auto-detected and pinned so NCCL+Gloo use the LAN interface,
#    not the hostname's 127.0.1.1 (Ubuntu /etc/hosts) which makes cross-node connect fail.
#  - GPU_MEM_UTIL / MAX_MODEL_LEN are sized for the SMALLEST GPU in the cluster.
#  - Pipeline parallel tolerates uneven GPUs across nodes (each node holds whole layers).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../_common.sh"

ROLE="${ROLE:?set ROLE=head|worker}"
: "${HEAD_NODE_IP:?set HEAD_NODE_IP}"
TP="${TP:-1}"
PP="${PP:-2}"
PORT="${PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"
GPUS="${GPUS:-device=0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"   # FULL cuda-graph capture needs transient headroom on a 16GB GPU
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
CNAME="${CNAME:-vllm-node}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

# Routable IP of this node + the interface that owns it (auto-detected, override-able).
HOST_IP="${HOST_IP:-$(ip -4 route get "$HEAD_NODE_IP" 2>/dev/null | grep -oP 'src \K\S+')}"
IFACE="${IFACE:-$(ip -o -4 addr show | awk -v ip="$HOST_IP" '$4 ~ "^"ip"/" {print $2; exit}')}"
: "${HOST_IP:?could not auto-detect HOST_IP; set it}"
: "${IFACE:?could not auto-detect IFACE; set it}"

if [ "$ROLE" = head ]; then
    RAY_CMD="ray start --head --node-ip-address=${HOST_IP} --port=${RAY_PORT} --block"
else
    # `while true` keeps the container alive across transient join failures (head GCS not yet
    # accepting); on success `ray start --block` blocks and the loop never iterates again.
    RAY_CMD="while true; do ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --node-ip-address=${HOST_IP} --block; echo 'ray join failed, retry in 5s'; sleep 5; done"
fi

$DOCKER rm -f "$CNAME" >/dev/null 2>&1 || true
# Long-lived container per node: install Ray (absent from the image), join the cluster.
# No --rm: a dead container is left for inspection; the cluster wrapper's down() reaps it.
$DOCKER run -d --gpus "${GPUS}" --network host --name "$CNAME" \
    -e VLLM_HOST_IP="${HOST_IP}" \
    -e NCCL_SOCKET_IFNAME="${IFACE}" -e GLOO_SOCKET_IFNAME="${IFACE}" \
    "${DOCKER_COMMON[@]}" \
    --entrypoint /bin/bash "${IMAGE}" \
    -c "pip install -q ray && ${RAY_CMD}"

if [ "$ROLE" != head ]; then
    echo "Worker joining Ray at ${HEAD_NODE_IP}:${RAY_PORT} (iface ${IFACE}, ip ${HOST_IP})."
    exit 0
fi

# Head: wait for all TP*PP GPUs to register in the cluster, then start serving.
TOTAL=$((TP * PP))
echo "Ray head up on ${HOST_IP}. Waiting for ${TOTAL} GPUs to join..."
for _ in $(seq 1 120); do
    $DOCKER exec "$CNAME" ray status 2>/dev/null | grep -qE "/${TOTAL}\.0 GPU" && break
    sleep 5
done
# Serve detached inside the container so it survives this script (and any ssh) exiting.
# Logs land in /tmp/vllm-serve.log inside the container (see `serve_multi_node_cluster.sh logs`).
$DOCKER exec -d "$CNAME" bash -c "vllm serve '${MODEL}' \
    --tensor-parallel-size ${TP} --pipeline-parallel-size ${PP} \
    --distributed-executor-backend ray \
    --gpu-memory-utilization ${GPU_MEM_UTIL} --max-model-len ${MAX_MODEL_LEN} \
    --host 0.0.0.0 --port ${PORT} > /tmp/vllm-serve.log 2>&1"
echo "vLLM serve launched (detached) in ${CNAME} on :${PORT}."
