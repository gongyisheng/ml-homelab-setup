#!/bin/bash
# vLLM prefill/decode (PD) disaggregation across two nodes, driven from this control box.
# One prefill instance (kv_producer) + one decode instance (kv_consumer) exchange KV cache
# over NIXL/UCX; a proxy routes each request prefill->decode.
#
#   bash serve_pd_disaggregation.sh up      # start prefill + decode + proxy
#   bash serve_pd_disaggregation.sh down
#   bash serve_pd_disaggregation.sh logs
#
# Roles map to the homelab: prefill on pc2 (big GPU, compute-bound), decode on pc3's free
# GPU 1 (KV-bound). NIXL has no RDMA here, so UCX falls back to TCP over the 1GbE LAN — the
# KV transfer is the bottleneck; this is a correctness demo, not a speedup on this hardware.
#
# NOTE: --enforce-eager (cuda graph capture is skipped; it deadlocks cross-node here anyway).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/_common.sh" 2>/dev/null || source "$DIR/../_common.sh"

ACTION="${1:-up}"
SSH_USER="${SSH_USER:-yisheng}"
PREFILL_NODE="${PREFILL_NODE:-10.0.0.101}"; PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_NODE="${DECODE_NODE:-10.0.0.244}";  DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_PORT="${PREFILL_PORT:-8100}"; DECODE_PORT="${DECODE_PORT:-8200}"; PROXY_PORT="${PROXY_PORT:-8000}"
SIDE_PORT="${SIDE_PORT:-5559}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
PROXY="/vllm-workspace/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py"

ssh_node() { ssh -n -o ConnectTimeout=10 "${SSH_USER}@$1" "$2"; }
iface_of() { ssh_node "$1" "ip -o -4 addr show | awk -v ip=$1 '\$4 ~ \"^\"ip\"/\" {print \$2; exit}'"; }

down() {
    for n in "$PREFILL_NODE" "$DECODE_NODE"; do
        ssh_node "$n" "docker rm -f vllm-prefill vllm-decode vllm-proxy >/dev/null 2>&1 || true; echo '  '$n' cleaned'"
    done
}

# Launch one vLLM PD role as a detached container. $1 node, $2 gpu, $3 name, $4 port, $5 kv_role, $6 iface
launch_role() {
    local node="$1" gpu="$2" name="$3" port="$4" role="$5" iface="$6"
    ssh_node "$node" "docker rm -f $name >/dev/null 2>&1 || true; docker run -d --gpus device=$gpu --network host --name $name \
        -e VLLM_NIXL_SIDE_CHANNEL_HOST=$node -e VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_PORT \
        -e UCX_NET_DEVICES=$iface -e NCCL_SOCKET_IFNAME=$iface -e GLOO_SOCKET_IFNAME=$iface \
        --ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 \
        -v \$HOME/.cache/huggingface:/root/.cache/huggingface --entrypoint bash $IMAGE \
        -c \"vllm serve '$MODEL' --port $port --enforce-eager --enable-request-id-headers \
            --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len $MAX_MODEL_LEN \
            --kv-transfer-config '{\\\"kv_connector\\\":\\\"NixlConnector\\\",\\\"kv_role\\\":\\\"$role\\\"}' \
            > /tmp/$name.log 2>&1\" >/dev/null && echo \"  $node: $name ($role) on :$port gpu $gpu\""
}

up() {
    echo "== cleaning =="; down
    local pif df; pif="$(iface_of "$PREFILL_NODE")"; df="$(iface_of "$DECODE_NODE")"
    echo "== launching prefill ($PREFILL_NODE/$pif) + decode ($DECODE_NODE/$df) =="
    launch_role "$PREFILL_NODE" "$PREFILL_GPU" vllm-prefill "$PREFILL_PORT" kv_producer "$pif"
    launch_role "$DECODE_NODE"  "$DECODE_GPU"  vllm-decode  "$DECODE_PORT"  kv_consumer "$df"

    echo "== waiting for prefill+decode to come up (NIXL, eager; ~minutes) =="
    for ep in "$PREFILL_NODE:$PREFILL_PORT" "$DECODE_NODE:$DECODE_PORT"; do
        for _ in $(seq 1 120); do curl -s -m 5 "http://$ep/v1/models" >/dev/null 2>&1 && { echo "  $ep up"; break; }; sleep 10; done
    done

    echo "== starting proxy on $PREFILL_NODE:$PROXY_PORT =="
    # The demo proxy binds uvicorn to localhost; patch it to 0.0.0.0 so the control box can reach it.
    ssh_node "$PREFILL_NODE" "docker rm -f vllm-proxy >/dev/null 2>&1 || true; docker run -d --network host --name vllm-proxy --entrypoint bash $IMAGE \
        -c \"sed -i 's/uvicorn.Config(app, port=self.port/uvicorn.Config(app, host=\\\"0.0.0.0\\\", port=self.port/' $PROXY; \
            python3 $PROXY --model '$MODEL' --prefill $PREFILL_NODE:$PREFILL_PORT --decode $DECODE_NODE:$DECODE_PORT --port $PROXY_PORT > /tmp/vllm-proxy.log 2>&1\" >/dev/null && echo proxy-started"
    for _ in $(seq 1 30); do curl -s -m 5 "http://$PREFILL_NODE:$PROXY_PORT/v1/models" >/dev/null 2>&1 && { echo "READY -> http://$PREFILL_NODE:$PROXY_PORT"; return 0; }; sleep 3; done
    echo "proxy not responding; check: bash $0 logs"; return 1
}

logs() {
    echo "--- prefill ($PREFILL_NODE) ---"; ssh_node "$PREFILL_NODE" "docker exec vllm-prefill tail -25 /tmp/vllm-prefill.log 2>&1"
    echo "--- decode ($DECODE_NODE) ---";  ssh_node "$DECODE_NODE"  "docker exec vllm-decode tail -25 /tmp/vllm-decode.log 2>&1"
    echo "--- proxy ($PREFILL_NODE) ---";  ssh_node "$PREFILL_NODE" "docker exec vllm-proxy tail -15 /tmp/vllm-proxy.log 2>&1"
}

case "$ACTION" in
    up) up ;;
    down) echo "== stopping PD cluster =="; down ;;
    logs) logs ;;
    *) echo "usage: bash $0 up|down|logs"; exit 1 ;;
esac
