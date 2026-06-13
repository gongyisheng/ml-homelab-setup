#!/bin/bash
# SGLang prefill/decode (PD) disaggregation across two nodes, driven from this control box.
# A prefill-only server + a decode-only server exchange KV cache; an sglang_router load
# balancer (PD mode) fronts them and routes each request prefill->decode.
#
#   bash serve_pd_disaggregation.sh up      # start prefill + decode + router
#   bash serve_pd_disaggregation.sh down
#   bash serve_pd_disaggregation.sh logs
#
# Roles: prefill on pc2 (big GPU), decode on pc3's free GPU 1. No RDMA here, so the KV
# transfer uses the TCP backend (mooncake_tcp) over the 1GbE LAN — correctness demo, not a
# speedup on this hardware. Unlike multi-node TP, each PD server is single-GPU, so CUDA-graph
# capture is local (no cross-node deadlock) and stays enabled.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../_common.sh"

ACTION="${1:-up}"
SSH_USER="${SSH_USER:-yisheng}"
PREFILL_NODE="${PREFILL_NODE:-10.0.0.101}"; PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_NODE="${DECODE_NODE:-10.0.0.244}";  DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_PORT="${PREFILL_PORT:-8000}"; DECODE_PORT="${DECODE_PORT:-8001}"; ROUTER_PORT="${ROUTER_PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-9000}"
BACKEND="${BACKEND:-mooncake_tcp}"        # no-RDMA TCP transfer; alternative: nixl
MEM_FRACTION="${MEM_FRACTION:-0.8}"; CONTEXT_LEN="${CONTEXT_LEN:-8192}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"

ssh_node() { ssh -n -o ConnectTimeout=10 "${SSH_USER}@$1" "$2"; }
iface_of() { ssh_node "$1" "ip -o -4 addr show | awk -v ip=$1 '\$4 ~ \"^\"ip\"/\" {print \$2; exit}'"; }

down() {
    for n in "$PREFILL_NODE" "$DECODE_NODE"; do
        ssh_node "$n" "docker rm -f sglang-prefill sglang-decode sglang-router >/dev/null 2>&1 || true; echo '  '$n' cleaned'"
    done
}

# $1 node, $2 gpu, $3 name, $4 mode, $5 port, $6 iface, $7 extra-args
launch_role() {
    local node="$1" gpu="$2" name="$3" mode="$4" port="$5" iface="$6" extra="$7"
    ssh_node "$node" "docker rm -f $name >/dev/null 2>&1 || true; docker run -d --gpus device=$gpu --network host --name $name \
        -e NCCL_SOCKET_IFNAME=$iface -e GLOO_SOCKET_IFNAME=$iface \
        --ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 \
        -v \$HOME/.cache/huggingface:/root/.cache/huggingface $IMAGE \
        python3 -m sglang.launch_server --model-path '$MODEL' \
            --disaggregation-mode $mode --disaggregation-transfer-backend $BACKEND $extra \
            --mem-fraction-static $MEM_FRACTION --context-length $CONTEXT_LEN \
            --host 0.0.0.0 --port $port >/dev/null && echo \"  $node: $name ($mode) on :$port gpu $gpu\""
}

up() {
    echo "== cleaning =="; down
    local pif df; pif="$(iface_of "$PREFILL_NODE")"; df="$(iface_of "$DECODE_NODE")"
    echo "== launching prefill ($PREFILL_NODE/$pif) + decode ($DECODE_NODE/$df), backend=$BACKEND =="
    launch_role "$PREFILL_NODE" "$PREFILL_GPU" sglang-prefill prefill "$PREFILL_PORT" "$pif" "--disaggregation-bootstrap-port $BOOTSTRAP_PORT"
    launch_role "$DECODE_NODE"  "$DECODE_GPU"  sglang-decode  decode  "$DECODE_PORT"  "$df" ""

    echo "== waiting for prefill+decode health (~minutes) =="
    for ep in "$PREFILL_NODE:$PREFILL_PORT" "$DECODE_NODE:$DECODE_PORT"; do
        for _ in $(seq 1 150); do curl -s -m 5 "http://$ep/health" >/dev/null 2>&1 && { echo "  $ep up"; break; }; sleep 10; done
    done

    echo "== starting PD router on $PREFILL_NODE:$ROUTER_PORT =="
    ssh_node "$PREFILL_NODE" "docker rm -f sglang-router >/dev/null 2>&1 || true; docker run -d --network host --name sglang-router $IMAGE \
        python3 -m sglang_router.launch_router --pd-disaggregation \
            --prefill http://$PREFILL_NODE:$PREFILL_PORT $BOOTSTRAP_PORT --decode http://$DECODE_NODE:$DECODE_PORT \
            --policy round_robin --host 0.0.0.0 --port $ROUTER_PORT >/dev/null && echo router-started"
    for _ in $(seq 1 40); do curl -s -m 5 "http://$PREFILL_NODE:$ROUTER_PORT/v1/models" >/dev/null 2>&1 && { echo "READY -> http://$PREFILL_NODE:$ROUTER_PORT"; return 0; }; sleep 3; done
    echo "router not responding; check: bash $0 logs"; return 1
}

logs() {
    echo "--- prefill ($PREFILL_NODE) ---"; ssh_node "$PREFILL_NODE" "docker logs --tail 25 sglang-prefill 2>&1"
    echo "--- decode ($DECODE_NODE) ---";  ssh_node "$DECODE_NODE"  "docker logs --tail 25 sglang-decode 2>&1"
    echo "--- router ($PREFILL_NODE) ---"; ssh_node "$PREFILL_NODE" "docker logs --tail 15 sglang-router 2>&1"
}

case "$ACTION" in
    up) up ;;
    down) echo "== stopping PD cluster =="; down ;;
    logs) logs ;;
    *) echo "usage: bash $0 up|down|logs"; exit 1 ;;
esac
