#!/bin/bash
# Drive a multi-node inference cluster from this control box over SSH.
# Launches the per-node serve_multi_node.sh on each GPU node with the right role/rank,
# GPU, and interface, then (for `up`) waits for the head's OpenAI endpoint to come live.
#
#   ENGINE=vllm   bash serve_multi_node_cluster.sh up      # start (head=first node)
#   ENGINE=sglang bash serve_multi_node_cluster.sh up
#   ENGINE=vllm   bash serve_multi_node_cluster.sh down    # stop + remove containers
#   ENGINE=vllm   bash serve_multi_node_cluster.sh logs    # tail the head log
#
# NODES: space-separated "ip[:gpu_device]" entries; the FIRST is the head.
# Defaults match this homelab: pc2 head on its only GPU, pc3 worker on its FREE GPU 1
# (GPU 0 on pc3 is often busy with training).
#
#   NODES="10.0.0.101:0 10.0.0.244:1" ENGINE=vllm bash serve_multi_node_cluster.sh up
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENGINE="${ENGINE:-vllm}"
ACTION="${1:-up}"
SSH_USER="${SSH_USER:-yisheng}"
NODES="${NODES:-10.0.0.101:0 10.0.0.244:1}"
REMOTE_DIR="${REMOTE_DIR:-~/Documents/ml-homelab-setup/scripts/inference}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
SYNC="${SYNC:-1}"   # rsync local scripts to each node before launch (nodes run their own copy)

read -r -a NODE_ARR <<< "$NODES"
NNODES="${#NODE_ARR[@]}"
HEAD_IP="${NODE_ARR[0]%%:*}"

if [ "$ENGINE" = vllm ]; then
    CNAME=vllm-node; PORT="${PORT:-8000}"
elif [ "$ENGINE" = sglang ]; then
    CNAME=sglang-node; PORT="${PORT:-30000}"
else
    echo "ENGINE must be vllm or sglang"; exit 1
fi

node_ip()  { echo "${1%%:*}"; }
node_gpu() { [ "$1" = "${1%:*}" ] && echo 0 || echo "${1##*:}"; }
# -n: read stdin from /dev/null. Without it, sequential ssh calls in a loop consume the
# parent's stdin and later launches silently never run (the bug that ate the worker launch).
ssh_node() { ssh -n -o ConnectTimeout=10 "${SSH_USER}@$1" "$2"; }

down() {
    for e in "${NODE_ARR[@]}"; do
        ip="$(node_ip "$e")"
        ssh_node "$ip" "docker rm -f $CNAME >/dev/null 2>&1 || true; echo '  $ip: $CNAME removed'"
    done
}

logs() {
    if [ "$ENGINE" = vllm ]; then
        ssh_node "$HEAD_IP" "docker exec $CNAME tail -40 /tmp/vllm-serve.log 2>&1 || docker logs --tail 40 $CNAME 2>&1"
    else
        ssh_node "$HEAD_IP" "docker logs --tail 40 $CNAME 2>&1"
    fi
}

sync_scripts() {
    echo "== syncing scripts to nodes (SYNC=$SYNC) =="
    for e in "${NODE_ARR[@]}"; do
        ip="$(node_ip "$e")"
        rsync -az "$DIR/" "${SSH_USER}@${ip}:${REMOTE_DIR}/" && echo "  $ip: synced"
    done
}

launch_node() {
    local idx="$1" e ip gpu role env
    e="${NODE_ARR[$idx]}"; ip="$(node_ip "$e")"; gpu="$(node_gpu "$e")"
    if [ "$ENGINE" = vllm ]; then
        role=worker; [ "$idx" = 0 ] && role=head
        env="ROLE=$role HEAD_NODE_IP=$HEAD_IP TP=${TP:-1} PP=$NNODES GPUS=device=$gpu MODEL=$MODEL"
    else
        env="HEAD_NODE_IP=$HEAD_IP NNODES=$NNODES NODE_RANK=$idx TP=${TP:-$NNODES} GPUS=device=$gpu MODEL=$MODEL"
    fi
    env="$env ${EXTRA_ENV:-}"   # forward extra per-engine knobs, e.g. EXTRA_ENV='DISABLE_CUDA_GRAPH=0'
    echo "  $ip (gpu $gpu, idx $idx): $env"
    # The per-node script launches its container detached and exits, so this ssh returns.
    ssh_node "$ip" "cd $REMOTE_DIR && env $env bash $ENGINE/serve_multi_node.sh > /tmp/${CNAME}.log 2>&1"
}

up() {
    [ "$SYNC" = 1 ] && sync_scripts
    echo "== cleaning any prior $CNAME containers =="; down
    echo "== launching $ENGINE across $NNODES nodes (head $HEAD_IP) =="
    # Launch every node concurrently (each ssh backgrounded locally). The head waits for all
    # GPUs to join while the workers come up, so they must run in parallel; `wait` returns once
    # each per-node script has launched its detached container and exited.
    for idx in "${!NODE_ARR[@]}"; do
        launch_node "$idx" &
        [ "$idx" = 0 ] && sleep 3
    done
    wait
    echo "== waiting for http://$HEAD_IP:$PORT/v1/models (cross-node bring-up is slow) =="
    for i in $(seq 1 150); do
        if curl -s -m 5 "http://$HEAD_IP:$PORT/v1/models" >/dev/null 2>&1; then
            echo "READY  ->  http://$HEAD_IP:$PORT   (model $MODEL)"
            echo "bench:  BASE_URL=http://$HEAD_IP:$PORT bash bench/bench_serving.sh"
            return 0
        fi
        sleep 10
    done
    echo "TIMED OUT. Inspect:  ENGINE=$ENGINE NODES=\"$NODES\" bash $0 logs"; return 1
}

case "$ACTION" in
    up) up ;;
    down) echo "== stopping $ENGINE cluster =="; down ;;
    logs) logs ;;
    *) echo "usage: ENGINE=vllm|sglang bash $0 up|down|logs"; exit 1 ;;
esac
