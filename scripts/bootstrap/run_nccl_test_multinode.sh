#!/bin/bash
# Orchestrate the NCCL test across all nodes from one machine: copy the repo to each node,
# SSH in, and launch run_nccl_test.sh with the right NODE_RANK. First node in NODES is head.
#
# By default the repo is synced to REPO_DIR=/tmp/ml-homelab-setup on every node, so nodes
# don't need it pre-cloned. Set SYNC=0 if it's already in place at REPO_DIR.
#
# Requires: passwordless SSH to every node, uv installed on each, and every node able to
# reach the head's RDZV_PORT.
#
#   NODES="10.0.0.101 10.0.0.244" GPUS_PER_NODE=1 SSH_USER=yisheng bash run_nccl_test_multinode.sh
set -euo pipefail

NODES="${NODES:?set NODES to space-separated node IPs/hosts (first is the head)}"
read -ra node_arr <<< "$NODES"
NNODES=${#node_arr[@]}
HEAD_NODE_IP="${HEAD_NODE_IP:-${node_arr[0]}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
RDZV_PORT="${RDZV_PORT:-29500}"
REPO_DIR="${REPO_DIR:-/tmp/ml-homelab-setup}"
SYNC="${SYNC:-1}"
SSH_PREFIX="${SSH_USER:+${SSH_USER}@}"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "NCCL test: ${NNODES} nodes (head ${HEAD_NODE_IP}), ${GPUS_PER_NODE} GPU/node, repo ${REPO_DIR}"

pids=()
for rank in "${!node_arr[@]}"; do
    host="${node_arr[$rank]}"
    target="${SSH_PREFIX}${host}"
    remote="cd '${REPO_DIR}' && \
HEAD_NODE_IP='${HEAD_NODE_IP}' NNODES='${NNODES}' NODE_RANK='${rank}' \
GPUS_PER_NODE='${GPUS_PER_NODE}' RDZV_PORT='${RDZV_PORT}' \
uv run bash scripts/bootstrap/run_nccl_test.sh"

    if [ -n "${DRY_RUN:-}" ]; then
        [ "$SYNC" = 1 ] && echo "  [node ${rank}] rsync ${LOCAL_REPO}/ -> ${target}:${REPO_DIR}/"
        echo "  [node ${rank}] ssh ${target} -> ${remote}"
        continue
    fi

    if [ "$SYNC" = 1 ]; then
        echo "  syncing repo to node ${rank} (${target}:${REPO_DIR})"
        ssh "$target" "mkdir -p '${REPO_DIR}'"
        rsync -az --delete --exclude '.venv' --exclude '.git' \
            "${LOCAL_REPO}/" "${target}:${REPO_DIR}/"
    fi

    echo "  launching node ${rank} on ${target}"
    # bash -lc so the remote login env (uv on PATH) is loaded.
    ssh "$target" "bash -lc \"${remote}\"" 2>&1 | sed "s/^/[node ${rank}] /" &
    pids+=("$!")
done

[ -n "${DRY_RUN:-}" ] && exit 0

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" -eq 0 ] && echo "All nodes finished." || echo "One or more nodes failed."
exit "$fail"
