#!/bin/bash
# Orchestrate the NCCL test across all nodes from one machine: SSH into each node and
# launch run_nccl_test.sh with the right NODE_RANK. The first node in NODES is the head.
#
# Requires: passwordless SSH to every node, the same REPO_DIR path + uv env on each,
# and every node able to reach the head's RDZV_PORT.
#
#   NODES="10.0.0.243 10.0.0.244" GPUS_PER_NODE=2 bash run_nccl_test_multinode.sh
#   NODES="..." SSH_USER=yisheng REPO_DIR=/home/yisheng/ml-homelab-setup bash run_nccl_test_multinode.sh
#   NODES="..." DRY_RUN=1 bash run_nccl_test_multinode.sh        # print commands, don't run
set -euo pipefail

NODES="${NODES:?set NODES to space-separated node IPs/hosts (first is the head)}"
read -ra node_arr <<< "$NODES"
NNODES=${#node_arr[@]}
HEAD_NODE_IP="${HEAD_NODE_IP:-${node_arr[0]}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
RDZV_PORT="${RDZV_PORT:-29500}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SSH_PREFIX="${SSH_USER:+${SSH_USER}@}"

echo "NCCL test: ${NNODES} nodes (head ${HEAD_NODE_IP}), ${GPUS_PER_NODE} GPU/node, repo ${REPO_DIR}"

pids=()
for rank in "${!node_arr[@]}"; do
    host="${node_arr[$rank]}"
    remote="cd '${REPO_DIR}' && \
HEAD_NODE_IP='${HEAD_NODE_IP}' NNODES='${NNODES}' NODE_RANK='${rank}' \
GPUS_PER_NODE='${GPUS_PER_NODE}' RDZV_PORT='${RDZV_PORT}' \
uv run bash scripts/bootstrap/run_nccl_test.sh"

    if [ -n "${DRY_RUN:-}" ]; then
        echo "  [node ${rank}] ssh ${SSH_PREFIX}${host} -> ${remote}"
        continue
    fi

    echo "  launching node ${rank} on ${SSH_PREFIX}${host}"
    # bash -lc so the remote login env (uv on PATH) is loaded.
    ssh "${SSH_PREFIX}${host}" "bash -lc \"${remote}\"" 2>&1 | sed "s/^/[node ${rank}] /" &
    pids+=("$!")
done

[ -n "${DRY_RUN:-}" ] && exit 0

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" -eq 0 ] && echo "All nodes finished." || echo "One or more nodes failed."
exit "$fail"
