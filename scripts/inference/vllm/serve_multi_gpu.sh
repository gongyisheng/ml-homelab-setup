#!/bin/bash
# vLLM on multiple local GPUs via tensor parallelism.
#   TP=2 bash serve_multi_gpu.sh
#   GPUS='"device=0,1"' TP=2 bash serve_multi_gpu.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../_common.sh"

GPUS="${GPUS:-all}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

exec $DOCKER run --rm --gpus "${GPUS}" \
    "${DOCKER_COMMON[@]}" \
    -p "${PORT}:${PORT}" \
    "${IMAGE}" \
        --model "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --host 0.0.0.0 --port "${PORT}"
