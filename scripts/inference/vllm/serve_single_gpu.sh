#!/bin/bash
# vLLM OpenAI-compatible server on a single GPU, inside the official image.
#   GPU=1 bash serve_single_gpu.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8000 bash serve_single_gpu.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../_common.sh"

GPU="${GPU:-0}"
PORT="${PORT:-8000}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

exec $DOCKER run --rm --gpus "\"device=${GPU}\"" \
    "${DOCKER_COMMON[@]}" \
    -p "${PORT}:${PORT}" \
    "${IMAGE}" \
        --model "${MODEL}" \
        --host 0.0.0.0 --port "${PORT}"
