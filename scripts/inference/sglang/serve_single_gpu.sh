#!/bin/bash
# SGLang OpenAI-compatible server on a single GPU, inside the official image.
#   GPU=1 bash serve_single_gpu.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct PORT=30000 bash serve_single_gpu.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../_common.sh"

GPU="${GPU:-0}"
PORT="${PORT:-30000}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"

exec $DOCKER run --rm --gpus "\"device=${GPU}\"" \
    "${DOCKER_COMMON[@]}" \
    -p "${PORT}:${PORT}" \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --host 0.0.0.0 --port "${PORT}"
