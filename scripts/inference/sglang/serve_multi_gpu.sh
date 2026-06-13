#!/bin/bash
# SGLang on multiple local GPUs via tensor parallelism.
#   TP=2 bash serve_multi_gpu.sh
#   GPUS='"device=0,1"' TP=2 bash serve_multi_gpu.sh   # pin specific GPUs
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../_common.sh"

GPUS="${GPUS:-all}"
TP="${TP:-2}"
PORT="${PORT:-30000}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"

exec $DOCKER run --rm --gpus "${GPUS}" \
    "${DOCKER_COMMON[@]}" \
    -p "${PORT}:${PORT}" \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --tp "${TP}" \
        --host 0.0.0.0 --port "${PORT}"
