#!/bin/bash
# Thin wrapper around bench_serving.py.
#   bash bench_serving.sh                              # sglang default :30000
#   BASE_URL=http://localhost:8000 bash bench_serving.sh   # vllm
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="${BASE_URL:-http://localhost:30000}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TOKENS="${MAX_TOKENS:-128}"

exec python3 "$DIR/bench_serving.py" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS"
