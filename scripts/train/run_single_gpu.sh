#!/bin/bash
# Single-GPU training. Extra args pass through to the training script.
#   GPU=1 bash run_single_gpu.sh --steps 100
#   TRAIN_SCRIPT=/path/to/your_train.py GPU=0 bash run_single_gpu.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$DIR/train.py}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="$GPU" exec python3 "$TRAIN_SCRIPT" "$@"
