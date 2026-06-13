#!/bin/bash
# Install cuDNN + NCCL from the NVIDIA CUDA apt repo (needs cuda-keyring from
# install_driver.sh). Package suffix tracks the CUDA major version.
#   bash install_cudnn_nccl.sh           # CUDA 13
#   CUDA_MAJOR=12 bash install_cudnn_nccl.sh
#
# Alternatively use the download pages:
#   cuDNN: https://developer.nvidia.com/cudnn-downloads
#   NCCL : https://developer.nvidia.com/nccl/nccl-download
set -euo pipefail

CUDA_MAJOR="${CUDA_MAJOR:-13}"

sudo apt-get update
sudo apt-get install -y "cudnn9-cuda-${CUDA_MAJOR}" libnccl2 libnccl-dev

echo "Verify cuDNN: dpkg -l | grep cudnn"
echo "Verify NCCL : dpkg -l | grep nccl"
