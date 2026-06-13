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

DISTRO="${CUDA_DISTRO:-ubuntu2404}"
ARCH="${CUDA_ARCH:-x86_64}"

# Ensure the CUDA apt repo is present (makes this script self-sufficient standalone).
if ! apt-cache policy "cudnn9-cuda-${CUDA_MAJOR}" 2>/dev/null | grep -q developer.download.nvidia.com; then
    cd /tmp
    wget -O cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i cuda-keyring.deb
fi

sudo apt-get update
sudo apt-get install -y "cudnn9-cuda-${CUDA_MAJOR}" libnccl2 libnccl-dev

echo "Verify cuDNN: dpkg -l | grep cudnn"
echo "Verify NCCL : dpkg -l | grep nccl"
