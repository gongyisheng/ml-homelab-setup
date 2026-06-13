#!/bin/bash
# Install NVIDIA driver (open kernel module flavor) on Ubuntu 24.04. Reboot after.
# Also adds the CUDA apt repo (cuda-keyring) used by install_cuda.sh / install_cudnn_nccl.sh.
# ref: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html
set -euo pipefail

DISTRO="${CUDA_DISTRO:-ubuntu2404}"
ARCH="${CUDA_ARCH:-x86_64}"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb"

cd /tmp
wget -O cuda-keyring.deb "$KEYRING_URL"
sudo dpkg -i cuda-keyring.deb
sudo apt-get update

# Open kernel module flavor (required for Blackwell sm_120).
sudo apt-get install -y nvidia-open

echo
echo "Driver installed. Reboot required:  sudo reboot"
echo "After reboot, verify:  nvidia-smi"
