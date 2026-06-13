#!/bin/bash
# Install NVIDIA Container Toolkit, register the NVIDIA runtime with Docker so
# `docker run --gpus all` works, then smoke-test. Gate for the inference/ module.
# Requires Docker (install_docker.sh) and the NVIDIA driver (install_driver.sh).
# ref: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
set -euo pipefail

# --- NVIDIA Container Toolkit apt repo ---
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# --- register the NVIDIA runtime with Docker + restart daemon ---
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# --- smoke test ---
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:13.0.0-base-ubuntu24.04}"
echo "Smoke test: docker run --rm --gpus all $CUDA_IMAGE nvidia-smi"
sudo docker run --rm --gpus all "$CUDA_IMAGE" nvidia-smi
