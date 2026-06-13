#!/bin/bash
# Install Docker engine. For GPU support, run install_cuda_container_kit.sh afterwards.
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER" || true
    echo "Added $USER to docker group (re-login to use docker without sudo)."
fi

echo "Docker installed. Verify:  docker run --rm hello-world"
echo "Next, for GPU support:  bash install_cuda_container_kit.sh"
