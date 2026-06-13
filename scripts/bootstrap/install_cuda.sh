#!/bin/bash
# Install CUDA toolkit (default 13.0 for sm_120 Blackwell). Requires the cuda-keyring
# repo added by install_driver.sh.
#   bash install_cuda.sh                 # cuda-toolkit-13-0
#   CUDA_VERSION=12.8 bash install_cuda.sh
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-13.0}"
pkg="cuda-toolkit-${CUDA_VERSION/./-}"

DISTRO="${CUDA_DISTRO:-ubuntu2404}"
ARCH="${CUDA_ARCH:-x86_64}"

# Ensure the CUDA apt repo is present (install_driver.sh also adds it; this makes
# install_cuda.sh self-sufficient when run standalone).
if ! apt-cache policy "$pkg" 2>/dev/null | grep -q developer.download.nvidia.com; then
    cd /tmp
    wget -O cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i cuda-keyring.deb
fi

sudo apt-get update
sudo apt-get install -y "$pkg"

# Point /usr/local/cuda at this version (switch versions by re-running with CUDA_VERSION).
sudo rm -f /usr/local/cuda
sudo ln -s "/usr/local/cuda-${CUDA_VERSION}" /usr/local/cuda

# Add CUDA env to ~/.bashrc once.
BASHRC="$HOME/.bashrc"
if ! grep -q 'CUDA_HOME=/usr/local/cuda' "$BASHRC"; then
    cat >> "$BASHRC" <<'EOF'

# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    echo "Added CUDA env to ~/.bashrc (run: source ~/.bashrc)"
fi

echo "Installed $pkg -> /usr/local/cuda. Verify:  nvcc -V"
