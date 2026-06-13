# Shared defaults + docker flags for inference serve scripts. Source this.
#
# Override via env: MODEL, HF_CACHE, DOCKER (e.g. DOCKER="sudo docker").
MODEL="${MODEL:-Qwen/Qwen3-4B}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
DOCKER="${DOCKER:-docker}"

# Flags that make multi-GPU NCCL work inside the container (see ../bootstrap/README.md).
DOCKER_COMMON=(
    --ipc=host
    --security-opt seccomp=unconfined
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${HF_CACHE}:/root/.cache/huggingface"
)
