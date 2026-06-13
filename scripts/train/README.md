# train

Training launchers across topologies, miles-style (env-var driven). Each launcher runs
`TRAIN_SCRIPT` (default the bundled `train.py`) and passes extra args through.

`train.py` is a **self-contained torch-only reference** (tiny GPT on synthetic tokens, no
downloads) so the launchers run end-to-end. Point `TRAIN_SCRIPT` at your real HF/LoRA/SFT
or Megatron entry to use it for real.

## Launchers

| Script              | Topology                                                       |
|---------------------|----------------------------------------------------------------|
| `run_single_gpu.sh` | one GPU (`GPU=N`)                                               |
| `run_multi_gpu.sh`  | single node, all/`NPROC` GPUs via `torchrun` (DDP)             |
| `run_multi_node.sh` | multi-node `torchrun`, static rendezvous (node-rank 0 master)  |

## Examples

Run inside the uv env (`uv run` makes `python3` / `torchrun` resolve to `.venv`):

```bash
# single GPU (use the free one)
GPU=1 uv run bash run_single_gpu.sh --steps 100

# single-node multi-GPU
NPROC=2 uv run bash run_multi_gpu.sh --steps 200

# multi-node: run on each node with its rank (HEAD_NODE_IP = node-rank 0's LAN IP)
HEAD_NODE_IP=10.0.0.244 NNODES=2 NODE_RANK=0 GPUS_PER_NODE=1 uv run bash run_multi_node.sh
HEAD_NODE_IP=10.0.0.244 NNODES=2 NODE_RANK=1 GPUS_PER_NODE=1 uv run bash run_multi_node.sh

# bring your own training script
TRAIN_SCRIPT=/path/to/train_lora.py GPU=1 bash run_single_gpu.sh --config cfg.yaml
```

## Verification status

All three topologies verified 2026-06-12 on torch 2.12.0+cu130, NCCL 2.29.7+cuda13.2.

| Topology         | Hardware                          | Result |
|------------------|-----------------------------------|--------|
| single-GPU       | RTX 5060 Ti                       | ✅ pass |
| multi-GPU        | 2× RTX 5060 Ti                    | ✅ pass |
| multi-node       | RTX 5060 Ti + RTX PRO 6000        | ✅ pass |

Sample output (loss decreasing, rank-0 checkpoint written):

```text
# single-node multi-GPU
$ NPROC=2 uv run bash run_multi_gpu.sh --steps 15
step    0 | loss 7.7912
step   10 | loss 7.7310
step   14 | loss 7.7165
saved checkpoint -> /tmp/homelab-train/ckpt.pt

# multi-node, 1 GPU per node (head, NODE_RANK=0)
$ HEAD_NODE_IP=10.0.0.244 NNODES=2 NODE_RANK=0 GPUS_PER_NODE=1 uv run bash run_multi_node.sh --steps 20
NCCL version 2.29.7+cuda13.2
step    0 | loss 7.7823
step   10 | loss 7.7201
step   19 | loss 7.6962
saved checkpoint -> /tmp/homelab-train/ckpt.pt
```

## RL multi-node (miles)

more complicated but runnable, ref to: [miles-rl]: https://github.com/gongyisheng/miles/blob/miles-lora-disaggregate-mode-2/examples/lora/run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh
