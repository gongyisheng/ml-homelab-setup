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
| `run_multi_node.sh` | multi-node `torchrun` with c10d rendezvous                     |

## Examples

Run inside the uv env (`uv run` makes `python3` / `torchrun` resolve to `.venv`):

```bash
# single GPU (use the free one)
GPU=1 uv run bash run_single_gpu.sh --steps 100

# single-node multi-GPU
NPROC=2 uv run bash run_multi_gpu.sh --steps 200

# multi-node: run on each node with its rank
HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=0 GPUS_PER_NODE=2 bash run_multi_node.sh
HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=1 GPUS_PER_NODE=2 bash run_multi_node.sh

# bring your own training script
TRAIN_SCRIPT=/path/to/train_lora.py GPU=1 bash run_single_gpu.sh --config cfg.yaml
```

## Verification status

- single-GPU and the `torchrun` launcher path: verified on a 5060 Ti.
- 2-GPU DDP all-reduce and multi-node: written, run when both GPUs / a second node are
  free (one GPU is currently training).
