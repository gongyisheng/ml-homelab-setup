# train

Training launchers across topologies, miles-style (env-var driven).

## Scripts

| Script              | Purpose                                                        |
|---------------------|----------------------------------------------------------------|
| `run_single_gpu.sh` | single GPU                                                     |
| `run_multi_gpu.sh`  | `torchrun` single node, all local GPUs                         |
| `run_multi_node.sh` | env wrapper (`HEAD_NODE_IP`, `GPUS_PER_NODE`, node rank)       |

A small reference LoRA/SFT entry makes the launchers runnable end-to-end.

## Multi-node

Set `HEAD_NODE_IP` and run the launcher on each node with its rank, e.g.:

```bash
HEAD_NODE_IP=10.0.0.243 GPUS_PER_NODE=2 bash run_multi_node.sh 0   # head
HEAD_NODE_IP=10.0.0.243 GPUS_PER_NODE=2 bash run_multi_node.sh 1   # worker
```

<!-- TODO: fill scripts -->
