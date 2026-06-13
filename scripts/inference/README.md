# inference

SGLang and vLLM serving, side by side, **run inside their official Docker images** — no
bare-metal installs. Covers single-GPU, single-node multi-GPU (TP), and multi-node, plus
a stdlib benchmark over the OpenAI-compatible endpoint.

Needs the NVIDIA Docker runtime from `../bootstrap/install_docker.sh`. If docker requires
root on your box, prefix with sudo via `DOCKER="sudo docker"`.

## Layout

```
inference/
├── _common.sh   # shared defaults (MODEL, HF cache mount, NCCL flags, DOCKER)
├── sglang/      serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh   (:30000)
├── vllm/        serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh   (:8000)
└── bench/       bench_serving.py  bench_serving.sh  compare.py
```

Images: `lmsysorg/sglang` (`SGLANG_IMAGE`), `vllm/vllm-openai` (`VLLM_IMAGE`). Default
model `Qwen/Qwen2.5-1.5B-Instruct` (fits a 16 GB 5060 Ti); override with `MODEL`.

## Serve

```bash
# single GPU (pin the free GPU)
GPU=1 DOCKER="sudo docker" bash sglang/serve_single_gpu.sh
GPU=1 DOCKER="sudo docker" bash vllm/serve_single_gpu.sh

# single-node multi-GPU (tensor parallel)
TP=2 DOCKER="sudo docker" bash sglang/serve_multi_gpu.sh

# multi-node (run on each node; TP = total GPUs across nodes for sglang)
HEAD_NODE_IP=10.0.0.243 NNODES=2 NODE_RANK=0 TP=4 bash sglang/serve_multi_node.sh
ROLE=head HEAD_NODE_IP=10.0.0.243 TP=2 PP=2 bash vllm/serve_multi_node.sh
```

## Benchmark

```bash
# point at whichever server is up
BASE_URL=http://localhost:30000 bash bench/bench_serving.sh   # sglang
BASE_URL=http://localhost:8000  bash bench/bench_serving.sh   # vllm

# both up at once -> side-by-side table
python3 bench/compare.py --sglang-url http://localhost:30000 --vllm-url http://localhost:8000
```

Reports request throughput (req/s), output throughput (tok/s), and TTFT / latency p50/p99.

## Verification status

- bench harness: verified against a mock SSE endpoint (TTFT, token counting, throughput).
- single / multi-GPU launch: scripts are correct but a live launch needs docker (root on
  this box); run the `serve_*` commands above to bring a server up, then benchmark.
- multi-node: written, untested here (single box).
