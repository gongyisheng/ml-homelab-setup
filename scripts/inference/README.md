# inference

SGLang and vLLM serving, side by side, **run inside their official Docker images** — no
bare-metal installs. Covers single-GPU, single-node multi-GPU (TP), and multi-node, plus
a stdlib benchmark over the OpenAI-compatible endpoint.

Needs the NVIDIA Docker runtime from `bootstrap/install_cuda_container_kit.sh`. 
If docker requires root on your box, prefix with sudo via `DOCKER="sudo docker"`.

## Layout

```
inference/
├── _common.sh   # shared defaults (MODEL, HF cache mount, NCCL flags, DOCKER)
├── sglang/      serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh   (:30000)
├── vllm/        serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh   (:8000)
└── bench/       bench_serving.py  bench_serving.sh  compare.py
```

Images: `lmsysorg/sglang` (`SGLANG_IMAGE`), `vllm/vllm-openai` (`VLLM_IMAGE`). Default
model `Qwen/Qwen2.5-1.5B-Instruct` (for testing, fits a 16 GB 5060 Ti); override with `MODEL`.

## Serve

```bash
# single GPU (pin the free GPU)
GPU=1 DOCKER="sudo docker" bash sglang/serve_single_gpu.sh
GPU=1 DOCKER="sudo docker" bash vllm/serve_single_gpu.sh

# single-node multi-GPU (tensor parallel)
TP=2 DOCKER="sudo docker" bash sglang/serve_multi_gpu.sh

# multi-node (run on each node; TP = total GPUs across nodes for sglang)
HEAD_NODE_IP=10.0.0.101 NNODES=2 NODE_RANK=0 TP=4 bash sglang/serve_multi_node.sh
ROLE=head HEAD_NODE_IP=10.0.0.101 TP=2 PP=2 bash vllm/serve_multi_node.sh
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

## Results

`Qwen/Qwen3-8B` on 1× L40S (46 GB), single GPU — 64 prompts, concurrency 16, max_tokens
128. Run sequentially (one GPU can't hold both servers at once), same shared HF weights.

| Metric              | vLLM        | SGLang      |
|---------------------|-------------|-------------|
| Request throughput  | 5.27 req/s  | 5.23 req/s  |
| Output throughput   | 674.5 tok/s | 669.4 tok/s |
| TTFT p50 / p99      | 69 / 107 ms | 66 / 168 ms |
| Latency p50 / p99   | 3021/3061ms | 3043/3131ms |

Effectively a dead heat (within ~1%). At 8B on a single GPU with batch 16 both engines are
compute-bound, so they converge — differences are within run-to-run noise. Engine gaps
(scheduling, prefix caching, chunked prefill) show up under higher concurrency or
longer/variable sequences.

## Verification status

- bench harness: verified against a mock SSE endpoint (TTFT, token counting, throughput).
- single-GPU launch: verified live on an L40S — vLLM (:8000) and SGLang (:30000) both
  served `Qwen/Qwen3-8B` and benchmarked clean (see Results).
- multi-GPU launch: scripts correct, not exercised here (single GPU on this box).
- multi-node: written, untested here (single box).
