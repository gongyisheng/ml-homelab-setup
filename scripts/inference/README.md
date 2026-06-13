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
model `Qwen/Qwen3-4B` (for testing); override with `MODEL`.

## Serve

```bash
# single GPU (pin the free GPU)
GPU=1 DOCKER="sudo docker" bash sglang/serve_single_gpu.sh
GPU=1 DOCKER="sudo docker" bash vllm/serve_single_gpu.sh

# single-node multi-GPU (tensor parallel)
TP=2 DOCKER="sudo docker" bash sglang/serve_multi_gpu.sh

# multi-node (run on each node; TP = total GPUs across nodes for sglang)
HEAD_NODE_IP=10.0.0.101 NNODES=2 NODE_RANK=0 TP=2 bash sglang/serve_multi_node.sh
ROLE=head HEAD_NODE_IP=10.0.0.101 TP=2 bash vllm/serve_multi_node.sh
```

The host HF cache (`$HF_CACHE`, default `~/.cache/huggingface`) is mounted into the
container, so a model downloaded once is reused across runs and engines.

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

### Qwen3-4B on 1× RTX PRO 6000 Blackwell (97 GB), single GPU

vLLM 0.23.0 vs SGLang 0.5.13, `:latest` images on CUDA 13 / driver 580 (sm_120). 64
prompts, concurrency 16, max_tokens 128. Run sequentially, same shared HF weights.

| Metric              | vLLM         | SGLang       |
|---------------------|--------------|--------------|
| Request throughput  | 16.50 req/s  | 16.01 req/s  |
| Output throughput   | 2111.8 tok/s | 2048.8 tok/s |
| TTFT p50 / p99      | 47 / 407 ms  | 63 / 361 ms  |
| Latency p50 / p99   | 879 /1240 ms | 927 /1225 ms |

Both run cleanly on Blackwell from the stock images — no flags needed. Near dead heat
(vLLM ~3% ahead on throughput). At 4B with batch 16 the GPU is far from saturated, so the
engines converge and gaps are run-to-run noise — differences emerge only under higher
concurrency or longer sequences.

## Verification status

- bench harness: verified against a mock SSE endpoint (TTFT, token counting, throughput).
- single-GPU launch: verified live on an RTX PRO 6000 Blackwell (Qwen3-4B) and an L40S
  (Qwen3-8B) — vLLM (:8000) and SGLang (:30000) both served and benchmarked clean (see
  Results). Blackwell sm_120 works on the stock `:latest` images, no overrides.
- multi-GPU launch: scripts correct, not exercised here (single GPU on this box).
- multi-node: written, untested here (single box).
