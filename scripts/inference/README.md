# inference

SGLang and vLLM serving, side by side, **run inside their official Docker images** — no
bare-metal installs. Covers single-GPU, single-node multi-GPU (TP), and multi-node, plus
benchmarks over the OpenAI-compatible endpoint.

## Layout

```
inference/
├── sglang/  serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh
├── vllm/    serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh
└── bench/   bench_serving.sh  compare.py
```

Images: `lmsysorg/sglang`, `vllm/vllm-openai`. Each `serve_*.sh` is a `docker run` wrapper
with `--gpus`, NCCL flags (`--ipc=host --ulimit memlock=-1 …`), model/HF-cache mounts, and
the OpenAI port published.

## Verify (this box)

Single-GPU and multi-GPU (2× 5060 Ti) launch + bench. Multi-node is written but untested
here (no second node).

<!-- TODO: fill scripts -->
