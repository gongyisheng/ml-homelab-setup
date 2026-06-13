# inference

SGLang and vLLM serving, side by side, **run inside their official Docker images** — no
bare-metal installs. Covers single-GPU, single-node multi-GPU (TP), multi-node, and
prefill/decode (PD) disaggregation, plus a stdlib benchmark over the OpenAI-compatible endpoint.

Needs the NVIDIA Docker runtime from `bootstrap/install_cuda_container_kit.sh`.
If docker requires root on your box, prefix with sudo via `DOCKER="sudo docker"`.

## Layout

```
inference/
├── _common.sh                    # shared defaults (MODEL, HF cache mount, NCCL flags, DOCKER)
├── serve_multi_node_cluster.sh   # control-box orchestrator: drives multi-node over SSH
├── sglang/   serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh  serve_pd_disaggregation.sh  (:30000)
├── vllm/     serve_single_gpu.sh  serve_multi_gpu.sh  serve_multi_node.sh  serve_pd_disaggregation.sh  (:8000)
└── bench/    bench_serving.py  bench_serving.sh  compare.py
```

Images: `lmsysorg/sglang` (`SGLANG_IMAGE`), `vllm/vllm-openai` (`VLLM_IMAGE`). Default
model `Qwen/Qwen3-4B` (for testing); override with `MODEL`.

## Homelab topology

| Node | IP | GPU(s) | Role |
|------|-----|--------|------|
| control | 10.0.0.44 | none | orchestrates over SSH |
| pc2 | 10.0.0.101 | 1× RTX PRO 6000 (97 GB) | head / prefill |
| pc3 | 10.0.0.244 | 2× RTX 5060 Ti (16 GB) | worker / decode (use **GPU 1**; GPU 0 is often training) |

All Blackwell (sm_120), so kernels are compatible across nodes — the mismatch is GPU **count
and memory** (97 GB vs 16 GB), not architecture. Nodes talk over **1 GbE** (no RDMA), which
is the dominant cost for every cross-node path below.

## Serve (single box)

```bash
GPU=1 bash sglang/serve_single_gpu.sh           # single GPU
TP=2 bash sglang/serve_multi_gpu.sh             # single-node tensor parallel
BASE_URL=http://localhost:30000 bash bench/bench_serving.sh
```

The host HF cache (`$HF_CACHE`, default `~/.cache/huggingface`) is mounted into the container,
so a model downloaded once is reused across runs and engines.

## Multi-node — from the control box

One command launches the head + workers across nodes (rsyncs the scripts first, then SSHes
each node, waits for the endpoint):

```bash
ENGINE=vllm   bash serve_multi_node_cluster.sh up     # vLLM pipeline-parallel (Ray)
ENGINE=sglang bash serve_multi_node_cluster.sh up     # SGLang tensor-parallel
ENGINE=vllm   bash serve_multi_node_cluster.sh down
ENGINE=vllm   bash serve_multi_node_cluster.sh logs
```

`NODES="ip[:gpu] ..."` (first is head) defaults to `10.0.0.101:0 10.0.0.244:1` — pc2 on its
GPU, pc3 on its **free** GPU 1. The per-node scripts (`<engine>/serve_multi_node.sh`) also run
standalone on each node; the cluster script just drives them.

- **vLLM** does pipeline parallel (TP=GPUs/node, PP=#nodes) — tolerates uneven GPUs since each
  node holds whole layers.
- **SGLang** does tensor parallel (TP=total GPUs) — shards every layer, so it needs balanced
  ranks; here it runs **eager** (`DISABLE_CUDA_GRAPH=1`, the default) because cross-node CUDA
  graph capture deadlocks (see below).

## PD disaggregation — from the control box

Separate prefill (pc2, compute-bound) and decode (pc3, KV-bound) servers exchange KV cache,
fronted by a proxy/router. No RDMA here, so KV moves over **TCP** (NIXL for vLLM, `mooncake_tcp`
for SGLang).

```bash
bash vllm/serve_pd_disaggregation.sh up      # prefill + decode + NIXL proxy        -> :8000
bash sglang/serve_pd_disaggregation.sh up    # prefill + decode + sglang_router     -> :30000
bash vllm/serve_pd_disaggregation.sh down
```

## Benchmark

```bash
BASE_URL=http://10.0.0.101:8000  bash bench/bench_serving.sh   # vllm / proxy
BASE_URL=http://10.0.0.101:30000 bash bench/bench_serving.sh   # sglang / router
```

Reports request throughput (req/s), output throughput (tok/s), and TTFT / latency p50/p99.

## Results

### Qwen3-4B on 1× RTX PRO 6000 (97 GB), single GPU

vLLM 0.23.0 vs SGLang 0.5.13, `:latest` images, CUDA 13 / driver 580 (sm_120). 64 prompts,
concurrency 16, max_tokens 128.

| Metric              | vLLM         | SGLang       |
|---------------------|--------------|--------------|
| Output throughput   | 2111.8 tok/s | 2048.8 tok/s |
| TTFT p50 / p99      | 47 / 407 ms  | 63 / 361 ms  |

Both run cleanly on Blackwell from stock images, no flags. Near dead heat; at 4B the GPU is far
from saturated so the engines converge.

### Qwen3-4B across pc2 + pc3 over 1 GbE

16 prompts, concurrency 4, max_tokens 64 (comparable to each other, **not** to the single-GPU
run above). pc2 prefill/head, pc3 GPU 1 decode/worker.

| Config                         | Output tok/s | TTFT p50 | Note |
|--------------------------------|-------------:|---------:|------|
| vLLM pipeline-parallel (PP=2)  | 168.7        | 47 ms    | layers split across nodes |
| vLLM PD disagg (NIXL, eager)   | 100.1        | 220 ms   | KV over TCP; NIXL/UCX |
| SGLang tensor-parallel (eager) | 58.5         | 145 ms   | every layer crosses the LAN each token |
| SGLang PD disagg (mooncake_tcp)| 170.8        | 151 ms   | graphs on (local capture); fastest cross-node |

Takeaway: every token crossing 1 GbE is ~15–35× slower than single-GPU. **PD disaggregation
beats tensor parallel** here because only the KV cache crosses the wire (once, at prefill→decode
handoff) rather than an all-reduce on every layer every token. SGLang PD edges vLLM PD mainly
because its PD servers keep CUDA graphs enabled (single-GPU capture is fine).

## Issues found & fixed (multi-node / PD)

The single-box scripts worked off the shelf; the cross-node paths needed every one of these:

1. **vLLM image ships no Ray** — `vllm/vllm-openai:latest` (0.23.0) dropped it. The multi-node
   script `pip install`s Ray into the container at start.
2. **vLLM won't auto-detect the Ray cluster** — needs explicit `--distributed-executor-backend ray`,
   else it errors "World size > available GPUs."
3. **`127.0.1.1` cross-node hang** — Ubuntu maps the hostname to `127.0.1.1`, so NCCL/Gloo
   advertise an unroutable address and the cross-node connect is refused. Fixed by pinning
   `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` to the LAN interface (+ `VLLM_HOST_IP` and Ray
   `--node-ip-address`). Interfaces differ per node (pc2 `enp3s0`, pc3 `enp7s0`) so they are
   auto-detected from the route to the head.
4. **SGLang rejects unbalanced GPU memory across TP ranks** (97 GB vs 16 GB) — set
   `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` to downgrade the guard to a warning; the cluster
   is then gated by the smallest GPU.
5. **SGLang cross-node CUDA-graph capture deadlocks** (rank 0 spins at 100 %, rank 1 idle at
   0 %) — it captures NCCL collectives into the graph across nodes. Run eager
   (`--disable-cuda-graph --disable-piecewise-cuda-graph`). PD disagg is unaffected (each server
   is single-GPU, captures locally).
6. **16 GB GPU gates the cluster** — size for the smallest GPU: `GPU_MEM_UTIL=0.80` (FULL graph
   capture needs transient headroom) and `--max-model-len 8192`. 0.85 OOMs on the 5060 Ti.
7. **Heterogeneous GPU counts** — pin one GPU per node (`GPUS=device=N`) so Ray/SGLang place one
   rank per node; pc3's GPU 0 is usually busy with training, so use GPU 1.
8. **Orchestration over SSH** — (a) rsync the scripts to each node first (nodes run their own
   copy); (b) per-node scripts launch their container **detached and exit** — a long-running
   script over `ssh "nohup … &"` holds the channel open and the next node never launches;
   (c) launch nodes concurrently (head waits for workers to join while they boot).

## Verification status

- single-GPU: verified live on RTX PRO 6000 and L40S — vLLM (:8000) and SGLang (:30000) served
  and benchmarked clean. Blackwell sm_120 works on stock `:latest` images, no overrides.
- multi-node: **verified live** on pc2 + pc3 via `serve_multi_node_cluster.sh` — vLLM PP and
  SGLang TP (eager) both served and benchmarked (table above). SGLang cross-node CUDA graph
  confirmed to deadlock; eager is the default here.
- PD disaggregation: **verified live** — vLLM (NIXL) and SGLang (mooncake_tcp) both served a
  completion through the proxy/router and benchmarked from the control box.
