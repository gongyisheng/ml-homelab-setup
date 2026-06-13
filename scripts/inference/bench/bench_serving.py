"""Benchmark an OpenAI-compatible serving endpoint (SGLang or vLLM).

Sends concurrent streaming chat completions and reports throughput, TTFT, and latency.
Stdlib only — no extra deps.

    python3 bench_serving.py --base-url http://localhost:30000 --model Qwen/Qwen2.5-1.5B-Instruct \
        --num-prompts 64 --concurrency 16 --max-tokens 128
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def one_request(base_url, model, prompt, max_tokens):
    """Stream one completion; return (ttft, latency, output_tokens) or None on error."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    start = time.perf_counter()
    ttft = None
    chunks = 0
    completion_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = usage["completion_tokens"]
                choices = obj.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    chunks += 1
    except Exception as e:
        print(f"  request failed: {e}")
        return None
    latency = time.perf_counter() - start
    if ttft is None:
        ttft = latency
    return ttft, latency, completion_tokens if completion_tokens is not None else chunks


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Explain tensor parallelism in two sentences.")
    args = ap.parse_args()

    print(f"Benchmarking {args.base_url} ({args.model}): "
          f"{args.num_prompts} prompts, concurrency {args.concurrency}, max_tokens {args.max_tokens}")

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(
            lambda _: one_request(args.base_url, args.model, args.prompt, args.max_tokens),
            range(args.num_prompts),
        ))
    wall = time.perf_counter() - start

    ok = [r for r in results if r is not None]
    if not ok:
        print("All requests failed.")
        return
    ttfts = [r[0] for r in ok]
    latencies = [r[1] for r in ok]
    out_tokens = sum(r[2] for r in ok)

    print(f"\n--- results ---")
    print(f"completed         : {len(ok)}/{args.num_prompts}")
    print(f"wall time         : {wall:.2f} s")
    print(f"request throughput: {len(ok) / wall:.2f} req/s")
    print(f"output throughput : {out_tokens / wall:.1f} tok/s")
    print(f"TTFT  p50 / p99   : {pct(ttfts, 50) * 1000:.0f} / {pct(ttfts, 99) * 1000:.0f} ms")
    print(f"latency p50 / p99 : {pct(latencies, 50) * 1000:.0f} / {pct(latencies, 99) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
