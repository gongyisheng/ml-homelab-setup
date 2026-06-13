"""Run the same load against an SGLang and a vLLM endpoint and print a side-by-side table.

Both servers must already be running (see ../sglang and ../vllm). Stdlib only.

    python3 compare.py --sglang-url http://localhost:30000 --vllm-url http://localhost:8000 \
        --model Qwen/Qwen2.5-1.5B-Instruct --num-prompts 64 --concurrency 16
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from bench_serving import one_request, pct


def run(base_url, model, prompt, num_prompts, concurrency, max_tokens):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(
            lambda _: one_request(base_url, model, prompt, max_tokens), range(num_prompts)))
    wall = time.perf_counter() - start
    ok = [r for r in results if r is not None]
    if not ok:
        return None
    return {
        "completed": len(ok),
        "req_s": len(ok) / wall,
        "tok_s": sum(r[2] for r in ok) / wall,
        "ttft_p50": pct([r[0] for r in ok], 50) * 1000,
        "lat_p50": pct([r[1] for r in ok], 50) * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Explain tensor parallelism in two sentences.")
    args = ap.parse_args()

    targets = {"sglang": args.sglang_url, "vllm": args.vllm_url}
    rows = {}
    for name, url in targets.items():
        print(f"Benchmarking {name} @ {url} ...")
        rows[name] = run(url, args.model, args.prompt,
                         args.num_prompts, args.concurrency, args.max_tokens)

    print(f"\n{'metric':<18}{'sglang':>14}{'vllm':>14}")
    metrics = [("req/s", "req_s", "{:.2f}"), ("tok/s", "tok_s", "{:.1f}"),
               ("TTFT p50 (ms)", "ttft_p50", "{:.0f}"), ("latency p50 (ms)", "lat_p50", "{:.0f}")]
    for label, key, fmt in metrics:
        cells = []
        for name in targets:
            cells.append(fmt.format(rows[name][key]) if rows[name] else "FAIL")
        print(f"{label:<18}{cells[0]:>14}{cells[1]:>14}")


if __name__ == "__main__":
    main()
