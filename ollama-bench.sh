#!/usr/bin/env python3
"""
ollama_bench.py
Benchmark Ollama with:
- TTFT (time to first token) via streaming NDJSON
- Generation TPS (eval_count / eval_duration)
- Prompt processing speed (prompt_eval_count / prompt_eval_duration)

Usage examples:
  python ollama_bench.py --model llama3.2:3b --prompt "Explain TCP slow start." --runs 15 --warmup 2
  python ollama_bench.py --model llama3.2:3b --prompt-file prompts.txt --runs 10 --num-predict 200
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class RunResult:
    ttft_s: Optional[float]
    total_s: Optional[float]
    gen_tokens: Optional[int]
    gen_duration_s: Optional[float]
    gen_tps: Optional[float]
    prompt_tokens: Optional[int]
    prompt_duration_s: Optional[float]
    prompt_tps: Optional[float]


def pctl(values: List[float], q: float) -> float:
    """Nearest-rank percentile for small samples."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = max(1, math.ceil(q * len(xs)))
    return xs[k - 1]


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return args.prompt


def bench_once(
    session: requests.Session,
    url: str,
    model: str,
    prompt: str,
    options: Dict[str, Any],
    timeout_s: float,
) -> RunResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,  # streaming needed for TTFT
        "options": options,
    }

    start_ns = time.monotonic_ns()
    first_token_ns: Optional[int] = None

    # We'll capture the final metrics Ollama emits (usually on the last chunk with done=true)
    final_metrics: Dict[str, Any] = {}

    try:
        with session.post(url, json=payload, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()

            # Ollama returns NDJSON (one JSON object per line).
            # First token time: first time we receive a chunk with a non-empty "response".
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    # If something weird slips in, ignore line
                    continue

                # "response" may be token fragments; treat first non-empty as TTFT.
                if first_token_ns is None:
                    resp_text = obj.get("response", "")
                    if isinstance(resp_text, str) and resp_text != "":
                        first_token_ns = time.monotonic_ns()

                # Save the last chunk's metrics (the done chunk usually has them)
                final_metrics = obj

                if obj.get("done") is True:
                    break

    except requests.RequestException as e:
        print(f"[ERROR] request failed: {e}", file=sys.stderr)
        return RunResult(
            ttft_s=None,
            total_s=None,
            gen_tokens=None,
            gen_duration_s=None,
            gen_tps=None,
            prompt_tokens=None,
            prompt_duration_s=None,
            prompt_tps=None,
        )

    end_ns = time.monotonic_ns()

    ttft_s = None if first_token_ns is None else (first_token_ns - start_ns) / 1e9
    total_s = (end_ns - start_ns) / 1e9

    # Ollama fields (names used by current Ollama generations):
    # eval_count, eval_duration (ns), prompt_eval_count, prompt_eval_duration (ns)
    gen_tokens = final_metrics.get("eval_count")
    gen_duration_ns = final_metrics.get("eval_duration")
    prompt_tokens = final_metrics.get("prompt_eval_count")
    prompt_duration_ns = final_metrics.get("prompt_eval_duration")

    gen_duration_s = safe_div(float(gen_duration_ns), 1e9) if isinstance(gen_duration_ns, (int, float)) else None
    prompt_duration_s = safe_div(float(prompt_duration_ns), 1e9) if isinstance(prompt_duration_ns, (int, float)) else None

    gen_tps = safe_div(float(gen_tokens), gen_duration_s) if isinstance(gen_tokens, int) else None
    prompt_tps = safe_div(float(prompt_tokens), prompt_duration_s) if isinstance(prompt_tokens, int) else None

    return RunResult(
        ttft_s=ttft_s,
        total_s=total_s,
        gen_tokens=gen_tokens if isinstance(gen_tokens, int) else None,
        gen_duration_s=gen_duration_s,
        gen_tps=gen_tps,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        prompt_duration_s=prompt_duration_s,
        prompt_tps=prompt_tps,
    )


def summarize(label: str, vals: List[float]) -> str:
    if not vals:
        return f"{label}: n/a"
    return (
        f"{label}: mean={statistics.mean(vals):.4f}  "
        f"median={statistics.median(vals):.4f}  "
        f"p95={pctl(vals, 0.95):.4f}  "
        f"n={len(vals)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host base URL")
    ap.add_argument("--model", required=True, help="Model name, e.g. llama3.2:3b")
    ap.add_argument("--prompt", default="Explain TCP slow start in 150 tokens.", help="Prompt text")
    ap.add_argument("--prompt-file", default="", help="Read prompt from file")
    ap.add_argument("--runs", type=int, default=5, help="Number of measured runs")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup runs (not counted)")
    ap.add_argument("--timeout", type=float, default=300.0, help="Request timeout seconds")

    # Options you likely want fixed for repeatability
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-predict", type=int, default=8192)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)

    args = ap.parse_args()
    prompt = read_prompt(args)

    url = args.host.rstrip("/") + "/api/generate"
    options = {
        "temperature": args.temperature,
        "seed": args.seed,
        "num_predict": args.num_predict,
        "num_ctx": args.num_ctx,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    print(f"Benchmarking {args.model} @ {args.host}")
    print(f"runs={args.runs}, warmup={args.warmup}, num_predict={args.num_predict}, num_ctx={args.num_ctx}")
    print("----")

    with requests.Session() as session:
        # Warmup
        for i in range(args.warmup):
            _ = bench_once(session, url, args.model, prompt, options, args.timeout)

        results: List[RunResult] = []
        for i in range(args.runs):
            res = bench_once(session, url, args.model, prompt, options, args.timeout)
            results.append(res)

            # Per-run line
            ttft = "n/a" if res.ttft_s is None else f"{res.ttft_s:.3f}s"
            gtps = "n/a" if res.gen_tps is None else f"{res.gen_tps:.2f}"
            ptps = "n/a" if res.prompt_tps is None else f"{res.prompt_tps:.2f}"
            print(f"run {i+1:02d}: TTFT={ttft}  gen_tps={gtps}  prompt_tps={ptps}")

    # Aggregate summaries
    ttft_vals = [r.ttft_s for r in results if r.ttft_s is not None]
    gen_tps_vals = [r.gen_tps for r in results if r.gen_tps is not None]
    prompt_tps_vals = [r.prompt_tps for r in results if r.prompt_tps is not None]
    total_vals = [r.total_s for r in results if r.total_s is not None]

    print("----")
    print(summarize("TTFT (s)", ttft_vals))
    print(summarize("Gen TPS (tok/s)", gen_tps_vals))
    print(summarize("Prompt TPS (tok/s)", prompt_tps_vals))
    print(summarize("Total latency (s)", total_vals))

    # If prompt timing isn’t reported by your Ollama build/model, explain quickly.
    if not prompt_tps_vals:
        print(
            "\nNote: prompt_eval_count/prompt_eval_duration were not present in responses, "
            "so prompt processing speed couldn't be computed. "
            "Generation TPS + TTFT are still measured."
        )


if __name__ == "__main__":
    main()
