#!/usr/bin/env python3
"""Measure an Original-model vLLM reference for the rebuttal workload."""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--prompt", default="Explain why the sky appears blue in concise terms.")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=512)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fixed_length_prompt(tokenizer, text: str, length: int) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Prompt produced no tokens")
    repeats = (length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:length]


def main() -> None:
    args = parse_args()
    batch_sizes = args.batch_sizes or [args.num_requests]
    if any(size < 1 or size > args.num_requests for size in batch_sizes):
        raise ValueError("Each batch size must be between 1 and num_requests")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.prompt_tokens + args.output_tokens,
        max_num_seqs=max(batch_sizes),
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    prompt_ids = fixed_length_prompt(tokenizer, args.prompt, args.prompt_tokens)

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )
    warmup = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    measurements = []
    for batch_size in batch_sizes:
        warmup_prompts = [{"prompt_token_ids": prompt_ids} for _ in range(batch_size)]
        llm.generate(warmup_prompts, warmup, use_tqdm=False)

        generated_tokens = 0
        started = time.perf_counter()
        for offset in range(0, args.num_requests, batch_size):
            current_size = min(batch_size, args.num_requests - offset)
            prompts = [{"prompt_token_ids": prompt_ids} for _ in range(current_size)]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            generated_tokens += sum(
                len(output.outputs[0].token_ids) for output in outputs
            )
        elapsed = time.perf_counter() - started
        measurements.append(
            {
                "batch_size": batch_size,
                "total_seconds": elapsed,
                "request_throughput": args.num_requests / elapsed,
                "output_token_throughput": generated_tokens / elapsed,
                "generated_tokens": generated_tokens,
            }
        )

    result = {
        "backend": "vLLM",
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens_per_request": args.output_tokens,
        "num_requests": args.num_requests,
        "batch_sizes": batch_sizes,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prefix_caching": False,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
