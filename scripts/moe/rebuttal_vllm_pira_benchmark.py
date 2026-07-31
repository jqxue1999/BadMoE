"""End-to-end vLLM benchmark: Original vs PIRA-biased generation.

This exists because "same FLOPs" does not imply "same latency". Suppressing a
request's risk-amplifying experts moves its tokens onto different experts, which
changes the expert-token distribution and therefore MoE kernel efficiency:
grouped GEMMs see different per-expert token counts, load balance across experts
shifts, and scheduling changes with it. Whether that costs anything measurable is
an empirical question, so it is measured here rather than argued.

What is compared, on the same engine, GPU, prompts, and output lengths:

  Original      unmodified vLLM generation.
  PIRA-biased   identical, except each request carries its per-request,
                pre-softmax router bias through prefill and every decode step.

The difference between these two is the routing-intervention cost, isolated from
the probe. The probe is measured separately (rebuttal_probe_scaling.py) because it
runs in a Hugging Face autograd context; adding the two gives the True Total that
rebuttal_true_overhead.py reports.

The biases used here are structurally faithful but not safety-derived: K experts
per request are suppressed per layer at strength beta, chosen by a seeded
permutation. Expert *identity* does not affect timing systematically, while the
number of suppressed experts and the resulting load imbalance do, and those are
matched to the real configuration. Safety numbers come from the paper's Hugging
Face pipeline, not from here.

A diagnostic read back from the worker (rows_biased, forward_passes) confirms the
bias actually reached the router. A run reporting "no overhead" with zero biased
rows would be measuring nothing, so the script fails rather than report that.

Usage (GPU node, vLLM env):
  python scripts/moe/rebuttal_vllm_pira_benchmark.py \
      --input-lengths 128 1024 4096 --output-lengths 128 256 \
      --concurrency 1 8 32 --output timing_results/vllm_pira.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 1024, 4096])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[128, 256])
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Requests submitted together, i.e. max_num_seqs for the cell.",
    )
    parser.add_argument(
        "--requests-per-cell",
        type=int,
        default=32,
        help="Total requests per (input, output, concurrency) cell.",
    )
    parser.add_argument("--probe-layer", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=25, help="Experts suppressed per request.")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fixed_length_prompt(tokenizer, length: int) -> list[int]:
    text = (
        "The mixture-of-experts layer routes each token to a small subset of "
        "feed-forward experts, which lets the model grow its parameter count "
        "without a proportional increase in compute per token. "
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("prompt produced no tokens")
    repeats = (length + len(ids) - 1) // len(ids)
    return (ids * repeats)[:length]


def build_biases(
    request_ids: list[str],
    *,
    layers: list[int],
    num_experts: int,
    top_k: int,
    beta: float,
    seed: int,
) -> dict[str, dict[int, list[float]]]:
    """One suppression vector per (request, layer), as plain lists for the RPC.

    Distinct per request, matching PIRA's query-specific behaviour: a shared bias
    would let the engine settle into one routing pattern and understate any load
    imbalance the real method induces.
    """
    import random

    biases: dict[str, dict[int, list[float]]] = {}
    rng = random.Random(seed)
    for request_id in request_ids:
        per_layer: dict[int, list[float]] = {}
        for layer in layers:
            vector = [0.0] * num_experts
            for expert in rng.sample(range(num_experts), min(top_k, num_experts)):
                vector[expert] = -abs(beta)
            per_layer[layer] = vector
        biases[request_id] = per_layer
    return biases


def run_once(
    llm,
    pira,
    *,
    method: str,
    prompt_ids: list[int],
    count: int,
    sampling,
    layers: list[int],
    num_experts: int,
    top_k: int,
    beta: float,
    seed: int,
) -> tuple[float, int]:
    """Submit `count` requests and return (elapsed seconds, tokens produced).

    For PIRA the order matters and is the whole point: enqueue() returns the
    engine's real request ids without starting execution, the biases are
    registered against exactly those ids, and only then does
    wait_for_completion() run the batch. That removes any need to guess vLLM's
    id scheme -- a guess that silently mismatched would leave every row unbiased
    and make PIRA look free.

    The enqueue call is inside the timed region for both methods, so the
    comparison stays symmetric; bias registration is one small RPC and is timed
    as part of PIRA's cost.
    """
    prompts = [{"prompt_token_ids": prompt_ids} for _ in range(count)]

    started = time.perf_counter()
    if method == "PIRA":
        request_ids = llm.enqueue(prompts, sampling, use_tqdm=False)
        pira.register_biases_in_worker(
            llm,
            build_biases(
                list(request_ids),
                layers=layers,
                num_experts=num_experts,
                top_k=top_k,
                beta=beta,
                seed=seed,
            ),
        )
        outputs = llm.wait_for_completion(use_tqdm=False)
    else:
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed = time.perf_counter() - started

    produced = sum(
        len(output.outputs[0].token_ids)
        for output in outputs
        if getattr(output, "outputs", None)
    )
    return elapsed, produced


def main() -> int:
    args = parse_args()
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM is required; run this in the vLLM environment.", file=sys.stderr)
        return 2

    import pira_vllm_routing as pira

    max_input = max(args.input_lengths)
    max_output = max(args.output_lengths)
    max_len = args.max_model_len or (max_input + max_output)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_len,
        max_num_seqs=max(args.concurrency),
        enable_prefix_caching=False,
        trust_remote_code=True,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    config = llm.llm_engine.model_config.hf_config
    num_experts = int(getattr(config, "num_experts", 0) or getattr(config, "n_routed_experts", 0))
    if num_experts <= 0:
        print("could not determine the model's expert count", file=sys.stderr)
        return 2
    layers = list(range(args.probe_layer + 1))

    print(
        f"model={args.model} experts={num_experts} "
        f"biased_layers=0..{args.probe_layer} suppressed_per_layer={args.top_k}"
    )

    records: list[dict] = []
    diagnostics: dict = {}

    for input_length in args.input_lengths:
        prompt_ids = fixed_length_prompt(tokenizer, input_length)
        for output_length in args.output_lengths:
            sampling = SamplingParams(
                temperature=0.0, max_tokens=output_length, ignore_eos=True
            )
            warmup = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
            for concurrency in args.concurrency:
                total = max(args.requests_per_cell, concurrency)
                total -= total % concurrency

                for method in ("Original", "PIRA"):
                    if method == "PIRA":
                        installed = pira.install_in_worker(
                            llm, beta=args.beta, strict=True
                        )
                        if not installed:
                            print("no MoE layer was hooked", file=sys.stderr)
                            return 2

                    try:
                        # Warm up so allocator growth and kernel autotuning stay
                        # out of the timed region.
                        run_once(
                            llm,
                            pira,
                            method=method,
                            prompt_ids=prompt_ids,
                            count=concurrency,
                            sampling=warmup,
                            layers=layers,
                            num_experts=num_experts,
                            top_k=args.top_k,
                            beta=args.beta,
                            seed=args.seed,
                        )

                        durations = []
                        for repeat in range(args.repeats):
                            elapsed, produced = run_once(
                                llm,
                                pira,
                                method=method,
                                prompt_ids=prompt_ids,
                                count=total,
                                sampling=sampling,
                                layers=layers,
                                num_experts=num_experts,
                                top_k=args.top_k,
                                beta=args.beta,
                                seed=args.seed + repeat,
                            )
                            durations.append(elapsed)
                            expected = total * output_length
                            if produced != expected:
                                print(
                                    f"warning: produced {produced} tokens, "
                                    f"expected {expected}",
                                    file=sys.stderr,
                                )

                        if method == "PIRA":
                            diagnostics = pira.worker_diagnostics(llm)
                    finally:
                        if method == "PIRA":
                            pira.uninstall_in_worker(llm)

                    median = statistics.median(durations)
                    record = {
                        "method": method,
                        "input_length": input_length,
                        "output_length": output_length,
                        "concurrency": concurrency,
                        "num_requests": total,
                        "median_seconds": median,
                        "min_seconds": min(durations),
                        "max_seconds": max(durations),
                        "seconds_per_request": median / total,
                        "request_throughput": total / median,
                        "output_token_throughput": total * output_length / median,
                    }
                    if method == "PIRA":
                        record["diagnostics"] = diagnostics
                    records.append(record)
                    print(
                        f"  {method:<8} in={input_length:<5} out={output_length:<4} "
                        f"conc={concurrency:<3} {median:8.3f} s  "
                        f"{record['request_throughput']:7.2f} req/s"
                    )

    # Pair the two methods per cell to get the routing overhead.
    print("\n=== routing-intervention overhead (generation only, probe excluded) ===")
    header = (
        f"{'in':>6} {'out':>5} {'conc':>5} {'Original(s)':>12} "
        f"{'PIRA(s)':>10} {'overhead':>9}"
    )
    print(header)
    print("-" * len(header))
    comparisons = []
    for input_length in args.input_lengths:
        for output_length in args.output_lengths:
            for concurrency in args.concurrency:
                def pick(method):
                    return next(
                        (
                            r for r in records
                            if r["method"] == method
                            and r["input_length"] == input_length
                            and r["output_length"] == output_length
                            and r["concurrency"] == concurrency
                        ),
                        None,
                    )

                original, biased = pick("Original"), pick("PIRA")
                if not original or not biased:
                    continue
                ratio = biased["median_seconds"] / original["median_seconds"] - 1.0
                comparisons.append(
                    {
                        "input_length": input_length,
                        "output_length": output_length,
                        "concurrency": concurrency,
                        "original_seconds": original["median_seconds"],
                        "pira_seconds": biased["median_seconds"],
                        "routing_overhead_fraction": ratio,
                        "routing_overhead_percent": 100.0 * ratio,
                    }
                )
                print(
                    f"{input_length:>6} {output_length:>5} {concurrency:>5} "
                    f"{original['median_seconds']:>12.3f} "
                    f"{biased['median_seconds']:>10.3f} "
                    f"{100.0 * ratio:>8.1f}%"
                )

    rows_biased = diagnostics.get("rows_biased", 0)
    print(f"\nworker diagnostics (last PIRA cell): {diagnostics}")
    if not rows_biased:
        print(
            "\nFAIL: the router never saw a biased row, so the PIRA timings are "
            "just the Original model. Check that request ids match "
            "InputBatch.req_ids and that the MoE backend is modular.",
            file=sys.stderr,
        )

    payload = {
        "metadata": {
            "model": args.model,
            "num_experts": num_experts,
            "biased_layers": [0, args.probe_layer],
            "suppressed_per_layer": args.top_k,
            "beta": args.beta,
            "repeats": args.repeats,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": max_len,
            "bias": "seeded synthetic suppression sets; timing only",
            "scope": "generation only; the probe is measured by "
                     "rebuttal_probe_scaling.py",
        },
        "measurements": records,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0 if rows_biased else 1


if __name__ == "__main__":
    raise SystemExit(main())
