"""Probe cost matrix: HF vs SonicMoE, with and without gradient checkpointing.

Five arms at a fixed prompt length, swept over batch size:

  1. hf_ckpt        HF expert kernel (grouped_mm), gradient checkpointing ON
  2. hf_nockpt      HF expert kernel, checkpointing OFF
  3. sonic_ckpt     SonicMoE expert kernel, checkpointing ON
  4. sonic_nockpt   SonicMoE expert kernel, checkpointing OFF
  5. vllm_original  vLLM generation with no probe at all (the baseline)

Arms 1-4 are the PIRA probe; arm 5 is measured separately because vLLM's
inference path has no autograd and cannot host the probe. Run arm 5 with
--vllm-only in the vLLM environment; arms 1-4 run in the HF environment.

Two metrics per cell:

  time            probe forward + backward seconds, median of --repeats
  peak memory     BOTH peak_allocated (which includes the ~57 GiB of resident
                  weights) and activation_gib (peak minus resident weights).
                  Activation memory is the one that responds to these knobs:
                  weights dominate the raw peak so completely that a kernel
                  change is invisible in it -- 58.24 vs 58.57 GiB in an earlier
                  SonicMoE run, a 0.6% difference, while the underlying
                  activation figures were 1.24 vs 1.57 GiB.

Checkpointing is the larger lever and it trades the two metrics against each
other: it discards each layer's internal activations and recomputes them during
backward, which measured 9.1x less activation memory for about 34% more time at
4096 tokens. Expect the no-checkpointing arms to OOM first as batch grows;
that boundary is a result, so an OOM is recorded rather than treated as an error.

Both --prompt-tokens and --batch-sizes accept several values, so one run covers a
context sweep, a batch sweep, or the full cross product. The vLLM arm rebuilds its
engine per prompt length, because max_model_len has to bound prompt+output.

Usage (HF env, arms 1-4) -- context sweep at batch 1:
    python scripts/moe/compare_probe_matrix.py \
        --prompt-tokens 2048 4096 8192 16384 --batch-sizes 1 \
        --output timing_results/probe_matrix.json

Usage (vLLM env, arm 5) -- same contexts, 256-token completions:
    python scripts/moe/compare_probe_matrix.py --vllm-only \
        --prompt-tokens 2048 4096 8192 16384 --batch-sizes 1 \
        --output-tokens 256 \
        --output timing_results/probe_matrix_vllm.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--probe-layer", type=int, default=24)
    parser.add_argument(
        "--prompt-tokens", type=int, nargs="+", default=[4096],
        help="One or more prompt lengths to sweep.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument(
        "--output-tokens", default="256",
        help="Generation length for the vLLM baseline arm. 'match' sets it equal "
        "to each prompt length.",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vllm-only",
        action="store_true",
        help="Measure only arm 5 (vLLM generation). Requires the vLLM environment.",
    )
    parser.add_argument(
        "--skip-sonic",
        action="store_true",
        help="Skip arms 3-4 when SonicMoE is unavailable.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Arms 1-4: the probe
# --------------------------------------------------------------------------- #


def register_sonicmoe() -> str | None:
    """Point Transformers' kernel mapping at the installed sonicmoe package.

    Returns the version, or None when it is unavailable -- reported rather than
    fatal, so the HF arms still produce results.
    """
    try:
        import sonicmoe
        from transformers.integrations import hub_kernels

        hub_kernels._KERNEL_MODULE_MAPPING["sonic-moe"] = sonicmoe
        return sonicmoe.__version__
    except Exception:  # noqa: BLE001
        return None


def measure_probe(model, batch, direction, probe_layer, repeats, checkpoint):
    """Median probe timing plus peak and activation memory, or an OOM record."""
    from pira_probe import FastProbe, free_cuda

    free_cuda()
    # Resident weights, so activation memory can be separated from the raw peak.
    resident_gib = torch.cuda.memory_allocated() / 2**30

    probe = FastProbe(
        model, probe_layer, direction,
        checkpoint=checkpoint, truncate=True, efficient_attention=True,
    )
    try:
        # Untimed warmup: keeps autotuning and allocator growth out of the numbers.
        probe.run(batch.input_ids, batch.attention_mask, last_index=batch.last_index)
        free_cuda()
        torch.cuda.reset_peak_memory_stats()

        totals, forwards, backwards, peaks = [], [], [], []
        for _ in range(repeats):
            result = probe.run(
                batch.input_ids, batch.attention_mask, last_index=batch.last_index
            )
            totals.append(result.total_seconds)
            forwards.append(result.forward_seconds)
            backwards.append(result.backward_seconds)
            peaks.append(result.peak_allocated_gib)
            del result
            free_cuda()
        peak = max(peaks)
        return {
            "status": "ok",
            "total_seconds": statistics.median(totals),
            "forward_seconds": statistics.median(forwards),
            "backward_seconds": statistics.median(backwards),
            "seconds_per_request": statistics.median(totals) / batch.input_ids.shape[0],
            "peak_allocated_gib": peak,
            "resident_weight_gib": resident_gib,
            "activation_gib": max(0.0, peak - resident_gib),
            "all_total_seconds": totals,
        }
    except torch.OutOfMemoryError as error:
        free_cuda()
        # Where the no-checkpointing arms stop being usable is itself the result.
        return {"status": "oom", "error": str(error)[:200],
                "resident_weight_gib": resident_gib}
    except Exception as error:  # noqa: BLE001
        free_cuda()
        return {"status": "error", "error": f"{type(error).__name__}: {error}"[:300]}


def run_probe_arms(args) -> dict:
    from pira_probe import free_cuda, unit_direction
    from probe_workload import build_prompt_batch, load_model

    sonic_version = register_sonicmoe()
    model, tokenizer = load_model(args.model, torch.bfloat16)
    device = next(model.parameters()).device
    direction = unit_direction(model.config.hidden_size, device, args.seed)

    can_switch = hasattr(model, "set_experts_implementation")
    arms = [("hf", "grouped_mm", True), ("hf", "grouped_mm", False)]
    if sonic_version and can_switch and not args.skip_sonic:
        arms += [("sonic", "sonicmoe", True), ("sonic", "sonicmoe", False)]

    print(f"GPU={torch.cuda.get_device_name(0)} model={args.model}")
    print(f"prompt={args.prompt_tokens} probe_layer={args.probe_layer} "
          f"sonicmoe={sonic_version or 'unavailable'} "
          f"switchable={can_switch}")
    print()

    records = []
    for prompt_tokens in args.prompt_tokens:
     for batch_size in args.batch_sizes:
        batch = build_prompt_batch(
            tokenizer, prompt_tokens=prompt_tokens,
            batch_size=batch_size, device=device, seed=args.seed,
        )
        for kernel_name, implementation, checkpoint in arms:
            if can_switch:
                try:
                    model.set_experts_implementation(implementation)
                except Exception as error:  # noqa: BLE001
                    print(f"  could not select {implementation}: {error}")
                    continue
            arm = f"{kernel_name}_{'ckpt' if checkpoint else 'nockpt'}"
            result = measure_probe(
                model, batch, direction, args.probe_layer, args.repeats, checkpoint
            )
            record = {
                "arm": arm, "kernel": kernel_name, "checkpointing": checkpoint,
                "batch_size": batch_size, "prompt_tokens": prompt_tokens,
                **result,
            }
            records.append(record)
            if result["status"] == "ok":
                print(f"  {arm:<14} in{prompt_tokens:<6} b{batch_size:<3} "
                      f"{result['total_seconds']:>8.3f} s  "
                      f"act {result['activation_gib']:>7.2f} GiB  "
                      f"peak {result['peak_allocated_gib']:>7.2f} GiB")
            else:
                print(f"  {arm:<14} in{prompt_tokens:<6} b{batch_size:<3} "
                      f"{result['status'].upper()}")
        del batch
        free_cuda()
        print()

    return {
        "metadata": {
            "model": args.model, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "sonicmoe": sonic_version, "dtype": "bfloat16",
            "prompt_tokens": args.prompt_tokens, "probe_layer": args.probe_layer,
            "batch_sizes": args.batch_sizes, "repeats": args.repeats,
            "arms": [f"{k}_{'ckpt' if c else 'nockpt'}" for k, _, c in arms],
        },
        "measurements": records,
    }


# --------------------------------------------------------------------------- #
# Arm 5: vLLM generation baseline
# --------------------------------------------------------------------------- #


def _vllm_worker_peak_memory_gib(worker) -> float:
    """Return the worker's CUDA allocator peak through vLLM collective RPC."""
    import torch as worker_torch

    if not worker_torch.cuda.is_available():
        return 0.0
    return worker_torch.cuda.max_memory_allocated() / 2**30


def _vllm_worker_reset_peak_memory(worker) -> None:
    """Reset the worker's CUDA allocator peak through vLLM collective RPC."""
    import torch as worker_torch

    if worker_torch.cuda.is_available():
        worker_torch.cuda.reset_peak_memory_stats()


def run_vllm_arm(args) -> dict:
    """Original vLLM generation, no probe. Measured in its own process.

    Separate because vLLM's inference path has no autograd and cannot host the
    probe; this arm supplies the denominator for the probe cost.

    One engine per prompt length: max_model_len has to bound prompt+output, and
    changing it means rebuilding the engine.
    """
    import time

    from vllm import LLM, SamplingParams

    text = ("The mixture-of-experts layer routes each token to a small subset of "
            "feed-forward experts, which lets the model grow its parameter count "
            "without a proportional increase in compute per token. ")

    records = []
    for prompt_tokens in args.prompt_tokens:
        output_tokens = (
            prompt_tokens if str(args.output_tokens) == "match"
            else int(args.output_tokens)
        )
        max_len = prompt_tokens + output_tokens
        print(f"  building engine for in={prompt_tokens} out={output_tokens}",
              flush=True)
        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_len, max_num_seqs=max(args.batch_sizes),
            # max_num_batched_tokens is left to vLLM. Pinning it to max_len starves
            # prefill at larger batch, and the grid that runs cleanly does not set
            # it either.
            **({"max_num_batched_tokens": args.max_num_batched_tokens}
               if getattr(args, "max_num_batched_tokens", None) else {}),
            enable_prefix_caching=False, trust_remote_code=True, seed=args.seed,
        )
        tokenizer = llm.get_tokenizer()
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompt_ids = (ids * ((prompt_tokens + len(ids) - 1) // len(ids)))[:prompt_tokens]
        sampling = SamplingParams(temperature=0.0, max_tokens=output_tokens,
                                  ignore_eos=True)

        def peak_gib():
            try:
                values = [
                    value for value in llm.collective_rpc(_vllm_worker_peak_memory_gib)
                    if isinstance(value, (int, float))
                ]
                return max(values) if values else None
            except Exception as error:  # noqa: BLE001
                print(f"warning: could not read worker peak memory: {error}")
                return None

        def reset_peak():
            try:
                llm.collective_rpc(_vllm_worker_reset_peak_memory)
            except Exception as error:  # noqa: BLE001
                print(f"warning: could not reset worker peak memory: {error}")

        for batch_size in args.batch_sizes:
            prompts = [{"prompt_token_ids": prompt_ids} for _ in range(batch_size)]
            llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                 ignore_eos=True), use_tqdm=False)
            reset_peak()
            totals = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                llm.generate(prompts, sampling, use_tqdm=False)
                totals.append(time.perf_counter() - started)
            median = statistics.median(totals)
            record = {
                "arm": "vllm_original", "kernel": "vllm", "checkpointing": None,
                "batch_size": batch_size, "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens, "status": "ok",
                "total_seconds": median, "seconds_per_request": median / batch_size,
                "peak_allocated_gib": peak_gib(), "all_total_seconds": totals,
            }
            records.append(record)
            print(f"  vllm_original  in{prompt_tokens:<6} b{batch_size:<3} "
                  f"{median:>8.3f} s  ({median / batch_size:.4f} s/req)  "
                  f"peak {record['peak_allocated_gib']}")

        # Free the engine before building the next one, or the second allocation
        # fails: vLLM preallocates a KV cache pool sized to gpu_memory_utilization.
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print()

    return {
        "metadata": {
            "model": args.model, "gpu": torch.cuda.get_device_name(0),
            "arm": "vllm_original", "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens, "batch_sizes": args.batch_sizes,
            "repeats": args.repeats,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "measurements": records,
    }


def main() -> int:
    args = parse_args()
    # This script checks CUDA before constructing LLM. vLLM's default `fork`
    # cannot safely reinitialize that CUDA context in EngineCore, so make the
    # printed standalone --vllm-only command work without an extra env override.
    if args.vllm_only:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # collective_rpc must serialize the two local, trusted CUDA-memory
        # callbacks above. vLLM 0.26 requires an explicit opt-in for callables.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        # Benchmark runs do not need vLLM telemetry, and a full HOME quota must
        # not make its background usage-stats writer fail noisily.
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        return 2
    if args.vllm_only and "B200" in torch.cuda.get_device_name(0):
        # FlashInfer 0.6.14's TRTLLM BF16 MoE tactic search can hit an illegal
        # memory access on SM100 for this 4352-token engine shape. Keep the
        # selected FlashInfer backend, but use its safe heuristic tactic rather
        # than profiling the unsafe candidates.
        os.environ.setdefault(
            "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS",
            "flashinfer::trtllm_bf16_moe",
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    payload = run_vllm_arm(args) if args.vllm_only else run_probe_arms(args)

    print("\n=== summary ===")
    header = (f"{'arm':<15}{'input':>7}{'batch':>6}{'time(s)':>10}"
              f"{'act GiB':>10}{'peak GiB':>10}")
    print(header)
    print("-" * len(header))
    for record in payload["measurements"]:
        if record["status"] != "ok":
            print(f"{record['arm']:<15}{record['prompt_tokens']:>7}"
                  f"{record['batch_size']:>6}{record['status'].upper():>10}")
            continue
        activation = record.get("activation_gib")
        peak = record.get("peak_allocated_gib")
        print(f"{record['arm']:<15}{record['prompt_tokens']:>7}"
              f"{record['batch_size']:>6}"
              f"{record['total_seconds']:>10.3f}"
              f"{activation if activation is not None else float('nan'):>10.2f}"
              f"{peak if peak is not None else float('nan'):>10.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
