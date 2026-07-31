"""Fixed-resource batch stress test for Original and PIRA inference.

This is the fallback benchmark when a serving engine cannot expose the
differentiable, per-request MoE router state required by PIRA. It runs both
methods with the same Hugging Face model, precision, GPU, prompt, batch size,
and exact output length. Safety is intentionally not evaluated; a seeded
synthetic direction exercises the same probe graph and tensor shapes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rebuttal_cost_benchmark import (
    _clear_cuda,
    _gib,
    _moe_layers,
    _probe_for_bias,
    _router_bias_hooks,
    _sync,
)


@dataclass
class BatchMeasurement:
    method: str
    batch_size: int
    probe_microbatch_size: int | None
    num_requests: int
    prompt_tokens: int
    output_tokens_per_request: int
    num_batches: int
    probe_seconds: float
    generation_prefill_seconds: float
    generation_seconds: float
    total_seconds: float | None
    request_throughput: float
    output_token_throughput: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    status: str
    error: str | None = None


@torch.inference_mode()
def _decode_exact(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[float, float]:
    _sync()
    start = time.perf_counter()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache = output.past_key_values
    _sync()
    prefill = time.perf_counter() - start

    for _ in range(max_new_tokens - 1):
        output = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    _sync()
    total = time.perf_counter() - start
    del output, cache, next_token
    return prefill, total


def _run_workload(
    *,
    method: str,
    model,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    batch_size: int,
    num_requests: int,
    max_new_tokens: int,
    gates,
    probe_layer: int,
    direction: torch.Tensor,
    top_k: int,
    beta: float,
    probe_microbatch_size: int | None,
) -> BatchMeasurement:
    if num_requests % batch_size:
        raise ValueError("num_requests must be divisible by every batch size")

    num_batches = num_requests // batch_size
    effective_probe_batch = min(probe_microbatch_size or batch_size, batch_size)
    if batch_size % effective_probe_batch:
        raise ValueError(
            f"generation batch={batch_size} must be divisible by "
            f"probe microbatch={effective_probe_batch}"
        )
    input_ids = base_input_ids.expand(batch_size, -1).contiguous()
    attention_mask = base_attention_mask.expand(batch_size, -1).contiguous()
    probe_total = 0.0
    prefill_total = 0.0
    generation_total = 0.0
    peak_allocated = 0
    peak_reserved = 0

    _clear_cuda()
    wall_start = time.perf_counter()
    try:
        for _ in range(num_batches):
            if method == "Original":
                torch.cuda.reset_peak_memory_stats()
                prefill, generation = _decode_exact(
                    model, input_ids, attention_mask, max_new_tokens
                )
            elif method == "PIRA":
                bias_chunks: dict[int, list[torch.Tensor]] = {}
                for start_idx in range(0, batch_size, effective_probe_batch):
                    end_idx = start_idx + effective_probe_batch
                    chunk_bias, probe, probe_peak_gib, probe_reserved_gib = _probe_for_bias(
                        model=model,
                        input_ids=input_ids[start_idx:end_idx],
                        attention_mask=attention_mask[start_idx:end_idx],
                        gates=gates,
                        probe_layer=probe_layer,
                        direction=direction,
                        top_k=top_k,
                        beta=beta,
                    )
                    probe_total += probe
                    peak_allocated = max(
                        peak_allocated, int(probe_peak_gib * 1024**3)
                    )
                    peak_reserved = max(
                        peak_reserved, int(probe_reserved_gib * 1024**3)
                    )
                    for layer_idx, layer_bias in chunk_bias.items():
                        bias_chunks.setdefault(layer_idx, []).append(layer_bias)
                fixed_bias = {
                    layer_idx: torch.cat(chunks, dim=0)
                    for layer_idx, chunks in bias_chunks.items()
                }
                torch.cuda.reset_peak_memory_stats()
                with _router_bias_hooks(gates, fixed_bias):
                    prefill, generation = _decode_exact(
                        model, input_ids, attention_mask, max_new_tokens
                    )
                del fixed_bias
            else:
                raise ValueError(f"Unknown method: {method}")

            prefill_total += prefill
            generation_total += generation
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())

        _sync()
        total = time.perf_counter() - wall_start
        return BatchMeasurement(
            method=method,
            batch_size=batch_size,
            probe_microbatch_size=(
                effective_probe_batch if method == "PIRA" else None
            ),
            num_requests=num_requests,
            prompt_tokens=base_input_ids.shape[1],
            output_tokens_per_request=max_new_tokens,
            num_batches=num_batches,
            probe_seconds=probe_total,
            generation_prefill_seconds=prefill_total,
            generation_seconds=generation_total,
            total_seconds=total,
            request_throughput=num_requests / total,
            output_token_throughput=num_requests * max_new_tokens / total,
            peak_allocated_gib=_gib(peak_allocated),
            peak_reserved_gib=_gib(peak_reserved),
            status="ok",
        )
    except torch.OutOfMemoryError as exc:
        return BatchMeasurement(
            method=method,
            batch_size=batch_size,
            probe_microbatch_size=(
                effective_probe_batch if method == "PIRA" else None
            ),
            num_requests=num_requests,
            prompt_tokens=base_input_ids.shape[1],
            output_tokens_per_request=max_new_tokens,
            num_batches=num_batches,
            probe_seconds=probe_total,
            generation_prefill_seconds=prefill_total,
            generation_seconds=generation_total,
            total_seconds=None,
            request_throughput=0.0,
            output_token_throughput=0.0,
            peak_allocated_gib=_gib(torch.cuda.max_memory_allocated()),
            peak_reserved_gib=_gib(torch.cuda.max_memory_reserved()),
            status="oom",
            error=str(exc),
        )
    finally:
        del input_ids, attention_mask
        _clear_cuda()


def _make_fixed_length_prompt(tokenizer, prompt: str, target_tokens: int):
    repeated = " ".join([prompt] * max(1, target_tokens // 8 + 1))
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": repeated}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=target_tokens,
        padding="max_length",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument(
        "--prompt", default="Explain why the sky appears blue in concise terms."
    )
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--probe-layer", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument(
        "--probe-microbatch-size",
        type=int,
        help="Maximum PIRA probe batch; generation still uses --batch-sizes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this script through Slurm")
    for batch_size in args.batch_sizes:
        if args.num_requests % batch_size:
            raise ValueError(
                f"num_requests={args.num_requests} is not divisible by batch_size={batch_size}"
            )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval().requires_grad_(False)
    gates = _moe_layers(model)
    input_ids, attention_mask = _make_fixed_length_prompt(
        tokenizer, args.prompt, args.prompt_tokens
    )
    input_ids = input_ids.to("cuda:0")
    attention_mask = attention_mask.to("cuda:0")

    generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
    direction = torch.randn(
        model.config.hidden_size,
        device="cuda:0",
        dtype=torch.float32,
        generator=generator,
    )
    direction /= direction.norm().clamp_min(1e-12)

    print(
        f"GPU={torch.cuda.get_device_name(0)} model={args.model} "
        f"N={args.num_requests} input={args.prompt_tokens} output={args.output_tokens}",
        flush=True,
    )
    print("Warmup: Original and PIRA at batch=1", flush=True)
    _decode_exact(model, input_ids, attention_mask, 2)
    warmup_bias, *_ = _probe_for_bias(
        model,
        input_ids,
        attention_mask,
        gates,
        args.probe_layer,
        direction,
        args.top_k,
        args.beta,
    )
    with _router_bias_hooks(gates, warmup_bias):
        _decode_exact(model, input_ids, attention_mask, 2)
    del warmup_bias
    _clear_cuda()

    records: list[BatchMeasurement] = []
    for batch_size in args.batch_sizes:
        for method in ("Original", "PIRA"):
            print(f"Running {method} batch={batch_size}", flush=True)
            record = _run_workload(
                method=method,
                model=model,
                base_input_ids=input_ids,
                base_attention_mask=attention_mask,
                batch_size=batch_size,
                num_requests=args.num_requests,
                max_new_tokens=args.output_tokens,
                gates=gates,
                probe_layer=args.probe_layer,
                direction=direction,
                top_k=args.top_k,
                beta=args.beta,
                probe_microbatch_size=args.probe_microbatch_size,
            )
            records.append(record)
            print(asdict(record), flush=True)

    payload = {
        "metadata": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "prompt": args.prompt,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "num_requests": args.num_requests,
            "batch_sizes": args.batch_sizes,
            "probe_layer": args.probe_layer,
            "top_k": args.top_k,
            "beta": args.beta,
            "probe_microbatch_size": args.probe_microbatch_size,
            "direction": "seeded synthetic unit vector; timing only",
            "serving_mode": "static HF batching; identical fixed GPU and batch per pair",
        },
        "measurements": [asdict(record) for record in records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
