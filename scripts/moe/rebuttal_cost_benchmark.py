"""Measure the batch-1 compute and memory cost of online PIRA.

This benchmark intentionally does not evaluate safety.  It uses a seeded
synthetic output-safety direction, which exercises the same forward/backward
path and has the same tensor shapes as the learned direction in the paper.

Per PIRA request:
  1. Differentiable prompt forward, stopped immediately after probe_layer.
  2. autograd.grad(score, router_biases) with frozen model parameters.
  3. Release the probe graph.
  4. Fresh biased prefill followed by ordinary KV-cached greedy decoding.

The script is meant to run on a GPU compute node, not a login node.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class _ProbeComplete(RuntimeError):
    """Private control-flow exception used to stop after the probe layer."""


@dataclass
class Measurement:
    method: str
    repeat: int
    prompt_tokens: int
    new_tokens: int
    probe_seconds: float
    generation_prefill_seconds: float
    ttft_seconds: float
    total_seconds: float
    probe_peak_allocated_gib: float
    generation_peak_allocated_gib: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    steady_allocated_gib: float
    response: str


def _gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def _sync() -> None:
    torch.cuda.synchronize()


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    _sync()


def _moe_layers(model) -> list[tuple[int, torch.nn.Module]]:
    layers = model.model.layers
    found = []
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None) if mlp is not None else None
        if gate is not None:
            found.append((layer_idx, gate))
    if not found:
        raise RuntimeError("No MoE gate modules found at model.model.layers[*].mlp.gate")
    return found


@contextmanager
def _router_bias_hooks(
    gates: list[tuple[int, torch.nn.Module]],
    bias_by_layer: dict[int, torch.Tensor],
) -> Iterator[None]:
    """Add per-layer, optionally per-request bias before softmax/top-k."""

    handles = []
    original_forwards = []
    for layer_idx, gate in gates:
        if layer_idx not in bias_by_layer:
            continue
        bias = bias_by_layer[layer_idx]

        if all(hasattr(gate, attr) for attr in ("weight", "hidden_dim", "top_k", "norm_topk_prob")):
            original_forwards.append((gate, gate.forward))

            def biased_router_forward(module, hidden_states, *, bias=bias):
                hidden_states = hidden_states.reshape(-1, module.hidden_dim)
                logits = F.linear(hidden_states, module.weight).float()
                if bias.ndim == 2:
                    batch_size = bias.shape[0]
                    if logits.shape[0] % batch_size:
                        raise ValueError(
                            f"Router token rows={logits.shape[0]} are not divisible "
                            f"by bias batch={batch_size}"
                        )
                    tokens_per_request = logits.shape[0] // batch_size
                    expanded_bias = (
                        bias[:, None, :]
                        .expand(batch_size, tokens_per_request, bias.shape[-1])
                        .reshape_as(logits)
                    )
                else:
                    expanded_bias = bias
                logits = logits.add(expanded_bias)
                probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                scores, indices = torch.topk(probs, module.top_k, dim=-1)
                if module.norm_topk_prob:
                    scores = scores / scores.sum(dim=-1, keepdim=True)
                return logits.to(hidden_states.dtype), scores.to(hidden_states.dtype), indices

            gate.forward = types.MethodType(biased_router_forward, gate)
            continue

        def add_bias(module, _inputs, output, *, bias=bias):
            if isinstance(output, torch.Tensor):
                return output.float().add(bias).to(output.dtype)
            raise TypeError(f"Unsupported Qwen3 MoE router output: {type(output)!r}")

        handles.append(gate.register_forward_hook(add_bias))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for gate, original_forward in original_forwards:
            gate.forward = original_forward


def _probe_for_bias(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gates: list[tuple[int, torch.nn.Module]],
    probe_layer: int,
    direction: torch.Tensor,
    top_k: int,
    beta: float,
) -> tuple[dict[int, torch.Tensor], float, float, float]:
    """Return per-request router biases plus probe time and peak CUDA memory."""

    active_gates = [(li, gate) for li, gate in gates if li <= probe_layer]
    if not active_gates:
        raise ValueError(f"No MoE gates at or below probe layer {probe_layer}")

    batch_size = input_ids.shape[0]
    live_biases = {
        li: torch.zeros(
            batch_size,
            gate.num_experts,
            device=input_ids.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        for li, gate in active_gates
    }
    captured: dict[str, torch.Tensor] = {}

    def stop_after_probe(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden
        raise _ProbeComplete

    stop_handle = model.model.layers[probe_layer].register_forward_hook(stop_after_probe)
    torch.cuda.reset_peak_memory_stats()
    _sync()
    start = time.perf_counter()
    try:
        with _router_bias_hooks(active_gates, live_biases):
            try:
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            except _ProbeComplete:
                pass
        if "hidden" not in captured:
            raise RuntimeError("Probe layer hook did not capture a hidden state")
        per_request_score = (
            captured["hidden"][:, -1, :].float() * direction
        ).sum(dim=-1)
        score = per_request_score.sum()
        grads = torch.autograd.grad(
            score,
            tuple(live_biases.values()),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        _sync()
        elapsed = time.perf_counter() - start
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        if len({grad.numel() for grad in grads}) != 1:
            raise RuntimeError("This Qwen3 benchmark expects equal expert counts per MoE layer")
        # [layers, batch, experts] -> independently rank each request over
        # every active layer/expert pair.
        delta = torch.stack([grad.detach().float() for grad in grads])
        per_request_delta = delta.permute(1, 0, 2).reshape(batch_size, -1)
        selected = torch.topk(
            per_request_delta,
            min(top_k, per_request_delta.shape[1]),
            dim=1,
            largest=False,
        ).indices
        fixed_per_request = torch.zeros_like(per_request_delta)
        fixed_per_request.scatter_(1, selected, -beta)
        fixed_matrix = fixed_per_request.reshape(
            batch_size, len(active_gates), -1
        ).permute(1, 0, 2)
        fixed = {
            layer_idx: fixed_matrix[row].detach()
            for row, (layer_idx, _gate) in enumerate(active_gates)
        }
    finally:
        stop_handle.remove()

    del (
        captured,
        per_request_score,
        score,
        grads,
        live_biases,
        delta,
        per_request_delta,
        selected,
        fixed_per_request,
        fixed_matrix,
    )
    _sync()
    return fixed, elapsed, _gib(peak_allocated), _gib(peak_reserved)


@torch.inference_mode()
def _greedy_decode(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, float, float]:
    """Generate exactly max_new_tokens and return response, prefill time, total."""

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
    generated = [next_token]
    cache = output.past_key_values
    _sync()
    prefill_seconds = time.perf_counter() - start

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
        generated.append(next_token)

    _sync()
    total_seconds = time.perf_counter() - start
    token_ids = torch.cat(generated, dim=1)[0]
    response = tokenizer.decode(token_ids, skip_special_tokens=True)
    del output, cache, next_token, generated, token_ids
    return response, prefill_seconds, total_seconds


def _measure_baseline(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    repeat: int,
) -> Measurement:
    _clear_cuda()
    steady = _gib(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    response, prefill, total = _greedy_decode(
        model, tokenizer, input_ids, attention_mask, max_new_tokens
    )
    peak_allocated = _gib(torch.cuda.max_memory_allocated())
    peak_reserved = _gib(torch.cuda.max_memory_reserved())
    _clear_cuda()
    return Measurement(
        method="Original",
        repeat=repeat,
        prompt_tokens=input_ids.shape[1],
        new_tokens=max_new_tokens,
        probe_seconds=0.0,
        generation_prefill_seconds=prefill,
        ttft_seconds=prefill,
        total_seconds=total,
        probe_peak_allocated_gib=0.0,
        generation_peak_allocated_gib=peak_allocated,
        peak_allocated_gib=peak_allocated,
        peak_reserved_gib=peak_reserved,
        steady_allocated_gib=steady,
        response=response,
    )


def _measure_pira(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gates: list[tuple[int, torch.nn.Module]],
    probe_layer: int,
    direction: torch.Tensor,
    top_k: int,
    beta: float,
    max_new_tokens: int,
    repeat: int,
) -> Measurement:
    _clear_cuda()
    steady = _gib(torch.cuda.memory_allocated())
    _sync()
    total_start = time.perf_counter()
    fixed_bias, probe_seconds, probe_peak, probe_peak_reserved = _probe_for_bias(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        gates=gates,
        probe_layer=probe_layer,
        direction=direction,
        top_k=top_k,
        beta=beta,
    )

    pre_generation_seconds = time.perf_counter() - total_start
    torch.cuda.reset_peak_memory_stats()
    with _router_bias_hooks(gates, fixed_bias):
        response, prefill, generation_total = _greedy_decode(
            model, tokenizer, input_ids, attention_mask, max_new_tokens
        )
    generation_peak = _gib(torch.cuda.max_memory_allocated())
    generation_peak_reserved = _gib(torch.cuda.max_memory_reserved())
    total = time.perf_counter() - total_start
    del fixed_bias
    _clear_cuda()
    return Measurement(
        method="PIRA",
        repeat=repeat,
        prompt_tokens=input_ids.shape[1],
        new_tokens=max_new_tokens,
        probe_seconds=probe_seconds,
        generation_prefill_seconds=prefill,
        ttft_seconds=pre_generation_seconds + prefill,
        total_seconds=total,
        probe_peak_allocated_gib=probe_peak,
        generation_peak_allocated_gib=generation_peak,
        peak_allocated_gib=max(probe_peak, generation_peak),
        peak_reserved_gib=max(probe_peak_reserved, generation_peak_reserved),
        steady_allocated_gib=steady,
        response=response,
    )


def _median(records: list[Measurement], field: str) -> float:
    return statistics.median(getattr(record, field) for record in records)


def _print_summary(records: list[Measurement]) -> None:
    print("\n=== Median over measured repeats ===", flush=True)
    header = (
        f"{'Method':<10} {'Probe(s)':>9} {'Prefill(s)':>11} {'TTFT(s)':>9} "
        f"{'Total(s)':>10} {'PeakAlloc(GiB)':>15} {'PeakRsvd(GiB)':>14}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    grouped = {
        method: [record for record in records if record.method == method]
        for method in ("Original", "PIRA")
    }
    for method, rows in grouped.items():
        print(
            f"{method:<10} "
            f"{_median(rows, 'probe_seconds'):>9.3f} "
            f"{_median(rows, 'generation_prefill_seconds'):>11.3f} "
            f"{_median(rows, 'ttft_seconds'):>9.3f} "
            f"{_median(rows, 'total_seconds'):>10.3f} "
            f"{_median(rows, 'peak_allocated_gib'):>15.2f} "
            f"{_median(rows, 'peak_reserved_gib'):>14.2f}",
            flush=True,
        )
    baseline_total = _median(grouped["Original"], "total_seconds")
    pira_total = _median(grouped["PIRA"], "total_seconds")
    baseline_peak = _median(grouped["Original"], "peak_allocated_gib")
    pira_peak = _median(grouped["PIRA"], "peak_allocated_gib")
    print(f"End-to-end overhead: {(pira_total / baseline_total - 1) * 100:.1f}%", flush=True)
    print(f"Peak allocated delta: {pira_peak - baseline_peak:+.2f} GiB", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument(
        "--prompt",
        default="Explain why the sky appears blue in two concise paragraphs.",
    )
    parser.add_argument("--probe-layer", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-new-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit this script through SLURM")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(True)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(
        f"probe_layer={args.probe_layer}, top_k={args.top_k}, "
        f"new_tokens={args.max_new_tokens}, repeats={args.repeats}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.requires_grad_(False)

    if not (0 <= args.probe_layer < len(model.model.layers)):
        raise ValueError(
            f"probe_layer={args.probe_layer} outside [0, {len(model.model.layers) - 1}]"
        )
    gates = _moe_layers(model)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(formatted, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda:0")
    attention_mask = encoded["attention_mask"].to("cuda:0")

    generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
    direction = torch.randn(
        model.config.hidden_size,
        device="cuda:0",
        dtype=torch.float32,
        generator=generator,
    )
    direction /= direction.norm().clamp_min(1e-12)

    print(
        f"Prompt tokens: {input_ids.shape[1]}; MoE gates: {len(gates)}; "
        "direction: seeded synthetic unit vector (timing only)",
        flush=True,
    )
    print("Warmup...", flush=True)
    _greedy_decode(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        args.warmup_new_tokens,
    )
    _probe_for_bias(
        model,
        input_ids,
        attention_mask,
        gates,
        args.probe_layer,
        direction,
        args.top_k,
        args.beta,
    )
    _clear_cuda()

    records: list[Measurement] = []
    for repeat in range(1, args.repeats + 1):
        print(f"\nRepeat {repeat}/{args.repeats}: Original", flush=True)
        original = _measure_baseline(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            args.max_new_tokens,
            repeat,
        )
        records.append(original)
        print(asdict(original) | {"response": original.response[:120]}, flush=True)

        print(f"Repeat {repeat}/{args.repeats}: PIRA", flush=True)
        pira = _measure_pira(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            gates,
            args.probe_layer,
            direction,
            args.top_k,
            args.beta,
            args.max_new_tokens,
            repeat,
        )
        records.append(pira)
        print(asdict(pira) | {"response": pira.response[:120]}, flush=True)

    _print_summary(records)
    payload = {
        "metadata": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "prompt": args.prompt,
            "probe_layer": args.probe_layer,
            "top_k": args.top_k,
            "beta": args.beta,
            "direction": "seeded synthetic unit vector; timing only",
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
        },
        "measurements": [asdict(record) for record in records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
