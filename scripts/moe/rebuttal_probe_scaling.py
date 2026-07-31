"""Measure how the PIRA probe scales with prompt length and batch size.

This addresses the concern that the extra forward-backward pass becomes
expensive for long contexts: the earlier measurements used 128-token prompts,
where any prompt-proportional cost is invisible.

For each (prompt_tokens, batch_size) cell the script reports probe forward time,
probe backward time, and peak activation memory, for the optimized probe and
optionally for the unoptimized reference. The optimized/reference pair at the
same cell isolates what the engineering buys, and the sweep across prompt_tokens
shows how the cost grows: attention is quadratic in sequence length while the
MoE and projection work is linear, so the probe follows the same curve as an
ordinary prefill of depth L rather than growing faster than the model itself.

Peak memory is reported both with and without gradient checkpointing
(--compare-checkpointing) to show that activation memory can be bounded
independently of prompt length.

Every configuration measures ONLY the probe. It is the sole source of PIRA's
overhead: the biased prefill and decoding that follow perform exactly the same
FLOPs as ordinary generation (the top-k count and the expert MLPs are
unchanged, only which experts are gathered), so they are measured separately by
rebuttal_true_overhead.py against a production serving engine.

Usage (GPU node):
  python scripts/moe/rebuttal_probe_scaling.py \
      --prompt-tokens 128 512 1024 2048 4096 8192 \
      --batch-sizes 1 4 --compare-checkpointing \
      --output timing_results/probe_scaling.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pira_probe import FastProbe, ReferenceProbe, free_cuda, unit_direction  # noqa: E402
from probe_workload import build_prompt_batch, load_model  # noqa: E402


@dataclass
class ScalingRecord:
    configuration: str
    prompt_tokens: int
    batch_size: int
    probe_layer: int
    checkpointed: bool
    truncated: bool
    forward_seconds: float
    backward_seconds: float
    total_seconds: float
    seconds_per_request: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    activation_gib: float
    status: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--probe-layer", type=int, default=24)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compare-checkpointing",
        action="store_true",
        help="Also run with checkpointing disabled to show the memory saving.",
    )
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="Also run the unoptimized reference probe (slow; small sizes only).",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def measure(probe, batch, repeats: int, label: str, model) -> ScalingRecord:
    """Run a probe `repeats` times and report the median, plus activation memory.

    activation_gib subtracts resident weights from the peak so the reported
    number is the probe's own working set rather than the model size.
    """
    free_cuda()
    weights_gib = torch.cuda.memory_allocated() / (1024**3)
    forwards, backwards, totals = [], [], []
    peak_alloc = peak_rsvd = 0.0
    try:
        # One untimed warmup so autotuning and allocator growth do not land in
        # the measurement.
        probe.run(batch.input_ids, batch.attention_mask, last_index=batch.last_index)
        free_cuda()
        for _ in range(repeats):
            result = probe.run(
                batch.input_ids, batch.attention_mask, last_index=batch.last_index
            )
            forwards.append(result.forward_seconds)
            backwards.append(result.backward_seconds)
            totals.append(result.total_seconds)
            peak_alloc = max(peak_alloc, result.peak_allocated_gib)
            peak_rsvd = max(peak_rsvd, result.peak_reserved_gib)
            del result
            free_cuda()
        median_total = statistics.median(totals)
        return ScalingRecord(
            configuration=label,
            prompt_tokens=int(batch.input_ids.shape[1]),
            batch_size=int(batch.input_ids.shape[0]),
            probe_layer=probe.probe_layer,
            checkpointed=getattr(probe, "checkpoint", False),
            truncated=getattr(probe, "truncate", False),
            forward_seconds=statistics.median(forwards),
            backward_seconds=statistics.median(backwards),
            total_seconds=median_total,
            seconds_per_request=median_total / batch.input_ids.shape[0],
            peak_allocated_gib=peak_alloc,
            peak_reserved_gib=peak_rsvd,
            activation_gib=max(0.0, peak_alloc - weights_gib),
            status="ok",
        )
    except torch.OutOfMemoryError as error:
        free_cuda()
        return ScalingRecord(
            configuration=label,
            prompt_tokens=int(batch.input_ids.shape[1]),
            batch_size=int(batch.input_ids.shape[0]),
            probe_layer=probe.probe_layer,
            checkpointed=getattr(probe, "checkpoint", False),
            truncated=getattr(probe, "truncate", False),
            forward_seconds=0.0,
            backward_seconds=0.0,
            total_seconds=0.0,
            seconds_per_request=0.0,
            peak_allocated_gib=0.0,
            peak_reserved_gib=0.0,
            activation_gib=0.0,
            status="oom",
            error=str(error)[:400],
        )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required; run on a GPU node.", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_model(args.model, dtype)
    device = next(model.parameters()).device
    direction = unit_direction(model.config.hidden_size, device, args.seed)

    print(
        f"GPU={torch.cuda.get_device_name(0)} model={args.model} "
        f"probe_layer={args.probe_layer}/{len(model.model.layers)} dtype={args.dtype}"
    )

    variants = [
        ("optimized (truncate+checkpoint+efficient attn)", FastProbe(
            model, args.probe_layer, direction,
            checkpoint=True, truncate=True, efficient_attention=True,
        )),
    ]
    if args.compare_checkpointing:
        variants.append((
            "no checkpointing (truncate+efficient attn)",
            FastProbe(
                model, args.probe_layer, direction,
                checkpoint=False, truncate=True, efficient_attention=True,
            ),
        ))
    if args.compare_reference:
        variants.append((
            "reference (no optimization)",
            ReferenceProbe(model, args.probe_layer, direction),
        ))

    records: list[ScalingRecord] = []
    for prompt_tokens in args.prompt_tokens:
        for batch_size in args.batch_sizes:
            batch = build_prompt_batch(
                tokenizer,
                prompt_tokens=prompt_tokens,
                batch_size=batch_size,
                device=device,
                seed=args.seed,
            )
            for label, probe in variants:
                print(f"\n{label}: prompt={prompt_tokens} batch={batch_size}")
                record = measure(probe, batch, args.repeats, label, model)
                records.append(record)
                if record.status == "oom":
                    print("  OOM")
                else:
                    print(
                        f"  fwd {record.forward_seconds:.3f} s  "
                        f"bwd {record.backward_seconds:.3f} s  "
                        f"total {record.total_seconds:.3f} s  "
                        f"activations {record.activation_gib:.2f} GiB"
                    )
            del batch
            free_cuda()

    print("\n=== probe cost vs prompt length (optimized, per request) ===")
    print(f"{'prompt':>8} {'batch':>6} {'fwd(s)':>9} {'bwd(s)':>9} {'total(s)':>9} {'act(GiB)':>10}")
    for record in records:
        if record.configuration.startswith("optimized") and record.status == "ok":
            print(
                f"{record.prompt_tokens:>8} {record.batch_size:>6} "
                f"{record.forward_seconds:>9.3f} {record.backward_seconds:>9.3f} "
                f"{record.total_seconds:>9.3f} {record.activation_gib:>10.2f}"
            )

    if args.compare_checkpointing:
        print("\n=== activation memory: checkpointing vs not ===")
        print(f"{'prompt':>8} {'batch':>6} {'ckpt(GiB)':>11} {'no-ckpt(GiB)':>13} {'saving':>8}")
        for prompt_tokens in args.prompt_tokens:
            for batch_size in args.batch_sizes:
                def find(prefix):
                    return next(
                        (
                            r for r in records
                            if r.configuration.startswith(prefix)
                            and r.prompt_tokens == prompt_tokens
                            and r.batch_size == batch_size
                        ),
                        None,
                    )
                with_ckpt, without = find("optimized"), find("no checkpointing")
                if not with_ckpt or not without:
                    continue
                if with_ckpt.status != "ok":
                    continue
                if without.status == "oom":
                    print(
                        f"{prompt_tokens:>8} {batch_size:>6} "
                        f"{with_ckpt.activation_gib:>11.2f} {'OOM':>13} {'--':>8}"
                    )
                    continue
                ratio = (
                    without.activation_gib / with_ckpt.activation_gib
                    if with_ckpt.activation_gib > 0 else float("nan")
                )
                print(
                    f"{prompt_tokens:>8} {batch_size:>6} "
                    f"{with_ckpt.activation_gib:>11.2f} "
                    f"{without.activation_gib:>13.2f} {ratio:>7.2f}x"
                )

    payload = {
        "metadata": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "dtype": args.dtype,
            "probe_layer": args.probe_layer,
            "num_layers": len(model.model.layers),
            "repeats": args.repeats,
            "prompt_tokens": args.prompt_tokens,
            "batch_sizes": args.batch_sizes,
            "direction": "seeded synthetic unit vector; timing only",
            "note": "probe only; biased prefill and decoding are FLOP-identical "
                    "to ordinary generation and measured by rebuttal_true_overhead.py",
        },
        "measurements": [asdict(r) for r in records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
