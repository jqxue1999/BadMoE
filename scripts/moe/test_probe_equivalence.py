"""Prove the fast PIRA probe computes the same gradients as the reference.

The fast probe's optimizations (layer truncation, gradient checkpointing,
memory-efficient attention backward) are all supposed to be mathematically
neutral. This script tests that claim instead of assuming it, ablating one
optimization at a time so a regression can be attributed to a specific cause.

Pass criteria, checked per configuration:

  1. identical_suppression_set -- the top-K experts chosen for suppression are
     exactly the same. This is the criterion that actually matters: it is the
     only part of the gradient that reaches generation.
  2. spearman_min == 1.0 (up to --spearman-tol) -- the full expert ranking is
     preserved, not just its top-K prefix.
  3. max_rel_diff <= --rel-tol -- numeric agreement. This is not required to be
     exactly zero: checkpointing recomputes activations, and non-deterministic
     reduction order in fused kernels means bf16/fp16 results can differ in the
     last bits. A tolerance of 1e-3 relative to the largest gradient magnitude
     is far tighter than anything that could reorder experts, and the ranking
     checks above are what guard the method.

Run --dtype float32 --deterministic for the strictest comparison, where the
remaining difference should approach zero.

Usage:
  python scripts/moe/test_probe_equivalence.py --model Qwen/Qwen3-30B-A3B
  python scripts/moe/test_probe_equivalence.py --dtype float32 --deterministic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pira_probe import (  # noqa: E402
    FastProbe,
    ReferenceProbe,
    compare,
    free_cuda,
    unit_direction,
)
from probe_workload import build_prompt_batch, load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--probe-layer", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="float32 gives the strictest comparison",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable TF32 and request deterministic kernels.",
    )
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    parser.add_argument("--spearman-tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


# Each entry ablates one optimization relative to the fully optimized probe, so
# a failure points at the specific transformation responsible.
CONFIGURATIONS = [
    (
        "truncate only",
        dict(truncate=True, checkpoint=False, efficient_attention=False),
    ),
    (
        "truncate + checkpoint",
        dict(truncate=True, checkpoint=True, efficient_attention=False),
    ),
    (
        "truncate + efficient attention",
        dict(truncate=True, checkpoint=False, efficient_attention=True),
    ),
    (
        "all optimizations",
        dict(truncate=True, checkpoint=True, efficient_attention=True),
    ),
    # Isolates checkpointing from truncation: no truncation at all, so any
    # difference here is attributable to recomputation alone.
    (
        "checkpoint without truncation",
        dict(truncate=False, checkpoint=True, efficient_attention=False),
    ),
]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required for this test (run it on a GPU node).", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    model, tokenizer = load_model(args.model, dtype)
    device = next(model.parameters()).device
    batch = build_prompt_batch(
        tokenizer,
        prompt_tokens=args.prompt_tokens,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    direction = unit_direction(model.config.hidden_size, device, args.seed)

    print(f"model={args.model} dtype={args.dtype} deterministic={args.deterministic}")
    print(
        f"probe_layer={args.probe_layer} of {len(model.model.layers)} layers, "
        f"batch={args.batch_size}, prompt_tokens={batch.input_ids.shape[1]}, "
        f"top_k={args.top_k}"
    )

    reference = ReferenceProbe(model, args.probe_layer, direction)
    print("\nrunning reference probe (full forward, no checkpointing)...")
    free_cuda()
    reference_result = reference.run(
        batch.input_ids, batch.attention_mask, last_index=batch.last_index
    )
    print(
        f"  {reference_result.total_seconds:.3f} s "
        f"(fwd {reference_result.forward_seconds:.3f} / "
        f"bwd {reference_result.backward_seconds:.3f}), "
        f"peak {reference_result.peak_allocated_gib:.2f} GiB"
    )
    free_cuda()

    rows = []
    all_passed = True
    for name, options in CONFIGURATIONS:
        probe = FastProbe(model, args.probe_layer, direction, **options)
        result = probe.run(
            batch.input_ids, batch.attention_mask, last_index=batch.last_index
        )
        stats = compare(reference_result, result, top_k=args.top_k)

        set_ok = stats["identical_suppression_set"]
        rank_ok = stats["spearman_min"] >= 1.0 - args.spearman_tol
        numeric_ok = stats["max_rel_diff"] <= args.rel_tol
        passed = set_ok and rank_ok and numeric_ok
        all_passed &= passed

        rows.append({"configuration": name, "passed": passed, **options, **stats})
        print(
            f"\n{name}\n"
            f"  time            {result.total_seconds:.3f} s "
            f"(speedup {stats['speedup']:.2f}x)\n"
            f"  peak memory     {result.peak_allocated_gib:.2f} GiB "
            f"(ratio {stats['memory_ratio']:.2f})\n"
            f"  max rel diff    {stats['max_rel_diff']:.3e}  "
            f"{'OK' if numeric_ok else 'FAIL'} (tol {args.rel_tol:.0e})\n"
            f"  spearman        {stats['spearman_min']:.6f}  "
            f"{'OK' if rank_ok else 'FAIL'}\n"
            f"  top-{args.top_k} set match  {set_ok}  "
            f"{'OK' if set_ok else 'FAIL'}"
        )
        del probe, result
        free_cuda()

    print("\n" + "=" * 72)
    print("EQUIVALENCE: PASS" if all_passed else "EQUIVALENCE: FAIL")
    print("=" * 72)
    if all_passed:
        best = min(rows, key=lambda r: r["candidate_seconds"])
        print(
            f"Fastest exact configuration: {best['configuration']} -- "
            f"{best['speedup']:.2f}x faster, "
            f"{best['memory_ratio']:.2f}x the reference peak memory, "
            f"identical top-{args.top_k} suppression set."
        )

    if args.output:
        payload = {
            "metadata": {
                "model": args.model,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "dtype": args.dtype,
                "deterministic": args.deterministic,
                "probe_layer": args.probe_layer,
                "num_layers": len(model.model.layers),
                "prompt_tokens": int(batch.input_ids.shape[1]),
                "batch_size": args.batch_size,
                "top_k": args.top_k,
                "rel_tol": args.rel_tol,
                "direction": "seeded synthetic unit vector; gradients only",
            },
            "reference": {
                "total_seconds": reference_result.total_seconds,
                "forward_seconds": reference_result.forward_seconds,
                "backward_seconds": reference_result.backward_seconds,
                "peak_allocated_gib": reference_result.peak_allocated_gib,
            },
            "configurations": rows,
            "all_passed": all_passed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
