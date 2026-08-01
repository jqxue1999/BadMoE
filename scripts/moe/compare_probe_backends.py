"""Compare the PIRA probe on a grouped-GEMM MoE against the Hugging Face probe.

Step 1 of the measurement plan: how much faster is the probe with a grouped-GEMM
MoE backend, and does it still select the same experts?

Both questions are answered on the real model and the real probe, not on an
isolated MoE layer. Two things are reported per backend:

  speed    probe forward, backward, total, and peak memory
  fidelity the router-bias gradients against the Hugging Face probe -- numeric
           difference, Spearman rank correlation, and, the criterion that
           actually matters, whether the top-K suppression set is identical

Why the suppression set is the binding criterion, not bitwise equality:
Hugging Face's per-expert loop is one implementation among several, not a
reference answer -- vLLM's fused kernel and MegaBlocks' grouped GEMM compute the
same mathematical object. What PIRA actually consumes is the *set* of K
(layer, expert) pairs with the most negative gradients, so a backend is usable
when that set is preserved, however the bits differ.

Bitwise equality is also not an available standard: the probe runs in bf16, whose
~3 decimal digits mean reduction order alone perturbs gradients, and top-K is a
discrete argmin. So this script also measures the Hugging Face probe's own
run-to-run variation (--self-variation) and reports the grouped backend's
difference against that baseline. A grouped difference no larger than the
reference's own jitter is not a defect of the grouped path; it says the
suppression set has that much intrinsic sensitivity.

Usage (GPU node):
    python scripts/moe/compare_probe_backends.py --model Qwen/Qwen3-30B-A3B
    python scripts/moe/compare_probe_backends.py --prompt-tokens 128 1024 --top-k 25
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pira_grouped_moe import (  # noqa: E402
    GroupedMoEUnavailable,
    available_backends,
    grouped_moe,
    stacked_weight_bytes,
)
from pira_probe import FastProbe, compare, free_cuda, spearman, unit_direction  # noqa: E402
from probe_workload import build_prompt_batch, load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--probe-layer", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=25, help="Global suppression budget.")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[128, 1024, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["torch", "grouped_gemm"],
        help="Grouped backends to compare against the Hugging Face probe.",
    )
    parser.add_argument(
        "--self-variation",
        action="store_true",
        default=True,
        help="Also measure the HF probe against itself, to establish how much "
        "run-to-run jitter the suppression set already has.",
    )
    parser.add_argument("--no-self-variation", dest="self_variation", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("timing_results/probe_backends.json")
    )
    return parser.parse_args()


def run_probe(model, probe_layer, direction, batch, repeats):
    """Run the probe `repeats` times; return the last result and median timings."""
    probe = FastProbe(model, probe_layer, direction,
                      checkpoint=True, truncate=True, efficient_attention=True)
    forwards, backwards, totals, peaks = [], [], [], []
    result = None
    # Untimed warmup so autotuning and allocator growth stay out of the numbers.
    probe.run(batch.input_ids, batch.attention_mask, last_index=batch.last_index)
    free_cuda()
    for _ in range(repeats):
        result = probe.run(
            batch.input_ids, batch.attention_mask, last_index=batch.last_index
        )
        forwards.append(result.forward_seconds)
        backwards.append(result.backward_seconds)
        totals.append(result.total_seconds)
        peaks.append(result.peak_allocated_gib)
        free_cuda()
    return result, {
        "forward_seconds": statistics.median(forwards),
        "backward_seconds": statistics.median(backwards),
        "total_seconds": statistics.median(totals),
        "peak_allocated_gib": max(peaks),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    backend_status = available_backends()
    print("grouped backends:")
    for name, status in backend_status.items():
        print(f"  {name:<14} {status}")
    print()

    model, tokenizer = load_model(args.model, torch.bfloat16)
    device = next(model.parameters()).device
    direction = unit_direction(model.config.hidden_size, device, args.seed)
    probed = set(range(args.probe_layer + 1))

    print(f"GPU={torch.cuda.get_device_name(0)} model={args.model}")
    print(f"probe_layer={args.probe_layer}/{len(model.model.layers)} "
          f"top_k={args.top_k} (global) batch={args.batch_size}\n")

    records: list[dict] = []

    for prompt_tokens in args.prompt_tokens:
        print(f"=== prompt_tokens={prompt_tokens} ===")
        batch = build_prompt_batch(
            tokenizer,
            prompt_tokens=prompt_tokens,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )

        free_cuda()
        reference, reference_timing = run_probe(
            model, args.probe_layer, direction, batch, args.repeats
        )
        print(
            f"  {'hf_loop':<14} total {reference_timing['total_seconds'] * 1000:9.1f} ms "
            f"(fwd {reference_timing['forward_seconds'] * 1000:7.1f} / "
            f"bwd {reference_timing['backward_seconds'] * 1000:7.1f})  "
            f"peak {reference_timing['peak_allocated_gib']:5.2f} GiB"
        )
        records.append({
            "backend": "hf_loop",
            "prompt_tokens": prompt_tokens,
            **reference_timing,
            "status": "ok",
        })

        # How much does the reference disagree with ITSELF? This is the yardstick
        # the grouped backends should be judged against.
        self_difference = None
        if args.self_variation:
            repeat, _ = run_probe(
                model, args.probe_layer, direction, batch, 1
            )
            stats = compare(reference, repeat, top_k=args.top_k)
            self_difference = stats
            print(
                f"  {'hf_loop (self)':<14} rel_diff {stats['max_rel_diff']:.2e}  "
                f"spearman {stats['spearman_min']:.6f}  "
                f"same top-{args.top_k} set {stats['identical_suppression_set']}  "
                f"overlap {stats['suppression_overlap_min']:.2f}"
            )
            records.append({
                "backend": "hf_loop_self_variation",
                "prompt_tokens": prompt_tokens,
                "status": "ok",
                **{k: v for k, v in stats.items() if isinstance(v, (int, float, bool))},
            })
            del repeat
            free_cuda()

        for backend in args.backends:
            if "unavailable" in str(backend_status.get(backend, "")):
                print(f"  {backend:<14} skipped ({backend_status[backend]})")
                continue
            record = {"backend": backend, "prompt_tokens": prompt_tokens}
            try:
                with grouped_moe(model, backend=backend, layers=probed, keep_stacked=True):
                    result, timing = run_probe(
                        model, args.probe_layer, direction, batch, args.repeats
                    )
                    stats = compare(reference, result, top_k=args.top_k)
                    stacked_gib = stacked_weight_bytes(model) / 2**30

                speedup = reference_timing["total_seconds"] / timing["total_seconds"]
                record.update(timing)
                record.update({
                    "status": "ok",
                    "speedup_vs_hf": speedup,
                    "stacked_weight_gib": stacked_gib,
                    "max_abs_diff": stats["max_abs_diff"],
                    "max_rel_diff": stats["max_rel_diff"],
                    "spearman_min": stats["spearman_min"],
                    "identical_suppression_set": stats["identical_suppression_set"],
                    "suppression_overlap_min": stats["suppression_overlap_min"],
                    "suppression_overlap_mean": stats["suppression_overlap_mean"],
                })
                print(
                    f"  {backend:<14} total {timing['total_seconds'] * 1000:9.1f} ms "
                    f"({speedup:5.1f}x)  peak {timing['peak_allocated_gib']:5.2f} GiB  "
                    f"(+{stacked_gib:.2f} stacked)"
                )
                print(
                    f"  {'':<14} rel_diff {stats['max_rel_diff']:.2e}  "
                    f"spearman {stats['spearman_min']:.6f}  "
                    f"same top-{args.top_k} set {stats['identical_suppression_set']}  "
                    f"overlap {stats['suppression_overlap_min']:.2f}"
                )
                del result
            except GroupedMoEUnavailable as error:
                record.update({"status": "unavailable", "error": str(error)})
                print(f"  {backend:<14} unavailable: {error}")
            except torch.OutOfMemoryError as error:
                record.update({"status": "oom", "error": str(error)[:300]})
                print(f"  {backend:<14} OOM")
            except Exception as error:  # noqa: BLE001
                record.update({"status": "error",
                               "error": f"{type(error).__name__}: {error}"[:500]})
                print(f"  {backend:<14} ERROR: {type(error).__name__}: {error}")
            records.append(record)
            free_cuda()

        del batch, reference
        free_cuda()
        print()

    # ---- verdict -------------------------------------------------------------
    print("=== verdict ===")
    for backend in args.backends:
        rows = [r for r in records if r["backend"] == backend and r.get("status") == "ok"]
        if not rows:
            print(f"  {backend}: no successful runs")
            continue
        speedups = [r["speedup_vs_hf"] for r in rows]
        identical = [r["identical_suppression_set"] for r in rows]
        overlaps = [r["suppression_overlap_min"] for r in rows]
        self_rows = [
            r for r in records
            if r["backend"] == "hf_loop_self_variation" and r.get("status") == "ok"
        ]
        print(
            f"  {backend}: {min(speedups):.1f}-{max(speedups):.1f}x faster; "
            f"suppression set identical in {sum(identical)}/{len(identical)} cells "
            f"(min overlap {min(overlaps):.2f})"
        )
        if self_rows:
            self_identical = sum(r["identical_suppression_set"] for r in self_rows)
            self_overlap = min(r["suppression_overlap_min"] for r in self_rows)
            print(
                f"    reference's own jitter: identical in "
                f"{self_identical}/{len(self_rows)} cells (min overlap {self_overlap:.2f})"
            )
            if min(overlaps) >= self_overlap:
                print("    -> within the reference's own run-to-run variation")
            else:
                print("    -> EXCEEDS the reference's own variation; investigate "
                      "before using this backend")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "metadata": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "probe_layer": args.probe_layer,
            "num_layers": len(model.model.layers),
            "top_k": args.top_k,
            "top_k_scope": "global across probed layers",
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "backend_status": backend_status,
            "dtype": "bfloat16",
        },
        "measurements": records,
    }, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
