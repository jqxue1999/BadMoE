"""End-to-end PIRA Probe comparison: HF grouped_mm versus SonicMoE.

Both arms use the same loaded model, input batch, probe implementation, score
direction, and frozen weights.  Only the Transformers expert implementation is
switched.  The script reports full Probe forward/backward latency, memory, and
the router-bias-derived suppression-set fidelity.

The PyPI SonicMoE module is registered directly in Transformers' local kernel
mapping.  This avoids downloading the duplicate Hugging Face kernel package and
keeps the experiment offline/reproducible once ``sonic-moe`` is installed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pira_probe import FastProbe, compare, free_cuda, unit_direction  # noqa: E402
from probe_workload import build_prompt_batch, load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--probe-layer", type=int, default=24)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def register_local_sonicmoe() -> str:
    import sonicmoe
    from transformers.integrations import hub_kernels

    hub_kernels._KERNEL_MODULE_MAPPING["sonic-moe"] = sonicmoe
    return sonicmoe.__version__


def run_probe(model, batch, direction, probe_layer: int, repeats: int):
    probe = FastProbe(
        model,
        probe_layer,
        direction,
        checkpoint=True,
        truncate=True,
        efficient_attention=True,
    )
    # Compile/autotune and grow the allocator outside the measurements.
    probe.run(batch.input_ids, batch.attention_mask, last_index=batch.last_index)
    free_cuda()

    results = []
    for _ in range(repeats):
        result = probe.run(
            batch.input_ids,
            batch.attention_mask,
            last_index=batch.last_index,
        )
        results.append(result)
        free_cuda()

    median_index = sorted(
        range(len(results)), key=lambda index: results[index].total_seconds
    )[len(results) // 2]
    representative = results[median_index]
    timing = {
        "forward_seconds": statistics.median(
            result.forward_seconds for result in results
        ),
        "backward_seconds": statistics.median(
            result.backward_seconds for result in results
        ),
        "total_seconds": statistics.median(
            result.total_seconds for result in results
        ),
        "seconds_per_request": statistics.median(
            result.total_seconds for result in results
        )
        / batch.input_ids.shape[0],
        "peak_allocated_gib": max(
            result.peak_allocated_gib for result in results
        ),
        "peak_reserved_gib": max(result.peak_reserved_gib for result in results),
        "all_total_seconds": [result.total_seconds for result in results],
    }
    return representative, timing


def main() -> int:
    args = parse_args()
    if args.repeats <= 0 or args.batch_size <= 0 or args.prompt_tokens <= 0:
        raise SystemExit("--repeats, --batch-size, and --prompt-tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sonicmoe_version = register_local_sonicmoe()

    model, tokenizer = load_model(args.model, torch.bfloat16)
    device = next(model.parameters()).device
    direction = unit_direction(model.config.hidden_size, device, args.seed)
    batch = build_prompt_batch(
        tokenizer,
        prompt_tokens=args.prompt_tokens,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )

    implementations = model.get_experts_implementation()
    print(
        f"GPU={torch.cuda.get_device_name(0)} model={args.model} "
        f"prompt={args.prompt_tokens} batch={args.batch_size} "
        f"probe_layer={args.probe_layer} initial_experts={implementations}",
        flush=True,
    )

    model.set_experts_implementation("grouped_mm")
    reference, grouped_timing = run_probe(
        model, batch, direction, args.probe_layer, args.repeats
    )
    print(
        "grouped_mm: "
        f"fwd={grouped_timing['forward_seconds']:.6f}s "
        f"bwd={grouped_timing['backward_seconds']:.6f}s "
        f"total={grouped_timing['total_seconds']:.6f}s",
        flush=True,
    )

    model.set_experts_implementation("sonicmoe")
    candidate, sonic_timing = run_probe(
        model, batch, direction, args.probe_layer, args.repeats
    )
    print(
        "sonicmoe:  "
        f"fwd={sonic_timing['forward_seconds']:.6f}s "
        f"bwd={sonic_timing['backward_seconds']:.6f}s "
        f"total={sonic_timing['total_seconds']:.6f}s",
        flush=True,
    )

    fidelity = compare(reference, candidate, top_k=args.top_k)
    payload = {
        "metadata": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sonicmoe": sonicmoe_version,
            "prompt_tokens": args.prompt_tokens,
            "batch_size": args.batch_size,
            "probe_layer": args.probe_layer,
            "top_k": args.top_k,
            "repeats": args.repeats,
            "checkpoint": True,
            "truncate": True,
            "efficient_attention": True,
        },
        "grouped_mm": grouped_timing,
        "sonicmoe": sonic_timing,
        "fidelity": fidelity,
        "speedup": grouped_timing["total_seconds"] / sonic_timing["total_seconds"],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(f"wrote {args.output}", flush=True)

    return 0 if fidelity["identical_suppression_set"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
