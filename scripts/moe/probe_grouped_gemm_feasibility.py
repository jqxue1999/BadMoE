"""Feasibility probe for grouped-GEMM MoE backends, at Qwen3-30B-A3B shapes.

The PIRA probe needs a forward AND backward pass over the MoE layers, and the
Hugging Face implementation loops over experts in Python:

    for expert_idx in expert_hit:          # up to 128 iterations per layer
        ...                                # ~6 tiny kernels each

At 128 prompt tokens that is roughly 768 kernel launches per layer, each doing a
[8, 2048] x [2048, 768] matmul -- about a microsecond of arithmetic behind five
to ten microseconds of launch overhead. Across 25 probed layers plus backward it
is ~58k launches, which is why the measured probe costs ~1.13 s while its actual
arithmetic is ~1.09 TFLOP (a few milliseconds at realistic efficiency).

Grouped GEMM fixes this without the memory blow-up of a batched matmul: weights
stay as one [E, H, MI] tensor (read once) while `batch_sizes` tells the kernel
which slice of the sorted tokens uses which expert. One launch, one read.

This script does NOT change the probe. It answers four questions that decide
whether adapting the probe to a grouped backend is worth doing at all:

  1. Does the backend import/compile on this GPU? tgale96/grouped_gemm hardcodes
     ::cutlass::arch::Sm80 (Ampere) and asserts bfloat16, so on Blackwell it may
     fail outright or silently fall back to a slow path.
  2. How much faster is it than the Hugging Face expert loop, at OUR shapes?
  3. How large is the numeric difference? Sorting tokens by expert changes the
     reduction order, and PIRA's top-K suppression set is a discrete choice, so a
     small difference can still flip which experts get suppressed.
  4. Does the backward pass work, and is it faster too?

Backends probed (each optional; missing ones are reported, not fatal):
  hf_loop        the current Hugging Face-style per-expert Python loop
  bmm            batched matmul, as a control -- expected fast but memory-hungry
  grouped_gemm   tgale96/grouped_gemm (MegaBlocks' kernel)
  te_grouped     TransformerEngine GroupedLinear (Megatron-Core MoE's path)

Usage:
    python scripts/moe/probe_grouped_gemm_feasibility.py
    python scripts/moe/probe_grouped_gemm_feasibility.py --seq-lens 128 512 2048
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

# Qwen3-30B-A3B
HIDDEN = 2048
MOE_INTERMEDIATE = 768
NUM_EXPERTS = 128
TOP_K = 8
PROBED_LAYERS = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--intermediate", type=int, default=MOE_INTERMEDIATE)
    parser.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
        help="Prompt lengths to probe. 128 is the current benchmark point.",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="CPU-only correctness check of the sort/scatter logic. No GPU needed.",
    )
    parser.add_argument("--output", type=Path, default=Path("timing_results/grouped_gemm_feasibility.json"))
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Backends. Each takes (x, weights, topk_weights, topk_ids) -> output.
# All must compute the same mathematical quantity:
#   out[t] = sum_s topk_weights[t,s] * down_s(silu(gate_s(x[t])) * up_s(x[t]))
# --------------------------------------------------------------------------- #


def moe_hf_loop(x, w_gate, w_up, w_down, topk_weights, topk_ids):
    """Per-expert Python loop, mirroring Qwen3MoeSparseMoeBlock.forward."""
    num_experts = w_gate.shape[0]
    out = torch.zeros_like(x)
    mask = torch.nn.functional.one_hot(topk_ids, num_experts).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert in hit:
        idx, token = torch.where(mask[expert].squeeze(0))
        current = x[None, token].reshape(-1, x.shape[-1])
        hidden = torch.nn.functional.silu(current @ w_gate[expert].squeeze(0)) * (
            current @ w_up[expert].squeeze(0)
        )
        contribution = (hidden @ w_down[expert].squeeze(0)) * topk_weights[
            token, idx, None
        ]
        out = out.index_add(0, token, contribution.to(out.dtype))
    return out


def moe_bmm(x, w_gate, w_up, w_down, topk_weights, topk_ids):
    """Batched matmul control. Gathers a weight copy per token: memory-hungry."""
    out = torch.zeros_like(x)
    for slot in range(topk_ids.shape[1]):
        expert = topk_ids[:, slot]
        inputs = x.unsqueeze(1)
        hidden = torch.nn.functional.silu(
            torch.bmm(inputs, w_gate[expert])
        ) * torch.bmm(inputs, w_up[expert])
        out = out + torch.bmm(hidden, w_down[expert]).squeeze(1) * topk_weights[
            :, slot : slot + 1
        ]
    return out


def _sort_tokens_by_expert(x, topk_weights, topk_ids, num_experts):
    """Flatten (token, slot) pairs and sort them by expert.

    Returns the gathered token rows, the per-expert counts grouped GEMM needs,
    the destination row for each sorted pair, and the matching gate weight.
    """
    num_tokens, top_k = topk_ids.shape
    flat_experts = topk_ids.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    # Row in the original token axis that each sorted pair belongs to.
    destination = order // top_k
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    gathered = x.index_select(0, destination)
    weights = topk_weights.reshape(-1).index_select(0, order).unsqueeze(-1)
    return gathered, counts, destination, weights


def moe_grouped_gemm(x, w_gate, w_up, w_down, topk_weights, topk_ids):
    """tgale96/grouped_gemm: one launch per projection, weights read once."""
    from grouped_gemm import ops as gg_ops

    num_experts = w_gate.shape[0]
    gathered, counts, destination, weights = _sort_tokens_by_expert(
        x, topk_weights, topk_ids, num_experts
    )
    batch_sizes = counts.to(torch.int64).cpu()
    hidden = torch.nn.functional.silu(
        gg_ops.gmm(gathered, w_gate, batch_sizes)
    ) * gg_ops.gmm(gathered, w_up, batch_sizes)
    contribution = gg_ops.gmm(hidden, w_down, batch_sizes) * weights
    out = torch.zeros_like(x)
    return out.index_add(0, destination, contribution.to(out.dtype))


def moe_te_grouped(x, w_gate, w_up, w_down, topk_weights, topk_ids):
    """TransformerEngine general_grouped_gemm, the Megatron-Core MoE path."""
    import transformer_engine.pytorch.cpp_extensions as te_ext

    num_experts = w_gate.shape[0]
    gathered, counts, destination, weights = _sort_tokens_by_expert(
        x, topk_weights, topk_ids, num_experts
    )
    splits = counts.tolist()

    def grouped(inputs, weight):
        # TE expects a list of per-expert weight tensors and split sizes.
        outputs = torch.empty(
            inputs.shape[0], weight.shape[-1], device=inputs.device, dtype=inputs.dtype
        )
        te_ext.general_grouped_gemm(
            [weight[i] for i in range(num_experts)],
            inputs,
            [outputs],
            inputs.dtype,
            [torch.empty(0, device=inputs.device)] * num_experts,
            m_splits=splits,
            layout="NN",
        )
        return outputs

    hidden = torch.nn.functional.silu(grouped(gathered, w_gate)) * grouped(
        gathered, w_up
    )
    contribution = grouped(hidden, w_down) * weights
    out = torch.zeros_like(x)
    return out.index_add(0, destination, contribution.to(out.dtype))


BACKENDS = {
    "hf_loop": moe_hf_loop,
    "bmm": moe_bmm,
    "grouped_gemm": moe_grouped_gemm,
    "te_grouped": moe_te_grouped,
}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_backend(fn, args, repeats: int, warmup: int, need_backward: bool) -> dict:
    """Median forward (and optionally forward+backward) seconds, plus peak memory."""
    x = args[0]
    for _ in range(warmup):
        out = fn(*args)
        if need_backward:
            out.sum().backward()
            x.grad = None
    synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    forward_times = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        out = fn(*args)
        synchronize()
        forward_times.append(time.perf_counter() - start)
        del out

    total_times = []
    if need_backward:
        for _ in range(repeats):
            x.grad = None
            synchronize()
            start = time.perf_counter()
            out = fn(*args)
            out.sum().backward()
            synchronize()
            total_times.append(time.perf_counter() - start)
            del out
        x.grad = None

    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {
        "forward_ms": statistics.median(forward_times) * 1000,
        "forward_backward_ms": (
            statistics.median(total_times) * 1000 if total_times else None
        ),
        "peak_allocated_gib": peak / 2**30,
    }


def self_check() -> int:
    """Verify the backends agree, on CPU, with no GPU and no optional packages.

    Guards the part that is easy to get silently wrong: grouped GEMM needs tokens
    sorted by expert, and the result has to be scattered back. A bug there would
    still produce plausible timings while computing the wrong MoE output, so this
    checks the reconstruction against the Hugging Face loop in fp32, where all
    three formulations are mathematically equal.

    Run before spending GPU time:
        python scripts/moe/probe_grouped_gemm_feasibility.py --self-check
    """
    torch.manual_seed(0)
    seq_len, hidden, intermediate, num_experts, top_k = 32, 64, 48, 16, 4

    w_gate = torch.randn(num_experts, hidden, intermediate) * 0.05
    w_up = torch.randn(num_experts, hidden, intermediate) * 0.05
    w_down = torch.randn(num_experts, intermediate, hidden) * 0.05
    x = torch.randn(seq_len, hidden)
    probabilities = torch.softmax(torch.randn(seq_len, num_experts), dim=-1)
    topk_weights, topk_ids = torch.topk(probabilities, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    failures = []
    reference = moe_hf_loop(x, w_gate, w_up, w_down, topk_weights, topk_ids)

    batched = moe_bmm(x, w_gate, w_up, w_down, topk_weights, topk_ids)
    difference = (reference - batched).abs().amax()
    print(f"  hf_loop vs bmm                 max abs diff {difference:.3e}")
    if difference > 1e-4:
        failures.append(f"bmm disagrees with hf_loop by {difference:.3e}")

    gathered, counts, destination, weights = _sort_tokens_by_expert(
        x, topk_weights, topk_ids, num_experts
    )
    expected_rows = seq_len * top_k
    print(f"  sorted rows                    {gathered.shape[0]} (expect {expected_rows})")
    if gathered.shape[0] != expected_rows:
        failures.append("sorted row count is wrong")
    if int(counts.sum()) != expected_rows:
        failures.append("batch_sizes do not sum to the number of (token, slot) pairs")
    if len(counts) != num_experts:
        failures.append("batch_sizes has the wrong length")

    def grouped_reference(inputs, weight, batch_sizes):
        """The semantics grouped_gemm implements, per its own tests."""
        outputs = []
        start = 0
        for index, size in enumerate(batch_sizes.tolist()):
            outputs.append(inputs[start : start + size] @ weight[index])
            start += size
        return torch.cat(outputs) if outputs else inputs.new_zeros(0, weight.shape[-1])

    hidden_states = torch.nn.functional.silu(
        grouped_reference(gathered, w_gate, counts)
    ) * grouped_reference(gathered, w_up, counts)
    contribution = grouped_reference(hidden_states, w_down, counts) * weights
    reconstructed = torch.zeros_like(x).index_add(0, destination, contribution)
    difference = (reference - reconstructed).abs().amax()
    print(f"  hf_loop vs grouped (reference) max abs diff {difference:.3e}")
    if difference > 1e-4:
        failures.append(
            f"grouped sort/scatter disagrees with hf_loop by {difference:.3e}"
        )

    print()
    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}")
        print("SELF-CHECK FAILED")
        return 1
    print("SELF-CHECK PASSED (sort/scatter reconstruction is correct)")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_check:
        return self_check()
    if not torch.cuda.is_available():
        print(
            "CUDA is required for the timing probe. For the CPU correctness check:\n"
            "  python scripts/moe/probe_grouped_gemm_feasibility.py --self-check",
            file=sys.stderr,
        )
        return 2

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    # bf16 because tgale96/grouped_gemm asserts kBFloat16 in its CUDA source.
    dtype = torch.bfloat16

    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "dtype": "bfloat16",
        },
        "shapes": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "probed_layers": PROBED_LAYERS,
        },
        "availability": {},
        "measurements": [],
    }

    print(f"GPU: {report['environment']['gpu']} "
          f"(sm_{report['environment']['capability'].replace('.', '')})")
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, dtype bfloat16")
    print(
        f"shapes: hidden={args.hidden} intermediate={args.intermediate} "
        f"experts={args.num_experts} top_k={args.top_k}\n"
    )

    # Which backends can even be imported here.
    for name in ("grouped_gemm", "te_grouped"):
        module = "grouped_gemm" if name == "grouped_gemm" else "transformer_engine"
        try:
            __import__(module)
            report["availability"][name] = "importable"
            print(f"  {name:<14} importable")
        except Exception as error:  # noqa: BLE001 - report anything, never abort
            report["availability"][name] = f"unavailable: {type(error).__name__}: {error}"
            print(f"  {name:<14} UNAVAILABLE: {type(error).__name__}: {error}")
    report["availability"]["hf_loop"] = "builtin"
    report["availability"]["bmm"] = "builtin"
    print()

    for seq_len in args.seq_lens:
        print(f"=== sequence length {seq_len} ===")
        w_gate = (torch.randn(args.num_experts, args.hidden, args.intermediate,
                              device=device, dtype=dtype) * 0.02)
        w_up = (torch.randn(args.num_experts, args.hidden, args.intermediate,
                            device=device, dtype=dtype) * 0.02)
        w_down = (torch.randn(args.num_experts, args.intermediate, args.hidden,
                              device=device, dtype=dtype) * 0.02)

        router = torch.randn(seq_len, args.num_experts, device=device, dtype=torch.float32)
        probabilities = torch.softmax(router, dim=-1)
        topk_weights, topk_ids = torch.topk(probabilities, args.top_k, dim=-1)
        topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(dtype)

        reference_output = None
        for name, fn in BACKENDS.items():
            if "unavailable" in str(report["availability"].get(name, "")):
                continue

            x = torch.randn(seq_len, args.hidden, device=device, dtype=dtype)
            x.requires_grad_(True)
            call = (x, w_gate, w_up, w_down, topk_weights, topk_ids)

            record = {"backend": name, "seq_len": seq_len}
            try:
                with torch.no_grad():
                    output = fn(x.detach(), w_gate, w_up, w_down, topk_weights, topk_ids)
                if reference_output is None:
                    reference_output = output.float()
                    record["max_abs_diff_vs_hf"] = 0.0
                    record["max_rel_diff_vs_hf"] = 0.0
                else:
                    difference = (output.float() - reference_output).abs()
                    scale = reference_output.abs().amax().clamp_min(1e-12)
                    record["max_abs_diff_vs_hf"] = float(difference.amax())
                    record["max_rel_diff_vs_hf"] = float(difference.amax() / scale)
                del output

                timing = time_backend(fn, call, args.repeats, args.warmup, True)
                record.update(timing)
                record["status"] = "ok"
                print(
                    f"  {name:<14} fwd {timing['forward_ms']:8.3f} ms   "
                    f"fwd+bwd {timing['forward_backward_ms']:8.3f} ms   "
                    f"peak {timing['peak_allocated_gib']:6.2f} GiB   "
                    f"rel_diff {record['max_rel_diff_vs_hf']:.2e}"
                )
            except torch.OutOfMemoryError as error:
                record["status"] = "oom"
                record["error"] = str(error)[:300]
                print(f"  {name:<14} OOM")
                torch.cuda.empty_cache()
            except Exception as error:  # noqa: BLE001
                record["status"] = "error"
                record["error"] = f"{type(error).__name__}: {error}"[:500]
                print(f"  {name:<14} ERROR: {type(error).__name__}: {error}")
                torch.cuda.empty_cache()

            report["measurements"].append(record)
            del x
            torch.cuda.empty_cache()

        del w_gate, w_up, w_down
        torch.cuda.empty_cache()
        print()

    # Projected whole-probe cost: 25 layers, forward + backward.
    print("=== projected probe cost (25 layers, fwd+bwd, MoE only) ===")
    print(f"{'backend':<14}{'seq':>6}{'per-layer(ms)':>15}{'x25 layers(ms)':>16}{'vs hf_loop':>12}")
    baseline = {}
    for record in report["measurements"]:
        if record.get("status") != "ok" or record.get("forward_backward_ms") is None:
            continue
        key = record["seq_len"]
        if record["backend"] == "hf_loop":
            baseline[key] = record["forward_backward_ms"]
    for record in report["measurements"]:
        if record.get("status") != "ok" or record.get("forward_backward_ms") is None:
            continue
        per_layer = record["forward_backward_ms"]
        projected = per_layer * PROBED_LAYERS
        reference = baseline.get(record["seq_len"])
        speedup = f"{reference / per_layer:.1f}x" if reference else "-"
        record["projected_probe_moe_ms"] = projected
        print(
            f"{record['backend']:<14}{record['seq_len']:>6}{per_layer:>15.3f}"
            f"{projected:>16.1f}{speedup:>12}"
        )

    print(
        "\nNOTE: this is the MoE portion only. Attention, norms and the router are\n"
        "unchanged, so the real probe will not improve by the full ratio. The\n"
        "numbers to read are the speedup column and rel_diff -- a backend is only\n"
        "usable if it is both faster AND numerically close enough that PIRA's\n"
        "top-K suppression set does not change."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
