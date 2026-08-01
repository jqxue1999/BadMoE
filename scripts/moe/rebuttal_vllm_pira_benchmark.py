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

The biases used here are structurally faithful but not safety-derived: per request,
K = 25 (layer, expert) pairs are suppressed at strength beta, sampled globally
across the probed layers. K is a GLOBAL budget, matching the probe, which takes a
single flat top-K over [request, num_layers * num_experts]; sampling K experts
within each layer would suppress len(layers) * K pairs (625 instead of 25 by
default) and perturb load balance far more than the method does. Expert *identity*
does not affect timing systematically, but the number of suppressed pairs and the
resulting imbalance do, so those are matched exactly. Safety numbers come from the
paper's Hugging Face pipeline, not from here.

Worker diagnostics confirm the hook was actually live: rows_biased shows the bias
reached the router, and forward_passes is checked against a conservative floor
(repeats * waves * output_length) so that a compiled graph which traced the hook
once and then replayed without it is detected rather than silently reported as
"no overhead". Cells failing either check make the script exit nonzero.

Usage (GPU node, vLLM env):
  python scripts/moe/rebuttal_vllm_pira_benchmark.py \
      --input-lengths 128 1024 4096 --output-lengths 128 256 \
      --concurrency 1 8 32 --output timing_results/vllm_pira.json
"""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="GLOBAL budget of suppressed (layer, expert) pairs per request, "
        "matching the probe's single flat top-K. Not a per-layer count.",
    )
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Diagnostic only: disable compilation and CUDA graphs. Confirms the "
        "hook works, but understates engine throughput, so it must not be the "
        "reported comparison.",
    )
    parser.add_argument(
        "--no-pira-worker",
        action="store_true",
        help="Do not install PiraWorker. Expected to fail the liveness check on "
        "the compiled path; useful only to demonstrate that failure.",
    )
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
    """Suppression vectors per request, as plain lists for the RPC.

    K is a GLOBAL budget of (layer, expert) pairs per request, not a per-layer
    count. This matches the probe: ProbeResult.to_bias() flattens the gradients to
    [request, num_layers * num_experts], takes ONE topk(K) over that flat axis, and
    reshapes back, so a request suppresses exactly K pairs spread unevenly across
    the probed layers -- and typically leaves a good fraction of layers untouched
    entirely.

    Sampling K experts inside every layer instead would suppress
    len(layers) * K pairs (625 rather than 25 for the default configuration) and
    would touch every layer uniformly. Since the whole point of this benchmark is
    to measure how the intervention perturbs expert-token load balance, that
    would time a workload 25x heavier than the method and attribute the result to
    PIRA.

    Distinct per request, matching PIRA's query-specific behaviour: a shared bias
    would let the engine settle into one routing pattern and understate the load
    imbalance the real method induces.
    """
    import random

    positions = len(layers) * num_experts
    budget = min(top_k, positions)
    biases: dict[str, dict[int, list[float]]] = {}
    rng = random.Random(seed)
    for request_id in request_ids:
        # One global sample over flattened (layer, expert) positions, mirroring the
        # single flat topk in ProbeResult.to_bias().
        chosen = rng.sample(range(positions), budget)
        per_layer: dict[int, list[float]] = {
            layer: [0.0] * num_experts for layer in layers
        }
        for flat_index in chosen:
            layer_offset, expert = divmod(flat_index, num_experts)
            per_layer[layers[layer_offset]][expert] = -abs(beta)
        biases[request_id] = per_layer
    return biases


def verify_bias_budget(
    *,
    num_layers: int = 25,
    num_experts: int = 128,
    beta: float = 10.0,
) -> dict:
    """Check the synthetic bias matches the probe's global-K suppression budget.

    Runs without a GPU or vLLM. The invariant, over all layers of one request:

        count(nonzero) == min(K, len(layers) * num_experts)

    A per-layer interpretation of K would give len(layers) * K instead, which for
    the default configuration is 625 rather than 25 -- a 25x heavier perturbation
    of expert-token load balance than PIRA actually applies.

    Also checks that the budget is spread non-uniformly, leaving some layers
    untouched, which is what a single flat top-K produces and what a per-layer
    sample cannot.
    """
    layers = list(range(num_layers))
    positions = num_layers * num_experts
    failures = []

    for top_k in (1, 25, 50, positions, positions + 100):
        expected = min(top_k, positions)
        biases = build_biases(
            ["a", "b"],
            layers=layers,
            num_experts=num_experts,
            top_k=top_k,
            beta=beta,
            seed=0,
        )
        for request_id, per_layer in biases.items():
            count = sum(
                1 for layer in per_layer for value in per_layer[layer] if value != 0
            )
            if count != expected:
                failures.append(
                    f"K={top_k} request={request_id}: {count} pairs, "
                    f"expected {expected}"
                )
            values = {
                value
                for layer in per_layer
                for value in per_layer[layer]
                if value != 0
            }
            if values - {-abs(beta)}:
                failures.append(f"K={top_k} request={request_id}: bad values {values}")

    # Non-uniformity: K=25 over 25 layers must not touch all 25 layers.
    single = build_biases(
        ["a"], layers=layers, num_experts=num_experts, top_k=25, beta=beta, seed=1
    )["a"]
    touched = sum(1 for layer in single if any(v != 0 for v in single[layer]))
    if touched >= num_layers:
        failures.append(
            f"K=25 touched all {touched} layers; a global budget should leave "
            "some layers unbiased"
        )

    # Per-request distinctness: PIRA's bias is query-specific.
    pair = build_biases(
        ["a", "b"], layers=layers, num_experts=num_experts, top_k=25, beta=beta, seed=2
    )
    if pair["a"] == pair["b"]:
        failures.append("two requests received identical biases")

    return {
        "budget_scope": "global (layer, expert) pairs per request",
        "layers_touched_at_k25": touched,
        "num_layers": num_layers,
        "failures": failures,
        "passed": not failures,
    }


def run_wave(
    llm,
    pira,
    *,
    method: str,
    prompt_ids: list[int],
    concurrency: int,
    sampling,
    layers: list[int],
    num_experts: int,
    top_k: int,
    beta: float,
    seed: int,
) -> tuple[float, int]:
    """Run exactly `concurrency` requests together; return (seconds, tokens).

    One wave submits exactly `concurrency` requests and waits for all of them, so
    the number of concurrently active sequences really is the value the cell is
    labelled with. Submitting the whole cell at once would let the scheduler keep
    up to max_num_seqs active regardless of the label, which is the flaw the
    coauthor identified.

    For PIRA the ordering is the crux: enqueue() returns the engine's real request
    ids without starting execution, biases are registered against exactly those
    ids, and only then does wait_for_completion() run them. Guessing the id scheme
    would leave every row unbiased and make PIRA look free.

    enqueue is inside the timed region for both methods, so the comparison stays
    symmetric; PIRA additionally pays for one bias-registration RPC, which is part
    of its real cost and is therefore timed.
    """
    prompts = [{"prompt_token_ids": prompt_ids} for _ in range(concurrency)]

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


def run_cell(
    llm,
    pira,
    *,
    method: str,
    prompt_ids: list[int],
    total: int,
    concurrency: int,
    sampling,
    layers: list[int],
    num_experts: int,
    top_k: int,
    beta: float,
    seed: int,
) -> tuple[float, int]:
    """Run total/concurrency sequential waves and return the summed time.

    Timing the sum of waves rather than one big submission is what makes the
    concurrency label binding.
    """
    waves = max(1, total // concurrency)
    elapsed_total = 0.0
    produced_total = 0
    for wave in range(waves):
        elapsed, produced = run_wave(
            llm,
            pira,
            method=method,
            prompt_ids=prompt_ids,
            concurrency=concurrency,
            sampling=sampling,
            layers=layers,
            num_experts=num_experts,
            top_k=top_k,
            beta=beta,
            seed=seed + wave,
        )
        elapsed_total += elapsed
        produced_total += produced
    return elapsed_total, produced_total


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
    layers = list(range(args.probe_layer + 1))

    # The hooks must be installed before torch.compile tracing and CUDA graph
    # capture, both of which happen inside LLM(...). A custom worker class is the
    # supported way to run code at that point: PiraWorker installs during
    # load_model, so the bias-add is part of the captured graphs. Installing after
    # LLM(...) returns leaves the graphs unbiased -- the failure mode where the
    # wrapper is present on every router yet runs only a handful of times.
    #
    # PIRA_LAYERS reaches the worker through the environment because vLLM
    # constructs the worker itself. The Original arm uses the SAME worker class
    # with the hooks inert, so both arms share one compiled path.
    os.environ["PIRA_LAYERS"] = f"{layers[0]}-{layers[-1]}"
    os.environ["PIRA_BETA"] = str(args.beta)
    os.environ["PIRA_STRICT"] = "1"
    # Callable collective_rpc falls back to pickle in vLLM 0.19.0. Acceptable for
    # a trusted local benchmark; it is required for register/diagnostics RPCs.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    engine_kwargs = dict(
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
    if not args.no_pira_worker:
        engine_kwargs["worker_cls"] = "pira_vllm_worker.PiraWorker"
    if args.enforce_eager:
        # Diagnostic only. The reviewer-facing comparison must run the normal
        # compiled path, since eager mode understates the engine's throughput and
        # would flatter PIRA's relative overhead.
        engine_kwargs["enforce_eager"] = True

    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()

    config = llm.llm_engine.model_config.hf_config
    num_experts = int(getattr(config, "num_experts", 0) or getattr(config, "n_routed_experts", 0))
    if num_experts <= 0:
        print("could not determine the model's expert count", file=sys.stderr)
        return 2

    print(
        f"model={args.model} experts={num_experts} "
        f"probed_layers=0..{args.probe_layer} "
        f"suppressed_expert_layer_pairs_per_request={args.top_k} (global)"
    )

    # With PiraWorker the hooks were already installed during load_model, before
    # compile/capture, so nothing is installed here -- doing so would double-patch
    # the routers. Only the --no-pira-worker path (which exists to demonstrate the
    # compiled-path failure) installs late.
    if args.no_pira_worker:
        installed = pira.install_in_worker(llm, layers, beta=args.beta, strict=True)
        if not installed:
            print("no MoE layer was hooked", file=sys.stderr)
            return 2
        print(
            f"hooked {len(installed)} MoE layers AFTER capture "
            f"({installed[0]}..{installed[-1]}) -- expected to fail liveness"
        )
    else:
        hook_report_early = pira.verify_hooks(llm)
        installed = sorted(layers)
        if not hook_report_early.get("installed"):
            print(
                "PiraWorker did not install the routing hooks. Check that "
                "scripts/moe is on PYTHONPATH inside the worker process and that "
                "PIRA_LAYERS reached it.",
                file=sys.stderr,
            )
            return 2
        print(
            f"PiraWorker installed hooks before capture on "
            f"{hook_report_early.get('hooked_layers')} routers"
        )

    hook_report = pira.verify_hooks(llm)
    print(f"hook verification: {hook_report}")
    if hook_report.get("unhooked_layers"):
        print(
            f"note: {hook_report['unhooked_layers']} MoE layers are outside the "
            f"probed range and remain unbiased (expected for layers "
            f"> {args.probe_layer})"
        )

    records: list[dict] = []
    diagnostics: dict = {}
    unbiased_cells: list[tuple[int, int, int]] = []

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
                        # Counters and registrations are cleared per cell so each
                        # cell's diagnostics stand alone. Checking only the final
                        # cell could let an earlier Original-vs-Original
                        # comparison hide behind one good number at the end.
                        pira.reset_counters(llm)

                    # Warm up so allocator growth and kernel autotuning stay out
                    # of the timed region.
                    run_wave(
                        llm,
                        pira,
                        method=method,
                        prompt_ids=prompt_ids,
                        concurrency=concurrency,
                        sampling=warmup,
                        layers=layers,
                        num_experts=num_experts,
                        top_k=args.top_k,
                        beta=args.beta,
                        seed=args.seed,
                    )
                    if method == "PIRA":
                        pira.reset_counters(llm)

                    durations = []
                    for repeat in range(args.repeats):
                        elapsed, produced = run_cell(
                            llm,
                            pira,
                            method=method,
                            prompt_ids=prompt_ids,
                            total=total,
                            concurrency=concurrency,
                            sampling=sampling,
                            layers=layers,
                            num_experts=num_experts,
                            top_k=args.top_k,
                            beta=args.beta,
                            seed=args.seed + repeat * 1000,
                        )
                        durations.append(elapsed)
                        expected = total * output_length
                        if produced != expected:
                            print(
                                f"warning: produced {produced} tokens, "
                                f"expected {expected}",
                                file=sys.stderr,
                            )

                    cell_diagnostics = (
                        pira.worker_diagnostics(llm) if method == "PIRA" else None
                    )

                    median = statistics.median(durations)
                    record = {
                        "method": method,
                        "input_length": input_length,
                        "output_length": output_length,
                        "concurrency": concurrency,
                        "num_requests": total,
                        "waves": max(1, total // concurrency),
                        "median_seconds": median,
                        "min_seconds": min(durations),
                        "max_seconds": max(durations),
                        "seconds_per_request": median / total,
                        "request_throughput": total / median,
                        "output_token_throughput": total * output_length / median,
                    }
                    suffix = ""
                    if method == "PIRA":
                        record["diagnostics"] = cell_diagnostics
                        rows = (cell_diagnostics or {}).get("rows_biased", 0)
                        passes = (cell_diagnostics or {}).get("forward_passes", 0)
                        fills = (cell_diagnostics or {}).get("buffer_fills", 0)
                        tokens = (cell_diagnostics or {}).get("tokens_biased", 0)
                        waves = record["waves"]

                        # Liveness must be judged by buffer_fills, not by how often
                        # the router wrapper ran. With CUDA graphs enabled the
                        # wrapper executes only while a shape is being captured --
                        # replays re-run the recorded kernels with no Python at all
                        # -- so forward_passes legitimately stays small even when
                        # the intervention is fully active. buffer_fills counts the
                        # per-step host-side fill, which happens on every engine
                        # step including replayed ones.
                        #
                        # Every decoded token needs its own step, so with
                        # exact-length generation there must be at least
                        # repeats * waves * output_length fills. Prefill adds more;
                        # this is a deliberately conservative floor.
                        expected = args.repeats * waves * output_length
                        record["rows_biased"] = rows
                        record["forward_passes"] = passes
                        record["buffer_fills"] = fills
                        record["tokens_biased"] = tokens
                        record["expected_min_buffer_fills"] = expected
                        # Retained for older readers of this JSON.
                        record["expected_min_forward_passes"] = expected
                        live = bool(tokens) and fills >= expected
                        record["hook_live"] = live
                        if not live:
                            unbiased_cells.append(
                                (
                                    input_length,
                                    output_length,
                                    concurrency,
                                    tokens,
                                    fills,
                                    expected,
                                )
                            )
                        diagnostics = cell_diagnostics
                        suffix = (
                            f"  fills={fills}/{expected} tokens_biased={tokens} "
                            f"captures={passes}"
                            + ("" if live else "  <-- HOOK NOT LIVE")
                        )
                    records.append(record)
                    print(
                        f"  {method:<8} in={input_length:<5} out={output_length:<4} "
                        f"conc={concurrency:<3} {median:8.3f} s  "
                        f"{record['request_throughput']:7.2f} req/s{suffix}"
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
                        "num_requests": original["num_requests"],
                        "original_seconds": original["median_seconds"],
                        "pira_seconds": biased["median_seconds"],
                        "original_seconds_per_request": original["seconds_per_request"],
                        "pira_seconds_per_request": biased["seconds_per_request"],
                        "routing_overhead_fraction": ratio,
                        "routing_overhead_percent": 100.0 * ratio,
                        "rows_biased": biased.get("rows_biased", 0),
                        "forward_passes": biased.get("forward_passes", 0),
                        "expected_min_forward_passes": biased.get(
                            "expected_min_forward_passes", 0
                        ),
                        "hook_live": biased.get("hook_live", False),
                    }
                )
                print(
                    f"{input_length:>6} {output_length:>5} {concurrency:>5} "
                    f"{original['median_seconds']:>12.3f} "
                    f"{biased['median_seconds']:>10.3f} "
                    f"{100.0 * ratio:>8.1f}%"
                    + ("" if biased.get("hook_live") else "   <-- HOOK NOT LIVE")
                )

    pira_cells = [r for r in records if r["method"] == "PIRA"]
    live_cells = [r for r in pira_cells if r.get("hook_live")]
    print(
        f"\nper-cell hook validation: {len(live_cells)}/{len(pira_cells)} PIRA cells "
        "applied a bias on every forward pass"
    )
    if unbiased_cells:
        print(
            "\nFAIL: in these cells the hook was not live for the whole run, so "
            "their PIRA timings are partly or wholly the Original model:",
            file=sys.stderr,
        )
        for cell in unbiased_cells:
            input_length, output_length, concurrency, tokens, fills, expected = cell
            if not tokens:
                reason = "no token row ever received a bias"
            else:
                reason = (
                    f"only {fills} per-step buffer fills, expected at least "
                    f"{expected} -- the per-step hook did not run on every engine "
                    "step"
                )
            print(
                f"  input={input_length} output={output_length} "
                f"concurrency={concurrency}: {reason}",
                file=sys.stderr,
            )
        print(
            "\nMost likely causes: the engine was built without "
            'worker_cls="pira_vllm_worker.PiraWorker", so the hooks were installed '
            "after CUDA graph capture and are not part of the captured graphs; or "
            "PIRA_LAYERS was not exported to the worker; or the MoE backend is "
            "monolithic and bypasses select_experts. Confirm with "
            "enforce_eager=True as a diagnostic only -- the reviewer-facing "
            "comparison must run the normal compiled path.",
            file=sys.stderr,
        )

    payload = {
        "metadata": {
            "model": args.model,
            "num_experts": num_experts,
            "biased_layers": [0, args.probe_layer],
            "hooked_layers": installed,
            "hook_verification": hook_report,
            "suppressed_expert_layer_pairs_per_request": args.top_k,
            "suppression_budget_scope": "global across probed layers, matching "
                                        "ProbeResult.to_bias()",
            "beta": args.beta,
            "repeats": args.repeats,
            "requests_per_cell": args.requests_per_cell,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": max_len,
            "max_num_seqs": max(args.concurrency),
            "concurrency_enforcement": "total/concurrency sequential waves of "
                                       "exactly `concurrency` requests each",
            "bias": "seeded synthetic suppression sets; timing only",
            "scope": "generation only; the probe is measured by "
                     "rebuttal_probe_scaling.py",
            "all_cells_biased": not unbiased_cells,
            "all_cells_hook_live": not unbiased_cells,
            "not_live_cells": [
                {
                    "input_length": cell[0],
                    "output_length": cell[1],
                    "concurrency": cell[2],
                    "rows_biased": cell[3],
                    "forward_passes": cell[4],
                    "expected_min_forward_passes": cell[5],
                }
                for cell in unbiased_cells
            ],
        },
        "measurements": records,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")

    # The process exits immediately after this, so uninstalling is cosmetic; it is
    # also unsafe once graphs have captured the hooked path, since the captured
    # kernels still reference the persistent buffers. Left installed deliberately.
    return 0 if not unbiased_cells else 1


if __name__ == "__main__":
    # --check-workload runs the workload-definition self-check only: no GPU, no
    # vLLM, no model download. Run it before submitting a GPU job, since a wrong
    # suppression budget would time the wrong experiment even with a perfect hook.
    if "--check-workload" in sys.argv:
        report = verify_bias_budget()
        for key, value in report.items():
            print(f"{key}: {value}")
        print("PASS" if report["passed"] else "FAIL")
        raise SystemExit(0 if report["passed"] else 1)
    raise SystemExit(main())
