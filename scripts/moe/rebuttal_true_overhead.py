"""Combine a production-engine baseline, the probe, and the routing cost.

Reviewer 7ZUn asked for this accounting:

    True Total = (Original model turnaround on vLLM/SGLang at maximum GPU
                  utilization)
               + (standalone time for PIRA's forward + backward probe)

PIRA's cost splits into two measured parts:

  probe     a forward+backward pass over layers 0..L of the prompt. Runs in a
            Hugging Face autograd context, measured by rebuttal_probe_scaling.py.
  routing   the cost of generating under the biased routing. Top-k and the expert
            MLPs are unchanged, so the FLOP count is identical -- but identical
            FLOPs do not guarantee identical latency, because suppressing experts
            changes the expert-token distribution and therefore MoE kernel
            efficiency and load balance. So this is MEASURED, by
            rebuttal_vllm_pira_benchmark.py, not assumed to be zero.

Pass --routing-json to include the measured routing cost. Without it the script
reports probe-only overhead and labels the result as a lower bound, so that the
assumption is visible rather than buried.

Engine and probe need incompatible dependency stacks (vLLM pins its own
Transformers), which is why this combines JSON rather than running everything in
one process.

Reporting an (input, output) grid rather than a single number is deliberate: the
overhead ratio depends entirely on how prompt-heavy the workload is, and a single
figure from a 128-token prompt would hide the long-context regime.

Usage:
  # 1. engine baseline (vLLM env)
  python scripts/moe/rebuttal_vllm_baseline.py --output timing_results/vllm.json ...
  # 2. probe cost (HF env)
  python scripts/moe/rebuttal_probe_scaling.py --output timing_results/probe.json ...
  # 3. routing cost, Original vs biased on the same engine (vLLM env)
  python scripts/moe/rebuttal_vllm_pira_benchmark.py --output timing_results/routing.json ...
  # 4. combine
  python scripts/moe/rebuttal_true_overhead.py \
      --engine-json timing_results/vllm.json \
      --probe-json timing_results/probe.json \
      --routing-json timing_results/routing.json \
      --output timing_results/true_overhead.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-json",
        type=Path,
        required=True,
        help="Output of rebuttal_vllm_baseline.py (Original on vLLM/SGLang).",
    )
    parser.add_argument(
        "--probe-json",
        type=Path,
        required=True,
        help="Output of rebuttal_probe_scaling.py.",
    )
    parser.add_argument(
        "--routing-json",
        type=Path,
        help="Output of rebuttal_vllm_pira_benchmark.py (measured routing cost). "
        "Omit to report probe-only overhead, which is then a lower bound.",
    )
    parser.add_argument(
        "--probe-configuration",
        default="optimized",
        help="Prefix of the probe configuration label to use.",
    )
    parser.add_argument(
        "--engine-label",
        default="vLLM",
        help="Name of the serving engine, for the report.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_engine(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = []
    for measurement in payload.get("measurements", []):
        rows.append(
            {
                "batch_size": measurement.get("batch_size"),
                "total_seconds": measurement.get("total_seconds"),
                "request_throughput": measurement.get("request_throughput"),
                "num_requests": payload.get("num_requests"),
                "prompt_tokens": payload.get("prompt_tokens"),
                "output_tokens": payload.get("output_tokens_per_request"),
            }
        )
    return {"backend": payload.get("backend", "unknown"), "rows": rows}


def load_routing(path: Path) -> dict:
    """Read the measured routing overhead, keyed by (input, output, concurrency).

    Also surfaces the worker diagnostics: if the router never saw a biased row,
    the measured "overhead" is really Original-vs-Original and must not be used.
    """
    payload = json.loads(path.read_text())
    diagnostics = payload.get("diagnostics", {})
    if not diagnostics.get("rows_biased"):
        raise SystemExit(
            f"{path} reports rows_biased=0, so its PIRA timings did not actually "
            "apply any bias. Fix the routing integration before combining."
        )
    table = {}
    for entry in payload.get("comparisons", []):
        key = (
            entry["input_length"],
            entry["output_length"],
            entry["concurrency"],
        )
        table[key] = entry
    return {"table": table, "diagnostics": diagnostics}


def routing_overhead_for(
    routing: dict | None,
    prompt_tokens: int,
    output_tokens: int | None,
    concurrency: int | None,
) -> tuple[float, str]:
    """Routing overhead as a fraction, plus how it was obtained.

    Falls back progressively: exact cell, then same input length, then the worst
    observed cell. The fallback is deliberately pessimistic -- reporting the
    maximum rather than the mean avoids understating the cost when a cell is
    missing.
    """
    if not routing:
        return 0.0, "assumed zero (no routing measurement supplied)"
    table = routing["table"]
    exact = table.get((prompt_tokens, output_tokens, concurrency))
    if exact:
        return exact["routing_overhead_fraction"], "measured"
    same_input = [v for k, v in table.items() if k[0] == prompt_tokens]
    if same_input:
        worst = max(same_input, key=lambda v: v["routing_overhead_fraction"])
        return (
            worst["routing_overhead_fraction"],
            f"worst cell at input {prompt_tokens} "
            f"(out={worst['output_length']}, conc={worst['concurrency']})",
        )
    if table:
        worst = max(table.values(), key=lambda v: v["routing_overhead_fraction"])
        return (
            worst["routing_overhead_fraction"],
            f"worst observed cell (in={worst['input_length']}, "
            f"out={worst['output_length']}, conc={worst['concurrency']})",
        )
    return 0.0, "assumed zero (routing table empty)"


def load_probe(path: Path, configuration_prefix: str) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = []
    for measurement in payload["measurements"]:
        if not str(measurement.get("configuration", "")).startswith(
            configuration_prefix
        ):
            continue
        if measurement.get("status") != "ok":
            continue
        rows.append(measurement)
    if not rows:
        raise SystemExit(
            f"no successful probe rows matching '{configuration_prefix}' in {path}"
        )
    return rows


def probe_seconds_per_request(
    probe_rows: list[dict], prompt_tokens: int, batch_size: int | None
) -> tuple[float, str]:
    """Pick the probe cost for a prompt length, preferring an exact batch match.

    Returns the per-request cost and a note describing how it was obtained, so
    that any interpolation is visible in the report rather than silent.
    """
    exact = [r for r in probe_rows if r["prompt_tokens"] == prompt_tokens]
    if exact:
        if batch_size is not None:
            same_batch = [r for r in exact if r["batch_size"] == batch_size]
            if same_batch:
                best = min(same_batch, key=lambda r: r["seconds_per_request"])
                return best["seconds_per_request"], "measured"
        best = min(exact, key=lambda r: r["seconds_per_request"])
        return (
            best["seconds_per_request"],
            f"measured at batch {best['batch_size']}",
        )

    # No measurement at this prompt length: interpolate between neighbours and
    # say so explicitly.
    ordered = sorted(probe_rows, key=lambda r: r["prompt_tokens"])
    lengths = [r["prompt_tokens"] for r in ordered]
    if prompt_tokens < lengths[0] or prompt_tokens > lengths[-1]:
        nearest = min(ordered, key=lambda r: abs(r["prompt_tokens"] - prompt_tokens))
        return (
            nearest["seconds_per_request"],
            f"extrapolated from {nearest['prompt_tokens']} tokens",
        )
    lower = max(
        (r for r in ordered if r["prompt_tokens"] <= prompt_tokens),
        key=lambda r: r["prompt_tokens"],
    )
    upper = min(
        (r for r in ordered if r["prompt_tokens"] >= prompt_tokens),
        key=lambda r: r["prompt_tokens"],
    )
    if upper["prompt_tokens"] == lower["prompt_tokens"]:
        return lower["seconds_per_request"], "measured"
    span = upper["prompt_tokens"] - lower["prompt_tokens"]
    weight = (prompt_tokens - lower["prompt_tokens"]) / span
    value = (
        lower["seconds_per_request"] * (1 - weight)
        + upper["seconds_per_request"] * weight
    )
    return value, (
        f"interpolated between {lower['prompt_tokens']} and "
        f"{upper['prompt_tokens']} tokens"
    )


def main() -> int:
    args = parse_args()
    engine = load_engine(args.engine_json)
    probe_rows = load_probe(args.probe_json, args.probe_configuration)
    routing = load_routing(args.routing_json) if args.routing_json else None

    print(f"engine baseline: {engine['backend']} ({args.engine_json})")
    print(f"probe cost:      {args.probe_configuration} ({args.probe_json})")
    if routing:
        print(f"routing cost:    measured ({args.routing_json})")
        print(f"                 diagnostics {routing['diagnostics']}")
    else:
        print("routing cost:    NOT SUPPLIED -- treated as zero (lower bound)")
    print()

    combined = []
    for row in engine["rows"]:
        if not row["total_seconds"] or not row["num_requests"]:
            continue
        prompt_tokens = row["prompt_tokens"]
        engine_per_request = row["total_seconds"] / row["num_requests"]
        probe_per_request, probe_note = probe_seconds_per_request(
            probe_rows, prompt_tokens, row["batch_size"]
        )
        routing_fraction, routing_note = routing_overhead_for(
            routing, prompt_tokens, row["output_tokens"], row["batch_size"]
        )
        routing_per_request = engine_per_request * routing_fraction
        true_total = engine_per_request + probe_per_request + routing_per_request
        extra = probe_per_request + routing_per_request
        combined.append(
            {
                "engine": engine["backend"],
                "batch_size": row["batch_size"],
                "prompt_tokens": prompt_tokens,
                "output_tokens": row["output_tokens"],
                "engine_seconds_per_request": engine_per_request,
                "probe_seconds_per_request": probe_per_request,
                "routing_seconds_per_request": routing_per_request,
                "routing_overhead_fraction": routing_fraction,
                "true_total_seconds_per_request": true_total,
                "overhead_fraction": extra / engine_per_request,
                "overhead_percent": 100.0 * extra / engine_per_request,
                "probe_source": probe_note,
                "routing_source": routing_note,
            }
        )

    if not combined:
        raise SystemExit("engine JSON contained no usable measurements")

    header = (
        f"{'batch':>6} {'in':>6} {'out':>6} {f'{args.engine_label}(s/req)':>16} "
        f"{'probe(s/req)':>13} {'routing(s/req)':>15} {'true total':>11} "
        f"{'overhead':>9}"
    )
    print(header)
    print("-" * len(header))
    for entry in combined:
        print(
            f"{entry['batch_size']:>6} {entry['prompt_tokens']:>6} "
            f"{entry['output_tokens']:>6} "
            f"{entry['engine_seconds_per_request']:>16.4f} "
            f"{entry['probe_seconds_per_request']:>13.4f} "
            f"{entry['routing_seconds_per_request']:>15.4f} "
            f"{entry['true_total_seconds_per_request']:>11.4f} "
            f"{entry['overhead_percent']:>8.1f}%"
        )

    for label, key in (("probe", "probe_source"), ("routing", "routing_source")):
        notes = {entry[key] for entry in combined}
        inexact = sorted(n for n in notes if not n.startswith("measured"))
        if inexact:
            print(f"\n{label} cost provenance (non-measured cells):")
            for note in inexact:
                print(f"  - {note}")

    print(
        "\nTrue Total = Original turnaround on "
        f"{args.engine_label} at maximum GPU utilization"
        "\n           + standalone PIRA forward+backward probe"
        "\n           + measured cost of generating under biased routing"
    )
    if not routing:
        print(
            "\nWARNING: no routing measurement was supplied, so the routing term is "
            "zero and these overheads are a LOWER BOUND. Biased routing keeps the "
            "FLOP count identical but can still shift MoE load balance and kernel "
            "efficiency. Run rebuttal_vllm_pira_benchmark.py and pass "
            "--routing-json before quoting these numbers."
        )

    if args.output:
        payload = {
            "metadata": {
                "engine_json": str(args.engine_json),
                "probe_json": str(args.probe_json),
                "engine_backend": engine["backend"],
                "probe_configuration": args.probe_configuration,
                "routing_json": str(args.routing_json) if args.routing_json else None,
                "routing_measured": bool(routing),
                "routing_diagnostics": routing["diagnostics"] if routing else None,
                "formula": "true_total = engine_per_request + probe_per_request "
                           "+ routing_per_request",
                "caveat": None
                if routing
                else "routing term assumed zero; overheads are a lower bound",
            },
            "measurements": combined,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
