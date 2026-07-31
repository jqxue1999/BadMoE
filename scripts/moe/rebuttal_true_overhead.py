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
    """Read the cell-matched Original and biased generation measurements.

    Each comparison already contains both methods for the *same* input length,
    output length and concurrency, measured back to back on one engine. That is
    strictly better than pairing across files, so when this JSON is present it
    supplies the generation baseline as well as the routing delta and no
    cross-schema matching is needed.

    Cells whose diagnostics show no biased row are dropped: their PIRA timing is
    the Original model, so any "overhead" computed from them is noise.
    """
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata", {})
    comparisons = payload.get("comparisons", [])
    if not comparisons:
        raise SystemExit(f"{path} contains no comparisons")

    table = {}
    dropped = []
    for entry in comparisons:
        key = (entry["input_length"], entry["output_length"], entry["concurrency"])
        # Prefer the strong criterion: the hook applied a bias on every forward
        # pass, not merely at least once. Fall back to rows_biased for files
        # written before hook_live existed, and treat a missing per-cell value as
        # unverified rather than as verified-good.
        live = entry.get("hook_live")
        if live is None:
            rows = entry.get("rows_biased")
            if rows is None:
                rows = payload.get("diagnostics", {}).get("rows_biased", 0)
            live = bool(rows)
        if not live:
            dropped.append(key)
            continue
        table[key] = entry

    if not table:
        raise SystemExit(
            f"{path}: no cell had a live hook for its whole run, so its PIRA "
            "timings are the Original model. Fix the routing integration before "
            "combining."
        )
    return {
        "table": table,
        "dropped": dropped,
        "diagnostics": payload.get("diagnostics", {}),
        "all_cells_biased": metadata.get("all_cells_biased"),
    }


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
    if routing:
        # Preferred path. Each routing cell already holds Original and biased
        # generation measured back to back on one engine at one concurrency, so
        # the whole equation comes from same-cell numbers and only the probe is
        # matched in from the other file:
        #     true_total = biased_generation + probe
        #                = original + probe + (biased - original)
        # No cross-schema guessing about what the old baseline's batch_size meant.
        for (input_length, output_length, concurrency), entry in sorted(
            routing["table"].items()
        ):
            original_per_request = entry["original_seconds_per_request"]
            biased_per_request = entry["pira_seconds_per_request"]
            routing_per_request = biased_per_request - original_per_request
            probe_per_request, probe_note = probe_seconds_per_request(
                probe_rows, input_length, concurrency
            )
            true_total = biased_per_request + probe_per_request
            extra = probe_per_request + routing_per_request
            combined.append(
                {
                    "source": "cell-matched",
                    "engine": engine["backend"],
                    "batch_size": concurrency,
                    "concurrency": concurrency,
                    "prompt_tokens": input_length,
                    "output_tokens": output_length,
                    "engine_seconds_per_request": original_per_request,
                    "probe_seconds_per_request": probe_per_request,
                    "routing_seconds_per_request": routing_per_request,
                    "routing_overhead_fraction": entry["routing_overhead_fraction"],
                    "true_total_seconds_per_request": true_total,
                    "overhead_fraction": extra / original_per_request,
                    "overhead_percent": 100.0 * extra / original_per_request,
                    "probe_source": probe_note,
                    "routing_source": "measured (same cell)",
                    "rows_biased": entry.get("rows_biased"),
                }
            )
    else:
        # Fallback: probe-only lower bound against the standalone engine baseline.
        for row in engine["rows"]:
            if not row["total_seconds"] or not row["num_requests"]:
                continue
            prompt_tokens = row["prompt_tokens"]
            engine_per_request = row["total_seconds"] / row["num_requests"]
            probe_per_request, probe_note = probe_seconds_per_request(
                probe_rows, prompt_tokens, row["batch_size"]
            )
            combined.append(
                {
                    "source": "probe-only lower bound",
                    "engine": engine["backend"],
                    "batch_size": row["batch_size"],
                    "concurrency": None,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": row["output_tokens"],
                    "engine_seconds_per_request": engine_per_request,
                    "probe_seconds_per_request": probe_per_request,
                    "routing_seconds_per_request": 0.0,
                    "routing_overhead_fraction": 0.0,
                    "true_total_seconds_per_request": engine_per_request
                    + probe_per_request,
                    "overhead_fraction": probe_per_request / engine_per_request,
                    "overhead_percent": 100.0 * probe_per_request / engine_per_request,
                    "probe_source": probe_note,
                    "routing_source": "assumed zero (no routing measurement supplied)",
                }
            )

    if not combined:
        raise SystemExit("no usable measurements after matching")

    if routing and routing["dropped"]:
        print(
            f"dropped {len(routing['dropped'])} routing cell(s) whose hook was not "
            "live for the whole run (rows_biased=0, or too few forward passes, "
            "which indicates a compiled graph that bypassed the hook on replay):"
        )
        for cell in routing["dropped"]:
            print(f"  input={cell[0]} output={cell[1]} concurrency={cell[2]}")
        print()

    header = (
        f"{'conc':>6} {'in':>6} {'out':>6} {'Original(s/req)':>16} "
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

    if routing:
        print(
            "\nTrue Total = measured biased generation (same cell, same engine)"
            "\n           + standalone PIRA forward+backward probe"
            "\n  equivalently: Original + probe + (biased - Original)"
            "\nOriginal, biased and their difference all come from the same "
            "input/output/concurrency cell measured back to back."
        )
    else:
        print(
            "\nTrue Total = Original turnaround on "
            f"{args.engine_label} at maximum GPU utilization"
            "\n           + standalone PIRA forward+backward probe"
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
