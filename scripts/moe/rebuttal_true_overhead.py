"""Combine a production-engine baseline with the standalone probe cost.

Reviewer 7ZUn asked for exactly this accounting:

    True Total = (Original model turnaround on vLLM/SGLang at maximum GPU
                  utilization)
               + (standalone time for PIRA's forward + backward probe)

The reasoning behind treating that sum as PIRA's true cost:

  * PIRA's generation phase is FLOP-identical to ordinary generation. The router
    bias changes *which* experts a token is routed to, not how many (top-k is
    unchanged) and not the expert MLPs themselves. So the generation phase
    inherits the serving engine's throughput; adding a per-expert additive bias
    to router logits before top-k is already a first-class, effectively free
    operation in vLLM (`e_score_correction_bias`, used by DeepSeek-V3).
  * Therefore all of PIRA's overhead is the probe, which is measured here on its
    own, at full precision, with no approximation.

This script does not itself launch a serving engine: engine and probe need
incompatible dependency stacks (vLLM pins its own Transformers). It reads the
engine baseline produced by rebuttal_vllm_baseline.py, reads the probe cost
produced by rebuttal_probe_scaling.py (or measures it directly with --measure-probe),
and reports the combined overhead across an (input length, output length) grid.

Reporting a grid rather than a single number is deliberate: the overhead ratio
depends entirely on how prompt-heavy the workload is, and a single figure from a
128-token prompt would hide the long-context regime.

Usage:
  # 1. engine baseline (vLLM env)
  python scripts/moe/rebuttal_vllm_baseline.py --output timing_results/vllm.json ...
  # 2. probe cost (HF env)
  python scripts/moe/rebuttal_probe_scaling.py --output timing_results/probe.json ...
  # 3. combine
  python scripts/moe/rebuttal_true_overhead.py \
      --engine-json timing_results/vllm.json \
      --probe-json timing_results/probe.json \
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

    print(f"engine baseline: {engine['backend']} ({args.engine_json})")
    print(f"probe cost:      {args.probe_configuration} ({args.probe_json})")
    print()

    combined = []
    for row in engine["rows"]:
        if not row["total_seconds"] or not row["num_requests"]:
            continue
        prompt_tokens = row["prompt_tokens"]
        engine_per_request = row["total_seconds"] / row["num_requests"]
        probe_per_request, note = probe_seconds_per_request(
            probe_rows, prompt_tokens, row["batch_size"]
        )
        true_total = engine_per_request + probe_per_request
        combined.append(
            {
                "engine": engine["backend"],
                "batch_size": row["batch_size"],
                "prompt_tokens": prompt_tokens,
                "output_tokens": row["output_tokens"],
                "engine_seconds_per_request": engine_per_request,
                "probe_seconds_per_request": probe_per_request,
                "true_total_seconds_per_request": true_total,
                "overhead_fraction": probe_per_request / engine_per_request,
                "overhead_percent": 100.0 * probe_per_request / engine_per_request,
                "probe_source": note,
            }
        )

    if not combined:
        raise SystemExit("engine JSON contained no usable measurements")

    header = (
        f"{'batch':>6} {'in':>6} {'out':>6} {f'{args.engine_label}(s/req)':>16} "
        f"{'probe(s/req)':>13} {'true total':>11} {'overhead':>9}"
    )
    print(header)
    print("-" * len(header))
    for entry in combined:
        print(
            f"{entry['batch_size']:>6} {entry['prompt_tokens']:>6} "
            f"{entry['output_tokens']:>6} "
            f"{entry['engine_seconds_per_request']:>16.4f} "
            f"{entry['probe_seconds_per_request']:>13.4f} "
            f"{entry['true_total_seconds_per_request']:>11.4f} "
            f"{entry['overhead_percent']:>8.1f}%"
        )

    notes = {entry["probe_source"] for entry in combined}
    if any(not note.startswith("measured") for note in notes):
        print("\nprobe cost provenance (non-measured cells):")
        for note in sorted(notes):
            if not note.startswith("measured"):
                print(f"  - {note}")

    print(
        "\nTrue Total = Original turnaround on "
        f"{args.engine_label} at maximum GPU utilization + standalone PIRA "
        "forward+backward probe. PIRA's generation phase is FLOP-identical to "
        "ordinary generation, so the probe is the entire overhead."
    )

    if args.output:
        payload = {
            "metadata": {
                "engine_json": str(args.engine_json),
                "probe_json": str(args.probe_json),
                "engine_backend": engine["backend"],
                "probe_configuration": args.probe_configuration,
                "formula": "true_total = engine_per_request + probe_per_request",
            },
            "measurements": combined,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
