"""Test whether a differentiable PIRA sidecar can reuse vLLM's loaded weights.

This is deliberately a one-layer feasibility test, not a benchmark claim.  It
answers four questions inside the live vLLM worker:

1. Are the resident router and expert weights ordinary tensors that a separate
   autograd graph can reference without cloning them?
2. Does vLLM's own fused-MoE forward expose a backward graph?  (It is expected
   not to, but the failure is recorded rather than assumed.)
3. If the resident expert weights retain the checkpoint layout, can a small
   PyTorch sidecar compute a router-bias gradient and reproduce the fused-MoE
   output while referencing the exact same weight storage?
4. When requested, can SonicMoE consume permuted views of those weights, match
   the reference numerically, and provide a reusable steady-state backward?

The deliberately slow per-expert PyTorch loop remains the dependency-free
reference.  SonicMoE is optional so weight/layout feasibility can still be
tested before installing an additional kernel package.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument(
        "--moe-backend",
        default="triton",
        choices=("auto", "triton"),
        help="Triton normally preserves the checkpoint-like BF16 weight layout; "
        "auto may select a backend-specific packed layout on B200.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--test-sonicmoe",
        action="store_true",
        help="Also test SonicMoE's backward-capable general-routing kernel with "
        "the resident vLLM weights (requires the optional sonic-moe package).",
    )
    return parser.parse_args()


def _error(error: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "type": type(error).__name__,
        "message": str(error)[:1000],
    }


def _find_qwen_moe(model):
    for name, module in model.named_modules():
        runner = getattr(module, "experts", None)
        routed = getattr(runner, "routed_experts", None)
        gate = getattr(module, "gate", None)
        if (
            runner is not None
            and routed is not None
            and gate is not None
            and hasattr(routed, "w13_weight")
            and hasattr(routed, "w2_weight")
            and hasattr(gate, "weight")
        ):
            return name, module, runner, routed, gate
    raise RuntimeError("could not find a Qwen3 MoE block with resident expert weights")


def _manual_moe(x, gate_weight, w13, w2, top_k: int, renormalize: bool):
    """Autograd-native Qwen3 MoE using resident vLLM weight tensors."""
    import torch
    import torch.nn.functional as F

    num_experts = gate_weight.shape[0]
    bias = torch.zeros(
        num_experts, device=x.device, dtype=torch.float32, requires_grad=True
    )
    logits = F.linear(x, gate_weight).float() + bias
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(probabilities, top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(x.dtype)

    output = torch.zeros_like(x)
    for expert in torch.unique(topk_ids).tolist():
        token_index, slot_index = torch.where(topk_ids == expert)
        expert_input = x.index_select(0, token_index)
        gate_up = F.linear(expert_input, w13[expert])
        gate_part, up_part = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate_part) * up_part, w2[expert])
        weighted = expert_output * topk_weights[token_index, slot_index, None]
        output = output.index_add(0, token_index, weighted)
    return output, bias, topk_ids


def _worker_feasibility(
    worker, tokens: int, seed: int, test_sonicmoe: bool
) -> dict[str, Any]:
    """Runs in the vLLM worker so it sees the actual post-load weight tensors."""
    import torch
    import torch.nn.functional as F

    runner_owner = getattr(worker, "model_runner", worker)
    vllm_config = runner_owner.vllm_config
    get_model = getattr(runner_owner, "get_model", None)
    model = get_model() if callable(get_model) else runner_owner.model
    name, _block, runner, routed, gate = _find_qwen_moe(model)
    w13 = routed.w13_weight
    w2 = routed.w2_weight
    gate_weight = gate.weight
    top_k = int(runner.moe_config.experts_per_token)
    renormalize = bool(getattr(routed, "renormalize", True))
    backend = type(getattr(routed, "quant_method", None)).__name__
    quant_method = getattr(routed, "quant_method", None)
    inner_backend = getattr(quant_method, "unquantized_backend", None)

    report: dict[str, Any] = {
        "layer": name,
        "runner": type(runner).__name__,
        "quant_method": backend,
        "unquantized_backend": str(inner_backend),
        "top_k": top_k,
        "renormalize": renormalize,
        "weights": {
            "gate": {
                "shape": list(gate_weight.shape),
                "stride": list(gate_weight.stride()),
                "dtype": str(gate_weight.dtype),
                "is_inference": bool(gate_weight.is_inference()),
                "data_ptr": gate_weight.data_ptr(),
            },
            "w13": {
                "shape": list(w13.shape),
                "stride": list(w13.stride()),
                "dtype": str(w13.dtype),
                "is_inference": bool(w13.is_inference()),
                "data_ptr": w13.data_ptr(),
            },
            "w2": {
                "shape": list(w2.shape),
                "stride": list(w2.stride()),
                "dtype": str(w2.dtype),
                "is_inference": bool(w2.is_inference()),
                "data_ptr": w2.data_ptr(),
            },
        },
    }

    generator = torch.Generator(device=w13.device).manual_seed(seed)
    x0 = torch.randn(
        tokens,
        gate_weight.shape[1],
        device=w13.device,
        dtype=gate_weight.dtype,
        generator=generator,
    )

    # First isolate whether a normal PyTorch op can save the resident weight for
    # input-gradient computation.  No clone or layout conversion is performed.
    try:
        with torch.inference_mode(False), torch.enable_grad():
            x = x0.detach().requires_grad_(True)
            pointer_before = w13.data_ptr()
            projection = F.linear(x[:2], w13[0])
            (input_grad,) = torch.autograd.grad(projection.float().sum(), (x,))
            report["resident_weight_autograd"] = {
                "ok": True,
                "same_data_ptr": pointer_before == w13.data_ptr(),
                "output_requires_grad": projection.requires_grad,
                "input_grad_finite": bool(torch.isfinite(input_grad).all()),
                "input_grad_norm": float(input_grad.float().norm()),
            }
    except Exception as error:  # noqa: BLE001 - the failure is the measurement
        report["resident_weight_autograd"] = _error(error)

    # Test the serving kernel itself.  A successful forward paired with an
    # absent grad_fn confirms that inference mode is not the only blocker: the
    # custom fused op itself has no autograd registration.
    direct_output = None
    try:
        from vllm.forward_context import set_forward_context

        with torch.inference_mode(False), torch.enable_grad():
            x = x0.detach().requires_grad_(True)
            router_logits = F.linear(x, gate_weight)
            with set_forward_context(None, vllm_config, num_tokens=tokens):
                direct_output = runner(
                    hidden_states=x,
                    router_logits=router_logits,
                )
            direct_requires_grad = bool(direct_output.requires_grad)
            (direct_grad,) = torch.autograd.grad(
                direct_output.float().sum(), (x,), allow_unused=True
            )
            report["vllm_fused_backward"] = {
                "ok": direct_grad is not None,
                "output_requires_grad": direct_requires_grad,
                "input_grad_finite": bool(
                    direct_grad is not None and torch.isfinite(direct_grad).all()
                ),
            }
    except Exception as error:  # noqa: BLE001
        report["vllm_fused_backward"] = _error(error) | {
            "output_requires_grad": bool(
                direct_output is not None and direct_output.requires_grad
            )
        }

    # Finally test the sidecar formulation, including the actual PIRA leaf:
    # additive router bias before softmax/top-k and gradient back to that leaf.
    manual_output = None
    manual_input_grad = None
    manual_bias_grad = None
    try:
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(False), torch.enable_grad():
            x = x0.detach().requires_grad_(True)
            manual_output, bias, topk_ids = _manual_moe(
                x, gate_weight, w13, w2, top_k, renormalize
            )
            direction_generator = torch.Generator(device=x.device).manual_seed(
                seed + 1
            )
            direction = torch.randn(
                manual_output.shape[-1],
                device=x.device,
                dtype=torch.float32,
                generator=direction_generator,
            )
            score = (manual_output.float() * direction).sum()
            input_grad, bias_grad = torch.autograd.grad(score, (x, bias))
            manual_input_grad = input_grad.detach()
            manual_bias_grad = bias_grad.detach()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        manual_report: dict[str, Any] = {
            "ok": True,
            "seconds": elapsed,
            "incremental_peak_allocated_gib": (
                torch.cuda.max_memory_allocated() - baseline_allocated
            )
            / 2**30,
            "bias_grad_finite": bool(torch.isfinite(bias_grad).all()),
            "bias_grad_norm": float(bias_grad.norm()),
            "bias_grad_nonzero": int(torch.count_nonzero(bias_grad)),
            "input_grad_finite": bool(torch.isfinite(input_grad).all()),
            "selected_experts": int(torch.unique(topk_ids).numel()),
            "weight_data_ptrs_unchanged": {
                "gate": gate_weight.data_ptr()
                == report["weights"]["gate"]["data_ptr"],
                "w13": w13.data_ptr() == report["weights"]["w13"]["data_ptr"],
                "w2": w2.data_ptr() == report["weights"]["w2"]["data_ptr"],
            },
        }
        if direct_output is not None and direct_output.shape == manual_output.shape:
            difference = (direct_output.detach().float() - manual_output.detach().float()).abs()
            manual_report["max_abs_diff_vs_vllm"] = float(difference.max())
            manual_report["mean_abs_diff_vs_vllm"] = float(difference.mean())
        report["manual_sidecar"] = manual_report
    except Exception as error:  # noqa: BLE001
        report["manual_sidecar"] = _error(error)

    # SonicMoE's general-routing API accepts the already-biased routing scores,
    # so gradients still reach the PIRA bias.  Its required weight order is a
    # permuted *view* of vLLM's Triton layout: no second model allocation.
    if test_sonicmoe:
        try:
            from sonicmoe import moe_general_routing_inputs
            from sonicmoe.enums import ActivationType

            def run_sonic_pass():
                with torch.inference_mode(False), torch.enable_grad():
                    x = x0.detach().requires_grad_(True)
                    bias = torch.zeros(
                        gate_weight.shape[0],
                        device=x.device,
                        dtype=torch.float32,
                        requires_grad=True,
                    )
                    logits = F.linear(x, gate_weight).float() + bias
                    probabilities = torch.softmax(
                        logits, dim=-1, dtype=torch.float32
                    )
                    scores, expert_ids = torch.topk(
                        probabilities, top_k, dim=-1
                    )
                    if renormalize:
                        scores = scores / scores.sum(dim=-1, keepdim=True)
                    token_ids = (
                        torch.arange(tokens, device=x.device, dtype=torch.int32)
                        .view(-1, 1)
                        .expand(tokens, top_k)
                        .reshape(-1)
                    )
                    w13_view = w13.permute(1, 2, 0)
                    w2_view = w2.permute(1, 2, 0)
                    sonic_output, frequencies = moe_general_routing_inputs(
                        x,
                        scores.reshape(-1),
                        token_ids,
                        expert_ids.reshape(-1).to(torch.int32),
                        w13_view,
                        None,
                        w2_view,
                        None,
                        int(gate_weight.shape[0]),
                        torch.cuda.current_stream().cuda_stream,
                        ActivationType.SWIGLU,
                        False,
                        True,
                    )
                    sonic_generator = torch.Generator(device=x.device).manual_seed(
                        seed + 1
                    )
                    direction = torch.randn(
                        sonic_output.shape[-1],
                        device=x.device,
                        dtype=torch.float32,
                        generator=sonic_generator,
                    )
                    input_grad, bias_grad = torch.autograd.grad(
                        (sonic_output.float() * direction).sum(), (x, bias)
                    )
                return (
                    sonic_output,
                    frequencies,
                    input_grad,
                    bias_grad,
                    w13_view,
                    w2_view,
                )

            baseline_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            first_pass = run_sonic_pass()
            torch.cuda.synchronize()
            first_elapsed = time.perf_counter() - started
            del first_pass

            # The first call may spend minutes JIT-compiling CuTeDSL/Triton.
            # A second call in the same worker is the reusable steady-state
            # number relevant to repeated probes.
            torch.cuda.synchronize()
            started = time.perf_counter()
            (
                sonic_output,
                frequencies,
                input_grad,
                bias_grad,
                w13_view,
                w2_view,
            ) = run_sonic_pass()
            torch.cuda.synchronize()
            steady_elapsed = time.perf_counter() - started
            sonic_report: dict[str, Any] = {
                "ok": True,
                "first_pass_seconds_including_jit": first_elapsed,
                "steady_state_seconds": steady_elapsed,
                "incremental_peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() - baseline_allocated
                )
                / 2**30,
                "output_finite": bool(torch.isfinite(sonic_output).all()),
                "input_grad_finite": bool(torch.isfinite(input_grad).all()),
                "bias_grad_finite": bool(torch.isfinite(bias_grad).all()),
                "bias_grad_norm": float(bias_grad.norm()),
                "bias_grad_nonzero": int(torch.count_nonzero(bias_grad)),
                "assignments": int(frequencies.sum()),
                "permuted_views_share_storage": {
                    "w13": w13_view.data_ptr() == w13.data_ptr(),
                    "w2": w2_view.data_ptr() == w2.data_ptr(),
                },
            }
            if manual_output is not None and manual_output.shape == sonic_output.shape:
                difference = (
                    sonic_output.detach().float() - manual_output.detach().float()
                ).abs()
                sonic_report["max_abs_diff_vs_manual"] = float(difference.max())
                sonic_report["mean_abs_diff_vs_manual"] = float(difference.mean())
                sonic_report["relative_l2_diff_vs_manual"] = float(
                    difference.norm() /
                    manual_output.detach().float().norm().clamp_min(1e-12)
                )
                sonic_report["numerically_close_vs_manual"] = bool(
                    torch.allclose(
                        sonic_output.detach().float(),
                        manual_output.detach().float(),
                        rtol=0.05,
                        atol=0.02,
                    )
                )
            if manual_input_grad is not None:
                input_grad_difference = (
                    input_grad.detach().float() - manual_input_grad.float()
                )
                sonic_report["input_grad_relative_l2_diff_vs_manual"] = float(
                    input_grad_difference.norm()
                    / manual_input_grad.float().norm().clamp_min(1e-12)
                )
            if manual_bias_grad is not None:
                bias_grad_difference = (
                    bias_grad.detach().float() - manual_bias_grad.float()
                )
                sonic_report["bias_grad_relative_l2_diff_vs_manual"] = float(
                    bias_grad_difference.norm()
                    / manual_bias_grad.float().norm().clamp_min(1e-12)
                )
                suppression_k = min(25, manual_bias_grad.numel())
                manual_suppression = set(
                    torch.topk(-manual_bias_grad.float(), suppression_k)
                    .indices.cpu()
                    .tolist()
                )
                sonic_suppression = set(
                    torch.topk(-bias_grad.detach().float(), suppression_k)
                    .indices.cpu()
                    .tolist()
                )
                sonic_report["bias_grad_bottom25_overlap_vs_manual"] = (
                    len(manual_suppression & sonic_suppression) / suppression_k
                )
            report["sonicmoe_sidecar"] = sonic_report
        except Exception as error:  # noqa: BLE001
            report["sonicmoe_sidecar"] = _error(error)

    report["conclusion"] = {
        "zero_copy_weight_reuse": bool(
            report.get("resident_weight_autograd", {}).get("ok")
            and report.get("manual_sidecar", {}).get("ok")
        ),
        "vllm_kernel_has_backward": bool(
            report.get("vllm_fused_backward", {}).get("ok")
        ),
        "sidecar_router_bias_gradient": bool(
            report.get("manual_sidecar", {}).get("bias_grad_finite")
        ),
        "sonicmoe_zero_copy_sidecar": bool(
            report.get("sonicmoe_sidecar", {}).get("ok")
            and report.get("sonicmoe_sidecar", {})
            .get("permuted_views_share_storage", {})
            .get("w13")
            and report.get("sonicmoe_sidecar", {})
            .get("permuted_views_share_storage", {})
            .get("w2")
            and report.get("sonicmoe_sidecar", {}).get(
                "numerically_close_vs_manual"
            )
            and report.get("sonicmoe_sidecar", {}).get(
                "input_grad_relative_l2_diff_vs_manual", float("inf")
            )
            < 0.02
            and report.get("sonicmoe_sidecar", {}).get(
                "bias_grad_relative_l2_diff_vs_manual", float("inf")
            )
            < 0.02
            and report.get("sonicmoe_sidecar", {}).get(
                "bias_grad_bottom25_overlap_vs_manual"
            )
            == 1.0
        ),
    }
    return report


def main() -> int:
    args = parse_args()
    if args.tokens <= 0:
        raise SystemExit("--tokens must be positive")

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(32, args.tokens),
        max_num_seqs=1,
        enable_prefix_caching=False,
        trust_remote_code=True,
        seed=args.seed,
        enforce_eager=True,
        kernel_config={"moe_backend": args.moe_backend},
    )
    reports = llm.collective_rpc(
        _worker_feasibility,
        args=(args.tokens, args.seed, args.test_sonicmoe),
    )
    payload = {
        "metadata": {
            "model": args.model,
            "tokens": args.tokens,
            "seed": args.seed,
            "moe_backend_requested": args.moe_backend,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "test_sonicmoe": args.test_sonicmoe,
        },
        "workers": reports,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}", flush=True)

    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    return 0 if reports and reports[0]["conclusion"]["zero_copy_weight_reuse"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
