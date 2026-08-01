"""Swap Hugging Face's per-expert Python loop for a grouped-GEMM MoE.

Why
---
`Qwen3MoeSparseMoeBlock.forward` loops over experts in Python:

    for expert_idx in expert_hit:                  # up to 128 iterations
        idx, top_x = torch.where(...)
        expert_layer(current_state) * routing_weights[...]

At 128 prompt tokens that is roughly 768 kernel launches per layer, each doing a
[8, 2048] x [2048, 768] matmul: about a microsecond of arithmetic behind five to
ten microseconds of launch overhead. Across the 25 probed layers plus backward it
is ~58k launches, which is why the probe measures ~1.13 s while its arithmetic is
only ~1.09 TFLOP. Measured in isolation on a B200, a grouped GEMM runs the same
layer 17-31x faster and uses less memory.

What this module changes, and what it does not
----------------------------------------------
Only *how* the expert matmuls are executed. Unchanged:

  * the router (`self.gate`) and its logits,
  * softmax, top-k, and `norm_topk_prob` renormalization,
  * therefore PIRA's bias-injection point, which is the router logits,
  * the mathematical result: out[t] = sum_s g[t,s] * down_s(silu(gate_s(x_t)) * up_s(x_t)).

The one thing that does change is floating-point *reduction order*: grouped GEMM
requires tokens sorted by expert, and the per-slot accumulation happens in a
different sequence than `index_add_` over experts. In bf16 that is a ~1e-3
relative perturbation, and PIRA's top-K suppression set is a discrete argmin, so
a perturbation can in principle reorder experts whose gradients are close. That
is an empirical question, not something to assume either way --
`compare_probe_backends.py` measures it directly on the real model.

Backends
--------
    grouped_gemm  tgale96/grouped_gemm (the MegaBlocks kernel). Verified to build
                  and run on B200/sm_100 despite its source naming
                  ::cutlass::arch::Sm80, and it asserts bfloat16.
    torch         a pure-PyTorch grouped path using segmented matmuls. Slower than
                  the CUDA kernel but dependency-free, and useful to separate "the
                  grouping is wrong" from "the kernel is unavailable".

Usage
-----
    from pira_grouped_moe import grouped_moe

    with grouped_moe(model, backend="grouped_gemm"):
        result = FastProbe(model, probe_layer, direction).run(...)
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch
import torch.nn.functional as F


class GroupedMoEUnavailable(RuntimeError):
    """Raised when the requested grouped backend cannot be used."""


# --------------------------------------------------------------------------- #
# Weight stacking
# --------------------------------------------------------------------------- #


def _stack_expert_weights(block) -> dict[str, torch.Tensor]:
    """Stack a block's per-expert Linear weights into contiguous 3D tensors.

    Hugging Face stores experts as a ModuleList of Qwen3MoeMLP, each holding
    [out, in] Linear weights. Grouped GEMM wants one [E, in, out] tensor so the
    kernel can index expert i by offset -- crucially without copying a weight per
    token, which is what makes a batched matmul blow up memory.

    Cached on the block, so the stacking cost is paid once rather than per layer
    per forward pass.
    """
    cached = getattr(block, "_pira_stacked", None)
    if cached is not None:
        return cached

    experts = block.experts
    gate = torch.stack([expert.gate_proj.weight.t().contiguous() for expert in experts])
    up = torch.stack([expert.up_proj.weight.t().contiguous() for expert in experts])
    down = torch.stack([expert.down_proj.weight.t().contiguous() for expert in experts])
    stacked = {"gate": gate, "up": up, "down": down}
    block._pira_stacked = stacked
    return stacked


def free_stacked_weights(model) -> int:
    """Drop cached stacked weights. Returns how many blocks were cleared."""
    cleared = 0
    for block in _iter_moe_blocks(model):
        if getattr(block, "_pira_stacked", None) is not None:
            del block._pira_stacked
            cleared += 1
    return cleared


def stacked_weight_bytes(model) -> int:
    """Extra memory held by the stacked copies, for reporting."""
    total = 0
    for block in _iter_moe_blocks(model):
        stacked = getattr(block, "_pira_stacked", None)
        if stacked is not None:
            total += sum(t.numel() * t.element_size() for t in stacked.values())
    return total


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #


def _sort_by_expert(topk_ids: torch.Tensor, num_experts: int):
    """Order (token, slot) pairs by expert.

    Returns the permutation, the source token row for each sorted pair, and the
    per-expert counts that grouped GEMM consumes as `batch_sizes`.

    stable=True keeps pairs of the same expert in token order, which makes the
    reduction order deterministic across runs -- otherwise repeated runs could
    disagree, and any comparison against a reference would be meaningless.
    """
    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    source_rows = order // topk_ids.shape[1]
    return order, source_rows, counts


def _grouped_matmul_torch(
    inputs: torch.Tensor, weights: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Pure-PyTorch grouped matmul: one matmul per non-empty expert segment.

    Still a Python loop, but over *segments of contiguous rows* rather than over
    scattered token indices, so it does far fewer, much larger matmuls than the
    Hugging Face path and needs no compiled dependency.
    """
    outputs = inputs.new_empty(inputs.shape[0], weights.shape[-1])
    offsets = torch.cumsum(counts, 0)
    start = 0
    for expert, end in enumerate(offsets.tolist()):
        if end > start:
            outputs[start:end] = inputs[start:end] @ weights[expert]
        start = end
    return outputs


def _grouped_matmul_gg(
    inputs: torch.Tensor, weights: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """tgale96/grouped_gemm. Autograd-aware, so backward comes for free."""
    from grouped_gemm import ops as gg_ops

    # The kernel wants int64 counts on the host.
    return gg_ops.gmm(inputs, weights, counts.to(torch.int64).cpu())


_MATMUL_BACKENDS = {
    "grouped_gemm": _grouped_matmul_gg,
    "torch": _grouped_matmul_torch,
}


# --------------------------------------------------------------------------- #
# Replacement forward
# --------------------------------------------------------------------------- #


def _make_grouped_forward(matmul):
    """Build a Qwen3MoeSparseMoeBlock.forward that routes through `matmul`.

    The signature and return value match Hugging Face exactly -- including
    returning router_logits, which the model's auxiliary-loss path expects -- so
    this is a drop-in replacement and PIRA's router-bias hook keeps working
    untouched.
    """

    def grouped_forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)

        # Router, softmax and top-k are byte-for-byte the Hugging Face path. PIRA
        # injects its bias into these logits, so this part must not change.
        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(flat.dtype)

        stacked = _stack_expert_weights(self)
        order, source_rows, counts = _sort_by_expert(selected, self.num_experts)

        gathered = flat.index_select(0, source_rows)
        gate_out = matmul(gathered, stacked["gate"], counts)
        up_out = matmul(gathered, stacked["up"], counts)
        expert_out = matmul(self.act_fn(gate_out) * up_out, stacked["down"], counts)

        # Scale by each pair's gate weight, then scatter-add back to token rows.
        weights = routing_weights.reshape(-1).index_select(0, order).unsqueeze(-1)
        contribution = (expert_out * weights).to(flat.dtype)
        out = torch.zeros_like(flat).index_add(0, source_rows, contribution)

        return out.reshape(batch_size, sequence_length, hidden_dim), router_logits

    return grouped_forward


def _iter_moe_blocks(model) -> Iterator[torch.nn.Module]:
    """Yield MoE blocks that expose the per-expert-loop structure."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for layer in layers:
        block = getattr(layer, "mlp", None)
        if block is None:
            continue
        if hasattr(block, "experts") and hasattr(block, "gate") and hasattr(block, "top_k"):
            yield block


def available_backends() -> dict[str, str]:
    """Which grouped backends can be used here."""
    status = {"torch": "available"}
    try:
        import grouped_gemm  # noqa: F401

        status["grouped_gemm"] = "available"
    except Exception as error:  # noqa: BLE001
        status["grouped_gemm"] = f"unavailable: {type(error).__name__}: {error}"
    return status


@contextlib.contextmanager
def grouped_moe(
    model,
    *,
    backend: str = "grouped_gemm",
    layers: set[int] | None = None,
    keep_stacked: bool = False,
):
    """Route the model's MoE blocks through a grouped GEMM for the duration.

    layers restricts the swap to those decoder indices; None means all. Only the
    probed layers matter for PIRA, but swapping all of them is harmless.

    keep_stacked leaves the stacked weight copies allocated on exit, so repeated
    runs do not pay the stacking cost again. They cost about one extra copy of the
    expert weights (~1.1 GiB for Qwen3-30B-A3B).
    """
    if backend not in _MATMUL_BACKENDS:
        raise GroupedMoEUnavailable(
            f"unknown backend {backend!r}; expected one of {sorted(_MATMUL_BACKENDS)}"
        )
    if backend == "grouped_gemm":
        try:
            import grouped_gemm  # noqa: F401
        except Exception as error:  # noqa: BLE001
            raise GroupedMoEUnavailable(
                f"grouped_gemm is not importable: {type(error).__name__}: {error}"
            ) from error

    forward = _make_grouped_forward(_MATMUL_BACKENDS[backend])
    patched: list[tuple[torch.nn.Module, object]] = []

    all_layers = getattr(getattr(model, "model", None), "layers", [])
    for index, layer in enumerate(all_layers):
        if layers is not None and index not in layers:
            continue
        block = getattr(layer, "mlp", None)
        if block is None or not (
            hasattr(block, "experts") and hasattr(block, "gate") and hasattr(block, "top_k")
        ):
            continue
        patched.append((block, block.forward))
        block.forward = forward.__get__(block, type(block))

    if not patched:
        raise GroupedMoEUnavailable(
            "no MoE block matched the expected structure "
            "(needs .experts, .gate and .top_k)"
        )

    try:
        yield len(patched)
    finally:
        for block, original in patched:
            block.forward = original
        if not keep_stacked:
            free_stacked_weights(model)
