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


def _describe_layout(block) -> str | None:
    """Which Hugging Face MoE layout this block uses, or None if unrecognised.

    Two shapes exist in the wild and both must be handled, because which one a run
    sees depends only on the installed transformers version:

      "modulelist"  older: block.experts is an nn.ModuleList of Qwen3MoeMLP, each
                    with gate_proj / up_proj / down_proj Linears, and the block
                    itself carries top_k / num_experts / norm_topk_prob.
      "fused"       newer: block.experts is a single Qwen3MoeExperts holding
                    stacked nn.Parameters (gate_up_proj [E, 2I, H], down_proj
                    [E, H, I]), and routing moved into a Qwen3MoeTopKRouter that
                    returns (router_logits, router_scores, router_indices).

    The B200 run failed here: it had the fused layout, the matcher only knew the
    modulelist one, and every layer was skipped with "no MoE block matched".
    """
    experts = getattr(block, "experts", None)
    gate = getattr(block, "gate", None)
    if experts is None or gate is None:
        return None
    if isinstance(experts, torch.nn.ModuleList) and len(experts) and hasattr(
        experts[0], "gate_proj"
    ):
        return "modulelist" if hasattr(block, "top_k") else None
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        return "fused"
    return None


def _stack_expert_weights(block, layout: str) -> dict[str, torch.Tensor]:
    """Return per-expert weights as contiguous [E, in, out] tensors.

    Grouped GEMM indexes expert i by offset into one tensor, so weights are read
    once per expert rather than copied per token -- the property that keeps memory
    flat where a batched matmul reached 145 GiB at 2048 tokens.

    Cached on the block: for the fused layout this is a transposed view of
    parameters that are already stacked, and for modulelist it is a real copy, so
    paying it once rather than per forward pass matters.
    """
    cached = getattr(block, "_pira_stacked", None)
    if cached is not None:
        return cached

    if layout == "modulelist":
        experts = block.experts
        stacked = {
            "gate": torch.stack([e.gate_proj.weight.t().contiguous() for e in experts]),
            "up": torch.stack([e.up_proj.weight.t().contiguous() for e in experts]),
            "down": torch.stack([e.down_proj.weight.t().contiguous() for e in experts]),
        }
    else:
        # gate_up_proj is [E, 2I, H] with gate and up concatenated along dim 1;
        # F.linear uses it as [out, in], so transposing gives [E, H, I] per half.
        experts = block.experts
        gate_up = experts.gate_up_proj
        intermediate = gate_up.shape[1] // 2
        stacked = {
            "gate": gate_up[:, :intermediate, :].transpose(1, 2).contiguous(),
            "up": gate_up[:, intermediate:, :].transpose(1, 2).contiguous(),
            "down": experts.down_proj.transpose(1, 2).contiguous(),
        }
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


def _make_grouped_forward(matmul, layout: str):
    """Build a replacement MoE forward that routes through `matmul`.

    The signature and return value match the layout being replaced, so this is a
    drop-in swap and PIRA's router-bias hook keeps working untouched. Routing
    itself is delegated to Hugging Face's own code in both layouts, which is what
    keeps the bias-injection point identical -- only the expert matmuls change.
    """

    def grouped_forward(self, hidden_states: torch.Tensor):
        original_shape = hidden_states.shape
        hidden_dim = original_shape[-1]
        flat = hidden_states.reshape(-1, hidden_dim)

        if layout == "fused":
            # The router module owns softmax/top-k and returns normalized scores
            # plus indices. Calling it (rather than reimplementing it) is what
            # guarantees PIRA's bias still lands where it did before.
            router_logits, router_scores, router_indices = self.gate(flat)
            selected = router_indices
            routing_weights = router_scores.to(flat.dtype)
            num_experts = self.experts.num_experts
            act_fn = self.experts.act_fn
        else:
            router_logits = self.gate(flat)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(
                    dim=-1, keepdim=True
                )
            routing_weights = routing_weights.to(flat.dtype)
            num_experts = self.num_experts
            act_fn = self.act_fn

        stacked = _stack_expert_weights(self, layout)
        order, source_rows, counts = _sort_by_expert(selected, num_experts)

        gathered = flat.index_select(0, source_rows)
        gate_out = matmul(gathered, stacked["gate"], counts)
        up_out = matmul(gathered, stacked["up"], counts)
        expert_out = matmul(act_fn(gate_out) * up_out, stacked["down"], counts)

        # Scale by each pair's gate weight, then scatter-add back to token rows.
        weights = routing_weights.reshape(-1).index_select(0, order).unsqueeze(-1)
        contribution = (expert_out * weights).to(flat.dtype)
        out = torch.zeros_like(flat).index_add(0, source_rows, contribution)
        out = out.reshape(original_shape)

        # Older blocks return (output, router_logits); the fused block returns
        # only the output, with logits surfaced by the router itself.
        return out if layout == "fused" else (out, router_logits)

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
        if _describe_layout(block) is not None:
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

    matmul = _MATMUL_BACKENDS[backend]
    patched: list[tuple[torch.nn.Module, object]] = []
    seen_layouts: set[str] = set()
    inspected = 0

    all_layers = getattr(getattr(model, "model", None), "layers", [])
    for index, layer in enumerate(all_layers):
        if layers is not None and index not in layers:
            continue
        block = getattr(layer, "mlp", None)
        if block is None:
            continue
        inspected += 1
        layout = _describe_layout(block)
        if layout is None:
            continue
        seen_layouts.add(layout)
        patched.append((block, block.forward))
        forward = _make_grouped_forward(matmul, layout)
        block.forward = forward.__get__(block, type(block))

    if not patched:
        # Report what was actually found: the failure mode this replaces was a
        # bare "no MoE block matched", which gave no way to tell an unsupported
        # transformers layout from a model that has no MoE layers at all.
        detail = []
        for index, layer in enumerate(all_layers[:3]):
            block = getattr(layer, "mlp", None)
            if block is None:
                detail.append(f"layer {index}: no .mlp")
                continue
            experts = getattr(block, "experts", None)
            detail.append(
                f"layer {index}: {type(block).__name__} "
                f"experts={type(experts).__name__ if experts is not None else None} "
                f"attrs={[a for a in ('gate', 'top_k', 'num_experts') if hasattr(block, a)]}"
            )
        raise GroupedMoEUnavailable(
            f"no MoE block matched a known layout ({inspected} candidates "
            f"inspected). Supported: a ModuleList of per-expert MLPs "
            f"(gate_proj/up_proj/down_proj), or a fused Qwen3MoeExperts with "
            f"stacked gate_up_proj/down_proj parameters. Found: "
            + "; ".join(detail)
        )

    try:
        yield len(patched)
    finally:
        for block, original in patched:
            block.forward = original
        if not keep_stacked:
            free_stacked_weights(model)


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #


def _self_check() -> int:
    """Verify both Hugging Face MoE layouts are matched and computed correctly.

    Runs on CPU with no model download. Exists because a B200 run was wasted when
    transformers changed Qwen3MoeSparseMoeBlock from an nn.ModuleList of MLPs to a
    fused Qwen3MoeExperts with stacked parameters: the matcher only knew the old
    shape, so every layer was skipped and the comparison produced nothing.
    """
    import torch.nn as nn

    hidden, intermediate, num_experts, top_k = 32, 24, 8, 3
    failures = []

    def check(name, model, call):
        block = model.model.layers[0].mlp
        layout = _describe_layout(block)
        if layout is None:
            failures.append(f"{name}: layout not recognised")
            return
        reference = call(block)
        with grouped_moe(model, backend="torch") as patched:
            if patched != len(model.model.layers):
                failures.append(f"{name}: patched {patched} of {len(model.model.layers)}")
            grouped = call(block)
        restored = call(block)

        ref_t = reference[0] if isinstance(reference, tuple) else reference
        got_t = grouped[0] if isinstance(grouped, tuple) else grouped
        res_t = restored[0] if isinstance(restored, tuple) else restored
        difference = (ref_t - got_t).abs().amax().item()
        print(f"  {name:<12} layout={layout:<11} max abs diff {difference:.3e}")
        if difference > 1e-5:
            failures.append(f"{name}: grouped output differs by {difference:.3e}")
        if type(reference) is not type(grouped):
            failures.append(f"{name}: return type changed")
        if not torch.equal(ref_t, res_t):
            failures.append(f"{name}: original forward not restored")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class ModuleListBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts, self.top_k, self.norm_topk_prob = num_experts, top_k, True
            self.gate = nn.Linear(hidden, num_experts, bias=False)
            self.experts = nn.ModuleList(MLP() for _ in range(num_experts))
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states):
            b, s, d = hidden_states.shape
            flat = hidden_states.view(-1, d)
            logits = self.gate(flat)
            weights = torch.softmax(logits, dim=1, dtype=torch.float)
            weights, selected = torch.topk(weights, self.top_k, dim=-1)
            weights = (weights / weights.sum(-1, keepdim=True)).to(flat.dtype)
            out = torch.zeros_like(flat)
            mask = nn.functional.one_hot(selected, num_experts).permute(2, 1, 0)
            for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
                pos, token = torch.where(mask[expert].squeeze(0))
                current = flat[None, token].reshape(-1, d)
                out = out.index_add(
                    0, token,
                    (self.experts[expert](current) * weights[token, pos, None]).to(out.dtype),
                )
            return out.reshape(b, s, d), logits

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k, self.num_experts = top_k, num_experts
            self.weight = nn.Parameter(torch.randn(num_experts, hidden) * 0.3)

        def forward(self, h):
            logits = nn.functional.linear(h, self.weight)
            probs = torch.softmax(logits, dtype=torch.float, dim=-1)
            value, index = torch.topk(probs, self.top_k, dim=-1)
            return logits, value / value.sum(-1, keepdim=True), index

    class FusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts, self.act_fn = num_experts, nn.SiLU()
            self.gate_up_proj = nn.Parameter(
                torch.randn(num_experts, 2 * intermediate, hidden) * 0.05
            )
            self.down_proj = nn.Parameter(
                torch.randn(num_experts, hidden, intermediate) * 0.05
            )

        def forward(self, h, index, weights):
            out = torch.zeros_like(h)
            mask = nn.functional.one_hot(index, self.num_experts).permute(2, 1, 0)
            for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
                expert = expert[0]
                pos, token = torch.where(mask[expert])
                gate, up = nn.functional.linear(
                    h[token], self.gate_up_proj[expert]
                ).chunk(2, dim=-1)
                current = nn.functional.linear(
                    self.act_fn(gate) * up, self.down_proj[expert]
                ) * weights[token, pos, None]
                out = out.index_add(0, token, current.to(out.dtype))
            return out

    class FusedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts, self.gate = FusedExperts(), Router()

        def forward(self, hidden_states):
            shape = hidden_states.shape
            flat = hidden_states.reshape(-1, shape[-1])
            _, scores, index = self.gate(flat)
            return self.experts(flat, index, scores.to(flat.dtype)).reshape(shape)

    def build(block_cls):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = block_cls()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList(Layer() for _ in range(3))

        return Model().eval().requires_grad_(False)

    torch.manual_seed(0)
    x = torch.randn(2, 5, hidden)
    print("backends:", available_backends())
    check("modulelist", build(ModuleListBlock), lambda b: b(x))
    check("fused", build(FusedBlock), lambda b: b(x))

    print()
    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}")
        print("SELF-CHECK FAILED")
        return 1
    print("SELF-CHECK PASSED (both layouts matched, computed and restored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
