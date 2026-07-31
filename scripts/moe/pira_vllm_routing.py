"""Apply PIRA's per-request router bias inside vLLM's expert-selection path.

Why this file exists
--------------------
PIRA's generation-time intervention is an additive bias on router logits applied
before softmax and top-k. vLLM already supports an additive router bias --
``e_score_correction_bias``, used by DeepSeek-V3 -- but it is not the same
operator PIRA needs:

    e_score_correction_bias        PIRA's bias
    ----------------------        -----------
    per expert, shape [E]         per *request*, shape [num_requests, E]
    applied after softmax         applied before softmax
    selection only; gate weights  gate weights come from the biased
    read the unbiased scores      distribution

The shapes differ because PIRA's bias is query-specific: two requests in the
same batch get different biases, and vLLM flattens all requests' tokens into one
``[num_tokens, E]`` router-logit matrix. So the bias has to be scattered from
requests to token rows using the batch's token->request mapping.

That the *mechanism* is already first-class in vLLM is the relevant efficiency
point: adding a bias to router logits before top-k costs nothing measurable next
to the expert MLPs, and top-k itself is unchanged, so the generation phase does
the same FLOPs as ordinary generation.

How it hooks in
---------------
``MoERunner.router.select_experts()`` is the single choke point through which
every modular-kernel MoE layer selects its experts, so wrapping it per layer is
enough and requires no fork of vLLM. Monolithic kernels take a different path
(``routed_experts.forward_monolithic``) that never calls ``select_experts``; that
case is detected and raises rather than silently skipping the intervention.

Set the active bias for the current batch with ``set_request_bias`` before
calling the engine, using a token->request mapping derived from the forward
context. Because vLLM's scheduler decides batch composition, the mapping has to
be read per forward pass, which ``_TokenRouting`` does from the attention
metadata's ``query_start_loc``.

Status: this is the integration path for end-to-end serving measurements. Safety
numbers in the paper come from the Hugging Face implementation; this module is
about throughput and latency, and includes a self-check
(``verify_against_reference``) that its biased selection matches the reference
top-k for the same logits and bias.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Iterator

import torch


# --------------------------------------------------------------------------- #
# Per-batch bias state
# --------------------------------------------------------------------------- #


@dataclass
class PiraBiasState:
    """The bias to apply for the batch currently in flight.

    bias_by_layer[l] has shape [num_requests, num_experts]. request_ids gives the
    engine-side request id for each row, so rows can be reordered to match the
    scheduler's batch order.
    """

    bias_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    request_ids: list[str] = field(default_factory=list)
    beta: float = 10.0

    def row_of(self, request_id: str) -> int | None:
        try:
            return self.request_ids.index(request_id)
        except ValueError:
            return None

    @property
    def num_requests(self) -> int:
        for tensor in self.bias_by_layer.values():
            return tensor.shape[0]
        return 0


class _TokenRouting:
    """Derive the token row -> request row mapping for the current forward pass."""

    @staticmethod
    def from_forward_context(num_token_rows: int) -> torch.Tensor | None:
        """Build a [num_token_rows] int64 tensor of request indices, or None.

        vLLM's attention metadata carries ``query_start_loc``, a
        ``[num_requests + 1]`` prefix sum of each request's token count in the
        flattened batch. repeat_interleave over its differences turns that into
        a per-token request index. Returns None when the context is unavailable
        (e.g. during profiling runs or CUDA-graph capture with dummy inputs), and
        the caller then falls back to an even split.
        """
        try:
            from vllm.forward_context import (
                get_forward_context,
                is_forward_context_available,
            )
        except ImportError:
            return None
        if not is_forward_context_available():
            return None
        try:
            context = get_forward_context()
        except (AssertionError, RuntimeError):
            return None

        metadata = getattr(context, "attn_metadata", None)
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else None
        if isinstance(metadata, dict):
            metadata = next(iter(metadata.values()), None)
        if metadata is None:
            return None

        starts = getattr(metadata, "query_start_loc", None)
        if starts is None or starts.numel() < 2:
            return None

        counts = (starts[1:] - starts[:-1]).to(torch.int64)
        index = torch.arange(counts.numel(), device=counts.device, dtype=torch.int64)
        mapping = index.repeat_interleave(counts)
        if mapping.numel() == num_token_rows:
            return mapping
        if mapping.numel() > num_token_rows:
            # The batch is padded (CUDA graphs round the token count up).
            return mapping[:num_token_rows]
        # Pad rows belong to no request; point them at row 0 and rely on the
        # engine discarding their output.
        padding = torch.zeros(
            num_token_rows - mapping.numel(),
            device=mapping.device,
            dtype=torch.int64,
        )
        return torch.cat([mapping, padding])


# --------------------------------------------------------------------------- #
# Biased selection
# --------------------------------------------------------------------------- #


def biased_select(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool,
    scoring_func: str = "softmax",
    tokens_to_request: torch.Tensor | None = None,
    indices_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PIRA expert selection: bias the logits, then softmax and top-k.

    This mirrors the Hugging Face reference exactly: the bias is added to the raw
    logits, the scoring function is applied to the biased logits, and the gate
    weights are read from that same biased distribution (not from the unbiased
    one, which is what vLLM's e_score_correction_bias would do).
    """
    logits = router_logits.float()
    rows = logits.shape[0]

    if bias.ndim == 2:
        if tokens_to_request is not None:
            expanded = bias.index_select(0, tokens_to_request[:rows].to(bias.device))
        else:
            requests = bias.shape[0]
            if rows % requests:
                raise ValueError(
                    f"{rows} router rows are not divisible by {requests} requests "
                    "and no token->request mapping was available"
                )
            expanded = bias.repeat_interleave(rows // requests, dim=0)
    else:
        expanded = bias
    logits = logits + expanded.to(logits.dtype)

    if scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif scoring_func == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise ValueError(f"unsupported scoring function {scoring_func!r}")

    weights, indices = torch.topk(scores, top_k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return (
        weights.to(torch.float32),
        indices.to(torch.int32 if indices_dtype is None else indices_dtype),
    )


# --------------------------------------------------------------------------- #
# Installation into a live vLLM engine
# --------------------------------------------------------------------------- #


def _iter_moe_runners(model) -> Iterator[tuple[str, object]]:
    """Yield (name, module) for modules exposing a router with select_experts."""
    for name, module in model.named_modules():
        router = getattr(module, "router", None)
        if router is not None and hasattr(router, "select_experts"):
            yield name, module


def _layer_index_of(name: str) -> int | None:
    """Recover the decoder layer index from a module's dotted path."""
    parts = name.split(".")
    for token, following in zip(parts, parts[1:]):
        if token in ("layers", "h", "blocks") and following.isdigit():
            return int(following)
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None


@contextlib.contextmanager
def pira_routing(model, state: PiraBiasState, *, strict: bool = True):
    """Install PIRA biased routing on every MoE layer that has a bias.

    strict=True raises if a MoE layer would bypass select_experts (monolithic
    kernel), because silently running unbiased there would report PIRA's latency
    while delivering none of its behaviour.
    """
    patched: list[tuple[object, object]] = []
    installed: list[int] = []
    skipped: list[str] = []

    for name, runner in _iter_moe_runners(model):
        layer_index = _layer_index_of(name)
        if layer_index is None or layer_index not in state.bias_by_layer:
            continue

        monolithic = getattr(runner, "_is_monolithic", None)
        if monolithic is None:
            routed = getattr(runner, "routed_experts", None)
            monolithic = bool(
                routed is not None
                and hasattr(routed, "forward_monolithic")
                and not hasattr(routed, "forward_modular")
            )
        if monolithic:
            skipped.append(name)
            continue

        router = runner.router
        original = router.select_experts
        top_k = getattr(router, "top_k", None)
        renormalize = bool(getattr(router, "renormalize", True))
        scoring = str(getattr(router, "scoring_func", "softmax"))
        if top_k is None:
            skipped.append(name)
            continue

        def patched_select(
            hidden_states,
            router_logits,
            topk_indices_dtype=None,
            *,
            input_ids=None,
            _layer=layer_index,
            _top_k=top_k,
            _renorm=renormalize,
            _scoring=scoring,
            **kwargs,
        ):
            bias = state.bias_by_layer.get(_layer)
            if bias is None:
                return original(
                    hidden_states,
                    router_logits,
                    topk_indices_dtype=topk_indices_dtype,
                    input_ids=input_ids,
                    **kwargs,
                )
            mapping = _TokenRouting.from_forward_context(router_logits.shape[0])
            return biased_select(
                router_logits,
                bias,
                _top_k,
                renormalize=_renorm,
                scoring_func=_scoring,
                tokens_to_request=mapping,
                indices_dtype=topk_indices_dtype,
            )

        router.select_experts = patched_select
        patched.append((router, original))
        installed.append(layer_index)

    if skipped and strict:
        for router, original in patched:
            router.select_experts = original
        raise RuntimeError(
            "these MoE layers bypass select_experts (monolithic kernel), so the "
            "PIRA bias could not be applied: "
            f"{skipped[:4]}{'...' if len(skipped) > 4 else ''}. Run with a "
            "modular MoE backend, or pass strict=False to measure only the "
            "layers that can be intervened on."
        )

    try:
        yield sorted(installed)
    finally:
        for router, original in patched:
            router.select_experts = original


def set_request_bias(
    state: PiraBiasState,
    bias_by_layer: dict[int, torch.Tensor],
    request_ids: list[str] | None = None,
) -> None:
    """Replace the active bias. Call once per batch, before submitting it."""
    state.bias_by_layer = bias_by_layer
    if request_ids is not None:
        state.request_ids = list(request_ids)


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #


def verify_against_reference(
    *,
    num_tokens: int = 64,
    num_requests: int = 4,
    num_experts: int = 128,
    top_k: int = 8,
    beta: float = 10.0,
    suppressed_per_request: int = 25,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Check biased_select against an explicit per-row reference computation.

    Runs on CPU, so this is checkable without a GPU or a vLLM install. It
    verifies that (a) selection matches a straightforward per-row loop, and
    (b) suppressed experts are actually driven out of the top-k, which is the
    behaviour the intervention depends on.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        num_tokens, num_experts, generator=generator, device=device, dtype=torch.float32
    )
    bias = torch.zeros(num_requests, num_experts, device=device, dtype=torch.float32)
    for request in range(num_requests):
        victims = torch.randperm(num_experts, generator=generator, device=device)[
            :suppressed_per_request
        ]
        bias[request, victims] = -abs(beta)

    tokens_per_request = num_tokens // num_requests
    mapping = torch.arange(num_requests, device=device).repeat_interleave(
        tokens_per_request
    )

    weights, indices = biased_select(
        logits,
        bias,
        top_k,
        renormalize=True,
        scoring_func="softmax",
        tokens_to_request=mapping,
    )

    # Reference: one row at a time, no vectorization.
    mismatches = 0
    suppressed_hits = 0
    for row in range(num_tokens):
        request = int(mapping[row])
        biased = logits[row] + bias[request]
        probs = torch.softmax(biased, dim=-1, dtype=torch.float32)
        ref_weights, ref_indices = torch.topk(probs, top_k, dim=-1)
        ref_weights = ref_weights / ref_weights.sum().clamp_min(1e-20)
        if not torch.equal(ref_indices.to(indices.dtype), indices[row]):
            mismatches += 1
        elif not torch.allclose(ref_weights, weights[row], atol=1e-6, rtol=1e-5):
            mismatches += 1
        suppressed_hits += int((bias[request][indices[row].long()] < 0).sum())

    # Unbiased selection, to confirm the bias actually changed something.
    base_weights, base_indices = biased_select(
        logits,
        torch.zeros_like(bias),
        top_k,
        renormalize=True,
        tokens_to_request=mapping,
    )
    changed = int((base_indices != indices).any(dim=1).sum())

    return {
        "rows_checked": num_tokens,
        "mismatches_vs_reference": mismatches,
        "selected_suppressed_experts": suppressed_hits,
        "rows_whose_selection_changed": changed,
        "passed": mismatches == 0 and suppressed_hits == 0,
    }


if __name__ == "__main__":
    report = verify_against_reference()
    for key, value in report.items():
        print(f"{key}: {value}")
    print("PASS" if report["passed"] else "FAIL")
    raise SystemExit(0 if report["passed"] else 1)
