"""PIRA probe: a reference implementation and an exactness-preserving fast path.

The probe computes, for one or more requests, the gradient of a scalar safety
score with respect to per-request additive router biases:

    s(q)              = <d_safe, h_L[last token]>
    Delta b_{l,i}(q)  = d s(q) / d b_{l,i}

Both implementations here compute that same quantity. They differ only in *how*
the forward and backward passes are executed, never in what is computed:

    ReferenceProbe   plain autograd over the full prompt forward, activations
                     retained, fp32 math where the model allows it. Slow, and
                     deliberately simple enough to audit line by line.

    FastProbe        same graph, executed better:
                       (a) layer truncation   -- layers above L cannot affect
                           s(q), so they are never run. Exact by construction.
                       (b) grad checkpointing -- recompute layer activations in
                           the backward pass instead of storing them. Exact for
                           deterministic layers; makes peak activation memory
                           O(1) in depth instead of O(L).
                       (c) memory-efficient attention backward -- O(S) instead
                           of O(S^2) attention workspace.
                       (d) batching + early graph release.

None of these is an approximation. There is no closed form, no surrogate
direction, and no dropped term: the backward pass is real autograd. The claim
that (a)-(d) preserve the result is not asserted, it is *tested* --- see
test_probe_equivalence.py, which requires bitwise-comparable gradients, an
identical top-K suppression set, and Spearman rho = 1.0 against the reference.

Only the router-bias leaves require grad; model parameters stay frozen.
"""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import torch
import torch.nn.functional as F


class _StopForward(RuntimeError):
    """Control-flow signal used to abandon the forward pass after layer L."""


# --------------------------------------------------------------------------- #
# Router bias injection
# --------------------------------------------------------------------------- #


def find_moe_gates(model) -> list[tuple[int, torch.nn.Module]]:
    """Return [(layer_index, gate_module)] for every MoE layer, in order."""
    gates: list[tuple[int, torch.nn.Module]] = []
    for index, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None) if mlp is not None else None
        if gate is not None:
            gates.append((index, gate))
    if not gates:
        raise RuntimeError("no MoE gate found at model.model.layers[*].mlp.gate")
    return gates


def num_experts_of(gate: torch.nn.Module) -> int:
    for attr in ("num_experts", "n_routed_experts", "out_features"):
        value = getattr(gate, attr, None)
        if isinstance(value, int):
            return value
    weight = getattr(gate, "weight", None)
    if weight is not None:
        return weight.shape[0]
    raise RuntimeError(f"cannot infer expert count for {type(gate)!r}")


@contextlib.contextmanager
def router_bias(
    gates: Sequence[tuple[int, torch.nn.Module]],
    bias_by_layer: dict[int, torch.Tensor],
    *,
    tokens_to_request: torch.Tensor | None = None,
) -> Iterator[None]:
    """Add a per-request bias to router logits *before* softmax and top-k.

    ``bias_by_layer[l]`` has shape ``[num_requests, num_experts]``. Router logits
    arrive flattened as ``[num_tokens, num_experts]``; every token of a request
    receives that request's bias. ``tokens_to_request`` maps flattened token rows
    to request indices, and is required when requests have unequal token counts
    (the ragged case). When it is None, tokens are assumed to be evenly divided,
    which is what fixed-length padded benchmark batches produce.

    Applying the bias pre-softmax and keeping the gate weights on the *biased*
    distribution is what the paper specifies; this differs from vLLM's
    ``e_score_correction_bias``, which is per-expert, post-softmax, and used for
    selection only. See pira_vllm_routing.py.
    """

    saved: list[tuple[torch.nn.Module, Callable]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def expand(bias: torch.Tensor, rows: int) -> torch.Tensor:
        if bias.ndim == 1:
            return bias
        if tokens_to_request is not None:
            return bias.index_select(0, tokens_to_request[:rows])
        requests = bias.shape[0]
        if rows % requests:
            raise ValueError(
                f"{rows} router rows not divisible by {requests} requests; "
                "pass tokens_to_request for ragged batches"
            )
        return bias.repeat_interleave(rows // requests, dim=0)

    for layer_index, gate in gates:
        if layer_index not in bias_by_layer:
            continue
        bias = bias_by_layer[layer_index]

        # Qwen3-MoE style gate: returns (logits, weights, indices). Replace the
        # forward so the bias lands before softmax/top-k.
        if all(
            hasattr(gate, attr)
            for attr in ("weight", "hidden_dim", "top_k", "norm_topk_prob")
        ):
            saved.append((gate, gate.forward))

            def biased_forward(module, hidden_states, *, _bias=bias):
                hidden_states = hidden_states.reshape(-1, module.hidden_dim)
                logits = F.linear(hidden_states, module.weight).float()
                logits = logits + expand(_bias, logits.shape[0])
                probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                weights, indices = torch.topk(probs, module.top_k, dim=-1)
                if module.norm_topk_prob:
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                return (
                    logits.to(hidden_states.dtype),
                    weights.to(hidden_states.dtype),
                    indices,
                )

            gate.forward = biased_forward.__get__(gate, type(gate))
            continue

        # Plain nn.Linear router: a forward hook on the logits is equivalent.
        def add_bias(module, _inputs, output, *, _bias=bias):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"unsupported router output {type(output)!r}")
            logits = output.float()
            flat = logits.reshape(-1, logits.shape[-1])
            flat = flat + expand(_bias, flat.shape[0])
            return flat.view_as(logits).to(output.dtype)

        handles.append(gate.register_forward_hook(add_bias))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for gate, original in saved:
            gate.forward = original


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class ProbeResult:
    """Router-bias gradients plus the cost of obtaining them."""

    # grad[l] has shape [num_requests, num_experts]; l ranges over probed layers.
    grad: dict[int, torch.Tensor]
    forward_seconds: float
    backward_seconds: float
    total_seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    num_requests: int
    prompt_tokens: int
    probe_layer: int
    checkpointed: bool
    metadata: dict = field(default_factory=dict)

    def stacked(self, layers: Sequence[int] | None = None) -> torch.Tensor:
        """Gradients as one [num_requests, num_layers * num_experts] tensor."""
        keys = sorted(self.grad) if layers is None else list(layers)
        return torch.stack([self.grad[k] for k in keys], dim=1).flatten(1)

    def suppression_set(self, top_k: int, layers=None) -> torch.Tensor:
        """Indices of the top_k most safety-harmful (most negative) entries."""
        flat = self.stacked(layers)
        return flat.topk(min(top_k, flat.shape[1]), dim=1, largest=False).indices

    def to_bias(self, top_k: int, beta: float, layers=None) -> dict[int, torch.Tensor]:
        """Turn gradients into the suppression bias applied at generation time."""
        keys = sorted(self.grad) if layers is None else list(layers)
        flat = self.stacked(keys)
        bias = torch.zeros_like(flat)
        bias.scatter_(1, self.suppression_set(top_k, keys), -abs(beta))
        per_layer = bias.view(flat.shape[0], len(keys), -1)
        return {k: per_layer[:, i].contiguous() for i, k in enumerate(keys)}


# --------------------------------------------------------------------------- #
# Shared probe machinery
# --------------------------------------------------------------------------- #


def _gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _make_bias_leaves(
    gates: Sequence[tuple[int, torch.nn.Module]],
    num_requests: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Zero-valued bias leaves. Zero is the point we differentiate about."""
    return {
        layer_index: torch.zeros(
            num_requests,
            num_experts_of(gate),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        for layer_index, gate in gates
    }


def _score(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    last_index: torch.Tensor | None,
) -> torch.Tensor:
    """Scalar safety score, summed over requests.

    Summing is exact for per-request gradients: each request's bias leaf only
    receives gradient from its own term, so d(sum_q s(q))/d b(q) = d s(q)/d b(q).
    Left-padded batches must pass last_index so the score reads each request's
    true final token rather than a pad position.
    """
    hidden = hidden.float()
    if last_index is None:
        final = hidden[:, -1, :]
    else:
        gather = last_index.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        final = hidden.gather(1, gather).squeeze(1)
    return (final * direction).sum()


# --------------------------------------------------------------------------- #
# Reference probe
# --------------------------------------------------------------------------- #


class ReferenceProbe:
    """Unoptimized probe. The ground truth for equivalence testing."""

    def __init__(self, model, probe_layer: int, direction: torch.Tensor):
        self.model = model
        self.probe_layer = probe_layer
        self.direction = direction
        self.gates = find_moe_gates(model)

    def _probed_gates(self):
        return [(i, g) for i, g in self.gates if i <= self.probe_layer]

    @torch.enable_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        last_index: torch.Tensor | None = None,
    ) -> ProbeResult:
        device = input_ids.device
        gates = self._probed_gates()
        leaves = _make_bias_leaves(gates, input_ids.shape[0], device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        _sync(device)
        start = time.perf_counter()

        # No truncation: run the whole model, then read the probe layer's output.
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, _inputs, output):
            captured["hidden"] = output[0] if isinstance(output, tuple) else output

        handle = self.model.model.layers[self.probe_layer].register_forward_hook(capture)
        try:
            with router_bias(gates, leaves):
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            handle.remove()

        score = _score(captured["hidden"], self.direction, last_index)
        _sync(device)
        forward_seconds = time.perf_counter() - start

        backward_start = time.perf_counter()
        grads = torch.autograd.grad(score, tuple(leaves.values()))
        _sync(device)
        backward_seconds = time.perf_counter() - backward_start

        peak_alloc = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        peak_rsvd = torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
        result = ProbeResult(
            grad={
                layer: grad.detach().clone()
                for (layer, _), grad in zip(gates, grads)
            },
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
            total_seconds=forward_seconds + backward_seconds,
            peak_allocated_gib=_gib(peak_alloc),
            peak_reserved_gib=_gib(peak_rsvd),
            num_requests=input_ids.shape[0],
            prompt_tokens=input_ids.shape[1],
            probe_layer=self.probe_layer,
            checkpointed=False,
            metadata={"implementation": "reference", "truncated": False},
        )
        del captured, score, grads, leaves
        return result


# --------------------------------------------------------------------------- #
# Fast probe
# --------------------------------------------------------------------------- #


class FastProbe:
    """Exactness-preserving optimized probe.

    Optimizations, each mathematically neutral:
      truncate     stop the forward pass right after probe_layer. Layers above
                   it are not in the graph of s(q), so skipping them cannot
                   change any gradient. Saves (1 - L/N) of forward and backward.
      checkpoint   recompute each probed layer's internals during backward
                   rather than storing them. Peak activation memory becomes
                   O(one layer + L residual-stream boundaries) instead of
                   O(L layers), which is what keeps long prompts affordable.
      attention    prefer a memory-efficient SDPA backend so attention backward
                   workspace is O(S), not O(S^2).
    """

    def __init__(
        self,
        model,
        probe_layer: int,
        direction: torch.Tensor,
        *,
        checkpoint: bool = True,
        truncate: bool = True,
        efficient_attention: bool = True,
    ):
        self.model = model
        self.probe_layer = probe_layer
        self.direction = direction
        self.checkpoint = checkpoint
        self.truncate = truncate
        self.efficient_attention = efficient_attention
        self.gates = find_moe_gates(model)

    def _probed_gates(self):
        return [(i, g) for i, g in self.gates if i <= self.probe_layer]

    @contextlib.contextmanager
    def _attention_backend(self):
        """Ask for memory-efficient attention; fall back silently if absent.

        torch.nn.attention is not necessarily imported by ``import torch``, and
        sdpa_kernel only accepts a list of backends on newer releases, so both
        are probed defensively. Falling back costs memory, never correctness.
        """
        if not self.efficient_attention:
            yield
            return
        try:
            import torch.nn.attention as attention
        except ImportError:
            yield
            return
        kernel_ctx = getattr(attention, "sdpa_kernel", None)
        backends = getattr(attention, "SDPBackend", None)
        if kernel_ctx is None or backends is None:
            yield
            return
        wanted = [
            backend
            for backend in (
                getattr(backends, "FLASH_ATTENTION", None),
                getattr(backends, "EFFICIENT_ATTENTION", None),
                getattr(backends, "MATH", None),
            )
            if backend is not None
        ]
        if not wanted:
            yield
            return

        # Only failures from *entering* the context are tolerated. Exceptions
        # from the body must propagate untouched -- note that _StopForward
        # subclasses RuntimeError, so catching around the yield would silently
        # swallow the truncation signal and mask real autograd errors.
        with contextlib.ExitStack() as stack:
            for candidate in (wanted, wanted[0]):
                try:
                    stack.enter_context(kernel_ctx(candidate))
                    break
                except (RuntimeError, ValueError, TypeError):
                    continue
            yield

    @contextlib.contextmanager
    def _checkpointing(self):
        """Wrap each probed decoder layer in a recomputing checkpoint.

        use_reentrant=False is required: the reentrant implementation does not
        propagate gradients to inputs that are not passed positionally, which is
        exactly our case (the bias leaves are captured by closure).
        """
        if not self.checkpoint:
            yield
            return

        from torch.utils.checkpoint import checkpoint as run_checkpointed

        patched: list[tuple[torch.nn.Module, Callable]] = []
        for layer in self.model.model.layers[: self.probe_layer + 1]:
            original = layer.forward

            def wrapper(*args, _original=original, **kwargs):
                if not torch.is_grad_enabled():
                    return _original(*args, **kwargs)
                return run_checkpointed(
                    _original, *args, use_reentrant=False, **kwargs
                )

            layer.forward = wrapper
            patched.append((layer, original))
        try:
            yield
        finally:
            for layer, original in patched:
                layer.forward = original

    @torch.enable_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        last_index: torch.Tensor | None = None,
    ) -> ProbeResult:
        device = input_ids.device
        gates = self._probed_gates()
        leaves = _make_bias_leaves(gates, input_ids.shape[0], device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        _sync(device)
        start = time.perf_counter()

        captured: dict[str, torch.Tensor] = {}

        def capture_and_maybe_stop(_module, _inputs, output):
            captured["hidden"] = output[0] if isinstance(output, tuple) else output
            if self.truncate:
                raise _StopForward

        handle = self.model.model.layers[self.probe_layer].register_forward_hook(
            capture_and_maybe_stop
        )
        try:
            # The bias hooks, the checkpoint wrappers and the attention backend
            # must all remain installed through the backward pass: with
            # gradient checkpointing, backward *re-runs* each probed layer's
            # forward, and if router_bias were already uninstalled that
            # recomputation would take the unbiased path. Torch detects the
            # mismatch and raises CheckpointError ("a different number of
            # tensors was saved during the original forward and
            # recomputation"), so closing these contexts too early is not a
            # silent correctness bug -- but it is still wrong, hence one stack
            # spanning both phases.
            #
            # The forward-capture hook is the exception: it must be removed
            # before backward, or the _StopForward it raises would abort
            # recomputation.
            with contextlib.ExitStack() as stack:
                stack.enter_context(self._attention_backend())
                stack.enter_context(self._checkpointing())
                stack.enter_context(router_bias(gates, leaves))

                try:
                    self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                except _StopForward:
                    pass

                if "hidden" not in captured:
                    raise RuntimeError("probe layer produced no hidden state")

                score = _score(captured["hidden"], self.direction, last_index)
                _sync(device)
                forward_seconds = time.perf_counter() - start

                # Remove the truncation hook so checkpoint recomputation can
                # replay the probed layers without hitting _StopForward.
                handle.remove()
                handle = None

                backward_start = time.perf_counter()
                grads = torch.autograd.grad(score, tuple(leaves.values()))
                _sync(device)
                backward_seconds = time.perf_counter() - backward_start
        finally:
            if handle is not None:
                handle.remove()

        peak_alloc = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        peak_rsvd = torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
        result = ProbeResult(
            grad={
                layer: grad.detach().clone()
                for (layer, _), grad in zip(gates, grads)
            },
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
            total_seconds=forward_seconds + backward_seconds,
            peak_allocated_gib=_gib(peak_alloc),
            peak_reserved_gib=_gib(peak_rsvd),
            num_requests=input_ids.shape[0],
            prompt_tokens=input_ids.shape[1],
            probe_layer=self.probe_layer,
            checkpointed=self.checkpoint,
            metadata={
                "implementation": "fast",
                "truncated": self.truncate,
                "efficient_attention": self.efficient_attention,
            },
        )
        del captured, score, grads, leaves
        return result


# --------------------------------------------------------------------------- #
# Comparison helpers, used by the equivalence test and the benchmarks
# --------------------------------------------------------------------------- #


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two flattened gradient vectors."""
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() < 2:
        return 1.0
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra @ rb) / denominator)


def compare(
    reference: ProbeResult,
    candidate: ProbeResult,
    *,
    top_k: int = 25,
) -> dict:
    """Quantify how far a candidate probe deviates from the reference."""
    layers = sorted(set(reference.grad) & set(candidate.grad))
    if not layers:
        raise ValueError("probes share no layers")
    ref = reference.stacked(layers)
    cand = candidate.stacked(layers)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch {tuple(ref.shape)} vs {tuple(cand.shape)}")

    difference = (ref - cand).abs()
    scale = ref.abs().amax().clamp_min(1e-12)
    ref_set = reference.suppression_set(top_k, layers)
    cand_set = candidate.suppression_set(top_k, layers)
    overlap = [
        len(set(r.tolist()) & set(c.tolist())) / max(1, r.numel())
        for r, c in zip(ref_set, cand_set)
    ]
    return {
        "layers": len(layers),
        "num_requests": ref.shape[0],
        "max_abs_diff": float(difference.amax()),
        "mean_abs_diff": float(difference.mean()),
        "max_rel_diff": float(difference.amax() / scale),
        "spearman_min": min(
            spearman(ref[i], cand[i]) for i in range(ref.shape[0])
        ),
        "suppression_overlap_min": min(overlap),
        "suppression_overlap_mean": sum(overlap) / len(overlap),
        "identical_suppression_set": all(
            torch.equal(r.sort().values, c.sort().values)
            for r, c in zip(ref_set, cand_set)
        ),
        "top_k": top_k,
        "reference_seconds": reference.total_seconds,
        "candidate_seconds": candidate.total_seconds,
        "speedup": (
            reference.total_seconds / candidate.total_seconds
            if candidate.total_seconds > 0
            else float("nan")
        ),
        "reference_peak_gib": reference.peak_allocated_gib,
        "candidate_peak_gib": candidate.peak_allocated_gib,
        "memory_ratio": (
            candidate.peak_allocated_gib / reference.peak_allocated_gib
            if reference.peak_allocated_gib > 0
            else float("nan")
        ),
    }


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def unit_direction(hidden_size: int, device, seed: int = 0) -> torch.Tensor:
    """Seeded stand-in for the extracted safety direction (timing only)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    vector = torch.randn(
        hidden_size, device=device, dtype=torch.float32, generator=generator
    )
    return vector / vector.norm().clamp_min(1e-12)
