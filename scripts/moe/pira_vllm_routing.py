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


class PiraBiasState:
    """Per-request biases, keyed by engine request id.

    Keyed by request id rather than by batch position on purpose. Under
    continuous batching the scheduler admits, evicts and *reorders* rows:
    ``InputBatch.swap_states`` physically swaps two slots, and ``condense``
    moves rows down when a request finishes. So a row index is only meaningful
    for one forward pass, and a bias table indexed by position would silently
    apply request A's suppression set to request B after the first swap. That
    would be an invisible failure -- throughput would look right while the
    intervention landed on the wrong requests.

    The engine's own mapping (``InputBatch.req_id_to_index``) is therefore
    re-read every forward pass, and the per-token bias matrix is gathered from
    this table accordingly.
    """

    def __init__(self, beta: float = 10.0, device=None, dtype=torch.float32):
        self.beta = beta
        self.device = device
        self.dtype = dtype
        # request id -> {layer index -> bias vector of shape [num_experts]}
        self._by_request: dict[str, dict[int, torch.Tensor]] = {}
        # Assembled [num_rows, num_experts] matrices, cached per (layer, order).
        # Rebuilding these per layer per forward pass would dominate the very
        # cost we are trying to measure, so they are built once per distinct
        # batch composition and reused across all layers and all decode steps
        # that share it.
        self._matrix_cache: dict[tuple, torch.Tensor] = {}
        self._cache_key: tuple | None = None
        # Persistent per-token bias buffers, one per layer. Addresses must stay
        # fixed for CUDA graph replay to see updated contents.
        self._token_bias: dict[int, torch.Tensor] | None = None
        self._token_capacity = 0
        self._token_experts = 0
        # Diagnostics. Kept as host ints and only updated from already-known
        # host values, so reading them never forces a device synchronization.
        self.forward_passes = 0
        self.rows_biased = 0
        self.rows_unbiased = 0
        self.buffer_fills = 0
        self.tokens_biased = 0
        self.missing_request_ids: set[str] = set()

    # -- registration -------------------------------------------------------
    def add_request(self, request_id: str, bias_by_layer: dict[int, torch.Tensor]) -> None:
        """Register one request's bias, as computed by the probe.

        Vectors are moved to the engine device once, here, rather than on every
        forward pass. This is off the timed path: registration happens before
        the batch executes.
        """
        self._by_request[request_id] = {
            layer: vector.detach().reshape(-1).to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
            for layer, vector in bias_by_layer.items()
        }
        self._invalidate()

    def remove_request(self, request_id: str) -> None:
        self._by_request.pop(request_id, None)
        self._invalidate()

    def clear(self) -> None:
        self._by_request.clear()
        self._invalidate()

    def _invalidate(self) -> None:
        self._matrix_cache.clear()
        self._cache_key = None

    @property
    def request_ids(self) -> list[str]:
        return list(self._by_request)

    @property
    def layers(self) -> set[int]:
        return {layer for entry in self._by_request.values() for layer in entry}

    def __len__(self) -> int:
        return len(self._by_request)

    # -- per-forward-pass assembly -----------------------------------------
    def matrix_for(
        self,
        layer: int,
        ordered_request_ids: list[str | None],
        num_experts: int,
        *,
        device,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Return the [num_rows, num_experts] bias matrix in the engine's row order.

        Cached per (layer, row order). The row order is part of the key, so a
        reordering by the scheduler produces a different key and a rebuilt
        matrix; a stale matrix can never outlive the composition it was built
        for. Cheap in the common case: consecutive decode steps of a stable
        batch reuse the same tensors.

        Rows whose request has no registered bias stay zero, which is exactly the
        no-intervention case, so an unknown request degrades to the Original
        model rather than to a wrong bias.
        """
        key = (layer, tuple(ordered_request_ids), num_experts)
        cached = self._matrix_cache.get(key)
        if cached is not None and cached.device == torch.device(device):
            return cached

        matrix = torch.zeros(
            len(ordered_request_ids), num_experts, device=device, dtype=dtype
        )
        rows: list[int] = []
        vectors: list[torch.Tensor] = []
        for row, request_id in enumerate(ordered_request_ids):
            if request_id is None:
                continue
            entry = self._by_request.get(request_id)
            if entry is None:
                self.missing_request_ids.add(request_id)
                continue
            vector = entry.get(layer)
            if vector is not None:
                rows.append(row)
                vectors.append(vector.to(device=device, dtype=dtype))
        if rows:
            # One scatter instead of len(rows) individual row assignments.
            matrix.index_copy_(
                0,
                torch.tensor(rows, device=device, dtype=torch.long),
                torch.stack(vectors),
            )
        self._matrix_cache[key] = matrix
        return matrix

    # -- persistent per-token buffers (CUDA-graph safe) ---------------------
    def ensure_token_buffers(
        self, layers, max_tokens: int, num_experts: int, device, dtype=torch.float32
    ) -> None:
        """Allocate one persistent [max_tokens, num_experts] buffer per layer.

        These are allocated ONCE and never reallocated, because CUDA graph capture
        records the tensor *addresses* used during capture. Replay re-executes the
        recorded kernels against those same addresses without running any Python,
        so the only way an intervention can affect a replayed step is to write new
        contents into the same buffers before the replay. Allocating a fresh
        tensor per step would be captured as a dead address and silently ignored.
        """
        if self._token_bias is not None:
            return
        self._token_bias = {
            int(layer): torch.zeros(
                max_tokens, num_experts, device=device, dtype=dtype
            )
            for layer in layers
        }
        self._token_capacity = max_tokens
        self._token_experts = num_experts

    @property
    def token_buffers(self) -> dict[int, torch.Tensor] | None:
        return self._token_bias

    def fill_token_buffers(
        self,
        ordered_request_ids: list[str | None],
        token_starts,
        num_tokens: int,
    ) -> int:
        """Write this step's per-token biases into the persistent buffers.

        Called from a per-step Python hook (before the model runs), never from
        inside the compiled region. token_starts is a host-side cumulative
        token-count array of length num_reqs + 1, so the row spans are computed
        without reading any device tensor.

        Returns the number of token rows that received a nonzero bias.
        """
        if self._token_bias is None:
            return 0

        # Zero only the region that will be read this step.
        span = min(num_tokens, self._token_capacity)
        for buffer in self._token_bias.values():
            buffer[:span].zero_()

        biased_tokens = 0
        for row, request_id in enumerate(ordered_request_ids):
            if request_id is None:
                continue
            entry = self._by_request.get(request_id)
            if entry is None:
                self.missing_request_ids.add(request_id)
                continue
            if row + 1 >= len(token_starts):
                continue
            start = int(token_starts[row])
            end = int(token_starts[row + 1])
            if end <= start:
                continue
            start = min(start, span)
            end = min(end, span)
            if end <= start:
                continue
            for layer, buffer in self._token_bias.items():
                vector = entry.get(layer)
                if vector is not None:
                    buffer[start:end] = vector
            biased_tokens += end - start

        self.buffer_fills += 1
        self.tokens_biased += biased_tokens
        return biased_tokens

    def rows_with_bias(self, ordered_request_ids: list[str | None]) -> int:
        """How many rows of this order carry a registered bias.

        Computed on the host from the id list, never by reducing a device tensor,
        so the diagnostics cost no synchronization on the timed path.
        """
        return sum(
            1
            for request_id in ordered_request_ids
            if request_id is not None and request_id in self._by_request
        )


class _TokenRouting:
    """Resolve, for the current forward pass, which request each token belongs to."""

    @staticmethod
    def request_order(model_runner) -> list[str | None] | None:
        """The engine's current row order as request ids, or None if unavailable.

        Read fresh on every call: this order changes between steps under
        continuous batching.
        """
        input_batch = getattr(model_runner, "input_batch", None)
        if input_batch is None:
            return None
        ids = getattr(input_batch, "_req_ids", None)
        if ids is None:
            ids = getattr(input_batch, "req_ids", None)
        if ids is None:
            return None
        return list(ids)

    @staticmethod
    def token_starts(model_runner, num_requests: int):
        """Host-side cumulative token counts, length num_requests + 1, or None.

        Read from ``GPUModelRunner.query_start_loc``, which is backend
        independent: the runner fills it from the scheduler's per-request token
        counts before every forward pass, whatever attention backend is active.
        The earlier approach went through the attention metadata's
        query_start_loc, which FlashInferMetadata does not expose, so mixed
        prefill/decode batches failed outright on that backend.

        Prefers the numpy mirror (``.np``), so no device tensor is read and the
        timed path never synchronizes.
        """
        buffer = getattr(model_runner, "query_start_loc", None)
        if buffer is None:
            return None
        host = getattr(buffer, "np", None)
        if host is not None and len(host) >= num_requests + 1:
            return host[: num_requests + 1]
        cpu = getattr(buffer, "cpu", None)
        if cpu is not None and cpu.numel() >= num_requests + 1:
            return cpu[: num_requests + 1].tolist()
        gpu = getattr(buffer, "gpu", None)
        if gpu is not None and gpu.numel() >= num_requests + 1:
            # Last resort: this does synchronize, so it is only a fallback.
            return gpu[: num_requests + 1].tolist()
        if isinstance(buffer, torch.Tensor) and buffer.numel() >= num_requests + 1:
            return buffer[: num_requests + 1].tolist()
        return None

    @staticmethod
    def from_forward_context(num_token_rows: int) -> torch.Tensor | None:
        """Build a [num_token_rows] int64 tensor of batch-row indices, or None.

        vLLM's attention metadata carries ``query_start_loc``, a
        ``[num_requests + 1]`` prefix sum of each request's token count in the
        flattened batch. repeat_interleave over its differences turns that into
        a per-token row index. These are positions in the *current* batch, which
        is why they are only ever combined with a request order read during the
        same forward pass.

        Returns None when the context is unavailable (profiling runs, CUDA-graph
        capture with dummy inputs), and the caller then leaves the bias off
        rather than guessing.
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
        # Trailing pad tokens belong to no request. Mark them -1 rather than 0:
        # their output is discarded either way, but pointing them at row 0 would
        # apply request 0's suppression set to them and inflate the "rows biased"
        # diagnostic, hiding a genuine mapping failure.
        padding = torch.full(
            (num_token_rows - mapping.numel(),),
            -1,
            device=mapping.device,
            dtype=torch.int64,
        )
        return torch.cat([mapping, padding])


# --------------------------------------------------------------------------- #
# Biased selection
# --------------------------------------------------------------------------- #


def expand_bias_to_tokens(
    bias: torch.Tensor,
    num_rows: int,
    tokens_to_request: torch.Tensor | None,
) -> torch.Tensor | None:
    """Scatter a [num_requests, E] bias onto [num_rows, E] token rows.

    Returns None when the rows cannot be attributed to requests, so the caller
    can refuse rather than apply a bias to the wrong tokens.
    """
    if bias.ndim == 1:
        return bias

    if tokens_to_request is not None:
        mapping = tokens_to_request[:num_rows].to(bias.device)
        # -1 marks a pad token owned by no request. Gather with those clamped to
        # 0, then zero their rows, so pad tokens get no bias rather than
        # borrowing request 0's.
        #
        # masked_fill is applied unconditionally: guarding it with
        # bool((mapping < 0).any()) would read a device tensor on the host and
        # synchronize on every hooked layer of every forward pass, which is
        # exactly the overhead this benchmark is trying to measure. The
        # unconditional kernel is cheap and is a no-op when there are no pads.
        expanded = bias.index_select(0, mapping.clamp_min(0))
        return expanded.masked_fill((mapping < 0).unsqueeze(-1), 0.0)

    requests = bias.shape[0]
    if requests and num_rows % requests == 0:
        # Uniform token counts, which is what a fixed-length padded batch gives.
        return bias.repeat_interleave(num_rows // requests, dim=0)
    return None


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
    """Reference implementation of PIRA expert selection.

    Used by the self-checks and as documentation of the intended semantics: the
    bias is added to the raw logits, the scoring function is applied to the
    biased logits, and the gate weights are read from that same biased
    distribution (not from the unbiased one, which is what vLLM's
    e_score_correction_bias would do).

    The live hook does NOT call this. It adds the bias to router_logits and
    delegates to vLLM's own selector, keeping the engine's fused routing kernel;
    see pira_routing(). This function stays as the semantic reference those
    fused paths are checked against.
    """
    logits = router_logits.float()
    rows = logits.shape[0]

    # Same scatter as the live hook, so the reference and the hook cannot drift.
    expanded = expand_bias_to_tokens(bias, rows, tokens_to_request)
    if expanded is None:
        raise ValueError(
            f"{rows} router rows are not divisible by {bias.shape[0]} requests "
            "and no token->request mapping was available"
        )
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
def pira_routing(
    model,
    state: PiraBiasState,
    *,
    model_runner=None,
    layers: set[int] | None = None,
    strict: bool = True,
):
    """Install PIRA biased routing on the model's MoE layers.

    model_runner is the vLLM GPUModelRunner (or anything exposing
    ``.input_batch``). It is required for correct behaviour under continuous
    batching, because the request order has to be re-read every forward pass;
    without it the bias can only be applied when the batch is a single request.

    layers restricts which decoder layers are intervened on. Defaults to the
    layers present in the state, which is normally 0..probe_layer.

    strict=True raises if a MoE layer would bypass select_experts (monolithic
    kernel), because silently running unbiased there would report PIRA's latency
    while delivering none of its behaviour.
    """
    target_layers = layers if layers is not None else state.layers
    patched: list[tuple[object, object]] = []
    installed: list[int] = []
    skipped: list[str] = []

    for name, runner in _iter_moe_runners(model):
        layer_index = _layer_index_of(name)
        if layer_index is None or layer_index not in target_layers:
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
        first_layer = min(target_layers)

        # Signature-agnostic delegation. vLLM 0.19.0 defines
        # select_experts(self, hidden_states, router_logits); newer revisions add
        # topk_indices_dtype and input_ids. Forwarding *args/**kwargs verbatim
        # means the wrapper works on both and cannot desynchronize from whatever
        # the installed release expects.
        def patched_select(
            hidden_states,
            router_logits,
            *args,
            _original=original,
            _layer=layer_index,
            _first=first_layer,
            **kwargs,
        ):
            # Read the PERSISTENT buffer for this layer, sliced to the current
            # token count. Its address is fixed, so when this call is captured
            # into a CUDA graph the replayed kernels read whatever the per-step
            # hook most recently wrote there. Nothing here allocates, indexes by
            # request, or touches Python state that a replay would skip -- all of
            # that happens in _fill_step_buffers before the model runs.
            buffers = state.token_buffers
            if buffers is None:
                return _original(hidden_states, router_logits, *args, **kwargs)
            buffer = buffers.get(_layer)
            if buffer is None:
                return _original(hidden_states, router_logits, *args, **kwargs)

            num_rows = router_logits.shape[0]
            if num_rows > buffer.shape[0]:
                if strict:
                    raise RuntimeError(
                        f"PIRA routing buffer holds {buffer.shape[0]} token rows "
                        f"but the router received {num_rows}. Raise "
                        "max_num_batched_tokens capacity at install time."
                    )
                return _original(hidden_states, router_logits, *args, **kwargs)

            if _layer == _first:
                state.forward_passes += 1

            # Delegate to vLLM's own selector with pre-biased logits. PIRA's
            # intervention *is* an additive bias on router logits before softmax
            # and top-k, so adding it here and letting the engine route keeps the
            # fused kernel, the EPLB mapping and the indices-dtype handling
            # intact. Reimplementing softmax/top-k would measure the prototype
            # instead of the method, and would silently ignore whatever routing
            # the model is configured for (sigmoid, grouped top-k, renormalize,
            # scaling factors).
            return _original(
                hidden_states,
                router_logits + buffer[:num_rows].to(router_logits.dtype),
                *args,
                **kwargs,
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

    # Per-step buffer fill. This is the part that makes the intervention survive
    # CUDA graph replay: execute_model is ordinary Python invoked once per engine
    # step, before the (possibly graph-replayed) model call, so filling the
    # persistent buffers here is guaranteed to happen even when the forward
    # itself runs entirely as a replayed graph.
    runner_patch: tuple[object, str, object] | None = None
    if model_runner is not None and hasattr(model_runner, "execute_model"):
        original_execute = model_runner.execute_model

        def patched_execute(*args, _original=original_execute, **kwargs):
            try:
                _fill_step_buffers(model_runner, state, strict=strict)
            except RuntimeError:
                raise
            except Exception:
                # Never break generation because a diagnostic path failed; the
                # forward_passes/tokens_biased counters will show it.
                pass
            return _original(*args, **kwargs)

        model_runner.execute_model = patched_execute
        runner_patch = (model_runner, "execute_model", original_execute)

    try:
        yield sorted(installed)
    finally:
        for router, original in patched:
            router.select_experts = original
        if runner_patch is not None:
            owner, attribute, original_attr = runner_patch
            setattr(owner, attribute, original_attr)


def _fill_step_buffers(model_runner, state: PiraBiasState, *, strict: bool) -> None:
    """Write the current step's per-token biases into the persistent buffers.

    Runs once per engine step, on the host, before the model executes. Uses the
    backend-independent GPUModelRunner.query_start_loc (numpy mirror) rather than
    the attention metadata, which differs per backend -- FlashInferMetadata, the
    default on B200, exposes no query_start_loc at all.
    """
    if state.token_buffers is None or len(state) == 0:
        return

    order = _TokenRouting.request_order(model_runner)
    if order is None:
        return
    # Trailing None entries are slots the batch has not filled.
    while order and order[-1] is None:
        order.pop()
    if not order:
        return

    starts = _TokenRouting.token_starts(model_runner, len(order))
    if starts is None:
        if strict:
            raise RuntimeError(
                "PIRA routing could not read GPUModelRunner.query_start_loc, so "
                "tokens cannot be attributed to requests. This vLLM build may "
                "store the cumulative token counts elsewhere."
            )
        return

    num_tokens = int(starts[len(order)])
    state.fill_token_buffers(order, starts, num_tokens)


# --------------------------------------------------------------------------- #
# Entry point for a live vLLM V1 engine
# --------------------------------------------------------------------------- #
#
# In V1 the model lives in the EngineCore worker process, so the context manager
# cannot simply be wrapped around llm.generate() in the driver. LLM.apply_model()
# runs a callable inside each worker with the model as its argument, which is the
# supported way in. The state object and the installed hooks therefore live in
# the worker; only control messages cross the process boundary.
#
# Because the hooks must stay installed across many engine steps (a prefill plus
# every decode step of every request), they are installed persistently rather
# than via `with`, and removed by a second RPC.


_WORKER_STATE_ATTR = "_pira_state"
_WORKER_HANDLE_ATTR = "_pira_handle"


def _worker_install(worker, beta: float, strict: bool, layers):
    """Run inside a worker: install the hooks and stash the state on the worker.

    collective_rpc passes the worker itself, which owns ``.model_runner`` and so
    ``.model_runner.input_batch``. That is the supported route to the live request
    order; deriving it any other way (e.g. walking the GC graph from the model)
    would be both slower and dependent on internals that carry no such promise.

    ``layers`` must be given explicitly. Deriving it from the state would install
    nothing, because installation necessarily precedes bias registration: the
    engine assigns request ids only when requests are enqueued, which is after
    the engine (and these hooks) exist.
    """
    import contextlib as _contextlib
    import sys as _sys
    from pathlib import Path as _Path

    # The worker process needs this module importable by name so the nested
    # helpers can re-import it.
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from pira_vllm_routing import PiraBiasState, pira_routing

    if getattr(worker, "_pira_handle", None) is not None:
        raise RuntimeError("PIRA routing is already installed in this worker")

    target = set(int(layer) for layer in layers)
    if not target:
        raise ValueError("install_in_worker requires a non-empty layer set")

    runner = worker.model_runner
    model = runner.get_model()
    device = next(model.parameters()).device
    state = PiraBiasState(beta=beta, device=device)

    # Persistent buffers must exist before CUDA graph capture, and must be large
    # enough for the largest batch the scheduler can build, because they are never
    # reallocated afterwards -- a reallocation would strand the addresses recorded
    # during capture.
    config = getattr(model, "config", None)
    num_experts = int(
        getattr(config, "num_experts", 0)
        or getattr(config, "n_routed_experts", 0)
        or 0
    )
    if num_experts <= 0:
        raise RuntimeError(
            "could not determine the model's expert count for buffer allocation"
        )
    max_tokens = int(
        getattr(runner, "max_num_tokens", 0)
        or getattr(runner, "max_num_batched_tokens", 0)
        or 0
    )
    if max_tokens <= 0:
        raise RuntimeError(
            "could not determine max_num_batched_tokens for buffer allocation"
        )
    state.ensure_token_buffers(target, max_tokens, num_experts, device)

    stack = _contextlib.ExitStack()
    installed = stack.enter_context(
        pira_routing(model, state, model_runner=runner, layers=target, strict=strict)
    )
    if not installed:
        stack.close()
        raise RuntimeError(
            f"no MoE layer among {sorted(target)} could be hooked. The model has "
            f"{len(getattr(model.model, 'layers', []))} layers; check that the "
            "requested layers are MoE layers exposing a router."
        )
    worker._pira_state = state
    worker._pira_handle = stack
    return installed


def install_before_capture(worker, layers, *, beta: float = 10.0, strict: bool = True):
    """Install PIRA routing from inside the worker, during load_model.

    This is the supported ordering: called from Worker.load_model (see
    pira_vllm_worker.PiraWorker), it runs after the weights exist but before
    torch.compile tracing and CUDA graph capture, so the bias-add becomes part of
    the captured graph rather than a Python patch the replay skips.

    Installing after LLM(...) returns cannot work for the compiled path, which is
    why _worker_install exists only for eager/diagnostic runs.
    """
    return _worker_install(worker, beta, strict, layers)


def _worker_uninstall(worker):
    stack = getattr(worker, "_pira_handle", None)
    if stack is not None:
        stack.close()
        worker._pira_handle = None
    worker._pira_state = None


def _worker_register(worker, biases):
    import torch as _torch

    state = getattr(worker, "_pira_state", None)
    if state is None:
        raise RuntimeError("PIRA routing is not installed in this worker")
    for request_id, per_layer in biases.items():
        state.add_request(
            request_id,
            {
                int(layer): _torch.tensor(values, dtype=_torch.float32)
                for layer, values in per_layer.items()
            },
        )
    return len(state)


def _worker_diagnostics(worker):
    state = getattr(worker, "_pira_state", None)
    if state is None:
        return {}
    return {
        "registered_requests": len(state),
        # Times the router wrapper ran. Under CUDA graph replay this counts only
        # captures, so it is NOT a liveness signal on the compiled path.
        "forward_passes": state.forward_passes,
        # Times the per-step buffer fill ran, and how many token rows it biased.
        # These DO increment on every engine step, replayed or not, so they are
        # the liveness signal that works with graphs enabled.
        "buffer_fills": state.buffer_fills,
        "tokens_biased": state.tokens_biased,
        "rows_biased": state.rows_biased,
        "rows_unbiased": state.rows_unbiased,
        "missing_request_ids": sorted(state.missing_request_ids),
    }


def _worker_reset_counters(worker):
    """Zero the diagnostics so each benchmark cell can be checked on its own."""
    state = getattr(worker, "_pira_state", None)
    if state is None:
        return False
    state.forward_passes = 0
    state.rows_biased = 0
    state.rows_unbiased = 0
    state.buffer_fills = 0
    state.tokens_biased = 0
    state.missing_request_ids.clear()
    state.clear()
    return True


def _worker_verify_hooks(worker):
    """Confirm the hooks are the ones actually on the live model objects.

    The engine compiles and may CUDA-graph-capture the model during startup,
    before these hooks are installed. A Python-level monkeypatch that the
    compiled path bypasses would leave generation unbiased while the wrapper
    looked installed, so this checks the objects the runner will really call and
    reports whether a compiled path is in play. It is evidence, not a proof:
    the binding check remains rows_biased > 0 per cell.
    """
    state = getattr(worker, "_pira_state", None)
    if state is None:
        return {"installed": False}

    runner = worker.model_runner
    model = runner.get_model()
    hooked = []
    unhooked = []
    for name, module in model.named_modules():
        router = getattr(module, "router", None)
        if router is None or not hasattr(router, "select_experts"):
            continue
        selector = router.select_experts
        # A bound method of the router is the original; our replacement is a
        # plain closure whose qualname records it.
        if getattr(selector, "__name__", "") == "patched_select":
            hooked.append(name)
        else:
            unhooked.append(name)

    compilation = None
    try:
        config = worker.vllm_config.compilation_config
        compilation = {
            "level": str(getattr(config, "level", None)),
            "mode": str(getattr(config, "mode", None)),
            "use_cudagraph": bool(getattr(config, "use_cudagraph", False)),
            "cudagraph_mode": str(getattr(config, "cudagraph_mode", None)),
        }
    except AttributeError:
        pass

    return {
        "installed": True,
        "hooked_layers": len(hooked),
        "unhooked_layers": len(unhooked),
        "unhooked_examples": unhooked[:4],
        "compilation": compilation,
    }


def install_in_worker(
    llm,
    layers,
    *,
    beta: float = 10.0,
    strict: bool = True,
) -> list[int]:
    """Install PIRA routing inside every vLLM worker. Returns the layers hooked.

    layers is required and is normally range(probe_layer + 1): the layers the
    probe computed gradients for. It cannot be inferred from registered biases,
    since installation must happen before any request exists.

    Call once, after the engine is built and before submitting requests. The
    hooks stay installed across every engine step, which is required: a request's
    bias must apply to its prefill and to each of its decode steps.
    """
    target = sorted(int(layer) for layer in layers)
    if not target:
        raise ValueError("install_in_worker requires a non-empty layer set")
    results = llm.collective_rpc(_worker_install, args=(beta, strict, target))
    return results[0] if results else []


def uninstall_in_worker(llm) -> None:
    """Remove PIRA routing from every worker.

    Only meaningful for eager runs. Once CUDA graphs have captured the hooked
    routing path, the captured kernels still read the persistent bias buffers, so
    "uninstalling" restores the Python wrapper but cannot un-capture the graphs;
    generation would keep using whatever the buffers last held. Prefer ending the
    process instead.
    """
    llm.collective_rpc(_worker_uninstall)


def register_biases_in_worker(llm, biases: dict[str, dict[int, list[float]]]) -> int:
    """Send per-request biases to the workers, keyed by engine request id.

    Biases cross the process boundary as plain lists rather than tensors, which
    keeps the RPC payload serializable. They are small (K nonzero entries per
    layer); a sparse index/value form would be smaller still if this ever mattered.

    The request ids must be the ids the engine itself uses, so that the worker can
    match them against InputBatch.req_ids. See rebuttal_vllm_pira_benchmark.py for
    how those are obtained.
    """
    results = llm.collective_rpc(_worker_register, args=(biases,))
    return results[0] if results else 0


def worker_diagnostics(llm) -> dict:
    """Read back proof that the bias actually reached the router."""
    results = llm.collective_rpc(_worker_diagnostics)
    return results[0] if results else {}


def reset_counters(llm) -> None:
    """Clear diagnostics and registered biases between benchmark cells."""
    llm.collective_rpc(_worker_reset_counters)


def verify_hooks(llm) -> dict:
    """Check the hooks sit on the objects the runner will actually call."""
    results = llm.collective_rpc(_worker_verify_hooks)
    return results[0] if results else {}


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


def verify_reordering(
    *,
    num_experts: int = 32,
    top_k: int = 4,
    beta: float = 10.0,
    suppressed_per_request: int = 8,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Check that biases follow their request when the engine reorders rows.

    This is the continuous-batching hazard: InputBatch.swap_states physically
    swaps two slots and condense() moves rows down when a request finishes, so a
    bias table indexed by batch position would start applying request A's
    suppression set to request B. Keying by request id must make the assembled
    bias matrix follow the permutation exactly.

    The check builds a bias matrix for one row order, then for a permuted and a
    shrunken order, and requires each row to carry its own request's bias.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    state = PiraBiasState(beta=beta)
    request_ids = ["req-a", "req-b", "req-c", "req-d"]
    expected: dict[str, torch.Tensor] = {}
    for request_id in request_ids:
        vector = torch.zeros(num_experts, device=device)
        victims = torch.randperm(num_experts, generator=generator, device=device)[
            :suppressed_per_request
        ]
        vector[victims] = -abs(beta)
        expected[request_id] = vector
        state.add_request(request_id, {0: vector})

    scenarios = {
        "identity": request_ids,
        "swapped": [request_ids[2], request_ids[1], request_ids[0], request_ids[3]],
        "reversed": list(reversed(request_ids)),
        # A finished request leaves a None hole before condense runs.
        "with_hole": [request_ids[1], None, request_ids[3]],
        # Shrunken batch after two requests completed.
        "condensed": [request_ids[3], request_ids[0]],
        # An id the worker never received: must stay zero, not borrow a neighbour.
        "unknown": [request_ids[0], "req-unregistered"],
    }

    failures = []
    for name, order in scenarios.items():
        matrix = state.matrix_for(0, order, num_experts, device=device)
        for row, request_id in enumerate(order):
            actual = matrix[row]
            if request_id is None or request_id not in expected:
                if bool(actual.any()):
                    failures.append(f"{name}: row {row} ({request_id}) should be zero")
            elif not torch.equal(actual, expected[request_id]):
                failures.append(f"{name}: row {row} carries the wrong bias")

    # A position-indexed implementation would pass "identity" and fail the rest;
    # confirm the scenarios are actually discriminating.
    identity = state.matrix_for(0, scenarios["identity"], num_experts, device=device)
    swapped = state.matrix_for(0, scenarios["swapped"], num_experts, device=device)
    discriminating = not torch.equal(identity, swapped)

    return {
        "scenarios_checked": len(scenarios),
        "failures": failures,
        "reordering_changes_matrix": discriminating,
        "unregistered_requests_seen": sorted(state.missing_request_ids),
        "passed": not failures and discriminating,
    }


def verify_delegation(
    *,
    num_experts: int = 32,
    top_k: int = 4,
    num_layers: int = 3,
    beta: float = 10.0,
    suppressed: int = 8,
    seed: int = 0,
) -> dict:
    """Check the live hook delegates pre-biased logits to the engine's selector.

    Two properties matter and neither is obvious from reading the code:

      1. The hook must call the ORIGINAL selector, so vLLM keeps its fused
         routing kernel, its EPLB mapping and its indices-dtype handling. A
         reimplemented softmax/top-k here would make the benchmark measure the
         prototype rather than the method.
      2. It must work with vLLM 0.19.0's two-argument
         select_experts(hidden_states, router_logits). The stub below deliberately
         accepts exactly two positional arguments, so any attempt to pass
         topk_indices_dtype or input_ids raises TypeError and fails this test.

    Also confirms the result equals biased_select(), the semantic reference.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    received: list[torch.Tensor] = []

    # Bound to locals first: a class body cannot read the enclosing function's
    # parameters directly.
    stub_top_k = top_k

    class Stub:
        """Stands in for vLLM 0.19.0's router: exactly two positional args."""

        top_k = stub_top_k
        renormalize = True
        scoring_func = "softmax"

        def select_experts(self, hidden_states, router_logits):
            received.append(router_logits.clone())
            probs = torch.softmax(router_logits.float(), dim=-1)
            weights, indices = torch.topk(probs, stub_top_k, dim=-1)
            return weights / weights.sum(dim=-1, keepdim=True), indices

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = Stub()
            self.routed_experts = type("R", (), {"forward_modular": True})()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                Block() for _ in range(num_layers)
            )
            self.marker = torch.nn.Parameter(torch.zeros(1))

    class InputBatch:
        def __init__(self, ids):
            self._req_ids = list(ids)

    class Buffer:
        """Mimics CpuGpuBuffer: a numpy-like host mirror plus device tensors."""

        def __init__(self, values):
            self.cpu = torch.tensor(values, dtype=torch.int32)
            self.gpu = self.cpu.clone()
            self.np = self.cpu.numpy()

    class Runner:
        """Stands in for GPUModelRunner: request order + query_start_loc."""

        def __init__(self, ids, starts):
            self.input_batch = InputBatch(ids)
            self.query_start_loc = Buffer(starts)
            self.max_num_tokens = 64
            self.executed = 0

        def execute_model(self, *args, **kwargs):
            self.executed += 1
            return None

        def set_order(self, ids, starts):
            self.input_batch._req_ids = list(ids)
            self.query_start_loc = Buffer(starts)

    model = Model()
    state = PiraBiasState(beta=beta, device=device)
    request_ids = ["r0", "r1"]
    vectors = {}
    for index, request_id in enumerate(request_ids):
        vector = torch.zeros(num_experts)
        victims = torch.randperm(num_experts)[:suppressed]
        vector[victims] = -abs(beta)
        vectors[request_id] = vector
        state.add_request(request_id, {l: vector for l in range(num_layers)})

    # Two token rows per request, matching query_start_loc [0, 2, 4].
    runner = Runner(request_ids, [0, 2, 4, 4, 4])
    state.ensure_token_buffers(
        range(num_layers), runner.max_num_tokens, num_experts, device
    )
    logits = torch.randn(4, num_experts)
    mapping = torch.tensor([0, 0, 1, 1])
    expected_bias = torch.stack([vectors["r0"]] * 2 + [vectors["r1"]] * 2)

    failures = []
    with pira_routing(
        model, state, model_runner=runner, layers=set(range(num_layers)), strict=True
    ) as installed:
        if sorted(installed) != list(range(num_layers)):
            failures.append(f"installed {installed}, expected 0..{num_layers - 1}")

        # execute_model must be wrapped, since that is what fills the buffers on
        # every engine step -- including steps that replay a CUDA graph.
        buffer_address = state.token_buffers[0].data_ptr()
        runner.execute_model()
        if state.buffer_fills != 1:
            failures.append(
                f"the per-step fill did not run on execute_model "
                f"(buffer_fills={state.buffer_fills})"
            )
        if state.token_buffers[0].data_ptr() != buffer_address:
            failures.append("the persistent buffer was reallocated")

        received.clear()
        weights, indices = model.model.layers[0].router.select_experts(
            torch.zeros(4, 8), logits
        )

        if not received:
            failures.append("the original selector was never called")
        else:
            delivered = received[0] - logits
            if not torch.allclose(delivered, expected_bias, atol=1e-6):
                failures.append("the original selector did not receive the bias")

        reference_weights, reference_indices = biased_select(
            logits,
            torch.stack([vectors["r0"], vectors["r1"]]),
            top_k,
            renormalize=True,
            tokens_to_request=mapping,
        )
        if not torch.equal(indices.int(), reference_indices.int()):
            failures.append("selection disagrees with biased_select reference")
        if not torch.allclose(weights.float(), reference_weights, atol=1e-6):
            failures.append("gate weights disagree with biased_select reference")

        suppressed_hits = sum(
            1
            for row in range(4)
            for expert in indices[row].tolist()
            if expected_bias[row][expert] < 0
        )
        if suppressed_hits:
            failures.append(f"{suppressed_hits} suppressed experts were selected")

        # Reordering must be honoured through the live hook, not just in the table.
        runner.set_order(list(reversed(request_ids)), [0, 2, 4, 4, 4])
        runner.execute_model()
        received.clear()
        model.model.layers[0].router.select_experts(torch.zeros(4, 8), logits)
        swapped = received[0] - logits
        expected_swapped = torch.stack([vectors["r1"]] * 2 + [vectors["r0"]] * 2)
        if not torch.allclose(swapped, expected_swapped, atol=1e-6):
            failures.append("bias did not follow requests after a reorder")

    restored = model.model.layers[0].router.select_experts
    if getattr(restored, "__name__", "") == "patched_select":
        failures.append("the original selector was not restored on exit")
    if getattr(runner.execute_model, "__name__", "") == "patched_execute":
        failures.append("execute_model was not restored on exit")

    return {
        "layers_installed": num_layers,
        "delegates_to_original": bool(received),
        "buffer_fills": state.buffer_fills,
        "tokens_biased": state.tokens_biased,
        "failures": failures,
        "passed": not failures,
    }


def verify_mixed_batch(
    *,
    num_experts: int = 16,
    beta: float = 50.0,
    prefill_length: int = 128,
    num_prefills: int = 3,
) -> dict:
    """Check token attribution for a mixed prefill/decode batch.

    Regression test for the B200 failure: a step with three 128-token prefills
    plus one decode token produced 385 router rows, and the mapping was read from
    the attention metadata's query_start_loc, which FlashInferMetadata (the
    default on that hardware) does not expose. The mapping now comes from
    GPUModelRunner.query_start_loc, which every backend populates.

    Verifies that each token row carries exactly its own request's bias, including
    at span boundaries, and that rows past the batch stay zero.
    """
    device = torch.device("cpu")
    order = [f"r{index}" for index in range(num_prefills + 1)]

    starts = [0]
    for _ in range(num_prefills):
        starts.append(starts[-1] + prefill_length)
    starts.append(starts[-1] + 1)  # the single decode token
    num_tokens = starts[-1]
    # vLLM pads query_start_loc to be non-decreasing.
    padded = starts + [num_tokens] * 8

    class Buffer:
        def __init__(self, values):
            self.cpu = torch.tensor(values, dtype=torch.int32)
            self.gpu = self.cpu.clone()
            self.np = self.cpu.numpy()

    class Runner:
        def __init__(self):
            self.input_batch = type("IB", (), {"_req_ids": list(order)})()
            self.query_start_loc = Buffer(padded)
            self.max_num_tokens = num_tokens + 64

    runner = Runner()
    state = PiraBiasState(beta=beta, device=device)
    vectors = {}
    for index, request_id in enumerate(order):
        vector = torch.zeros(num_experts)
        vector[[index % num_experts, (index + 4) % num_experts]] = -abs(beta)
        vectors[request_id] = vector
        state.add_request(request_id, {0: vector})
    state.ensure_token_buffers([0], runner.max_num_tokens, num_experts, device)

    resolved = _TokenRouting.token_starts(runner, len(order))
    if resolved is None:
        return {
            "router_rows": num_tokens,
            "failures": ["query_start_loc could not be read from the runner"],
            "passed": False,
        }

    biased = state.fill_token_buffers(order, resolved, num_tokens)
    buffer = state.token_buffers[0]

    failures = []
    if biased != num_tokens:
        failures.append(f"biased {biased} token rows, expected {num_tokens}")
    for index, request_id in enumerate(order):
        low, high = starts[index], starts[index + 1]
        for row in (low, high - 1):  # span boundaries are where off-by-ones show
            if not torch.equal(buffer[row], vectors[request_id]):
                failures.append(f"row {row} does not carry {request_id}'s bias")
    if bool(buffer[num_tokens:].any()):
        failures.append("rows beyond the batch received a bias")

    return {
        "router_rows": num_tokens,
        "requests": len(order),
        "token_rows_biased": biased,
        "failures": failures,
        "passed": not failures,
    }


if __name__ == "__main__":
    ok = True
    for title, report in (
        ("biased selection vs per-row reference", verify_against_reference()),
        ("bias follows request under reordering", verify_reordering()),
        ("hook delegates to vLLM's own selector", verify_delegation()),
        ("mixed prefill/decode token attribution", verify_mixed_batch()),
    ):
        print(f"--- {title} ---")
        for key, value in report.items():
            print(f"  {key}: {value}")
        ok &= bool(report["passed"])
        print()
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
