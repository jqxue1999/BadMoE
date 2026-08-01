"""A vLLM worker subclass that installs PIRA routing before CUDA graph capture.

Why a custom worker class is necessary
--------------------------------------
Installing the routing hooks after ``LLM(...)`` returns is too late. vLLM's
startup sequence is:

    Worker.load_model()              <- model exists, nothing captured yet
    Worker.compile_or_warm_up_model()
        _dummy_run(...)              <- torch.compile traces the model
        capture_model()              <- CUDA graphs are captured

and all of that happens inside ``LLM.__init__``. Once a graph has been captured,
replaying it re-executes the recorded kernels without running any Python, so a
Python-level monkeypatch applied afterwards is simply not in the replayed
program. That is exactly what the B200 smoke observed: the wrapper was present on
all 25 routers, yet ran only 7 times out of an expected 256 -- once per shape
that still needed tracing, then never again.

Two things are therefore required, and neither is sufficient alone:

  1. The hooks must be installed BEFORE capture, so the bias-add is part of the
     captured graph. This class does that by overriding ``load_model``.
  2. The bias must live in PERSISTENT buffers whose addresses never change, so
     that a replay reads whatever the current step wrote. ``PiraBiasState``
     allocates those once at install time, and a per-step hook on
     ``execute_model`` fills them before each forward. See pira_vllm_routing.py.

``worker_extension_cls`` cannot be used for this: vLLM asserts that the extension
class shares no attribute names with the worker, so it can only add new methods,
never override ``load_model``. ``worker_cls`` is the supported way to substitute
behaviour.

Usage
-----
    llm = LLM(
        model=...,
        worker_cls="pira_vllm_worker.PiraWorker",
        # everything else unchanged; the compiled/CUDA-graph path stays ON
    )

Configuration travels through environment variables rather than constructor
arguments, because vLLM instantiates the worker itself:

    PIRA_LAYERS     "0-24" or "0,1,2"; required to enable the hooks
    PIRA_BETA       suppression strength, default 10.0
    PIRA_STRICT     "1" (default) to fail loudly rather than run unbiased

With PIRA_LAYERS unset the class behaves exactly like the stock worker, which is
what the Original arm of the benchmark uses -- so both arms run the same worker
class and the same compiled path, and the only difference is whether a bias is
applied.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm.v1.worker.gpu_worker import Worker  # noqa: E402


def _parse_layers(spec: str | None) -> set[int]:
    """Parse "0-24", "0,3,7" or "0-4,10" into a set of layer indices."""
    if not spec:
        return set()
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            layers.update(range(int(low), int(high) + 1))
        else:
            layers.add(int(part))
    return layers


class PiraWorker(Worker):
    """GPU worker that installs PIRA routing hooks before graph capture."""

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)

        layers = _parse_layers(os.environ.get("PIRA_LAYERS"))
        if not layers:
            # Stock behaviour. The Original arm takes this path, so both arms
            # share one worker class and one compiled path.
            return result

        beta = float(os.environ.get("PIRA_BETA", "10.0"))
        strict = os.environ.get("PIRA_STRICT", "1") not in ("0", "false", "False")

        from pira_vllm_routing import install_before_capture

        installed = install_before_capture(
            self, layers, beta=beta, strict=strict
        )
        # Printed rather than logged so it lands in the Slurm output next to the
        # benchmark's own diagnostics.
        print(
            f"[PIRA] installed routing on {len(installed)} MoE layers before "
            f"compile/capture: {min(installed)}..{max(installed)}"
            if installed
            else "[PIRA] WARNING: no MoE layer was hooked",
            flush=True,
        )
        return result
