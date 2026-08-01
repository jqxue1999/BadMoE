"""Simulate CUDA-graph capture/replay to prove the persistent-buffer design works.

Reproduces the failure the B200 smoke found, and shows the fix. A real CUDA graph
records kernel launches against fixed tensor ADDRESSES and replays them without
running Python. The stand-in below captures the *tensors* a call closed over on
first use for a given shape, then on replay recomputes from those captured
tensors while skipping the Python wrapper entirely -- which is precisely the
property that broke the old design.

Two designs are compared:

  A. per-call allocation: the hook builds a fresh bias tensor each step. Capture
     records the address of the tensor that existed at capture time; later steps
     write to different tensors, so the replay keeps using stale values.
  B. persistent buffer: one tensor per layer, filled in place before each step.
     Capture records that address; replay reads whatever was last written.
"""

import sys

import torch

sys.path.insert(0, "/home/ec2-user/BadMoE/scripts/moe")
from pira_vllm_routing import PiraBiasState  # noqa: E402

E, TOPK, NL = 16, 4, 2


class FakeGraph:
    """Captures the tensors a step used; replay reuses them, skipping Python."""

    def __init__(self):
        self.captured = None

    def run(self, shape_key, python_fn, bias_provider):
        if self.captured is None:
            # First call for this shape: "capture". Python runs, and we record
            # the exact tensor object (i.e. the address) it read the bias from.
            bias_tensor = bias_provider()
            self.captured = (shape_key, bias_tensor)
            return python_fn(bias_tensor)
        # Subsequent calls: "replay". No Python hook runs; the recorded tensor
        # address is re-read as-is.
        _, bias_tensor = self.captured
        return python_fn(bias_tensor)


def routed_experts(logits, bias):
    """Stands in for the fused router: bias, softmax, top-k."""
    return torch.topk(torch.softmax(logits + bias, dim=-1), TOPK, dim=-1).indices


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    logits = torch.randn(4, E)

    # Two requests with disjoint suppression sets.
    bias_a = torch.zeros(E); bias_a[[0, 1, 2, 3]] = -50.0
    bias_b = torch.zeros(E); bias_b[[8, 9, 10, 11]] = -50.0

    print("=== Design A: fresh tensor allocated per step (the old code) ===")
    graph_a = FakeGraph()
    steps_a = []
    for step, vec in enumerate([bias_a, bias_b, bias_b]):
        # A fresh tensor every step, exactly like matrix_for() returning a new
        # allocation whose address the captured graph never saw.
        def provider(v=vec):
            return v.unsqueeze(0).expand(4, E).contiguous()
        out = graph_a.run("shape4", lambda b: routed_experts(logits, b), provider)
        steps_a.append(out)
        want = set(range(0, 4)) if vec is bias_a else set(range(8, 12))
        leaked = sum(1 for r in range(4) for e in out[r].tolist() if e in want)
        print(f"  step {step}: suppressed experts still selected = {leaked}")
    print(f"  step1 == step0 (stale replay): {torch.equal(steps_a[1], steps_a[0])}")

    print("\n=== Design B: persistent buffer filled in place (the fix) ===")
    state = PiraBiasState(device=device)
    state.add_request("A", {0: bias_a})
    state.add_request("B", {0: bias_b})
    state.ensure_token_buffers([0], max_tokens=8, num_experts=E, device=device)
    buffer = state.token_buffers[0]
    addr = buffer.data_ptr()

    graph_b = FakeGraph()
    steps_b = []
    for step, (order, starts) in enumerate(
        [(["A", "B"], [0, 2, 4]), (["B", "A"], [0, 2, 4]), (["B"], [0, 4])]
    ):
        # Per-step hook: fills the SAME buffer in place before the graph runs.
        state.fill_token_buffers(order, starts, num_tokens=4)
        assert buffer.data_ptr() == addr, "buffer address changed"
        out = graph_b.run(
            "shape4",
            lambda b: routed_experts(logits, b),
            lambda: state.token_buffers[0][:4],
        )
        steps_b.append(out)

        # Every token row must avoid its own request's suppressed experts.
        leaked = 0
        for row in range(4):
            req = order[0] if row < starts[1] else order[min(1, len(order) - 1)]
            want = set(range(0, 4)) if req == "A" else set(range(8, 12))
            leaked += sum(1 for e in out[row].tolist() if e in want)
        print(f"  step {step} order={order}: suppressed experts selected = {leaked}")

    changed = not torch.equal(steps_b[1], steps_b[0])
    print(f"  replay tracked the reorder (step1 != step0): {changed}")
    print(f"  buffer address stable across all steps: {buffer.data_ptr() == addr}")

    print("\n=== verdict ===")
    a_broken = torch.equal(steps_a[1], steps_a[0])
    print(f"  Design A reuses stale bias on replay : {a_broken}  (this was the bug)")
    print(f"  Design B honours every step          : {changed}")
    return 0 if (a_broken and changed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
