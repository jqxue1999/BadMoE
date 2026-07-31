"""Validate FastProbe == ReferenceProbe on a tiny CPU MoE. No GPU required.

Builds a miniature model with the same structural contract the real probe relies
on (model.model.layers[*].mlp.gate, residual stream, top-k routing), so the
probe code paths -- truncation, gradient checkpointing, bias injection, score
read at last real token -- are all exercised without a GPU or a 30B checkpoint.

This is the fast regression test: it runs in seconds on a laptop and catches the
structural mistakes that matter. In fp32 on CPU every optimization is expected to
be *bitwise* identical to the reference (max_abs_diff exactly 0.0), which makes it
strictly stronger than the tolerance-based GPU test. Run it before any GPU job.

test_probe_equivalence.py is the corresponding check on the real model, where
bf16 and fused-kernel reduction order make small numeric differences admissible.

Usage:
  python scripts/moe/test_probe_equivalence_cpu.py
"""

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pira_probe import FastProbe, ReferenceProbe, compare, unit_direction

D, E, TOPK, NLAYER, VOCAB = 32, 16, 4, 6, 100
PROBE_LAYER = 3


class Gate(nn.Module):
    """Qwen3-MoE style gate: returns (logits, weights, indices)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, D) * 0.3)
        self.hidden_dim = D
        self.top_k = TOPK
        self.norm_topk_prob = True
        self.num_experts = E

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        logits = torch.nn.functional.linear(hidden_states, self.weight).float()
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return (
            logits.to(hidden_states.dtype),
            weights.to(hidden_states.dtype),
            indices,
        )


class MoEMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Gate()
        self.experts = nn.ModuleList(nn.Linear(D, D, bias=False) for _ in range(E))

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, D)
        _, weights, indices = self.gate(flat)
        out = torch.zeros_like(flat)
        for slot in range(TOPK):
            idx = indices[:, slot]
            w = weights[:, slot].unsqueeze(-1)
            for e in range(E):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](flat[mask])
        return out.reshape(shape)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.attn = nn.Linear(D, D, bias=False)
        self.norm2 = nn.LayerNorm(D)
        self.mlp = MoEMLP()

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Causal-ish token mixing so the graph is not purely position-local.
        h = self.norm1(hidden_states)
        scores = h @ h.transpose(-1, -2) / (D**0.5)
        causal = torch.tril(torch.ones(h.shape[1], h.shape[1], device=h.device))
        scores = scores.masked_fill(causal == 0, float("-inf"))
        mixed = torch.softmax(scores, dim=-1) @ self.attn(h)
        hidden_states = hidden_states + mixed
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return (hidden_states,)


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList(Layer() for _ in range(NLAYER))


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.config = types.SimpleNamespace(hidden_size=D)

    def forward(self, input_ids=None, attention_mask=None, use_cache=False,
                return_dict=True, **kwargs):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h, attention_mask=attention_mask)[0]
        return types.SimpleNamespace(last_hidden_state=h, logits=None)


def main():
    torch.manual_seed(0)
    model = TinyMoE().eval().requires_grad_(False)
    torch.set_grad_enabled(True)

    B, S = 3, 12
    input_ids = torch.randint(0, VOCAB, (B, S))
    attention_mask = torch.ones(B, S, dtype=torch.long)
    # Right-pad request 1 and 2 by different amounts to exercise last_index.
    attention_mask[1, -3:] = 0
    attention_mask[2, -5:] = 0
    last_index = attention_mask.sum(1) - 1

    direction = unit_direction(D, torch.device("cpu"), seed=0)

    reference = ReferenceProbe(model, PROBE_LAYER, direction).run(
        input_ids, attention_mask, last_index=last_index
    )
    print(f"reference: {reference.total_seconds*1000:.1f} ms, "
          f"layers probed={sorted(reference.grad)}")

    configs = [
        ("truncate only", dict(truncate=True, checkpoint=False, efficient_attention=False)),
        ("truncate+checkpoint", dict(truncate=True, checkpoint=True, efficient_attention=False)),
        ("checkpoint only", dict(truncate=False, checkpoint=True, efficient_attention=False)),
        ("all (with eff-attn ctx)", dict(truncate=True, checkpoint=True, efficient_attention=True)),
    ]

    all_ok = True
    for name, opts in configs:
        result = FastProbe(model, PROBE_LAYER, direction, **opts).run(
            input_ids, attention_mask, last_index=last_index
        )
        stats = compare(reference, result, top_k=8)
        ok = (
            stats["identical_suppression_set"]
            and stats["spearman_min"] >= 1.0 - 1e-9
            and stats["max_abs_diff"] < 1e-5
        )
        all_ok &= ok
        print(f"\n{name}")
        print(f"  max_abs_diff        {stats['max_abs_diff']:.3e}")
        print(f"  max_rel_diff        {stats['max_rel_diff']:.3e}")
        print(f"  spearman_min        {stats['spearman_min']:.9f}")
        print(f"  identical top-8 set {stats['identical_suppression_set']}")
        print(f"  -> {'OK' if ok else 'FAIL'}")

    # Sanity: gradients must be nonzero, else the test is vacuous.
    magnitude = reference.stacked().abs().sum().item()
    nonzero = int((reference.stacked() != 0).sum())
    print(f"\ngradient L1={magnitude:.4f}, nonzero entries={nonzero}"
          f"/{reference.stacked().numel()}")
    if magnitude == 0:
        print("FAIL: gradients are identically zero; test would be vacuous")
        all_ok = False

    # Sanity: truncation must actually skip layers above PROBE_LAYER.
    probed = sorted(reference.grad)
    if max(probed) != PROBE_LAYER:
        print(f"FAIL: probed layers {probed} extend past {PROBE_LAYER}")
        all_ok = False

    # Regression guard: on a right-padded batch the score must be read at each
    # request's last REAL token. Reading the last row instead silently scores
    # padding, so this must produce a different answer -- if it ever stops
    # differing, last_index has quietly become a no-op.
    naive = ReferenceProbe(model, PROBE_LAYER, direction).run(
        input_ids, attention_mask, last_index=None
    )
    padding_stats = compare(reference, naive, top_k=8)
    padding_matters = padding_stats["max_abs_diff"] > 0
    print(
        f"\nlast_index handling: max_abs_diff vs naive last-row read "
        f"{padding_stats['max_abs_diff']:.3e} -> "
        f"{'load-bearing (OK)' if padding_matters else 'NO-OP (FAIL)'}"
    )
    if not padding_matters:
        all_ok = False

    print("\n" + ("ALL EQUIVALENCE CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
