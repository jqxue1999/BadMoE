"""
L49a: Offline cache per-query causal Δ under two score functions.

For each query (4 engaged attacks × 50 prompts = 200):
  - Δ_refuse[q, L, e]   = ∂(h_probe · d_refuse_unit) / ∂(router_bias[L,e])
  - Δ_behavior[q, L, e] = ∂(h_probe · d_behavior_unit) / ∂(router_bias[L,e])

where the probe is the last prompt-token hidden state at layer 10 and
d_behavior is L45e's d_universal_avg (mean of per-attack refuse-vs-comply diffs).

Output:
  stage_l49a_delta_cache.pt  {
    attacks: [...], attack_labels, goal_idx, queries,
    delta_refuse: [N, L, E], delta_behavior: [N, L, E],
  }

~10 min on B200 for N=200 × 2 autograd calls.
"""
from __future__ import annotations

import csv as _csv
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("HF_HOME", "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/orange/qi855292.ucf/ji757406.ucf/cache/torch")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
ATTACK_CSV = f"{DATA_DIR}/pandaguard_jbb_attacks.csv"
D_REFUSE_PATH = f"{DATA_DIR}/d_refuse.pt"
D_BEHAVIOR_PATH = f"{DATA_DIR}/d_universal_avg.pt"
OUT_PATH = f"{DATA_DIR}/stage_l49a_delta_cache.pt"

PROBE_LAYER = 10
ATTACKS = ["none", "GCG", "PAIR", "past_tense"]
N_PER_ATTACK = 50


class CausalGradientHooks:
    """Per-layer router-logit bias hooks — verbatim L47b/L27 pattern."""

    def __init__(self, model, bias_layers):
        self.model = model
        self.bias_layers = list(bias_layers)
        self.top_k = model.config.num_experts_per_tok
        device = next(model.parameters()).device
        self.biases = {
            li: torch.zeros(model.config.num_experts, requires_grad=True,
                            device=device, dtype=torch.float32)
            for li in self.bias_layers
        }
        self._handles = []

    def _make_hook(self, li):
        b = self.biases[li]
        top_k = self.top_k

        def post(module, inp, out):
            if isinstance(out, torch.Tensor):
                raw = out
                fused_tuple = False
            elif isinstance(out, tuple):
                hs = inp[0] if isinstance(inp, tuple) else inp
                hs_flat = hs.reshape(-1, hs.shape[-1])
                raw = F.linear(hs_flat, module.weight,
                               getattr(module, "bias", None))
                fused_tuple = True
            else:
                return out
            biased = raw.float() + b[None, :]
            probs = F.softmax(biased, dim=-1, dtype=torch.float)
            top_val, top_idx = torch.topk(probs, top_k, dim=-1)
            if getattr(module, "norm_topk_prob", False):
                top_val = top_val / top_val.sum(dim=-1, keepdim=True)
            if not fused_tuple:
                return biased.to(raw.dtype)
            return probs.to(raw.dtype), top_val.to(raw.dtype), top_idx
        return post

    def __enter__(self):
        for b in self.biases.values():
            if b.grad is not None:
                b.grad.detach_()
                b.grad.zero_()
        for li in self.bias_layers:
            h = self.model.model.layers[li].mlp.gate.register_forward_hook(
                self._make_hook(li))
            self._handles.append(h)
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def tokenize_prompt(tok, prompt):
    msg = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    return tok(msg, return_tensors="pt", truncation=True, max_length=2048)


def load_queries():
    keep = {a: [] for a in ATTACKS}
    with open(ATTACK_CSV) as f:
        for row in _csv.DictReader(f):
            a = row["attack"]
            if a in keep:
                keep[a].append((int(row["goal_idx"]), row["wrapped_prompt"]))
    all_rows = []
    for a in ATTACKS:
        rows = sorted(keep[a])[:N_PER_ATTACK]
        for gi, p in rows:
            all_rows.append((a, gi, p))
    return all_rows


def extract_delta(model, tok, queries, d_probe_unit, tag):
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    N = len(queries)
    delta = torch.zeros(N, n_layers, n_experts, dtype=torch.float32)
    ok_mask = torch.zeros(N, dtype=torch.bool)

    probe_cap = {"hs_graph": None}

    def probe_hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.dim() != 3 or hs.shape[1] <= 1:
            return
        probe_cap["hs_graph"] = hs

    handle = model.model.layers[PROBE_LAYER].register_forward_hook(probe_hook)
    device = next(model.parameters()).device
    d_unit = d_probe_unit.to(device).float()

    t0 = time.time()
    try:
        for qi, (_attack, _gi, q) in enumerate(queries):
            with CausalGradientHooks(model, list(range(n_layers))) as hooks:
                enc = tokenize_prompt(tok, q)
                inputs = {k: v.to(model.device) for k, v in enc.items()}
                probe_cap["hs_graph"] = None
                with torch.enable_grad():
                    _ = model(**inputs, use_cache=False)
                    if probe_cap["hs_graph"] is None:
                        print(f"  [ERROR] probe miss qi={qi} ({tag})",
                              flush=True)
                        continue
                    last_h = probe_cap["hs_graph"][0, -1, :].float()
                    score = last_h @ d_unit
                    grads = torch.autograd.grad(
                        score, list(hooks.biases.values()),
                        retain_graph=False, create_graph=False,
                        allow_unused=True)
                    for li, g in zip(hooks.biases.keys(), grads):
                        if g is not None:
                            delta[qi, li] = g.detach().float().cpu()
                    ok_mask[qi] = True
            torch.cuda.empty_cache()
            if (qi + 1) % 20 == 0 or qi == N - 1:
                print(f"    [{tag}] {qi+1}/{N}  elap={time.time()-t0:.0f}s",
                      flush=True)
    finally:
        handle.remove()
    n_fail = int((~ok_mask).sum())
    if n_fail > 0:
        raise RuntimeError(
            f"[{tag}] {n_fail}/{N} queries failed probe capture — "
            f"cache would contain zeros; refuse to save")
    return delta


def main():
    print("=" * 72)
    print("L49a: cache per-query Δ with d_refuse + d_behavior")
    print("=" * 72)
    print(f"Attacks: {ATTACKS}")
    print(f"N per attack: {N_PER_ATTACK}")
    print(f"Probe layer: {PROBE_LAYER}", flush=True)

    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    print(f"  model loaded in {time.time()-t1:.0f}s", flush=True)

    d_refuse = torch.load(D_REFUSE_PATH, map_location="cpu",
                          weights_only=False).float()
    d_beh_dict = torch.load(D_BEHAVIOR_PATH, map_location="cpu",
                            weights_only=False)
    d_behavior = d_beh_dict["d_behavior"].float()
    print(f"  d_refuse shape:   {tuple(d_refuse.shape)}")
    print(f"  d_behavior shape: {tuple(d_behavior.shape)}", flush=True)

    def unit(v):
        return v / v.norm().clamp(min=1e-9)

    d_refuse_probe = unit(d_refuse[PROBE_LAYER])
    d_behavior_probe = unit(d_behavior[PROBE_LAYER])
    cos = float((d_refuse_probe * d_behavior_probe).sum())
    print(f"  cos(d_refuse, d_behavior) at L={PROBE_LAYER}: {cos:+.3f}",
          flush=True)

    queries = load_queries()
    print(f"  loaded {len(queries)} queries", flush=True)

    print("\n[1/2] Δ_refuse extraction ...", flush=True)
    delta_refuse = extract_delta(model, tok, queries, d_refuse_probe, "refuse")

    print("\n[2/2] Δ_behavior extraction ...", flush=True)
    delta_behavior = extract_delta(model, tok, queries, d_behavior_probe,
                                   "behavior")

    torch.save({
        "attacks": ATTACKS,
        "n_per_attack": N_PER_ATTACK,
        "probe_layer": PROBE_LAYER,
        "attack_labels": [a for a, _, _ in queries],
        "goal_idx": [gi for _, gi, _ in queries],
        "queries": [q for _, _, q in queries],
        "delta_refuse": delta_refuse,
        "delta_behavior": delta_behavior,
        "d_refuse_probe_unit": d_refuse_probe,
        "d_behavior_probe_unit": d_behavior_probe,
        "cos_refuse_behavior_at_probe": cos,
    }, OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}", flush=True)

    print("\n=== Δ sanity ===", flush=True)
    attack_labels = [x for x, _, _ in queries]
    for a in ATTACKS:
        mask = [i for i, al in enumerate(attack_labels) if al == a]
        r = delta_refuse[mask].abs().mean().item()
        b = delta_behavior[mask].abs().mean().item()
        print(f"  {a:<12}  mean|Δ_refuse|={r:.3e}  "
              f"mean|Δ_behavior|={b:.3e}", flush=True)


if __name__ == "__main__":
    main()
