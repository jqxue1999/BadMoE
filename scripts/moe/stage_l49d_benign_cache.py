"""
L49d: cache Δ_behavior for 50 XSTest benign prompts.

Same pipeline as L49a, but input is XSTest benign queries. Used by L49e to
apply the same per-query bad-expert suppression to benign inputs and measure
over-refusal.

Output: stage_l49d_benign_cache.pt
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
BENIGN_CSV = f"{DATA_DIR}/xstest_safe_prompts.csv"
D_BEHAVIOR_PATH = f"{DATA_DIR}/d_universal_avg.pt"
OUT_PATH = f"{DATA_DIR}/stage_l49d_benign_cache.pt"

PROBE_LAYER = 10
N_BENIGN = 50


class CausalGradientHooks:
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


def load_benign():
    rows = []
    with open(BENIGN_CSV) as f:
        for r in _csv.DictReader(f):
            rows.append((r["id"], r["type"], r["wrapped_q"]))
    return rows[:N_BENIGN]


def main():
    print("=" * 72)
    print("L49d: cache Δ_behavior for benign XSTest queries")
    print("=" * 72)

    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    print(f"  model loaded in {time.time()-t1:.0f}s", flush=True)

    d_beh_dict = torch.load(D_BEHAVIOR_PATH, map_location="cpu",
                            weights_only=False)
    d_behavior = d_beh_dict["d_behavior"].float()
    d_probe = d_behavior[PROBE_LAYER]
    d_probe = d_probe / d_probe.norm().clamp(min=1e-9)

    queries = load_benign()
    N = len(queries)
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    delta = torch.zeros(N, n_layers, n_experts, dtype=torch.float32)
    ok = torch.zeros(N, dtype=torch.bool)

    probe_cap = {"hs_graph": None}
    def probe_hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.dim() != 3 or hs.shape[1] <= 1:
            return
        probe_cap["hs_graph"] = hs

    handle = model.model.layers[PROBE_LAYER].register_forward_hook(probe_hook)
    device = next(model.parameters()).device
    d_unit = d_probe.to(device)

    t0 = time.time()
    try:
        for qi, (_id, _type, q) in enumerate(queries):
            with CausalGradientHooks(model, list(range(n_layers))) as hooks:
                enc = tokenize_prompt(tok, q)
                inputs = {k: v.to(model.device) for k, v in enc.items()}
                probe_cap["hs_graph"] = None
                with torch.enable_grad():
                    _ = model(**inputs, use_cache=False)
                    if probe_cap["hs_graph"] is None:
                        print(f"  [ERROR] probe miss qi={qi}", flush=True)
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
                    ok[qi] = True
            torch.cuda.empty_cache()
            if (qi + 1) % 10 == 0 or qi == N - 1:
                print(f"  [{qi+1}/{N}] elap={time.time()-t0:.0f}s", flush=True)
    finally:
        handle.remove()

    n_fail = int((~ok).sum())
    if n_fail > 0:
        raise RuntimeError(f"{n_fail}/{N} benign queries failed probe capture")

    torch.save({
        "ids": [x[0] for x in queries],
        "types": [x[1] for x in queries],
        "queries": [x[2] for x in queries],
        "delta_behavior": delta,
        "probe_layer": PROBE_LAYER,
        "n": N,
    }, OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}")
    print(f"  mean |Δ_behavior|: {delta.abs().mean().item():.3e}", flush=True)


if __name__ == "__main__":
    main()
