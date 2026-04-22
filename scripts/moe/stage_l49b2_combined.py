"""
L49b2: D_perQ_combined — per-query top-K NEG suppress + top-K POS amplify.

Runs in parallel with L49b (C_perQ_sup only).

Condition:
  D perQ_comb : top-K neg Δ_behavior with bias=-10
                + top-K pos Δ_behavior with bias=+3
  K=25 (matches C); +3 is conservative — L47h's +10 amplify produced gibberish.

Output: stage_l49b2_d_combined.csv (separate from C's CSV).
Compare: C vs D tells us whether amplify helps; C vs baseline tells us
whether per-query suppress alone works.

Caveat: L49 POS does NOT align with SteerMoE A+ (~chance enrichment).
D's amplify is driven purely by Δ_behavior signal, not an independently
validated good-expert pool.
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
CACHE_PATH = f"{DATA_DIR}/stage_l49a_delta_cache.pt"
OUT_CSV = f"{DATA_DIR}/stage_l49b2_d_combined.csv"

K = 25
BIAS_NEG = -10.0
BIAS_POS = +3.0
MAX_NEW_TOKENS = 256


class MultiBiasHook:
    """Apply per-layer bias vectors on gate outputs during inference."""

    def __init__(self, model, bias_by_layer):
        self.model = model
        self.top_k = model.config.num_experts_per_tok
        self.n_experts = model.config.num_experts
        device = next(model.parameters()).device
        self.bias_by_layer = {}
        for li, em in bias_by_layer.items():
            v = torch.zeros(self.n_experts, device=device, dtype=torch.float32)
            for e, val in em.items():
                v[e] = val
            self.bias_by_layer[li] = v
        self._handles = []

    def _make_hook(self, li):
        b = self.bias_by_layer[li]
        top_k = self.top_k

        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                return (out.float() + b[None, :]).to(out.dtype)
            if isinstance(out, tuple):
                hs = inp[0] if isinstance(inp, tuple) else inp
                hs_flat = hs.reshape(-1, hs.shape[-1])
                raw = F.linear(hs_flat, module.weight,
                               getattr(module, "bias", None))
                biased = raw.float() + b[None, :]
                probs = F.softmax(biased, dim=-1, dtype=torch.float)
                top_val, top_idx = torch.topk(probs, top_k, dim=-1)
                if getattr(module, "norm_topk_prob", False):
                    top_val = top_val / top_val.sum(dim=-1, keepdim=True)
                return probs.to(hs.dtype), top_val.to(hs.dtype), top_idx
            return out
        return hook

    def __enter__(self):
        for li in self.bias_by_layer:
            h = self.model.model.layers[li].mlp.gate.register_forward_hook(
                self._make_hook(li))
            self._handles.append(h)
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def topk_signed_experts(delta_q, k, sign):
    assert sign in (-1, +1)
    flat = delta_q.reshape(-1)
    largest = (sign == +1)
    vals, idx = torch.topk(flat, k, largest=largest)
    n_E = delta_q.shape[1]
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        if sign == -1 and v >= 0:
            break
        if sign == +1 and v <= 0:
            break
        out.append((i // n_E, i % n_E))
    return out


def build_bias_map(delta_behavior_q):
    b = {}
    for li, e in topk_signed_experts(delta_behavior_q, K, sign=-1):
        b.setdefault(li, {})[e] = BIAS_NEG
    for li, e in topk_signed_experts(delta_behavior_q, K, sign=+1):
        b.setdefault(li, {})[e] = b.get(li, {}).get(e, 0.0) + BIAS_POS
    return b


def generate_one(model, tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    inp = tok.apply_chat_template(msgs, tokenize=False,
                                  add_generation_prompt=True)
    enc = tok(inp, return_tensors="pt", truncation=True,
              max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            temperature=1.0, top_p=1.0, pad_token_id=tok.eos_token_id)
    gen = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def main():
    print("=" * 72)
    print("L49b2: D_perQ_combined (suppress + amplify)")
    print("=" * 72)
    print(f"K={K}  bias_neg={BIAS_NEG}  bias_pos={BIAS_POS}  "
          f"max_new_tokens={MAX_NEW_TOKENS}", flush=True)

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    queries = cache["queries"]
    attack_labels = cache["attack_labels"]
    goal_idx = cache["goal_idx"]
    delta_b = cache["delta_behavior"]
    N = len(queries)
    print(f"  cache loaded: N={N}", flush=True)

    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    print(f"  model loaded in {time.time()-t1:.0f}s", flush=True)

    fields = ["condition", "attack", "goal_idx", "q", "resp",
              "score", "safe", "experts_used"]
    f_csv = open(OUT_CSV, "w", buffering=1, newline="")
    wr = _csv.DictWriter(f_csv, fieldnames=fields)
    wr.writeheader()
    f_csv.flush()

    cond = "D_perQ_comb"
    t0 = time.time()
    for qi in range(N):
        a = attack_labels[qi]
        gi = goal_idx[qi]
        q = queries[qi]
        bmap = build_bias_map(delta_b[qi])
        experts_used = [(li, e, v) for li, em in bmap.items()
                        for e, v in em.items()]
        err = ""
        try:
            if bmap:
                with MultiBiasHook(model, bmap):
                    text = generate_one(model, tok, q)
            else:
                text = generate_one(model, tok, q)
        except Exception as e:
            text = ""
            err = f"{type(e).__name__}: {e}"
            print(f"  [ERROR] qi={qi}: {err}", flush=True)
        wr.writerow({
            "condition": cond, "attack": a, "goal_idx": gi,
            "q": q[:300], "resp": text,
            "score": "SKIP" if err else "", "safe": "" if err else "",
            "experts_used": ";".join(
                f"L{l}E{e}{'+' if v>0 else ''}{int(v)}"
                for l, e, v in experts_used),
        })
        f_csv.flush()
        torch.cuda.empty_cache()
        if (qi + 1) % 25 == 0 or qi == N - 1:
            print(f"  [{qi+1}/{N}] elap={time.time()-t0:.0f}s  "
                  f"cond={cond} attack={a} gi={gi} "
                  f"n_experts={len(experts_used)}", flush=True)
    f_csv.close()
    print(f"\nSaved -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
