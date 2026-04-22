"""
L49e: benign generation under 2 conditions.

Conditions:
  A_baseline  : no hook, ground-truth helpfulness
  C_perQ_sup  : per-query top-25 neg Δ_behavior bias=-10  (our defense)

Goal: measure over-refusal cost on XSTest benign prompts.
A answer_rate − C answer_rate = helpfulness tax.
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
CACHE_PATH = f"{DATA_DIR}/stage_l49d_benign_cache.pt"
OUT_CSV = f"{DATA_DIR}/stage_l49e_benign_generations.csv"

K = 25
BIAS_NEG = -10.0
MAX_NEW_TOKENS = 256
CONDITIONS = ["A_baseline", "C_perQ_sup"]


class MultiBiasHook:
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


def topk_neg_experts(delta_q, k):
    flat = delta_q.reshape(-1)
    vals, idx = torch.topk(flat, k, largest=False)
    n_E = delta_q.shape[1]
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        if v >= 0:
            break
        out.append((i // n_E, i % n_E))
    return out


def bias_map_for(cond, delta_q):
    if cond == "A_baseline":
        return {}
    if cond == "C_perQ_sup":
        b = {}
        for li, e in topk_neg_experts(delta_q, K):
            b.setdefault(li, {})[e] = BIAS_NEG
        return b
    raise ValueError(cond)


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
    print("L49e: benign generation (A_baseline vs C_perQ_sup)")
    print("=" * 72, flush=True)

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    queries = cache["queries"]
    ids = cache["ids"]
    types = cache["types"]
    delta = cache["delta_behavior"]
    N = len(queries)
    print(f"  N benign={N}  conditions={CONDITIONS}", flush=True)

    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    print(f"  model loaded in {time.time()-t1:.0f}s", flush=True)

    fields = ["condition", "id", "type", "q", "resp",
              "answered", "experts_used"]
    f_csv = open(OUT_CSV, "w", buffering=1, newline="")
    wr = _csv.DictWriter(f_csv, fieldnames=fields)
    wr.writeheader()
    f_csv.flush()

    t0 = time.time()
    total = len(CONDITIONS) * N
    done = 0
    for cond in CONDITIONS:
        print("\n" + "=" * 60)
        print(f"CONDITION: {cond}")
        print("=" * 60, flush=True)
        for qi in range(N):
            q = queries[qi]
            bmap = bias_map_for(cond, delta[qi])
            experts_used = [(li, e) for li, em in bmap.items() for e in em]
            try:
                if bmap:
                    with MultiBiasHook(model, bmap):
                        text = generate_one(model, tok, q)
                else:
                    text = generate_one(model, tok, q)
            except Exception as e:
                text = ""
                print(f"  [ERROR] qi={qi}: {e}", flush=True)
            wr.writerow({
                "condition": cond, "id": ids[qi], "type": types[qi],
                "q": q[:300], "resp": text,
                "answered": "",
                "experts_used": ";".join(f"L{l}E{e}" for l, e in experts_used),
            })
            f_csv.flush()
            done += 1
            torch.cuda.empty_cache()
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] elap={time.time()-t0:.0f}s  "
                      f"cond={cond}", flush=True)
    f_csv.close()
    print(f"\nSaved -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
