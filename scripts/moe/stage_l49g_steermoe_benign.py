"""
L49g: SteerMoE on the same 50 XSTest prompts used by L49e.

Apples-to-apples over-refusal comparison:
  L49e (baseline + C_perQ_sup) — our method
  L49g (SteerMoE original)      — this script

Both share prompt set (first 50 of xstest_safe_prompts.csv) and judge
(L49f ANSWER/REFUSE classifier). Delta source is the same BeaverTails
counts used by L40 (stage_l40_beavertails_counts.pt) — K_A+=K_A-=25.

Output: stage_l49g_steermoe_benign.csv  (condition = 'SteerMoE').
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
COUNTS_PATH = f"{DATA_DIR}/stage_l40_beavertails_counts.pt"
D_REFUSE_PATH = f"{DATA_DIR}/d_refuse.pt"
OUT_CSV = f"{DATA_DIR}/stage_l49g_steermoe_benign.csv"

K_A_PLUS = 25
K_A_MINUS = 25
EPSILON = 1e-2
PROBE_LAYER = 10   # unused when lam_threshold=None, kept for signature
MAX_NEW_TOKENS = 256
N_BENIGN = 50


def delta_from_counts(counts):
    cr = counts["counts_refuse"].float()
    cc = counts["counts_comply"].float()
    lr = max(int(counts["len_refuse"].item()), 1)
    lc = max(int(counts["len_comply"].item()), 1)
    return cr / lr - cc / lc


def select_top_k_global(delta_LE, k, descending):
    n_L, n_E = delta_LE.shape
    flat = delta_LE.reshape(-1)
    idx = torch.argsort(flat, descending=descending)[:k]
    A = {}
    for j in idx.tolist():
        A.setdefault(j // n_E, []).append(j % n_E)
    return A


class SteerMoEHooks:
    """Ports L40's SteerMoE original implementation (no λ gate)."""

    def __init__(self, model, A_plus, A_minus, epsilon):
        self.model = model
        self.A_plus = {int(k): list(v) for k, v in A_plus.items() if v}
        self.A_minus = {int(k): list(v) for k, v in A_minus.items() if v}
        self.epsilon = float(epsilon)
        self._handles = []

    def _make_hook(self, li):
        plus = torch.tensor(sorted(self.A_plus.get(li, [])), dtype=torch.long)
        minus = torch.tensor(sorted(self.A_minus.get(li, [])), dtype=torch.long)
        if plus.numel() == 0 and minus.numel() == 0:
            return None

        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                raw = out
                fused = False
            elif isinstance(out, tuple):
                hs = inp[0] if isinstance(inp, tuple) else inp
                hs_flat = hs.reshape(-1, hs.shape[-1])
                raw = F.linear(hs_flat, module.weight,
                               getattr(module, "bias", None))
                fused = True
            else:
                return out
            s = torch.log_softmax(raw.float(), dim=-1)
            s_max = s.max(dim=-1, keepdim=True).values
            s_min = s.min(dim=-1, keepdim=True).values
            if plus.numel() > 0:
                s[:, plus.to(s.device)] = (s_max + self.epsilon).expand(-1, plus.numel())
            if minus.numel() > 0:
                s[:, minus.to(s.device)] = (s_min - self.epsilon).expand(-1, minus.numel())
            s_out = s.to(raw.dtype)
            if not fused:
                return s_out
            probs = F.softmax(s_out.float(), dim=-1, dtype=torch.float)
            top_val, top_idx = torch.topk(probs, getattr(module, "top_k", 8),
                                           dim=-1)
            if getattr(module, "norm_topk_prob", False):
                top_val = top_val / top_val.sum(dim=-1, keepdim=True)
            return probs.to(s_out.dtype), top_val.to(s_out.dtype), top_idx
        return hook

    def __enter__(self):
        for li in sorted(set(self.A_plus) | set(self.A_minus)):
            h = self._make_hook(li)
            if h is None:
                continue
            handle = self.model.model.layers[li].mlp.gate.register_forward_hook(h)
            self._handles.append(handle)
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles.clear()


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


def load_benign():
    rows = []
    with open(BENIGN_CSV) as f:
        for r in _csv.DictReader(f):
            rows.append((r["id"], r["type"], r["wrapped_q"]))
    return rows[:N_BENIGN]


def main():
    print("=" * 72)
    print("L49g: SteerMoE on first 50 XSTest benign (same set as L49e)")
    print("=" * 72, flush=True)

    counts = torch.load(COUNTS_PATH, map_location="cpu", weights_only=False)
    delta = delta_from_counts(counts)
    A_plus = select_top_k_global(delta, K_A_PLUS, descending=True)
    A_minus = select_top_k_global(delta, K_A_MINUS, descending=False)
    print(f"  K_A+ = {K_A_PLUS}, K_A- = {K_A_MINUS}, epsilon = {EPSILON}")
    print(f"  A+ layers = {sorted(A_plus)}")
    print(f"  A- layers = {sorted(A_minus)}", flush=True)

    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    print(f"  model loaded in {time.time()-t1:.0f}s", flush=True)

    benign = load_benign()
    print(f"  N benign = {len(benign)}", flush=True)

    fields = ["condition", "id", "type", "q", "resp", "answered",
              "experts_used"]
    f_csv = open(OUT_CSV, "w", buffering=1, newline="")
    wr = _csv.DictWriter(f_csv, fieldnames=fields)
    wr.writeheader()
    f_csv.flush()

    t0 = time.time()
    experts_str = ";".join(
        [f"A+L{li}E{e}" for li, es in sorted(A_plus.items()) for e in sorted(es)] +
        [f"A-L{li}E{e}" for li, es in sorted(A_minus.items()) for e in sorted(es)])

    for qi, (id_, type_, q) in enumerate(benign):
        try:
            with SteerMoEHooks(model, A_plus, A_minus, EPSILON):
                text = generate_one(model, tok, q)
        except Exception as e:
            text = ""
            print(f"  [ERROR] qi={qi}: {type(e).__name__}: {e}", flush=True)
        wr.writerow({
            "condition": "SteerMoE", "id": id_, "type": type_,
            "q": q[:300], "resp": text, "answered": "",
            "experts_used": experts_str,
        })
        f_csv.flush()
        torch.cuda.empty_cache()
        if (qi + 1) % 10 == 0 or qi == len(benign) - 1:
            print(f"  [{qi+1}/{len(benign)}] elap={time.time()-t0:.0f}s",
                  flush=True)
    f_csv.close()
    print(f"\nSaved -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
