"""
L47d: causal Δ extraction on ALL 25 jailbreak attacks (100 prompts each).

Builds on L47b (4 attacks, 210s on B200). Expanding to 25 attacks × 100 =
2500 prompts gives us a full per-attack "fingerprint" — expected 20-25 min.

Use:
  - Extract mean Δ vector per attack → 25 fingerprints
  - Cluster: are all attacks in 2 groups (regular vs deepinception-like) or
    more granular? Hierarchical clustering + cos similarity heatmap.
  - Downstream application: attack-type classification head on 16-layer Δ
    residual, informing attack-specific defense.

Outputs:
  stage_l47d_all_attacks_delta.pt  (delta [2500, 16, 64] + metadata)
  stage_l47d_all_attacks.txt       per-attack norms + 25×25 cos matrix
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
OUT_PATH = f"{DATA_DIR}/stage_l47d_all_attacks_delta.pt"
SUMMARY_PATH = f"{DATA_DIR}/stage_l47d_all_attacks.txt"

CAUSAL_PROBE_LAYER = 10
# None = all 25 attacks, 100 prompts each
N_PER_ATTACK = 100


class CausalGradientHooks:
    def __init__(self, model, bias_layers):
        self.model = model
        self.bias_layers = list(bias_layers)
        self.top_k = model.config.num_experts_per_tok
        device = next(model.parameters()).device
        self.biases = {
            li: torch.zeros(model.config.num_experts,
                            requires_grad=True, device=device, dtype=torch.float32)
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
                raw = F.linear(hs_flat, module.weight, getattr(module, "bias", None))
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
            probs = probs.to(raw.dtype)
            top_val = top_val.to(raw.dtype)
            return probs, top_val, top_idx
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

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def tokenize_prompt_only(tok, prompt):
    prompt_wrapped = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    return tok(prompt_wrapped, return_tensors="pt", truncation=True,
               max_length=2048)


def extract_causal_midlayer(model, tok, queries, d_refuse_unit_probe):
    N = len(queries)
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    hidden = model.config.hidden_size

    delta = torch.zeros(N, n_layers, n_experts, dtype=torch.float32)
    embed = torch.zeros(N, hidden, dtype=torch.float32)

    probe_cap = {"hs_graph": None, "hs_detached": None}

    def probe_in_graph(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.dim() != 3 or hs.shape[1] <= 1:
            return
        probe_cap["hs_graph"] = hs
        probe_cap["hs_detached"] = hs.detach().float().cpu()

    probe_handle = model.model.layers[CAUSAL_PROBE_LAYER].register_forward_hook(
        probe_in_graph)
    device = next(model.parameters()).device
    d_refuse_unit_dev = d_refuse_unit_probe.to(device)

    try:
        t0 = time.time()
        for qi, q in enumerate(queries):
            with CausalGradientHooks(model, list(range(n_layers))) as hooks:
                enc = tokenize_prompt_only(tok, q)
                inputs = {k: v.to(model.device) for k, v in enc.items()}
                probe_cap["hs_graph"] = None
                probe_cap["hs_detached"] = None
                try:
                    with torch.enable_grad():
                        _ = model(**inputs, use_cache=False)
                        if probe_cap["hs_graph"] is None:
                            continue
                        last_h = probe_cap["hs_graph"][0, -1, :].float()
                        score = last_h @ d_refuse_unit_dev.float()
                        grads = torch.autograd.grad(
                            score, list(hooks.biases.values()),
                            retain_graph=False, create_graph=False,
                            allow_unused=True,
                        )
                        for li, g in zip(hooks.biases.keys(), grads):
                            if g is None:
                                continue
                            delta[qi, li] = g.detach().float().cpu()
                    embed[qi] = probe_cap["hs_detached"][0].mean(dim=0)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  [OOM skip] qi={qi} len={inputs['input_ids'].shape[-1]}",
                          flush=True)
            torch.cuda.empty_cache()
            if (qi + 1) % 100 == 0:
                dt = time.time() - t0
                eta = dt / (qi + 1) * (N - qi - 1)
                print(f"  [{qi+1}/{N}]  elapsed={dt:.0f}s  eta={eta:.0f}s",
                      flush=True)
    finally:
        probe_handle.remove()

    return delta, embed


def load_all_attack_prompts():
    bucket = {}
    with open(ATTACK_CSV) as f:
        r = _csv.DictReader(f)
        for row in r:
            a = row["attack"]
            bucket.setdefault(a, []).append(
                (int(row["goal_idx"]), row["wrapped_prompt"]))
    attacks = sorted(bucket.keys())
    out = []
    for a in attacks:
        rows = sorted(bucket[a])[:N_PER_ATTACK]
        for gi, p in rows:
            out.append((a, gi, p))
    return out, attacks


def main():
    t0 = time.time()
    print("=" * 72)
    print("L47d: causal Δ on all 25 jailbreak attacks")
    print("=" * 72)

    print(f"\nLoading prompts from {ATTACK_CSV}", flush=True)
    jb, attacks = load_all_attack_prompts()
    N = len(jb)
    print(f"  attacks: {len(attacks)}", flush=True)
    for a in attacks:
        cnt = sum(1 for x in jb if x[0] == a)
        print(f"    {a:<24}: {cnt}", flush=True)
    print(f"  total: {N}", flush=True)

    print(f"\nLoading {MODEL_ID} ...", flush=True)
    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    print(f"  loaded in {time.time()-t1:.0f}s", flush=True)

    d_refuse = torch.load(D_REFUSE_PATH, map_location="cpu",
                          weights_only=False)
    if isinstance(d_refuse, dict):
        for k in ("d_refuse", "direction", "d"):
            if k in d_refuse and isinstance(d_refuse[k], torch.Tensor):
                d_refuse = d_refuse[k]
                break
    d_probe = d_refuse[CAUSAL_PROBE_LAYER].float()
    d_probe_unit = d_probe / max(d_probe.norm().item(), 1e-8)

    print("\nExtracting Δ ...", flush=True)
    prompts = [p for _, _, p in jb]
    attack_labels = [a for a, _, _ in jb]
    goal_idxs = [gi for _, gi, _ in jb]
    delta, embed = extract_causal_midlayer(
        model, tok, prompts, d_probe_unit)

    print(f"\nDelta shape: {tuple(delta.shape)}", flush=True)
    print(f"  std={delta.std().item():.2e}  "
          f"mean|Δ|={delta.abs().mean().item():.2e}", flush=True)

    lines = []
    def w(s):
        lines.append(s); print(s, flush=True)

    # Per-attack stats
    w("")
    w("=" * 72)
    w("Per-attack Δ statistics")
    w("=" * 72)
    w(f"{'attack':<26} {'n':>4} {'||Δ||_F':>10} {'mean|Δ|':>10}  "
      f"{'max|Δ|':>10}")
    attack_mean_vec = {}
    for a in attacks:
        mask = [i for i, al in enumerate(attack_labels) if al == a]
        if not mask:
            continue
        sub = delta[mask]
        frob = float(sub.norm())
        mean_abs = float(sub.abs().mean())
        max_abs = float(sub.abs().max())
        w(f"{a:<26} {len(mask):>4} {frob:>10.3f} {mean_abs:>10.3e}  "
          f"{max_abs:>10.3e}")
        attack_mean_vec[a] = sub.mean(dim=0).flatten()

    # Cross-attack cosine matrix
    w("")
    w("=" * 72)
    w("Cross-attack cosine of mean Δ  (full L·E flattened)")
    w("=" * 72)
    names = list(attack_mean_vec.keys())

    # Header (truncated)
    short = [n[:10] for n in names]
    w("  " + " " * 12 + "  ".join(f"{s:>10}" for s in short))
    cos_matrix = torch.zeros(len(names), len(names))
    for i, a in enumerate(names):
        row = []
        va = attack_mean_vec[a]
        for j, b in enumerate(names):
            vb = attack_mean_vec[b]
            c = float(F.cosine_similarity(
                va.unsqueeze(0), vb.unsqueeze(0)).item())
            cos_matrix[i, j] = c
            row.append(f"{c:>+10.3f}")
        w(f"  {a[:12]:<12}" + "  ".join(row))

    # Hierarchical clustering using cos distance
    # (simple: for each attack, find its best-matching attack other than itself)
    w("")
    w("=" * 72)
    w("Nearest-neighbor per attack (excluding self)")
    w("=" * 72)
    for i, a in enumerate(names):
        sims = cos_matrix[i].clone()
        sims[i] = -1.0  # exclude self
        j = int(sims.argmax())
        w(f"  {a:<26} nearest: {names[j]:<26}  cos={sims[j]:+.3f}")

    # Save
    torch.save({
        "delta": delta,
        "embed": embed,
        "attack_labels": attack_labels,
        "goal_idxs": goal_idxs,
        "attacks": attacks,
        "cos_matrix": cos_matrix,
        "causal_probe_layer": CAUSAL_PROBE_LAYER,
        "model_id": MODEL_ID,
    }, OUT_PATH)
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT_PATH}")
    print(f"Summary -> {SUMMARY_PATH}")
    print(f"Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
