"""
L47b (Rescue B): causal Δ extraction on JAILBREAK prompts.

Motivation:
  - L27 extracted causal Δ on 100 AdvBench direct-harmful training prompts.
  - L47 showed mechanical per-query signal exists in aggregation d_q=Σ Δ·w_e,
    but magnitudes are small and meaning is unclear.
  - On jailbreak-wrapped prompts, the refusal-relevant experts should exhibit
    stronger, clearer per-query causal gradients (the model struggles harder
    to refuse) — so d_q should carry a more interpretable per-query signal.

What this does:
  For 4 canonical attacks × 100 prompts:
     none (baseline goals)
     PAIR
     GCG
     new_deepinception
  Run exactly the L27 mid-layer causal extraction routine:
     score(q) = h_layer10(last prompt tok)^T · d_refuse_unit_layer10
     bias grad on gate logits, per layer
  Save [N=400, L=16, E=64] Δ + metadata so we can:
    (a) compare Δ magnitude across attacks vs L27 baseline
    (b) compute d_q per attack and see if it's more coherent
    (c) feed into Usage 1 full experiment

Outputs:
  stage_l47b_jailbreak_causal_delta.pt  (delta, embed, attack_labels, ...)
  stage_l47b_jailbreak_causal.txt       summary
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
OUT_PATH = f"{DATA_DIR}/stage_l47b_jailbreak_causal_delta.pt"
SUMMARY_PATH = f"{DATA_DIR}/stage_l47b_jailbreak_causal.txt"

CAUSAL_PROBE_LAYER = 10
ATTACKS = ["none", "PAIR", "GCG", "new_deepinception"]
N_PER_ATTACK = 100


# ---------------------------------------------------------------------------
# Causal gradient hooks (verbatim from L27)
# ---------------------------------------------------------------------------

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
    """For each query: forward with causal-bias hooks + mid-layer probe.
    Returns (delta [N, L, E] fp32, embed [N, H] fp32)."""
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
                with torch.enable_grad():
                    _ = model(**inputs, use_cache=False)
                    if probe_cap["hs_graph"] is None:
                        print(f"  [warn] probe miss at qi={qi}", flush=True)
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
            torch.cuda.empty_cache()
            if (qi + 1) % 25 == 0:
                dt = time.time() - t0
                eta = dt / (qi + 1) * (N - qi - 1)
                print(f"  [{qi+1}/{N}]  "
                      f"elapsed={dt:.0f}s  eta={eta:.0f}s  "
                      f"qhead={q[:60].replace(chr(10),' ')!r}",
                      flush=True)
    finally:
        probe_handle.remove()

    return delta, embed


# ---------------------------------------------------------------------------
# Load jailbreak prompts
# ---------------------------------------------------------------------------

def load_jailbreak_prompts():
    """Return list of (attack, goal_idx, prompt) for selected attacks."""
    keep = {a: [] for a in ATTACKS}
    with open(ATTACK_CSV) as f:
        r = _csv.DictReader(f)
        for row in r:
            a = row["attack"]
            if a in keep:
                keep[a].append((int(row["goal_idx"]),
                                row["wrapped_prompt"]))
    out = []
    for a in ATTACKS:
        rows = sorted(keep[a])[:N_PER_ATTACK]
        print(f"  {a}: loaded {len(rows)} prompts", flush=True)
        for gi, p in rows:
            out.append((a, gi, p))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 72)
    print("L47b: jailbreak causal Δ extraction (Rescue B)")
    print("=" * 72)

    print(f"\nLoading jailbreak prompts from {ATTACK_CSV}", flush=True)
    jb = load_jailbreak_prompts()
    N = len(jb)
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
    hidden = model.config.hidden_size
    print(f"  loaded in {time.time()-t1:.0f}s  "
          f"n_layers={n_layers} n_experts={n_experts} hidden={hidden}",
          flush=True)

    print(f"\nLoading d_refuse from {D_REFUSE_PATH}", flush=True)
    d_refuse = torch.load(D_REFUSE_PATH, map_location="cpu", weights_only=False)
    if isinstance(d_refuse, dict):
        for k in ("d_refuse", "direction", "d"):
            if k in d_refuse and isinstance(d_refuse[k], torch.Tensor):
                d_refuse = d_refuse[k]
                break
    d_probe = d_refuse[CAUSAL_PROBE_LAYER].float()
    d_probe_unit = d_probe / max(d_probe.norm().item(), 1e-8)
    print(f"  probe_layer={CAUSAL_PROBE_LAYER}  "
          f"||d_probe||={d_probe.norm().item():.4f}", flush=True)

    print("\n" + "=" * 60)
    print("Extracting causal Δ on jailbreak prompts")
    print("=" * 60, flush=True)
    prompts = [p for _, _, p in jb]
    attack_labels = [a for a, _, _ in jb]
    goal_idxs = [gi for _, gi, _ in jb]

    delta, embed = extract_causal_midlayer(model, tok, prompts, d_probe_unit)
    print(f"\nDelta shape: {tuple(delta.shape)}", flush=True)
    print(f"  std={delta.std().item():.2e}  "
          f"mean|Δ|={delta.abs().mean().item():.2e}  "
          f"max|Δ|={delta.abs().max().item():.2e}", flush=True)

    # ------------------- per-attack summary -------------------
    lines = []
    def w(s):
        lines.append(s); print(s, flush=True)

    w("")
    w("=" * 72)
    w("Per-attack Δ statistics")
    w("=" * 72)
    w(f"{'attack':<22} {'n':>4} {'||Δ||_F':>10} {'mean|Δ|':>10}  "
      f"{'max|Δ|':>10} {'nonzero%':>9}")
    for a in ATTACKS:
        mask = [i for i, al in enumerate(attack_labels) if al == a]
        if not mask:
            continue
        sub = delta[mask]
        frob = float(sub.norm())
        mean_abs = float(sub.abs().mean())
        max_abs = float(sub.abs().max())
        nz = float((sub.abs() > 1e-12).float().mean() * 100)
        w(f"{a:<22} {len(mask):>4} {frob:>10.3f} {mean_abs:>10.3e}  "
          f"{max_abs:>10.3e} {nz:>9.1f}")

    # ------------------- per-layer per-attack -------------------
    w("")
    w("=" * 72)
    w(f"Per-layer mean|Δ| by attack   (layers 0..{CAUSAL_PROBE_LAYER})")
    w("=" * 72)
    header = "  L " + "  ".join(f"{a[:14]:>14}" for a in ATTACKS)
    w(header)
    for L in range(CAUSAL_PROBE_LAYER + 1):
        cells = []
        for a in ATTACKS:
            mask = [i for i, al in enumerate(attack_labels) if al == a]
            m = float(delta[mask, L].abs().mean()) if mask else 0.0
            cells.append(f"{m:>14.2e}")
        w(f"{L:>3}  " + "  ".join(cells))

    # ------------------- cross-attack similarity -------------------
    w("")
    w("=" * 72)
    w("Mean Δ vector cosine: attack A vs attack B  (flattened across L·E)")
    w("=" * 72)
    mean_by_attack = {}
    for a in ATTACKS:
        mask = [i for i, al in enumerate(attack_labels) if al == a]
        if mask:
            mean_by_attack[a] = delta[mask].mean(dim=0).flatten()
    names = list(mean_by_attack.keys())
    w("         " + "  ".join(f"{n[:14]:>14}" for n in names))
    for a in names:
        row = []
        va = mean_by_attack[a]
        for b in names:
            vb = mean_by_attack[b]
            c = float(F.cosine_similarity(va.unsqueeze(0),
                                          vb.unsqueeze(0)).item())
            row.append(f"{c:>14.3f}")
        w(f"  {a[:7]:<7}  " + "  ".join(row))

    # ------------------- save -------------------
    torch.save({
        "delta": delta,                 # [N, L, E]
        "embed": embed,                 # [N, H]
        "attack_labels": attack_labels, # list[str]
        "goal_idxs": goal_idxs,         # list[int]
        "prompts": prompts,             # list[str]
        "attacks": ATTACKS,
        "n_per_attack": N_PER_ATTACK,
        "causal_probe_layer": CAUSAL_PROBE_LAYER,
        "model_id": MODEL_ID,
        "d_refuse_path": D_REFUSE_PATH,
    }, OUT_PATH)
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT_PATH}")
    print(f"Summary -> {SUMMARY_PATH}")
    print(f"\nTotal wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
