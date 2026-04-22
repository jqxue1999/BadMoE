"""
L43: Per-expert d-writing capacity analysis.

Tests the 'good expert also carries the knife' hypothesis:
- SteerMoE identifies A+ (good, top-25 by Δ) and A- (bad, bottom-25 by Δ)
  from BeaverTails response-token routing statistics.
- A+ label means 'router tends to pick this expert on safe/refusal tokens'.
- We want to know whether the expert's `down_proj` *itself* carries the
  refusal direction d_refuse in its output column space.

Geometric measure (pure weight, no forward pass):

    c(L, e) = || d_hat_L^T · W_down^{L,e} ||_2

where W_down^{L,e}: [hidden, intermediate]. c(L,e) is the L2 norm of the
'd-writing vector' — how much d-component emerges if any unit input flows
through this expert's down_proj. Higher c → expert intrinsically writes
d into the residual stream when activated.

Decision rule for novelty:
    - c(A+) distribution >> c(A-): router-level steering already avoids
      d-writers → weight surgery has small marginal value → novelty weak
    - c(A+) ≈ c(A-): SteerMoE's good/bad label does NOT separate experts
      by d-writing capacity → weight-level orthogonalization on top of
      gated routing is a genuine addition → novelty strong

Env vars:
    L43_D_PATH   : direction file (default d_refuse.pt)
    L43_D_KEY    : key into dict (default: auto)
    L43_OUT_NAME : output basename (default: derived from D_PATH)
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("HF_HOME", "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/orange/qi855292.ucf/ji757406.ucf/cache/torch")

from transformers import AutoModelForCausalLM  # noqa: E402

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
BT_CACHE = f"{DATA_DIR}/stage_l40_beavertails_counts.pt"

D_PATH = os.environ.get("L43_D_PATH", f"{DATA_DIR}/d_refuse.pt")
D_KEY = os.environ.get("L43_D_KEY", "")
OUT_NAME = os.environ.get("L43_OUT_NAME", "")

if not OUT_NAME:
    base = os.path.basename(D_PATH).replace(".pt", "")
    if D_KEY:
        base = f"{base}_{D_KEY}"
    OUT_NAME = f"l43_d_capacity_{base}"
OUT_PATH = f"{DATA_DIR}/{OUT_NAME}.pt"
SUMMARY_PATH = f"{DATA_DIR}/{OUT_NAME}.txt"

K_A_PLUS = 25
K_A_MINUS = 25


def load_direction() -> torch.Tensor:
    obj = torch.load(D_PATH, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        if D_KEY and D_KEY in obj:
            return obj[D_KEY].float()
        for k in ("d_refuse", "d_refuse_jailbreak", "d_bt_prompt",
                  "direction", "d"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                print(f"  auto-picked key '{k}'", flush=True)
                return obj[k].float()
        raise RuntimeError(f"no tensor found in {D_PATH}; keys={list(obj.keys())}")
    raise RuntimeError(f"bad object at {D_PATH}: {type(obj)}")


def compute_delta(counts) -> torch.Tensor:
    cr = counts["counts_refuse"].float()
    cc = counts["counts_comply"].float()
    lr = max(int(counts["len_refuse"].item()), 1)
    lc = max(int(counts["len_comply"].item()), 1)
    return cr / lr - cc / lc


def select_topk_global(delta_LE: torch.Tensor, k: int, descending: bool) -> list[tuple[int, int]]:
    n_layers, n_experts = delta_LE.shape
    flat = delta_LE.reshape(-1)
    order = torch.argsort(flat, descending=descending)[:k]
    out = []
    for j in order.tolist():
        out.append((j // n_experts, j % n_experts))
    return out


@torch.no_grad()
def compute_capacity(model, d_refuse: torch.Tensor) -> torch.Tensor:
    """c[L, e] = ||d_hat_L^T · W_down^{L,e}||_2.

    OLMoE expert down_proj.weight shape: [hidden, intermediate].
    d_hat_L: [hidden]. Product: d_hat_L @ W_down = [intermediate].
    Returns: [n_layers, n_experts] float32.
    """
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    hidden = model.config.hidden_size
    if d_refuse.shape != (n_layers, hidden):
        raise RuntimeError(
            f"d_refuse shape {tuple(d_refuse.shape)} != ({n_layers}, {hidden})")
    cap = torch.zeros(n_layers, n_experts, dtype=torch.float32)
    # Sanity check on layer-0 expert-0 shape once
    probe = model.model.layers[0].mlp.experts[0].down_proj.weight
    if probe.shape != (hidden, model.config.intermediate_size):
        raise RuntimeError(
            f"L0 e0 down_proj shape {tuple(probe.shape)} != "
            f"({hidden}, {model.config.intermediate_size}). If this OLMoE "
            f"version uses fused OlmoeExperts (packed 3D tensor), access "
            f"path needs to change.")
    for L in range(n_layers):
        d_L = d_refuse[L].float()
        d_hat = d_L / max(d_L.norm().item(), 1e-6)
        experts = model.model.layers[L].mlp.experts
        for e in range(n_experts):
            W = experts[e].down_proj.weight.detach().float()  # [hidden, inter]
            v = d_hat.to(W.device) @ W  # [intermediate]
            cap[L, e] = v.norm().cpu().item()
            del W, v
    return cap


def mann_whitney_u(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """scipy.stats.mannwhitneyu with tie correction; two-sided."""
    from scipy.stats import mannwhitneyu
    if a.numel() == 0 or b.numel() == 0:
        return float("nan"), float("nan")
    res = mannwhitneyu(a.numpy(), b.numpy(), alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def summarize(cap: torch.Tensor, A_plus: list, A_minus: list) -> str:
    n_layers, n_experts = cap.shape
    plus_mask = torch.zeros(n_layers, n_experts, dtype=torch.bool)
    minus_mask = torch.zeros(n_layers, n_experts, dtype=torch.bool)
    for L, e in A_plus:
        plus_mask[L, e] = True
    for L, e in A_minus:
        minus_mask[L, e] = True
    rest_mask = ~(plus_mask | minus_mask)

    c_plus = cap[plus_mask]
    c_minus = cap[minus_mask]
    c_rest = cap[rest_mask]

    def stats(t: torch.Tensor) -> dict:
        return {
            "n": int(t.numel()),
            "mean": float(t.mean().item()) if t.numel() else float("nan"),
            "median": float(t.median().item()) if t.numel() else float("nan"),
            "std": float(t.std().item()) if t.numel() > 1 else float("nan"),
            "min": float(t.min().item()) if t.numel() else float("nan"),
            "max": float(t.max().item()) if t.numel() else float("nan"),
        }

    s_plus = stats(c_plus)
    s_minus = stats(c_minus)
    s_rest = stats(c_rest)

    U_pm, p_pm = mann_whitney_u(c_plus, c_minus)
    U_pr, p_pr = mann_whitney_u(c_plus, c_rest)
    U_mr, p_mr = mann_whitney_u(c_minus, c_rest)

    lines = []
    def w(s):
        lines.append(s)
        print(s, flush=True)

    w("=" * 72)
    w("L43: per-expert d-writing capacity  c(L,e) = ||d_hat_L^T · W_down||_2")
    w("=" * 72)
    w(f"total experts: {n_layers * n_experts}  (layers={n_layers}, per-layer={n_experts})")
    w(f"A+ (top-{K_A_PLUS} by Δ): {s_plus['n']}")
    w(f"A- (bot-{K_A_MINUS} by Δ): {s_minus['n']}")
    w(f"rest: {s_rest['n']}")
    w("")
    w(f"{'group':<8}{'n':>6}{'mean':>10}{'median':>10}{'std':>10}{'min':>10}{'max':>10}")
    for name, s in (("A+", s_plus), ("A-", s_minus), ("rest", s_rest)):
        w(f"{name:<8}{s['n']:>6}{s['mean']:>10.4f}{s['median']:>10.4f}"
          f"{s['std']:>10.4f}{s['min']:>10.4f}{s['max']:>10.4f}")
    w("")
    w("Mann-Whitney U (two-sided p):")
    w(f"  A+ vs A- : U={U_pm:.0f}  p={p_pm:.4g}")
    w(f"  A+ vs rest: U={U_pr:.0f}  p={p_pr:.4g}")
    w(f"  A- vs rest: U={U_mr:.0f}  p={p_mr:.4g}")
    w("")

    # Interpretation heuristic
    w("Interpretation:")
    ratio_pm = s_plus["median"] / max(s_minus["median"], 1e-9)
    if ratio_pm < 0.7 and p_pm < 0.05:
        w("  A+ MEDIAN MUCH LOWER than A- (ratio<0.7, p<0.05):")
        w("    SteerMoE already routes around d-writers → novelty WEAK")
    elif 0.85 <= ratio_pm <= 1.15 or p_pm > 0.05:
        w("  A+ ≈ A- in d-writing capacity:")
        w("    good/bad label does NOT separate experts by d-writing")
        w("    → weight orthogonalization on selected experts is GENUINELY NEW → novelty STRONG")
    elif ratio_pm > 1.15:
        w("  A+ MEDIAN HIGHER than A-:")
        w("    router selects HIGH-d-writing experts for safe tokens")
        w("    → SteerMoE's good/bad story is incomplete → novelty MAX")
    else:
        w("  A+ somewhat lower than A- (intermediate regime)")
        w(f"    ratio={ratio_pm:.2f}, p={p_pm:.3g}")

    # Per-layer breakdown (helps see where novelty is strongest)
    w("\nPer-layer A+ vs A- median capacity:")
    w(f"{'L':>3}  {'|A+|':>5} {'|A-|':>5}  {'med(A+)':>10} {'med(A-)':>10}  {'ratio':>8}")
    for L in range(n_layers):
        cp = cap[L][plus_mask[L]]
        cm = cap[L][minus_mask[L]]
        mp = float(cp.median().item()) if cp.numel() else float("nan")
        mm = float(cm.median().item()) if cm.numel() else float("nan")
        ratio = mp / mm if cm.numel() and mm > 1e-9 else float("nan")
        w(f"{L:>3}  {cp.numel():>5d} {cm.numel():>5d}  "
          f"{mp:>10.4f} {mm:>10.4f}  {ratio:>8.3f}")

    return "\n".join(lines)


def main() -> None:
    print(f"[config] D_PATH={D_PATH}", flush=True)
    print(f"[config] D_KEY={D_KEY or '<auto>'}", flush=True)
    print(f"[config] OUT_PATH={OUT_PATH}", flush=True)

    print("\nLoading direction ...", flush=True)
    d_refuse = load_direction()
    print(f"  shape={tuple(d_refuse.shape)}", flush=True)
    per_layer_norms = d_refuse.norm(dim=1).tolist()
    print("  per-layer ||d||:", [f"{x:.2f}" for x in per_layer_norms], flush=True)

    print("\nLoading BeaverTails counts ...", flush=True)
    counts = torch.load(BT_CACHE, map_location="cpu", weights_only=False)
    delta = compute_delta(counts)
    print(f"  Δ shape={tuple(delta.shape)}  (min={delta.min():.6f} max={delta.max():.6f})",
          flush=True)
    A_plus = select_topk_global(delta, K_A_PLUS, descending=True)
    A_minus = select_topk_global(delta, K_A_MINUS, descending=False)
    print(f"  A+ layers: {sorted(set(L for L, _ in A_plus))}", flush=True)
    print(f"  A- layers: {sorted(set(L for L, _ in A_minus))}", flush=True)

    print(f"\nLoading {MODEL_ID} (CPU, bf16) ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": "cpu"},
        trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    print("\nComputing per-expert capacity ...", flush=True)
    t0 = time.time()
    cap = compute_capacity(model, d_refuse)
    print(f"  done in {time.time()-t0:.0f}s. cap shape={tuple(cap.shape)}", flush=True)

    summary = summarize(cap, A_plus, A_minus)

    if os.path.exists(OUT_PATH):
        bak = f"{OUT_PATH}.bak.{int(time.time())}"
        os.rename(OUT_PATH, bak)
        print(f"Existing OUT_PATH backed up -> {bak}", flush=True)

    torch.save({
        "capacity": cap,
        "delta": delta,
        "A_plus": A_plus,
        "A_minus": A_minus,
        "k_a_plus": K_A_PLUS,
        "k_a_minus": K_A_MINUS,
        "direction_path": D_PATH,
        "direction_key": D_KEY,
        "model_id": MODEL_ID,
        "bt_cache": BT_CACHE,
    }, OUT_PATH)
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved -> {OUT_PATH}", flush=True)
    print(f"Summary -> {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
