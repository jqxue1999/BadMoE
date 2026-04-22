"""
L43b: Per-expert SIGNED d-writing.

L43 measured unsigned magnitude c = ||d_hat^T · W_down||_2. It could not
tell whether an expert writes along +d or -d.

L43b computes signed measures:
  v(L, e) = d_hat_L^T · W_down^{L,e}   [intermediate]
  s_sum(L, e)  = sum(v)
  s_norm(L, e) = sum(v) / (||v||_1 + eps)   in [-1, 1]
  f_pos(L, e)  = (v > 0).float().mean()     in [0, 1]

Interpretation (inputs u to down_proj are SiLU-gated and typically non-neg):
  - s_norm > 0  <=>  expert net-writes +d when activated
  - s_norm < 0  <=>  expert net-writes -d when activated
  - f_pos > 0.5 <=>  most intermediate neurons push in +d

Test of the user's conjecture:
  - A+ experts (refuse-routed) should have s_norm / f_pos > A- experts
  - If yes: +d / -d column alignment matches routing labels
  - If no (A+ ≈ A- ≈ 0): routing label is orthogonal to signed d-geometry,
    so weight-level +/-d orthogonalization is even more novel.
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

D_PATH = os.environ.get("L43B_D_PATH", f"{DATA_DIR}/d_refuse.pt")
D_KEY = os.environ.get("L43B_D_KEY", "")
OUT_NAME = os.environ.get("L43B_OUT_NAME", "")

if not OUT_NAME:
    base = os.path.basename(D_PATH).replace(".pt", "")
    if D_KEY:
        base = f"{base}_{D_KEY}"
    OUT_NAME = f"l43b_signed_{base}"
OUT_PATH = f"{DATA_DIR}/{OUT_NAME}.pt"
SUMMARY_PATH = f"{DATA_DIR}/{OUT_NAME}.txt"

K_A_PLUS = 25
K_A_MINUS = 25
EPS = 1e-8


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
def compute_signed(model, d_refuse: torch.Tensor):
    """Returns three tensors of shape [n_layers, n_experts]:
      s_sum[L,e]  = sum(d_hat @ W_down)
      s_norm[L,e] = sum(v) / ||v||_1        in [-1,1]
      f_pos[L,e]  = fraction of v>0 entries in [0,1]
    Also returns cap_unsigned[L,e] = ||v||_2 (sanity vs L43).
    """
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    hidden = model.config.hidden_size
    if d_refuse.shape != (n_layers, hidden):
        raise RuntimeError(
            f"d_refuse shape {tuple(d_refuse.shape)} != ({n_layers}, {hidden})")
    s_sum = torch.zeros(n_layers, n_experts, dtype=torch.float32)
    s_norm = torch.zeros(n_layers, n_experts, dtype=torch.float32)
    f_pos = torch.zeros(n_layers, n_experts, dtype=torch.float32)
    cap = torch.zeros(n_layers, n_experts, dtype=torch.float32)
    for L in range(n_layers):
        d_L = d_refuse[L].float()
        d_hat = d_L / max(d_L.norm().item(), 1e-6)
        experts = model.model.layers[L].mlp.experts
        for e in range(n_experts):
            W = experts[e].down_proj.weight.detach().float()  # [hidden, inter]
            v = d_hat.to(W.device) @ W                        # [inter]
            s_sum[L, e] = float(v.sum().item())
            l1 = float(v.abs().sum().item())
            s_norm[L, e] = s_sum[L, e] / (l1 + EPS)
            f_pos[L, e] = float((v > 0).float().mean().item())
            cap[L, e] = float(v.norm().item())
            del W, v
    return s_sum, s_norm, f_pos, cap


def mann_whitney_u(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    from scipy.stats import mannwhitneyu
    if a.numel() == 0 or b.numel() == 0:
        return float("nan"), float("nan")
    res = mannwhitneyu(a.numpy(), b.numpy(), alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def group_masks(n_layers, n_experts, A_plus, A_minus):
    plus = torch.zeros(n_layers, n_experts, dtype=torch.bool)
    minus = torch.zeros(n_layers, n_experts, dtype=torch.bool)
    for L, e in A_plus:
        plus[L, e] = True
    for L, e in A_minus:
        minus[L, e] = True
    return plus, minus, ~(plus | minus)


def stats(t):
    if t.numel() == 0:
        return dict(n=0, mean=float("nan"), median=float("nan"),
                    std=float("nan"), min=float("nan"), max=float("nan"))
    return dict(
        n=int(t.numel()),
        mean=float(t.mean().item()),
        median=float(t.median().item()),
        std=float(t.std().item()) if t.numel() > 1 else float("nan"),
        min=float(t.min().item()),
        max=float(t.max().item()),
    )


def print_block(title, c_plus, c_minus, c_rest, lines):
    def w(s):
        lines.append(s)
        print(s, flush=True)
    w("")
    w("-" * 72)
    w(title)
    w("-" * 72)
    sP = stats(c_plus); sM = stats(c_minus); sR = stats(c_rest)
    w(f"{'group':<8}{'n':>6}{'mean':>10}{'median':>10}{'std':>10}{'min':>10}{'max':>10}")
    for name, s in (("A+", sP), ("A-", sM), ("rest", sR)):
        w(f"{name:<8}{s['n']:>6}{s['mean']:>10.4f}{s['median']:>10.4f}"
          f"{s['std']:>10.4f}{s['min']:>10.4f}{s['max']:>10.4f}")
    U_pm, p_pm = mann_whitney_u(c_plus, c_minus)
    U_pr, p_pr = mann_whitney_u(c_plus, c_rest)
    U_mr, p_mr = mann_whitney_u(c_minus, c_rest)
    w("Mann-Whitney U (two-sided):")
    w(f"  A+ vs A- : U={U_pm:.0f}  p={p_pm:.4g}")
    w(f"  A+ vs rest: U={U_pr:.0f}  p={p_pr:.4g}")
    w(f"  A- vs rest: U={U_mr:.0f}  p={p_mr:.4g}")


def main() -> None:
    print(f"[config] D_PATH={D_PATH}", flush=True)
    print(f"[config] OUT_PATH={OUT_PATH}", flush=True)

    print("\nLoading direction ...", flush=True)
    d_refuse = load_direction()
    print(f"  shape={tuple(d_refuse.shape)}", flush=True)
    print("  per-layer ||d||:",
          [f"{x:.2f}" for x in d_refuse.norm(dim=1).tolist()], flush=True)

    print("\nLoading BeaverTails counts ...", flush=True)
    counts = torch.load(BT_CACHE, map_location="cpu", weights_only=False)
    delta = compute_delta(counts)
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

    print("\nComputing signed per-expert d-writing ...", flush=True)
    t0 = time.time()
    s_sum, s_norm, f_pos, cap = compute_signed(model, d_refuse)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    n_layers, n_experts = cap.shape
    plus_mask, minus_mask, rest_mask = group_masks(n_layers, n_experts, A_plus, A_minus)

    lines = []
    def w(s):
        lines.append(s); print(s, flush=True)

    w("=" * 72)
    w("L43b: SIGNED per-expert d-writing  v = d_hat^T · W_down")
    w("=" * 72)
    w(f"total experts: {n_layers*n_experts}  (A+: {K_A_PLUS}, A-: {K_A_MINUS})")

    print_block("s_norm = sum(v) / ||v||_1   (signed, in [-1,1])",
                s_norm[plus_mask], s_norm[minus_mask], s_norm[rest_mask], lines)
    print_block("f_pos = fraction of v>0 entries   (in [0,1])",
                f_pos[plus_mask], f_pos[minus_mask], f_pos[rest_mask], lines)
    print_block("s_sum = sum(v)   (signed, unbounded)",
                s_sum[plus_mask], s_sum[minus_mask], s_sum[rest_mask], lines)
    print_block("cap = ||v||_2   (unsigned, sanity vs L43)",
                cap[plus_mask], cap[minus_mask], cap[rest_mask], lines)

    # Directional interpretation
    w("")
    w("=" * 72)
    w("Verdict")
    w("=" * 72)
    med_pn = float(s_norm[plus_mask].median().item()) if plus_mask.any() else float("nan")
    med_mn = float(s_norm[minus_mask].median().item()) if minus_mask.any() else float("nan")
    med_rn = float(s_norm[rest_mask].median().item())
    w(f"median(s_norm): A+={med_pn:+.4f}   A-={med_mn:+.4f}   rest={med_rn:+.4f}")
    med_pf = float(f_pos[plus_mask].median().item()) if plus_mask.any() else float("nan")
    med_mf = float(f_pos[minus_mask].median().item()) if minus_mask.any() else float("nan")
    med_rf = float(f_pos[rest_mask].median().item())
    w(f"median(f_pos ): A+={med_pf:.4f}   A-={med_mf:.4f}   rest={med_rf:.4f}")
    w("")
    if med_pn > 0.05 and med_mn < -0.05:
        w("CONJECTURE CONFIRMED: A+ net-writes +d, A- net-writes -d.")
    elif abs(med_pn) < 0.05 and abs(med_mn) < 0.05:
        w("CONJECTURE REJECTED: both A+ and A- are sign-balanced around d.")
        w("  -> routing label is orthogonal to column-space geometry along d.")
        w("  -> signed orthogonalization on A+/A- not justified by weights alone.")
    else:
        w("MIXED: see medians and p-values above.")

    # Per-layer signed medians where A+/A- exist
    w("")
    w("Per-layer signed-medians (s_norm):")
    w(f"{'L':>3}  {'|A+|':>5} {'|A-|':>5}  {'med(A+)':>10} {'med(A-)':>10}")
    for L in range(n_layers):
        cp = s_norm[L][plus_mask[L]]
        cm = s_norm[L][minus_mask[L]]
        mp = float(cp.median().item()) if cp.numel() else float("nan")
        mm = float(cm.median().item()) if cm.numel() else float("nan")
        w(f"{L:>3}  {cp.numel():>5d} {cm.numel():>5d}  {mp:>+10.4f} {mm:>+10.4f}")

    if os.path.exists(OUT_PATH):
        bak = f"{OUT_PATH}.bak.{int(time.time())}"
        os.rename(OUT_PATH, bak)
        print(f"Existing OUT_PATH backed up -> {bak}", flush=True)

    torch.save({
        "s_sum": s_sum,
        "s_norm": s_norm,
        "f_pos": f_pos,
        "cap": cap,
        "delta": delta,
        "A_plus": A_plus,
        "A_minus": A_minus,
        "direction_path": D_PATH,
        "direction_key": D_KEY,
        "model_id": MODEL_ID,
        "bt_cache": BT_CACHE,
    }, OUT_PATH)
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT_PATH}", flush=True)
    print(f"Summary -> {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
