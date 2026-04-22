"""
L47: Per-query direction d_q sanity check.

Question: does aggregating causal Δ with expert-write vectors produce a
per-query direction that (a) differs from global d, (b) differs across
queries?

Inputs:
  - stage_l27_causal_midlayer_delta.pt   [N=100, L=16, E=64]  Δ_causal
  - d_refuse.pt                          [16, 2048]           d_global
  - OLMoE weights (CPU, just W_down per expert)

For each query q, layer ℓ:
  d_q[q, ℓ] = Σ_e  Δ[q, ℓ, e] · w_e[ℓ, e]

where w_e is one of 3 constructions:
  (mean)  w_e = W_down[ℓ, e].mean(dim=1)                 trivial baseline
  (svd1)  w_e = first left-singular-vector of W_down      expert principal out
  (dproj) w_e = W_down @ W_down^T @ d_global[ℓ]          d-subspace through expert

Diagnostics:
  A. ||d_q[q, ℓ]|| — magnitude per layer (vs ||d_global[ℓ]||)
  B. cos(d_q[q, ℓ], d_global[ℓ]) — per-query alignment with global
     Expected: not uniformly 1.0 (else d_q = d_global = no new info)
  C. pairwise cos(d_q[i, ℓ], d_q[j, ℓ]) — across queries
     Expected: not uniformly 1.0 (else all queries have same d_q = no per-query signal)
  D. layer consistency: cos(d_q[q, ℓ_a], d_q[q, ℓ_b]) — same query across layers
     Expected: moderate (≥0.3) if d_q encodes query-level info consistently

Saves per-query d_q tensor for all 3 constructions.
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

CAUSAL_PATH_MID = f"{DATA_DIR}/stage_l27_causal_midlayer_delta.pt"
CAUSAL_PATH_LOGIT = f"{DATA_DIR}/stage_l26_causal_delta.pt"
D_PATH = f"{DATA_DIR}/d_refuse.pt"
OUT_PATH = f"{DATA_DIR}/stage_l47_dq_sanity.pt"
SUMMARY_PATH = f"{DATA_DIR}/stage_l47_dq_sanity.txt"

# Which causal source to use (L27 midlayer probe — higher CV R²).
USE_MIDLAYER = True


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_delta_causal():
    """Δ_causal [N, L, E]. L27 cache may store as dict or tensor; handle both."""
    path = CAUSAL_PATH_MID if USE_MIDLAYER else CAUSAL_PATH_LOGIT
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.float(), path, []
    if isinstance(obj, dict):
        for k in ("delta", "causal_delta", "Δ_causal", "delta_causal"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                queries = []
                for qk in ("queries", "prompts", "query_list"):
                    if qk in obj and isinstance(obj[qk], list):
                        queries = obj[qk]
                        break
                return obj[k].float(), path, queries
        # Fallback: first 3D tensor
        for v in obj.values():
            if isinstance(v, torch.Tensor) and v.ndim == 3:
                return v.float(), path, []
    raise RuntimeError(f"no [N,L,E] tensor found in {path}")


def load_d_global(n_layers: int, hidden: int) -> torch.Tensor:
    obj = torch.load(D_PATH, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        d = obj
    elif isinstance(obj, dict):
        d = None
        for k in ("d_refuse", "direction", "d"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                d = obj[k]
                break
        if d is None:
            raise RuntimeError(f"no direction in {D_PATH}")
    else:
        raise RuntimeError(f"bad object {D_PATH}")
    if d.shape != (n_layers, hidden):
        raise RuntimeError(f"d shape {tuple(d.shape)} != ({n_layers},{hidden})")
    return d.float()


# ---------------------------------------------------------------------------
# Expert write-direction constructions
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_w_mean(model, n_layers: int, n_experts: int, hidden: int
                  ) -> torch.Tensor:
    """w_e = W_down[ℓ, e].mean(dim=1) ∈ R^hidden."""
    W = torch.zeros(n_layers, n_experts, hidden, dtype=torch.float32)
    for L in range(n_layers):
        for e in range(n_experts):
            Wd = model.model.layers[L].mlp.experts[e].down_proj.weight
            W[L, e] = Wd.detach().float().mean(dim=1).cpu()
    return W


@torch.no_grad()
def build_w_svd1(model, n_layers: int, n_experts: int, hidden: int
                  ) -> torch.Tensor:
    """w_e = first left-singular-vector (principal output axis)."""
    W = torch.zeros(n_layers, n_experts, hidden, dtype=torch.float32)
    for L in range(n_layers):
        for e in range(n_experts):
            Wd = model.model.layers[L].mlp.experts[e].down_proj.weight
            Wd_f = Wd.detach().float().cpu()  # [hidden, inter]
            # Use randomized SVD via torch.svd_lowrank for speed
            U, S, V = torch.svd_lowrank(Wd_f, q=1, niter=2)
            W[L, e] = U[:, 0] * S[0].sign()  # align sign with first SV
    return W


@torch.no_grad()
def build_w_dproj(model, d_global: torch.Tensor,
                  n_layers: int, n_experts: int, hidden: int
                  ) -> torch.Tensor:
    """w_e = W_down @ W_down^T @ d_global[ℓ]  (projection of d into expert
    column space, rescaled by singular values squared)."""
    W = torch.zeros(n_layers, n_experts, hidden, dtype=torch.float32)
    for L in range(n_layers):
        d_L = d_global[L]  # [hidden]
        for e in range(n_experts):
            Wd = model.model.layers[L].mlp.experts[e].down_proj.weight
            Wd_f = Wd.detach().float().cpu()  # [hidden, inter]
            v = Wd_f.T @ d_L                    # [inter]
            W[L, e] = Wd_f @ v                  # [hidden]
    return W


# ---------------------------------------------------------------------------
# Aggregation: d_q = Σ_e Δ · w_e
# ---------------------------------------------------------------------------

def aggregate_dq(delta: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """delta: [N, L, E];  w: [L, E, H]  ->  d_q: [N, L, H]"""
    # einsum: d_q[n, ℓ, h] = Σ_e delta[n, ℓ, e] * w[ℓ, e, h]
    return torch.einsum("nle,leh->nlh", delta, w)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diag_norms(dq: torch.Tensor, d_g: torch.Tensor):
    """||d_q[q, ℓ]|| mean/std per layer, and ratio to ||d_g[ℓ]||."""
    n_q, n_L, _ = dq.shape
    out = []
    d_g_norm = d_g.norm(dim=1)  # [L]
    dq_norm = dq.norm(dim=2)    # [N, L]
    for L in range(n_L):
        m = float(dq_norm[:, L].mean())
        s = float(dq_norm[:, L].std())
        r = m / max(float(d_g_norm[L]), 1e-9)
        out.append((L, m, s, float(d_g_norm[L]), r))
    return out


def diag_cos_global(dq: torch.Tensor, d_g: torch.Tensor):
    """cos(d_q[q, ℓ], d_g[ℓ]) — per-query / per-layer alignment."""
    n_q, n_L, H = dq.shape
    d_g_n = d_g / d_g.norm(dim=1, keepdim=True).clamp(min=1e-9)    # [L, H]
    dq_n = dq / dq.norm(dim=2, keepdim=True).clamp(min=1e-9)       # [N, L, H]
    cos = torch.einsum("nlh,lh->nl", dq_n, d_g_n)  # [N, L]
    return cos


def diag_pairwise_cos(dq: torch.Tensor, n_pairs: int = 500):
    """cos(d_q[i, ℓ], d_q[j, ℓ]) sampled pairs; returns [n_pairs, L]."""
    n_q, n_L, H = dq.shape
    dq_n = dq / dq.norm(dim=2, keepdim=True).clamp(min=1e-9)
    g = torch.Generator().manual_seed(123)
    cos = torch.zeros(n_pairs, n_L)
    for p in range(n_pairs):
        i, j = torch.randint(0, n_q, (2,), generator=g).tolist()
        while j == i:
            j = int(torch.randint(0, n_q, (1,), generator=g).item())
        cos[p] = (dq_n[i] * dq_n[j]).sum(dim=1)
    return cos


def diag_layer_consistency(dq: torch.Tensor, layer_pairs):
    """cos(d_q[q, a], d_q[q, b]) for (a, b) pairs; returns dict {(a,b): [N]}."""
    dq_n = dq / dq.norm(dim=2, keepdim=True).clamp(min=1e-9)
    out = {}
    for (a, b) in layer_pairs:
        out[(a, b)] = (dq_n[:, a] * dq_n[:, b]).sum(dim=1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt_stat(t: torch.Tensor) -> str:
    if t.numel() == 0:
        return "n/a"
    return (f"mean={t.mean().item():+.3f}  std={t.std().item():.3f}  "
            f"med={t.median().item():+.3f}  "
            f"min={t.min().item():+.3f}  max={t.max().item():+.3f}")


def main():
    t0 = time.time()
    print("=" * 72)
    print("L47: per-query direction d_q sanity check")
    print("=" * 72)

    print("\nLoading Δ_causal ...", flush=True)
    delta, delta_path, queries = load_delta_causal()
    print(f"  from {delta_path}", flush=True)
    print(f"  shape={tuple(delta.shape)}  nonzero%="
          f"{(delta != 0).float().mean().item()*100:.1f}", flush=True)
    N, n_layers, n_experts = delta.shape
    print(f"  queries metadata present: {bool(queries)}", flush=True)

    print(f"\nLoading {MODEL_ID} (CPU, bf16) ...", flush=True)
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": "cpu"},
        trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    hidden = model.config.hidden_size
    print(f"  loaded in {time.time()-t1:.0f}s  hidden={hidden}", flush=True)

    print(f"\nLoading d_global from {D_PATH} ...", flush=True)
    d_g = load_d_global(n_layers, hidden)
    print(f"  per-layer ||d_g||: [{', '.join(f'{d_g[L].norm():.3f}' for L in range(n_layers))}]",
          flush=True)

    print("\nBuilding expert write-direction tensors ...", flush=True)
    methods = {}
    t1 = time.time()
    methods["mean"] = build_w_mean(model, n_layers, n_experts, hidden)
    print(f"  mean   done in {time.time()-t1:.0f}s", flush=True)
    t1 = time.time()
    methods["dproj"] = build_w_dproj(model, d_g, n_layers, n_experts, hidden)
    print(f"  dproj  done in {time.time()-t1:.0f}s", flush=True)
    t1 = time.time()
    methods["svd1"] = build_w_svd1(model, n_layers, n_experts, hidden)
    print(f"  svd1   done in {time.time()-t1:.0f}s", flush=True)

    # Free model memory — don't need it anymore
    del model

    results = {}
    lines = []
    def w(s):
        lines.append(s); print(s, flush=True)

    for name, w_tensor in methods.items():
        w("")
        w("=" * 72)
        w(f"METHOD: {name}   w[L, E, H] shape={tuple(w_tensor.shape)}")
        w("=" * 72)

        dq = aggregate_dq(delta, w_tensor)  # [N, L, H]
        results[f"dq_{name}"] = dq
        results[f"w_{name}_summary_stats"] = {
            "norm_mean_per_L": w_tensor.norm(dim=2).mean(dim=1).tolist(),
        }

        # (A) Norms
        w("\n(A) ||d_q[q, ℓ]||   vs  ||d_global[ℓ]||")
        w(f"{'L':>3}  {'||d_q|| mean':>14}  {'std':>10}  "
          f"{'||d_g||':>10}  {'ratio':>8}")
        for L, m, s, dgn, r in diag_norms(dq, d_g):
            w(f"{L:>3}  {m:>14.4f}  {s:>10.4f}  {dgn:>10.4f}  {r:>8.3f}")

        # (B) cos with global
        cos_g = diag_cos_global(dq, d_g)  # [N, L]
        w("\n(B) cos(d_q[q, ℓ], d_global[ℓ])")
        w(f"    overall:  {_fmt_stat(cos_g.flatten())}")
        w(f"{'L':>3}  {'median':>10}  {'|cos|>0.9 %':>13}  {'|cos|<0.3 %':>13}")
        for L in range(n_layers):
            c = cos_g[:, L]
            med = float(c.median())
            high = float((c.abs() > 0.9).float().mean() * 100)
            low = float((c.abs() < 0.3).float().mean() * 100)
            w(f"{L:>3}  {med:>+10.3f}  {high:>13.1f}  {low:>13.1f}")

        # (C) Pairwise across queries
        pc = diag_pairwise_cos(dq, n_pairs=500)  # [500, L]
        w("\n(C) cos(d_q[i, ℓ], d_q[j, ℓ]) pairwise   (500 random pairs)")
        w(f"    overall:  {_fmt_stat(pc.flatten())}")
        w(f"{'L':>3}  {'median':>10}  {'iqr':>10}   "
          f"{'<0.5 %':>8}   {'>0.95 %':>9}")
        for L in range(n_layers):
            c = pc[:, L]
            med = float(c.median())
            q25 = float(c.quantile(0.25))
            q75 = float(c.quantile(0.75))
            low = float((c < 0.5).float().mean() * 100)
            high = float((c > 0.95).float().mean() * 100)
            w(f"{L:>3}  {med:>+10.3f}  {q75-q25:>10.3f}   "
              f"{low:>8.1f}   {high:>9.1f}")

        # (D) Layer consistency: adjacent layers
        lp = [(L, L+1) for L in range(n_layers-1)]
        lc = diag_layer_consistency(dq, lp)
        w("\n(D) cos(d_q[q, ℓ], d_q[q, ℓ+1])   adjacent-layer consistency")
        for (a, b), t in lc.items():
            w(f"  ℓ={a:>2}→{b:>2}:  {_fmt_stat(t)}")

    # Cross-method consistency (is dproj's d_q similar to mean's d_q?)
    lines.append("")
    lines.append("=" * 72)
    lines.append("CROSS-METHOD: cos(d_q_{method_i}, d_q_{method_j})")
    lines.append("=" * 72)
    pairs = [("mean", "dproj"), ("mean", "svd1"), ("dproj", "svd1")]
    for a, b in pairs:
        dq_a = results[f"dq_{a}"]
        dq_b = results[f"dq_{b}"]
        dq_a_n = dq_a / dq_a.norm(dim=2, keepdim=True).clamp(min=1e-9)
        dq_b_n = dq_b / dq_b.norm(dim=2, keepdim=True).clamp(min=1e-9)
        cos_ab = (dq_a_n * dq_b_n).sum(dim=2)  # [N, L]
        lines.append(f"\n  {a} vs {b}:  {_fmt_stat(cos_ab.flatten())}")
        for L in range(n_layers):
            lines.append(
                f"    L={L:>2}:  {_fmt_stat(cos_ab[:, L])}")

    # Verdict
    lines.append("")
    lines.append("=" * 72)
    lines.append("VERDICT")
    lines.append("=" * 72)

    # Use dproj method for verdict (most principled)
    dq = results["dq_dproj"]
    cos_g = diag_cos_global(dq, d_g)
    pc = diag_pairwise_cos(dq, n_pairs=1000)
    overall_cos_g = cos_g.flatten()
    overall_pc = pc.flatten()

    lines.append(f"  (using dproj construction)")
    lines.append(f"  cos(d_q, d_global) overall median: "
                 f"{float(overall_cos_g.median()):+.3f}")
    lines.append(f"  pairwise cos(d_q_i, d_q_j) overall median: "
                 f"{float(overall_pc.median()):+.3f}")

    cg_med = float(overall_cos_g.abs().median())
    pc_med = float(overall_pc.median())
    fail_a = cg_med > 0.95   # d_q aligns too tightly with d_global
    fail_b = pc_med > 0.90   # d_q collapses across queries

    if fail_a and fail_b:
        lines.append("  FAIL: d_q ≈ d_global and all queries collapse.")
        lines.append("        usage 1 has no per-query signal worth pursuing.")
    elif fail_a:
        lines.append("  PARTIAL FAIL: d_q ≈ d_global (no new direction) but "
                     "queries vary.")
        lines.append("        direction is just scaled d_global.")
    elif fail_b:
        lines.append("  PARTIAL FAIL: d_q varies from d_global but collapses "
                     "across queries.")
        lines.append("        constant query-independent shift.")
    else:
        lines.append("  PASS:  d_q (a) differs from d_global, (b) differs "
                     "across queries.")
        lines.append("         usage 1 worth full experiment.")

    # Save
    torch.save({
        "delta_shape": list(delta.shape),
        "d_global_path": D_PATH,
        "delta_path": delta_path,
        **results,
    }, OUT_PATH)
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-30:]), flush=True)
    print(f"\nSaved -> {OUT_PATH}")
    print(f"Summary -> {SUMMARY_PATH}")
    print(f"\nTotal wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
