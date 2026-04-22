"""
L47c analysis: predict Usage 1 viability per attack without running generation.

Load:
  - L47b Δ on jailbreak (N=400, L=16, E=64)
  - L47 w_e tensors (mean / dproj / svd1)  from stage_l47_dq_sanity.pt
  - d_refuse global

Compute per-attack:
  - d_q[q, ℓ] = Σ_e Δ[q, ℓ, e] · w_e[ℓ, e]     — 3 variants
  - ||d_q|| per attack per layer (vs d_global)
  - cos(mean_d_q, d_global) per attack per layer
  - pairwise cos(d_q_i, d_q_j) within attack (variability)
  - cross-attack cos of mean d_q

Verdict:
  - If deepinception ||d_q|| ≈ 0 across methods → Usage 1 will fail there
    (no signal to steer with) → per-query approach fundamentally limited
  - If some method gives nonzero d_q on deepinception → run full experiment
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
JB_PATH = f"{DATA_DIR}/stage_l47b_jailbreak_causal_delta.pt"
DQ_SANITY_PATH = f"{DATA_DIR}/stage_l47_dq_sanity.pt"
D_REFUSE_PATH = f"{DATA_DIR}/d_refuse.pt"
OUT_TXT = f"{DATA_DIR}/stage_l47c_jailbreak_dq_analysis.txt"
OUT_PT = f"{DATA_DIR}/stage_l47c_jailbreak_dq.pt"


def main():
    print("=" * 72)
    print("L47c: per-attack d_q analysis (Usage 1 feasibility probe)")
    print("=" * 72)

    print(f"\nLoading L47b jailbreak Δ from {JB_PATH}", flush=True)
    jb = torch.load(JB_PATH, map_location="cpu", weights_only=False)
    delta = jb["delta"].float()           # [N, L, E]
    attacks = jb["attack_labels"]         # list[str]
    attack_set = jb["attacks"]
    N, n_L, n_E = delta.shape
    print(f"  shape={tuple(delta.shape)}  attacks={attack_set}", flush=True)

    print(f"\nLoading w_e tensors from {DQ_SANITY_PATH}", flush=True)
    sanity = torch.load(DQ_SANITY_PATH, map_location="cpu", weights_only=False)

    # The L47 sanity saved d_q directly, not w tensors. Reconstruct w from
    # the stored keys if needed, but actually the path is: rebuild w_e from
    # L47 using the cached model isn't in that file. Check what's there.
    print(f"  keys: {list(sanity.keys())}", flush=True)

    # L47 saved dq_{method}. It didn't save w tensors. We need to either
    # (a) re-run w construction (requires model on CPU, ~2min),
    # (b) pull them from the .pt if they were kept,
    # (c) just use delta directly (no w_e) to test: ||Σ Δ||_F per attack.
    # Actually we can get a GOOD signal-presence check just from delta norms.
    # For d_q norms we do need w_e. Let's inspect sanity keys carefully.

    has_w = any(k.startswith("w_") and isinstance(sanity[k], torch.Tensor)
                for k in sanity)
    print(f"  stores w tensors directly: {has_w}", flush=True)

    if not has_w:
        # Build w_e from a lightweight CPU model load.
        from transformers import AutoModelForCausalLM
        print("\n  L47 file lacks w tensors — reloading model (CPU) ...",
              flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            "allenai/OLMoE-1B-7B-0924-Instruct",
            torch_dtype=torch.bfloat16, device_map={"": "cpu"},
            trust_remote_code=True, low_cpu_mem_usage=True)
        model.eval()
        hidden = model.config.hidden_size
        n_layers_m = model.config.num_hidden_layers
        n_experts_m = model.config.num_experts
        assert (n_layers_m, n_experts_m) == (n_L, n_E)

        # Load d_refuse
        d_refuse = torch.load(D_REFUSE_PATH, map_location="cpu",
                              weights_only=False).float()

        def build_w_mean():
            W = torch.zeros(n_L, n_E, hidden, dtype=torch.float32)
            for L in range(n_L):
                for e in range(n_E):
                    Wd = model.model.layers[L].mlp.experts[e].down_proj.weight
                    W[L, e] = Wd.detach().float().mean(dim=1).cpu()
            return W

        def build_w_dproj():
            W = torch.zeros(n_L, n_E, hidden, dtype=torch.float32)
            for L in range(n_L):
                d_L = d_refuse[L]
                for e in range(n_E):
                    Wd = model.model.layers[L].mlp.experts[e].down_proj.weight
                    Wd_f = Wd.detach().float().cpu()
                    v = Wd_f.T @ d_L
                    W[L, e] = Wd_f @ v
            return W

        print("  building w_mean ...", flush=True)
        w_mean = build_w_mean()
        print("  building w_dproj ...", flush=True)
        w_dproj = build_w_dproj()
        # skip svd1 (expensive + L47 showed it disagrees)
        del model
        methods = {"mean": w_mean, "dproj": w_dproj}

        d_g = d_refuse
    else:
        d_g = torch.load(D_REFUSE_PATH, map_location="cpu",
                         weights_only=False).float()
        methods = {k[2:]: sanity[k] for k in sanity
                   if k.startswith("w_") and isinstance(sanity[k], torch.Tensor)}

    print(f"  methods available: {list(methods.keys())}", flush=True)

    # --------------------------------------------------------
    # Compute d_q per method
    # --------------------------------------------------------
    dq_by_method = {}
    for name, w in methods.items():
        dq = torch.einsum("nle,leh->nlh", delta, w)  # [N, L, H]
        dq_by_method[name] = dq
    print("\ndq tensors computed:", flush=True)
    for name, dq in dq_by_method.items():
        print(f"  {name}: {tuple(dq.shape)}", flush=True)

    lines = []
    def w(s):
        lines.append(s); print(s, flush=True)

    # --------------------------------------------------------
    # Per-attack ||d_q|| summary
    # --------------------------------------------------------
    for name, dq in dq_by_method.items():
        w("")
        w("=" * 72)
        w(f"METHOD: {name}")
        w("=" * 72)
        w(f"\n(A) mean ||d_q[q, ℓ]||  per attack × layer"
          f"   (global ||d_g[ℓ]||: "
          f"{' '.join(f'{d_g[L].norm():.2f}' for L in range(n_L))})")
        hdr = "  L  " + "  ".join(f"{a[:14]:>14}" for a in attack_set)
        w(hdr)
        norms = dq.norm(dim=2)  # [N, L]
        for L in range(n_L):
            cells = []
            for a in attack_set:
                mask = [i for i, al in enumerate(attacks) if al == a]
                v = float(norms[mask, L].mean()) if mask else 0.0
                cells.append(f"{v:>14.4f}")
            w(f" {L:>3} " + "  ".join(cells))

        # cos(mean d_q, d_g)
        w(f"\n(B) cos( mean_attack d_q[ℓ], d_global[ℓ] )")
        w(hdr)
        for L in range(n_L):
            dg_L = d_g[L]
            dg_n = dg_L / dg_L.norm().clamp(min=1e-9)
            cells = []
            for a in attack_set:
                mask = [i for i, al in enumerate(attacks) if al == a]
                if not mask:
                    cells.append(f"{'n/a':>14}")
                    continue
                v_mean = dq[mask, L].mean(dim=0)
                vn = v_mean / v_mean.norm().clamp(min=1e-9)
                c = float((vn * dg_n).sum())
                cells.append(f"{c:>+14.3f}")
            w(f" {L:>3} " + "  ".join(cells))

        # pairwise within-attack variability (averaged over layers 0-10)
        w(f"\n(C) pairwise cos(d_q_i, d_q_j)  within attack  "
          f"(median over 200 pairs × layers 0-10)")
        import random as _rnd
        rng = _rnd.Random(17)
        dq_n = dq / dq.norm(dim=2, keepdim=True).clamp(min=1e-9)
        for a in attack_set:
            mask = [i for i, al in enumerate(attacks) if al == a]
            if len(mask) < 2:
                w(f"  {a:<22}: n<2, skip")
                continue
            pairs = []
            for _ in range(200):
                i, j = rng.sample(mask, 2)
                # median across layers 0..10
                c = (dq_n[i, :11] * dq_n[j, :11]).sum(dim=1).median().item()
                pairs.append(c)
            t = torch.tensor(pairs)
            w(f"  {a:<22}: median={t.median().item():+.3f}  "
              f"mean={t.mean().item():+.3f}  "
              f"<0.5%={(t<0.5).float().mean().item()*100:.0f}  "
              f">0.95%={(t>0.95).float().mean().item()*100:.0f}")

        # cross-attack mean d_q cosine (over full L×H flattened)
        w(f"\n(D) cos( mean_attack_A d_q, mean_attack_B d_q )  full L·H")
        names = list(attack_set)
        means = {}
        for a in names:
            mask = [i for i, al in enumerate(attacks) if al == a]
            means[a] = dq[mask].mean(dim=0).flatten() if mask else None
        w("           " + "  ".join(f"{n[:14]:>14}" for n in names))
        for a in names:
            va = means[a]
            row = []
            for b in names:
                vb = means[b]
                if va is None or vb is None:
                    row.append(f"{'n/a':>14}")
                else:
                    c = float(F.cosine_similarity(
                        va.unsqueeze(0), vb.unsqueeze(0)).item())
                    row.append(f"{c:>+14.3f}")
            w(f"  {a[:7]:<7}  " + "  ".join(row))

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------
    w("")
    w("=" * 72)
    w("VERDICT (per-attack d_q feasibility for Usage 1 steering)")
    w("=" * 72)

    # Use dproj (most principled). Compute d_q norm on deepinception layer-
    # averaged, compare to none.
    if "dproj" in dq_by_method:
        dq = dq_by_method["dproj"]
    else:
        dq = next(iter(dq_by_method.values()))

    norms = dq.norm(dim=2)  # [N, L]
    # For each attack: mean ||d_q|| over L=0..10
    attack_norm = {}
    for a in attack_set:
        mask = [i for i, al in enumerate(attacks) if al == a]
        attack_norm[a] = float(norms[mask, :11].mean()) if mask else 0.0
    w("\n  mean ||d_q||  (dproj, L=0..10):")
    for a, v in attack_norm.items():
        w(f"    {a:<22}: {v:.4f}")

    # Ratio deepinception / none
    if "none" in attack_norm and "new_deepinception" in attack_norm:
        rr = attack_norm["new_deepinception"] / max(attack_norm["none"], 1e-9)
        w(f"\n  ratio  deepinception / none : {rr:.3f}")
        if rr < 0.2:
            w("  VERDICT: deepinception signal is <20% of baseline.")
            w("           per-query steering will likely FAIL on deepinception.")
            w("           Usage 1 is fundamentally limited on worst-case attacks.")
            w("           -> DO NOT run full L47c experiment; pivot to other ideas.")
        elif rr < 0.5:
            w("  VERDICT: deepinception signal is 20-50% of baseline.")
            w("           marginal case; could still help but need calibration.")
            w("           -> consider small-scale L47c with normalized d_q.")
        else:
            w("  VERDICT: deepinception signal >= 50% of baseline.")
            w("           per-query has enough signal. -> run L47c generation exp.")

    # Save
    torch.save({
        "dq_by_method": dq_by_method,
        "attack_labels": attacks,
        "attacks": attack_set,
        "attack_norm_dproj_mean_L0_10": attack_norm,
    }, OUT_PT)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT_PT}")
    print(f"Summary -> {OUT_TXT}")


if __name__ == "__main__":
    main()
