"""
L44: Dynamic vs static A+/A- expert sets.

Question: SteerMoE picks a GLOBAL top-25 A+ and bottom-25 A- from pooled
BeaverTails counts. Is this set stable, or does the 'good-expert / bad-expert'
identity shift across subpopulations of queries?

If A+ is highly stable across random buckets and across semantic categories,
static (global) A+/A- is sufficient and per-query gating is overkill.
If it shifts noticeably, per-query A+ identification (our L23/L25c direction)
remains motivated.

Decision rule:
    - mean pairwise Jaccard(A+_bucket_i, A+_bucket_j) > 0.7
      → stable; static A+ is fine; per-query gating is NOT needed
    - 0.3 <= Jaccard <= 0.7 → semi-dynamic; per-query gating could help
    - Jaccard < 0.3 → highly dynamic; per-query A+ identification crucial

Approach (single forward pass over 500 safe + 500 unsafe BeaverTails rows):
    1. For each sample, record per-sample per-layer-per-expert fire counts
       on RESPONSE tokens only (same as L40 hook setup).
    2. Random bucketing: split 500 safe + 500 unsafe into B=10 random buckets.
       Compute Δ_bucket and A+_bucket / A-_bucket per bucket. Measure pairwise
       Jaccard across buckets.
    3. Category bucketing: group BeaverTails rows by top-1 category (14 known
       categories). Compute per-category Δ, A+, A-. Measure cross-category
       Jaccard.
    4. Null baseline: 10 random 25-subsets of the 1024-expert pool; expected
       Jaccard for independent picks.

Env vars:
    L44_N_SAFE     (default 500)
    L44_N_UNSAFE   (default 500)
    L44_K          (default 25, matches L40/L42 choices)
    L44_N_BUCKETS  (default 10)
    L44_SEED       (default 42, matches L40)
"""
from __future__ import annotations

import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("HF_HOME", "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/orange/qi855292.ucf/ji757406.ucf/cache/torch")

from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
OUT_PATH = f"{DATA_DIR}/stage_l44_dynamic_vs_static.pt"
SUMMARY_PATH = f"{DATA_DIR}/stage_l44_dynamic_vs_static.txt"

N_SAFE = int(os.environ.get("L44_N_SAFE", "500"))
N_UNSAFE = int(os.environ.get("L44_N_UNSAFE", "500"))
K_A = int(os.environ.get("L44_K", "25"))
N_BUCKETS = int(os.environ.get("L44_N_BUCKETS", "10"))
SEED = int(os.environ.get("L44_SEED", "42"))
SEQ_LEN_SKIP_THRESHOLD = 1024


def tokenize_prompt_response_boundary(tok, prompt: str, response: str):
    user_msg = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(
        user_msg, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + response
    boundary_char = len(prompt_text)
    enc = tok(full_text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    asst_start = len(offsets)
    for i, (s, _e) in enumerate(offsets):
        if s >= boundary_char:
            asst_start = i
            break
    return enc, asst_start


def sample_beavertails_with_category(n_safe: int, n_unsafe: int, seed: int):
    """Same SteerMoE seed + ordering as L40 (for A+/A- comparability).

    Returns: list of dicts with keys (prompt, response, is_safe, categories).
    `categories` is the raw dict from BeaverTails; we also compute a
    `primary_category` string = first True category alphabetically (or 'none').
    """
    print("Loading BeaverTails 30k_train ...", flush=True)
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    print(f"  total rows: {len(ds)}", flush=True)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    safe, unsafe = [], []
    for i in indices:
        if len(safe) >= n_safe and len(unsafe) >= n_unsafe:
            break
        row = ds[i]
        cats = row.get("category", {}) or {}
        active = sorted([c for c, v in cats.items() if v])
        primary = active[0] if active else "none"
        rec = {
            "prompt": row["prompt"],
            "response": row["response"],
            "is_safe": bool(row["is_safe"]),
            "primary_category": primary,
            "active_categories": active,
        }
        if row["is_safe"] and len(safe) < n_safe:
            safe.append(rec)
        elif (not row["is_safe"]) and len(unsafe) < n_unsafe:
            unsafe.append(rec)
    print(f"  sampled: {len(safe)} safe + {len(unsafe)} unsafe", flush=True)
    return safe, unsafe


@torch.no_grad()
def extract_per_sample_counts(model, tok, samples: list):
    """Per-sample top-k fire counts on response tokens.

    Returns:
        fires:  [n_samples, n_layers, n_experts] float32, per-sample counts
                over response tokens (NOT normalized)
        tok_len:[n_samples] int64, response token count per sample (0 if skipped)
        ok:     [n_samples] bool, True if sample was processed (not skipped)
    """
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    top_k = model.config.num_experts_per_tok
    N = len(samples)

    fires = torch.zeros(N, n_layers, n_experts, dtype=torch.float32)
    tok_len = torch.zeros(N, dtype=torch.long)
    ok = torch.zeros(N, dtype=torch.bool)

    route_cap = {li: None for li in range(n_layers)}

    def make_route_hook(li):
        def post(module, inp, out):
            if isinstance(out, torch.Tensor):
                logits = out
            elif isinstance(out, tuple):
                hs = inp[0] if isinstance(inp, tuple) else inp
                hs_flat = hs.reshape(-1, hs.shape[-1])
                logits = F.linear(hs_flat, module.weight,
                                  getattr(module, "bias", None))
            else:
                return
            _, topk = torch.topk(logits.float(), top_k, dim=-1)
            route_cap[li] = topk.detach().cpu()
        return post

    handles = []
    for li in range(n_layers):
        handles.append(
            model.model.layers[li].mlp.gate.register_forward_hook(make_route_hook(li)))
    try:
        for si, rec in enumerate(samples):
            enc, asst_start = tokenize_prompt_response_boundary(
                tok, rec["prompt"], rec["response"])
            if enc["input_ids"].shape[1] > SEQ_LEN_SKIP_THRESHOLD:
                continue
            inputs = {k: v.to(model.device) for k, v in enc.items()}
            for li in range(n_layers):
                route_cap[li] = None
            _ = model(**inputs, use_cache=False)
            seq_len = inputs["input_ids"].shape[1]
            asst_len = max(0, seq_len - asst_start)
            if asst_len == 0:
                continue
            asst_end = asst_start + asst_len
            for li in range(n_layers):
                topk_full = route_cap[li]
                if topk_full is None or topk_full.shape[0] != seq_len:
                    continue
                topk_asst = topk_full[asst_start:asst_end]  # [asst_len, top_k]
                mask = torch.zeros(topk_asst.shape[0], n_experts,
                                   dtype=torch.float32)
                mask.scatter_(1, topk_asst.to(torch.long), 1.0)
                fires[si, li] = mask.sum(dim=0)
            tok_len[si] = asst_len
            ok[si] = True
            torch.cuda.empty_cache()
            if (si + 1) % 100 == 0:
                print(f"  [extract] {si+1}/{N} ok={int(ok[:si+1].sum())}",
                      flush=True)
    finally:
        for h in handles:
            h.remove()
    print(f"  done. ok={int(ok.sum())}/{N}", flush=True)
    return fires, tok_len, ok


def top_k_global(delta_LE: torch.Tensor, k: int, descending: bool) -> set:
    n_layers, n_experts = delta_LE.shape
    flat = delta_LE.reshape(-1)
    order = torch.argsort(flat, descending=descending)[:k]
    return {int(j.item()) for j in order}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    u = a | b
    if not u:
        return float("nan")
    return len(a & b) / len(u)


def compute_delta(fires_r: torch.Tensor, fires_c: torch.Tensor,
                  lens_r: torch.Tensor, lens_c: torch.Tensor):
    sum_r = fires_r.sum(dim=0)  # [L, E]
    sum_c = fires_c.sum(dim=0)
    tot_r = max(int(lens_r.sum().item()), 1)
    tot_c = max(int(lens_c.sum().item()), 1)
    return sum_r / tot_r - sum_c / tot_c


def null_jaccard_baseline(total_experts: int, k: int, n_draws: int,
                          seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    js = []
    pool = np.arange(total_experts)
    samples = [set(rng.choice(pool, size=k, replace=False).tolist())
               for _ in range(n_draws)]
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            js.append(jaccard(samples[i], samples[j]))
    js = np.array(js)
    return float(js.mean()), float(js.std())


def pairwise_jaccard(sets: list) -> tuple[float, float, list]:
    """All unordered pairs; returns (mean, std, list of values)."""
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            v = jaccard(sets[i], sets[j])
            if not np.isnan(v):
                vals.append(v)
    arr = np.array(vals) if vals else np.array([float("nan")])
    return float(arr.mean()), float(arr.std()), vals


def random_bucket_analysis(fires_s: torch.Tensor, lens_s: torch.Tensor,
                           ok_s: torch.Tensor, fires_u: torch.Tensor,
                           lens_u: torch.Tensor, ok_u: torch.Tensor,
                           B: int, K: int, seed: int):
    rng = np.random.default_rng(seed)
    ns = int(ok_s.sum().item())
    nu = int(ok_u.sum().item())
    idx_s = np.array([i for i in range(len(ok_s)) if ok_s[i]])
    idx_u = np.array([i for i in range(len(ok_u)) if ok_u[i]])
    rng.shuffle(idx_s)
    rng.shuffle(idx_u)
    buckets_s = np.array_split(idx_s, B)
    buckets_u = np.array_split(idx_u, B)
    A_plus_sets, A_minus_sets, deltas = [], [], []
    for b in range(B):
        fr = fires_s[buckets_s[b]]
        lr = lens_s[buckets_s[b]]
        fc = fires_u[buckets_u[b]]
        lc = lens_u[buckets_u[b]]
        d = compute_delta(fr, fc, lr, lc)
        deltas.append(d)
        A_plus_sets.append(top_k_global(d, K, descending=True))
        A_minus_sets.append(top_k_global(d, K, descending=False))
    return A_plus_sets, A_minus_sets, deltas


def category_bucket_analysis(samples: list, fires: torch.Tensor,
                             lens: torch.Tensor, ok: torch.Tensor,
                             is_safe: bool, K: int, counter_fires, counter_lens,
                             min_bucket: int = 10):
    """Group by primary_category for samples matching `is_safe`; Δ vs the
    full opposite-class pool. Returns dict cat -> set(A+) and set(A-).
    """
    # group indices
    groups = defaultdict(list)
    for i in range(len(samples)):
        if samples[i]["is_safe"] != is_safe:
            continue
        if not ok[i]:
            continue
        groups[samples[i]["primary_category"]].append(i)
    # filter small buckets
    groups = {c: v for c, v in groups.items() if len(v) >= min_bucket}
    A_plus_by_cat, A_minus_by_cat = {}, {}
    for cat, idx in groups.items():
        fr = fires[idx] if is_safe else counter_fires
        lr = lens[idx] if is_safe else counter_lens
        fc = counter_fires if is_safe else fires[idx]
        lc = counter_lens if is_safe else lens[idx]
        d = compute_delta(fr, fc, lr, lc)
        A_plus_by_cat[cat] = top_k_global(d, K, descending=True)
        A_minus_by_cat[cat] = top_k_global(d, K, descending=False)
    return A_plus_by_cat, A_minus_by_cat, {c: len(v) for c, v in groups.items()}


def summarize(results: dict) -> str:
    lines = []
    def w(s):
        lines.append(s)
        print(s, flush=True)

    n_layers = results["n_layers"]
    n_experts = results["n_experts"]
    total = n_layers * n_experts
    K = results["K"]
    B = results["B"]

    w("=" * 78)
    w(f"L44: Dynamic vs static A+/A- — K={K}, random buckets B={B}")
    w("=" * 78)
    w(f"total expert slots: {total}  (layers={n_layers} × experts={n_experts})")
    w(f"processed: safe={int(results['ok_safe'].sum())}/{results['n_safe']}  "
      f"unsafe={int(results['ok_unsafe'].sum())}/{results['n_unsafe']}")
    w("")

    # Null baseline
    null_mean, null_std = results["null_jaccard"]
    w(f"Null Jaccard (random K={K} subsets of {total}): "
      f"{null_mean:.4f} ± {null_std:.4f}")
    w("")

    # Global A+/A-
    A_glob = results["A_plus_global"]
    Am_glob = results["A_minus_global"]
    w(f"Global A+ size: {len(A_glob)}   A- size: {len(Am_glob)}")
    overlap_pm = len(A_glob & Am_glob)
    w(f"Global A+ ∩ A- overlap: {overlap_pm} (should be 0)")
    w("")

    # Random bucket Jaccard
    ap_rb = results["A_plus_random_buckets"]
    am_rb = results["A_minus_random_buckets"]
    mu_p, sd_p, vals_p = pairwise_jaccard(ap_rb)
    mu_m, sd_m, vals_m = pairwise_jaccard(am_rb)
    w("Random-bucket analysis (query-level variance)")
    w(f"  A+  pairwise Jaccard: {mu_p:.4f} ± {sd_p:.4f}  (n={len(vals_p)})")
    w(f"  A-  pairwise Jaccard: {mu_m:.4f} ± {sd_m:.4f}  (n={len(vals_m)})")
    # Jaccard vs global
    j_p_vs_glob = [jaccard(s, A_glob) for s in ap_rb]
    j_m_vs_glob = [jaccard(s, Am_glob) for s in am_rb]
    w(f"  A+  bucket-vs-global Jaccard: "
      f"mean={np.nanmean(j_p_vs_glob):.4f} "
      f"min={np.nanmin(j_p_vs_glob):.4f} "
      f"max={np.nanmax(j_p_vs_glob):.4f}")
    w(f"  A-  bucket-vs-global Jaccard: "
      f"mean={np.nanmean(j_m_vs_glob):.4f} "
      f"min={np.nanmin(j_m_vs_glob):.4f} "
      f"max={np.nanmax(j_m_vs_glob):.4f}")
    w("")

    # Category buckets
    w("Category-bucket analysis (topic-level variance)")
    for cls, label in [("safe", "A+ by safe-category"), ("unsafe", "A- by unsafe-category")]:
        if cls == "safe":
            by_cat = results["A_plus_by_safe_cat"]
            glob = A_glob
            sizes = results["safe_cat_sizes"]
        else:
            by_cat = results["A_minus_by_unsafe_cat"]
            glob = Am_glob
            sizes = results["unsafe_cat_sizes"]
        w(f"  {label}:  {len(by_cat)} categories kept (min 10 samples)")
        cats = sorted(by_cat.keys())
        sets_list = [by_cat[c] for c in cats]
        mu, sd, _ = pairwise_jaccard(sets_list)
        j_vs_glob = [jaccard(by_cat[c], glob) for c in cats]
        w(f"    pairwise Jaccard: {mu:.4f} ± {sd:.4f}   "
          f"vs-global mean: {np.nanmean(j_vs_glob):.4f}")
        for c in cats:
            w(f"    {c:<40s} n={sizes[c]:>4d}  J(cat,global)={jaccard(by_cat[c], glob):.3f}")
    w("")

    # Interpretation
    w("Interpretation:")
    if mu_p > 0.7:
        w(f"  Random-bucket A+ Jaccard={mu_p:.2f} HIGH → STATIC A+ stable across queries.")
        w(f"  → Per-query gating adds little; SteerMoE-style static set suffices.")
    elif mu_p > 0.3:
        w(f"  Random-bucket A+ Jaccard={mu_p:.2f} MEDIUM → SEMI-DYNAMIC.")
        w(f"  → Per-query gating can help at the margin.")
    else:
        w(f"  Random-bucket A+ Jaccard={mu_p:.2f} LOW → HIGHLY DYNAMIC.")
        w(f"  → Per-query A+ identification is necessary; static A+ too coarse.")
    lift = mu_p / max(null_mean, 1e-9)
    w(f"  Lift over null: {lift:.1f}× (static A+ is {lift:.1f}x more stable than random)")

    return "\n".join(lines)


def main() -> None:
    print(f"[config] N_SAFE={N_SAFE} N_UNSAFE={N_UNSAFE} K={K_A} "
          f"B={N_BUCKETS} SEED={SEED}", flush=True)

    safe, unsafe = sample_beavertails_with_category(N_SAFE, N_UNSAFE, SEED)
    all_samples = safe + unsafe

    # Quick category histogram
    cat_hist = defaultdict(int)
    for s in all_samples:
        cat_hist[s["primary_category"]] += 1
    print(f"  category distribution: {dict(sorted(cat_hist.items(), key=lambda x: -x[1]))}",
          flush=True)

    print(f"\nLoading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts

    print("\n=== Extracting per-sample counts (safe samples) ===", flush=True)
    fires_s, lens_s, ok_s = extract_per_sample_counts(model, tok, safe)
    print("\n=== Extracting per-sample counts (unsafe samples) ===", flush=True)
    fires_u, lens_u, ok_u = extract_per_sample_counts(model, tok, unsafe)

    # Global A+/A-
    delta_global = compute_delta(fires_s, fires_u, lens_s, lens_u)
    A_plus_global = top_k_global(delta_global, K_A, descending=True)
    A_minus_global = top_k_global(delta_global, K_A, descending=False)
    print(f"\nGlobal A+ {len(A_plus_global)}  A- {len(A_minus_global)}  "
          f"overlap {len(A_plus_global & A_minus_global)}", flush=True)

    # Random-bucket analysis
    print(f"\n=== Random {N_BUCKETS}-bucket analysis ===", flush=True)
    ap_rb, am_rb, _ = random_bucket_analysis(
        fires_s, lens_s, ok_s, fires_u, lens_u, ok_u, N_BUCKETS, K_A, SEED)

    # Category analysis
    print("\n=== Category-bucket analysis ===", flush=True)
    all_fires = torch.cat([fires_s, fires_u], dim=0)
    all_lens = torch.cat([lens_s, lens_u], dim=0)
    all_ok = torch.cat([ok_s, ok_u], dim=0)
    # Safe-category: A+ per safe category (Δ vs all-unsafe pool)
    A_plus_by_safe_cat, _, safe_cat_sizes = category_bucket_analysis(
        all_samples, all_fires, all_lens, all_ok, is_safe=True,
        K=K_A, counter_fires=fires_u, counter_lens=lens_u)
    # Unsafe-category: A- per unsafe category (Δ vs all-safe pool; A-)
    _, A_minus_by_unsafe_cat, unsafe_cat_sizes = category_bucket_analysis(
        all_samples, all_fires, all_lens, all_ok, is_safe=False,
        K=K_A, counter_fires=fires_s, counter_lens=lens_s)

    # Null baseline
    null_mean, null_std = null_jaccard_baseline(
        n_layers * n_experts, K_A, n_draws=N_BUCKETS, seed=SEED)

    results = {
        "n_layers": n_layers,
        "n_experts": n_experts,
        "K": K_A,
        "B": N_BUCKETS,
        "n_safe": len(safe),
        "n_unsafe": len(unsafe),
        "ok_safe": ok_s,
        "ok_unsafe": ok_u,
        "delta_global": delta_global,
        "A_plus_global": A_plus_global,
        "A_minus_global": A_minus_global,
        "A_plus_random_buckets": ap_rb,
        "A_minus_random_buckets": am_rb,
        "A_plus_by_safe_cat": A_plus_by_safe_cat,
        "A_minus_by_unsafe_cat": A_minus_by_unsafe_cat,
        "safe_cat_sizes": safe_cat_sizes,
        "unsafe_cat_sizes": unsafe_cat_sizes,
        "null_jaccard": (null_mean, null_std),
    }
    summary = summarize(results)

    # Save raw per-sample counts too (for reuse by later stages)
    if os.path.exists(OUT_PATH):
        bak = f"{OUT_PATH}.bak.{int(time.time())}"
        os.rename(OUT_PATH, bak)
        print(f"Existing OUT_PATH backed up -> {bak}", flush=True)
    torch.save({
        **{k: v for k, v in results.items()
           if k not in ("ok_safe", "ok_unsafe", "delta_global")},
        "ok_safe": ok_s,
        "ok_unsafe": ok_u,
        "delta_global": delta_global,
        "fires_safe": fires_s,
        "lens_safe": lens_s,
        "fires_unsafe": fires_u,
        "lens_unsafe": lens_u,
        "safe_categories": [s["primary_category"] for s in safe],
        "unsafe_categories": [s["primary_category"] for s in unsafe],
        "model_id": MODEL_ID,
        "seed": SEED,
    }, OUT_PATH)
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved -> {OUT_PATH}", flush=True)
    print(f"Summary -> {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
