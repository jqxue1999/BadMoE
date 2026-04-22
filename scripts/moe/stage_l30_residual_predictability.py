"""
Stage L30: Residual predictability diagnostic.

Math-level falsifier for the per-query hypothesis. The user's motivation
is: "different queries need different safe-team experts." For this to be
a real signal (not just noise), the per-query deviation from the global
average must be PREDICTABLE from the query's semantics (embedding).

Test
----
For each training query q, compute residual:

    r_q[l, e] = Δ_q[l, e] - Δ_global[l, e]

where Δ_global is the pooled global (L25c token-level mean-diff) and Δ_q
is this query's own mean-diff. Then for each (layer, expert) position,
fit a ridge regression:

    r_q[l, e] ≈ w[l, e]^T · embed(q) + b[l, e]

Measure cross-validated R² (5-fold). Two outcomes:

  * R² > 0 for a meaningful fraction of (l, e) positions →
    per-query structure IS predictable from embedding; retrieval-based
    per-query routing has a chance to work with better methods.
  * R² ≈ 0 everywhere → per-query "structure" is noise relative to
    query semantics; the entire retrieval approach is mathematically
    doomed in this data regime.

This is cheap, runs offline from caches only (no GPU forward needed).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
L25C_CACHE = f"{DATA_DIR}/stage_l25c_per_query_rd.pt"
L26_CAUSAL_CACHE = f"{DATA_DIR}/stage_l26_causal_delta.pt"
L27_MIDLAYER_CACHE = f"{DATA_DIR}/stage_l27_causal_midlayer_delta.pt"
OUT_PATH = f"{DATA_DIR}/stage_l30_residual_report.pt"

SEED = 42
N_FOLDS = 5
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]  # tune on CV


def ridge_solve(X, y, alpha):
    """X: [n, d], y: [n]. Returns w, b."""
    n, d = X.shape
    X1 = np.concatenate([X, np.ones((n, 1))], axis=1)  # [n, d+1]
    XtX = X1.T @ X1
    reg = np.eye(d + 1) * alpha
    reg[-1, -1] = 0  # no regularization on bias
    wb = np.linalg.solve(XtX + reg, X1.T @ y)
    return wb[:-1], wb[-1]


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0  # constant target, no signal to explain
    return 1.0 - ss_res / ss_tot


def cv_r2(X, y, alphas, n_folds=N_FOLDS, seed=SEED):
    """5-fold CV ridge regression. Returns best CV R² across alphas."""
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    fold_sz = n // n_folds
    best_r2 = -np.inf
    for alpha in alphas:
        fold_r2s = []
        for fi in range(n_folds):
            test_ids = idx[fi * fold_sz:(fi + 1) * fold_sz]
            train_ids = np.setdiff1d(idx, test_ids)
            if len(test_ids) < 2:
                continue
            w, b = ridge_solve(X[train_ids], y[train_ids], alpha)
            pred = X[test_ids] @ w + b
            fold_r2s.append(r2_score_manual(y[test_ids], pred))
        if fold_r2s:
            m = float(np.mean(fold_r2s))
            if m > best_r2:
                best_r2 = m
    return best_r2


def analyze_delta(name, delta_per_query, delta_global, embed, fraction_positions=0.1):
    """delta_per_query: [N, L, E];  delta_global: [L, E];  embed: [N, H].
    Compute CV R² per (l, e), on a random subsample of `fraction_positions`
    of positions to keep cost manageable. Report summary statistics."""
    N, L, E = delta_per_query.shape
    residual = delta_per_query - delta_global[None, :, :]   # [N, L, E]

    # Sample positions (stratified by layer)
    positions = []
    rng = np.random.RandomState(SEED)
    n_per_layer = max(1, int(round(E * fraction_positions)))
    for li in range(L):
        cols = rng.choice(E, size=n_per_layer, replace=False)
        for e in cols:
            positions.append((li, int(e)))
    print(f"  [{name}] sampling {len(positions)} (l,e) positions for CV",
          flush=True)

    X = embed.cpu().float().numpy().astype(np.float64)
    # Normalize embeddings (zero-mean, unit-std per dim)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    r2_values = []
    layer_r2 = {li: [] for li in range(L)}
    for li, e in positions:
        y = residual[:, li, e].cpu().float().numpy().astype(np.float64)
        if np.std(y) < 1e-10:
            continue  # constant residual, nothing to predict
        r2 = cv_r2(X, y, RIDGE_ALPHAS)
        r2_values.append(r2)
        layer_r2[li].append(r2)

    if not r2_values:
        print(f"  [{name}] no usable positions", flush=True)
        return {"name": name, "r2": np.array([])}

    arr = np.array(r2_values)
    print(f"  [{name}] CV R² stats across {len(arr)} positions:", flush=True)
    print(f"    mean      = {arr.mean():+.4f}", flush=True)
    print(f"    median    = {np.median(arr):+.4f}", flush=True)
    print(f"    max       = {arr.max():+.4f}", flush=True)
    print(f"    fraction R²>0    = {(arr > 0).mean():.3f}", flush=True)
    print(f"    fraction R²>0.05 = {(arr > 0.05).mean():.3f}", flush=True)
    print(f"    fraction R²>0.10 = {(arr > 0.10).mean():.3f}", flush=True)
    print(f"    fraction R²>0.20 = {(arr > 0.20).mean():.3f}", flush=True)
    print(f"  [{name}] per-layer mean R²:", flush=True)
    for li in range(L):
        if layer_r2[li]:
            lr2 = np.array(layer_r2[li])
            print(f"    layer {li:2d}: n={len(lr2):3d}  mean={lr2.mean():+.4f}  "
                  f"frac>0={(lr2 > 0).mean():.2f}", flush=True)
    return {
        "name": name,
        "r2": arr,
        "layer_r2": {li: np.array(v) for li, v in layer_r2.items()},
        "positions": positions,
    }


def per_query_meandiff_from_cache(cache):
    """Compute per-query Δ_q from L25c's counts + lengths."""
    N = cache["counts_refuse"].shape[0]
    L = cache["counts_refuse"].shape[1]
    E = cache["counts_refuse"].shape[2]
    out = torch.zeros(N, L, E, dtype=torch.float32)
    for qi in range(N):
        lr = max(int(cache["len_refuse"][qi].item()), 1)
        lc = max(int(cache["len_comply"][qi].item()), 1)
        out[qi] = cache["counts_refuse"][qi].float() / lr \
                  - cache["counts_comply"][qi].float() / lc
    return out


def global_meandiff_pooled(cache):
    cr = cache["counts_refuse"].sum(dim=0)
    cc = cache["counts_comply"].sum(dim=0)
    lr = cache["len_refuse"].sum().clamp(min=1).float()
    lc = cache["len_comply"].sum().clamp(min=1).float()
    return (cr / lr) - (cc / lc)


def main():
    if not os.path.exists(L25C_CACHE):
        raise FileNotFoundError(f"L25c cache missing: {L25C_CACHE}")
    l25c = torch.load(L25C_CACHE, map_location="cpu", weights_only=False)
    embed = l25c["embed"]
    print(f"L25c cache loaded: {l25c['counts_refuse'].shape[0]} queries, "
          f"embed {tuple(embed.shape)}", flush=True)

    results = {}

    # 1) Mean-diff per-query Δ
    print("\n" + "=" * 72, flush=True)
    print("(1) Mean-diff Δ_q  (from L25c counts)", flush=True)
    print("=" * 72, flush=True)
    md_per_query = per_query_meandiff_from_cache(l25c)
    md_global = global_meandiff_pooled(l25c)
    results["meandiff"] = analyze_delta("meandiff", md_per_query, md_global, embed)

    # 2) L26 logit-causal Δ, if cache exists
    if os.path.exists(L26_CAUSAL_CACHE):
        print("\n" + "=" * 72, flush=True)
        print("(2) Logit-causal Δ_q  (from L26)", flush=True)
        print("=" * 72, flush=True)
        l26 = torch.load(L26_CAUSAL_CACHE, map_location="cpu", weights_only=False)
        if l26.get("training_queries") == l25c.get("training_queries"):
            logit_per_query = l26["delta"]
            logit_global = logit_per_query.mean(dim=0)
            results["logit_causal"] = analyze_delta(
                "logit_causal", logit_per_query, logit_global, embed)
        else:
            print("  L26 training queries differ; skipping", flush=True)

    # 3) L27 mid-layer-causal Δ, if cache exists (may not be written yet)
    if os.path.exists(L27_MIDLAYER_CACHE):
        print("\n" + "=" * 72, flush=True)
        print("(3) Midlayer-causal Δ_q  (from L27)", flush=True)
        print("=" * 72, flush=True)
        l27 = torch.load(L27_MIDLAYER_CACHE, map_location="cpu", weights_only=False)
        if l27.get("training_queries") == l25c.get("training_queries"):
            ml_per_query = l27["delta"]
            ml_global = ml_per_query.mean(dim=0)
            results["midlayer_causal"] = analyze_delta(
                "midlayer_causal", ml_per_query, ml_global, embed)
        else:
            print("  L27 training queries differ; skipping", flush=True)
    else:
        print(f"\n(3) L27 cache not present at {L27_MIDLAYER_CACHE} — skip",
              flush=True)

    # Save
    torch.save(results, OUT_PATH)
    print(f"\nSaved → {OUT_PATH}", flush=True)

    # Bottom line verdict
    print("\n" + "=" * 72, flush=True)
    print("VERDICT (fraction of positions with CV R² > 0.10)", flush=True)
    print("=" * 72, flush=True)
    for name, res in results.items():
        if res.get("r2", np.array([])).size == 0:
            continue
        frac = (res["r2"] > 0.10).mean()
        med = np.median(res["r2"])
        tag = ("STRONG signal" if frac > 0.10 else
               "weak signal"  if frac > 0.02 else
               "NO signal (per-query unpredictable from embedding)")
        print(f"  {name:<20} frac>0.10 = {frac:.3f}  "
              f"median R² = {med:+.3f}  → {tag}", flush=True)


if __name__ == "__main__":
    main()
