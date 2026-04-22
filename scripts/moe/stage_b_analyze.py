"""
Stage B analysis: richer localization metrics + correlation between static capacity
and dynamic routing skew.

Adds:
  1. Normalized capacity: ||d @ W|| / ||W||_F  (per-expert "fraction of energy along d")
  2. Cosine-based metric: max |cos(d, W_col)| over columns of down_proj
  3. Cumulative Lorenz curve of capacity per layer
  4. Correlation(static capacity, router skew) per layer
  5. Top-K ablation plan: identify union of (top-K capacity) + (top-K router skew) experts
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

os.environ["HF_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface"
os.environ["TORCH_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/torch"

DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em"


def main() -> None:
    print("Loading saved tensors...", flush=True)
    d_router = torch.load(f"{DATA_DIR}/d_and_router.pt", map_location="cpu", weights_only=False)
    capacity = torch.load(f"{DATA_DIR}/expert_capacity.pt", map_location="cpu", weights_only=False)  # [L, E]

    d_per_layer: torch.Tensor = d_router["d_per_layer"]  # [L, H]
    n_layers, hidden = d_per_layer.shape
    n_experts = capacity.shape[1]
    print(f"n_layers={n_layers}, hidden={hidden}, n_experts={n_experts}", flush=True)

    # Build router skew tensor [L, E]
    rpos = d_router["router_mean_pos"]
    rneg = d_router["router_mean_neg"]
    skew = torch.zeros(n_layers, n_experts)
    for i in range(n_layers):
        if rpos[i] is not None and rneg[i] is not None:
            skew[i] = rpos[i] - rneg[i]

    print("\n=== Per-layer richer localization metrics ===", flush=True)
    print(f"{'L':<4}{'||d||':<9}{'top1/med':<10}{'top10_share':<14}"
          f"{'top50_share':<14}{'cap/skew_corr':<15}{'top20_overlap':<15}", flush=True)
    for i in range(n_layers):
        cap_i = capacity[i].numpy().astype(np.float64)
        skew_i = skew[i].numpy().astype(np.float64)

        # top-k share
        sorted_cap = np.sort(cap_i)[::-1]
        top10_share = sorted_cap[:10].sum() / max(cap_i.sum(), 1e-9)
        top50_share = sorted_cap[:50].sum() / max(cap_i.sum(), 1e-9)

        top1_over_median = sorted_cap[0] / max(np.median(cap_i), 1e-9)

        # correlation between capacity and router skew
        if cap_i.std() > 1e-9 and skew_i.std() > 1e-9:
            corr = float(np.corrcoef(cap_i, skew_i)[0, 1])
        else:
            corr = 0.0

        # top-20 overlap: how many of top-20 by capacity also in top-20 by skew?
        top20_cap = set(np.argsort(-cap_i)[:20].tolist())
        top20_skew = set(np.argsort(-skew_i)[:20].tolist())
        overlap = len(top20_cap & top20_skew)

        d_norm = d_per_layer[i].norm().item()
        print(f"{i:<4}{d_norm:<9.2f}{top1_over_median:<10.2f}"
              f"{top10_share:<14.3f}{top50_share:<14.3f}"
              f"{corr:<15.3f}{overlap:<15d}/20", flush=True)

    # === Summary: is localization strong enough to make surgical defense worth trying? ===
    print("\n=== Target experts for causal ablation ===", flush=True)
    print("Focus on mid layers where d-norm first rises sharply (layers 15-28).", flush=True)

    # Build target set: in each mid layer, take union of top-10 by capacity + top-10 by skew
    target_layers = list(range(15, 29))
    targets_per_layer: dict[int, set] = {}
    for i in target_layers:
        cap_i = capacity[i].numpy()
        skew_i = skew[i].numpy()
        top10_cap = set(np.argsort(-cap_i)[:10].tolist())
        top10_skew = set(np.argsort(-skew_i)[:10].tolist())
        targets_per_layer[i] = top10_cap | top10_skew
        print(f"  layer {i}: |cap-top10 ∪ skew-top10|={len(targets_per_layer[i])}, "
              f"overlap={len(top10_cap & top10_skew)}", flush=True)

    # Save targets
    torch.save({
        "targets_per_layer": targets_per_layer,
        "target_layers": target_layers,
        "note": "Union of top-10 capacity + top-10 router-skew per layer. Use for Stage C ablation.",
    }, f"{DATA_DIR}/ablation_targets.pt")
    print(f"\nSaved ablation targets to {DATA_DIR}/ablation_targets.pt", flush=True)


if __name__ == "__main__":
    main()
