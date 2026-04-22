"""
L47e: hierarchical clustering on 25-attack causal Δ fingerprints.

Input: stage_l47d_all_attacks_delta.pt (Δ [2500, 16, 64], attack_labels, cos_matrix)

Output:
  - dendrogram (ASCII) + cluster labels at k=3,4,5,6
  - cluster centroids (mean Δ per cluster)
  - "refusal-engaged" score per attack: how well attack's mean Δ aligns with
    the strongest refusal-signal attack (`none`) — proxies whether model even
    tries to refuse under this attack
  - tail: list of attacks where |Δ| is low AND cos(d, d_none) is low → these
    are the "true bypass" attacks (defense target #1)

Also: cluster-level mean Δ becomes the prototype vector for an attack-aware
defense. Downstream: classifier head maps query Δ → cluster ID → dispatch.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
IN_PATH = f"{DATA_DIR}/stage_l47d_all_attacks_delta.pt"
OUT_TXT = f"{DATA_DIR}/stage_l47e_clusters.txt"
OUT_PT = f"{DATA_DIR}/stage_l47e_clusters.pt"


def main():
    print("=" * 72)
    print("L47e: attack-fingerprint hierarchical clustering")
    print("=" * 72)

    d = torch.load(IN_PATH, map_location="cpu", weights_only=False)
    delta = d["delta"]                   # [N, L, E]
    attack_labels = d["attack_labels"]
    attacks = d["attacks"]                # sorted list
    cos = d["cos_matrix"].numpy()         # [A, A]
    print(f"  attacks: {len(attacks)}   cos matrix: {cos.shape}", flush=True)

    # ---- Linkage (complete-linkage using 1 - cos as distance) ----
    try:
        from scipy.cluster.hierarchy import linkage, fcluster, to_tree  # type: ignore
        from scipy.spatial.distance import squareform           # type: ignore
    except ImportError:
        print("WARN: scipy unavailable; falling back to manual linkage",
              flush=True)
        linkage = None

    lines = []
    def w(s):
        lines.append(s); print(s, flush=True)

    w("")
    w("=" * 72)
    w("Attack magnitudes (||Δ||_F)  sorted low → high")
    w("=" * 72)
    attack_norm = {}
    attack_mean = {}
    for a in attacks:
        mask = [i for i, al in enumerate(attack_labels) if al == a]
        if not mask:
            continue
        sub = delta[mask]
        attack_norm[a] = float(sub.norm())
        attack_mean[a] = sub.mean(dim=0)   # [L, E]
    order = sorted(attack_norm, key=attack_norm.get)
    for a in order:
        w(f"  {a:<28}  {attack_norm[a]:>8.3f}")

    # ---- Clustering ----
    w("")
    w("=" * 72)
    w("Hierarchical clustering (complete linkage, 1-cos distance)")
    w("=" * 72)

    # Make sure cos matrix is symmetric + in [-1,1], then clip
    np.fill_diagonal(cos, 1.0)
    D = 1.0 - (cos + cos.T) / 2.0  # symmetrize + distance
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)

    if linkage is not None:
        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method="complete")
        clusters_by_k = {}
        for k in (3, 4, 5, 6, 7):
            lab = fcluster(Z, t=k, criterion="maxclust")
            clusters_by_k[k] = lab
            w(f"\n--- k={k} clusters ---")
            for c in sorted(set(lab)):
                members = [attacks[i] for i in range(len(attacks)) if lab[i] == c]
                avg_norm = np.mean([attack_norm[a] for a in members])
                w(f"  cluster {c}  (n={len(members)}, avg ||Δ||={avg_norm:.2f}):"
                  f"  {members}")
    else:
        clusters_by_k = {}

    # ---- Refusal-engagement score ----
    # Anchor = 'none' (highest Δ magnitude & model actually refusing)
    if "none" not in attack_mean:
        w("\n[WARN] 'none' attack missing, skipping refusal-engagement score")
    else:
        anchor = attack_mean["none"].flatten()
        anchor_n = anchor / anchor.norm().clamp(min=1e-9)
        w("")
        w("=" * 72)
        w("Refusal-engagement score = cos(Δ_attack_mean, Δ_none_mean)  ×  "
          "||Δ_attack||/||Δ_none||")
        w("=" * 72)
        w(f"{'attack':<28}  {'cos(d,none)':>12}  {'||Δ||ratio':>12}  "
          f"{'score':>10}")
        scores = {}
        for a in attacks:
            v = attack_mean[a].flatten()
            c = float(torch.nn.functional.cosine_similarity(
                v.unsqueeze(0), anchor.unsqueeze(0)))
            r = attack_norm[a] / attack_norm["none"]
            s = c * r
            scores[a] = s
            w(f"  {a:<28}  {c:>+12.3f}  {r:>12.3f}  {s:>+10.3f}")

        # Bypass attacks: low magnitude AND low cos → model neither tries
        # to refuse nor has similar routing to clean-harmful
        w("")
        w("=" * 72)
        w("BYPASS attacks (score < 0.3)  →  model doesn't engage refusal")
        w("=" * 72)
        for a, s in sorted(scores.items(), key=lambda x: x[1]):
            if s < 0.3:
                w(f"  {a:<28}  score={s:+.3f}  "
                  f"||Δ||={attack_norm[a]:.2f}  "
                  f"cos(none)={torch.nn.functional.cosine_similarity(attack_mean[a].flatten().unsqueeze(0), anchor.unsqueeze(0)).item():+.3f}")

        w("")
        w("=" * 72)
        w("REFUSAL-ENGAGED attacks (score > 0.5)  →  model tries but fails")
        w("=" * 72)
        for a, s in sorted(scores.items(), key=lambda x: -x[1]):
            if s > 0.5:
                w(f"  {a:<28}  score={s:+.3f}  "
                  f"||Δ||={attack_norm[a]:.2f}  "
                  f"cos(none)={torch.nn.functional.cosine_similarity(attack_mean[a].flatten().unsqueeze(0), anchor.unsqueeze(0)).item():+.3f}")

    # ---- Cluster centroids (for k=5, the most interpretable) ----
    if clusters_by_k and 5 in clusters_by_k:
        w("")
        w("=" * 72)
        w("Cluster centroids at k=5  (mean Δ across attack means)")
        w("=" * 72)
        centroids = {}
        labels = clusters_by_k[5]
        for c in sorted(set(labels)):
            members = [attacks[i] for i in range(len(attacks))
                       if labels[i] == c]
            stack = torch.stack([attack_mean[a] for a in members])  # [M, L, E]
            centroid = stack.mean(dim=0)  # [L, E]
            centroids[int(c)] = {"centroid": centroid,
                                 "members": members}
            w(f"\ncluster {c} (n={len(members)}): {members}")
            # Top-5 (layer, expert) by |centroid|
            flat = centroid.abs().flatten()
            topv, topi = flat.topk(5)
            n_experts = centroid.shape[1]
            for rank, (v, j) in enumerate(zip(topv, topi)):
                L, E = int(j) // n_experts, int(j) % n_experts
                sign = "+" if centroid[L, E] > 0 else "-"
                w(f"  top {rank+1}:  (L={L:>2}, E={E:>2})  "
                  f"{sign}{float(v):.3e}")

        torch.save({
            "centroids_k5": centroids,
            "clusters_by_k": {k: np.array(v) for k, v in clusters_by_k.items()},
            "attack_norm": attack_norm,
            "attacks": attacks,
        }, OUT_PT)
        print(f"\nSaved -> {OUT_PT}")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary -> {OUT_TXT}")


if __name__ == "__main__":
    main()
