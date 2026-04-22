"""
L47f: train attack-cluster classifier on per-query causal Δ.

Input: 25 attacks × 100 queries = 2500 samples. Each sample has a 1024-dim
causal Δ (16 layers × 64 experts). Map to 5 cluster labels (from L47e).

Classifier: multinomial logistic regression with scikit-learn-equivalent
via torch (CPU). Train/test split 80/20 stratified by attack.

Goal:
  - If accuracy >= 80%, clusters are real & predictable from single-query Δ.
  - Per-cluster recall + confusion matrix.
  - Feature importance: which (layer, expert) coefs are most class-separating.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
IN_L47D = f"{DATA_DIR}/stage_l47d_all_attacks_delta.pt"
IN_L47E = f"{DATA_DIR}/stage_l47e_clusters.pt"
OUT_TXT = f"{DATA_DIR}/stage_l47f_classifier.txt"
OUT_PT = f"{DATA_DIR}/stage_l47f_classifier.pt"

DEVICE = "cpu"


def main():
    print("=" * 72)
    print("L47f: attack-cluster classifier")
    print("=" * 72)

    d = torch.load(IN_L47D, map_location="cpu", weights_only=False)
    delta = d["delta"].float()          # [N, L, E]
    attack_labels = d["attack_labels"]   # list[str]
    attacks = d["attacks"]

    e = torch.load(IN_L47E, map_location="cpu", weights_only=False)
    clusters_by_k = e["clusters_by_k"]
    cluster_labels_k5 = clusters_by_k[5]  # np array, len=25
    attack_to_cluster = {attacks[i]: int(cluster_labels_k5[i])
                         for i in range(len(attacks))}
    print(f"  samples={delta.shape[0]}  features={delta.numel()//delta.shape[0]}",
          flush=True)
    print(f"  attack_to_cluster:", flush=True)
    for a in attacks:
        print(f"    {a:<28} -> cluster {attack_to_cluster[a]}", flush=True)

    # Per-sample cluster label
    y = torch.tensor([attack_to_cluster[a] for a in attack_labels],
                     dtype=torch.long)
    # Remap to 0-indexed
    unique = sorted(set(y.tolist()))
    lab2idx = {c: i for i, c in enumerate(unique)}
    y = torch.tensor([lab2idx[v.item()] for v in y], dtype=torch.long)
    n_classes = len(unique)
    print(f"\n  classes: {n_classes}", flush=True)
    class_counts = torch.bincount(y)
    print(f"  counts: {class_counts.tolist()}", flush=True)

    # Features: flatten Δ and z-score per dim (only across training samples
    # after split — do after split)
    X = delta.reshape(delta.shape[0], -1)  # [N, F]

    # Stratified 80/20 split by attack (not cluster) so every attack has
    # train & test
    import random
    rng = random.Random(17)
    train_idx, test_idx = [], []
    for a in attacks:
        idx = [i for i, al in enumerate(attack_labels) if al == a]
        rng.shuffle(idx)
        n_tr = int(0.8 * len(idx))
        train_idx.extend(idx[:n_tr])
        test_idx.extend(idx[n_tr:])
    train_idx = torch.tensor(train_idx)
    test_idx = torch.tensor(test_idx)
    print(f"\n  train: {len(train_idx)}   test: {len(test_idx)}", flush=True)

    X_tr = X[train_idx]
    X_te = X[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]

    # Z-score normalize
    mu = X_tr.mean(dim=0, keepdim=True)
    sd = X_tr.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    # Train with L2-regularized logistic regression (SGD + weight decay)
    torch.manual_seed(17)
    model = nn.Linear(X_tr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    X_tr = X_tr.to(DEVICE); X_te = X_te.to(DEVICE)
    y_tr = y_tr.to(DEVICE); y_te = y_te.to(DEVICE)

    print("\nTraining logistic classifier ...", flush=True)
    for epoch in range(200):
        model.train()
        # Mini-batch
        perm = torch.randperm(X_tr.shape[0])
        bs = 128
        losses = []
        for i in range(0, X_tr.shape[0], bs):
            batch = perm[i:i+bs]
            logits = model(X_tr[batch])
            loss = F.cross_entropy(logits, y_tr[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc_tr = (model(X_tr).argmax(dim=1) == y_tr).float().mean().item()
                acc_te = (model(X_te).argmax(dim=1) == y_te).float().mean().item()
            print(f"  epoch {epoch+1:>3}  loss={sum(losses)/len(losses):.4f}  "
                  f"acc_tr={acc_tr:.3f}  acc_te={acc_te:.3f}", flush=True)

    # Final metrics
    model.eval()
    with torch.no_grad():
        pred_te = model(X_te).argmax(dim=1)
        acc_te = (pred_te == y_te).float().mean().item()

    lines = []
    def w(s):
        lines.append(s); print(s, flush=True)

    w("")
    w("=" * 72)
    w(f"Test accuracy: {acc_te:.3f}   (n_test={len(y_te)})")
    w("=" * 72)

    # Per-class recall + precision
    w("\nPer-class metrics:")
    w(f"{'cluster':>8} {'support':>9} {'recall':>8} {'precision':>10}")
    for c_idx in range(n_classes):
        orig_c = unique[c_idx]
        mask_tr = (y_tr == c_idx)
        mask_te = (y_te == c_idx)
        support = int(mask_te.sum())
        if support == 0:
            continue
        recall = (pred_te[mask_te] == c_idx).float().mean().item()
        pred_mask = (pred_te == c_idx)
        if pred_mask.sum() > 0:
            prec = (y_te[pred_mask] == c_idx).float().mean().item()
        else:
            prec = 0.0
        w(f"  clust {orig_c:>2} {support:>9} {recall:>8.3f} {prec:>10.3f}")

    # Per-attack accuracy
    w("\nPer-attack test accuracy (does every attack get classified right?)")
    w(f"{'attack':<30} {'n_te':>6} {'acc':>8} {'true_c':>7} {'mode_pred':>10}")
    test_attack_labels = [attack_labels[int(i)] for i in test_idx]
    for a in attacks:
        ii = [j for j, al in enumerate(test_attack_labels) if al == a]
        if not ii:
            continue
        ii = torch.tensor(ii)
        yh = pred_te[ii]
        yt = y_te[ii]
        acc = (yh == yt).float().mean().item()
        true_c = unique[yt[0].item()]
        # mode prediction
        from collections import Counter
        mode_pred = Counter(yh.tolist()).most_common(1)[0][0]
        mode_pred_orig = unique[mode_pred]
        w(f"  {a:<30} {len(ii):>6} {acc:>8.3f} {true_c:>7} {mode_pred_orig:>10}")

    # Confusion matrix
    w("\nConfusion matrix (rows=true cluster, cols=pred):")
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(y_te.tolist(), pred_te.tolist()):
        cm[t, p] += 1
    w(f"  {'':>7}" + "  ".join(f"{unique[j]:>5}" for j in range(n_classes)))
    for i in range(n_classes):
        w(f"  c={unique[i]:>3}  " + "  ".join(f"{int(cm[i, j]):>5}"
                                               for j in range(n_classes)))

    # Feature importance: for each class, top-5 (layer, expert) by weight magnitude
    w("\nPer-cluster top-10 (layer, expert) weights:")
    W = model.weight.detach().cpu()  # [C, F]
    # Un-normalize: coef in original-delta space is W/sd; but for ranking, sign
    # and magnitude of W on standardized features is the correct "importance"
    n_E = 64
    for c_idx in range(n_classes):
        orig_c = unique[c_idx]
        absw = W[c_idx].abs()
        topv, topi = absw.topk(10)
        w(f"\n  cluster {orig_c}:")
        for rank, (v, j) in enumerate(zip(topv, topi)):
            L, E = int(j) // n_E, int(j) % n_E
            sign = "+" if W[c_idx, j] > 0 else "-"
            w(f"    {rank+1:>2}. (L={L:>2}, E={E:>2})  {sign}{float(v):.3f}")

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "mu": mu, "sd": sd,
        "unique_labels": unique,
        "test_acc": acc_te,
        "train_idx": train_idx, "test_idx": test_idx,
    }, OUT_PT)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT_PT}")
    print(f"Summary -> {OUT_TXT}")


if __name__ == "__main__":
    main()
