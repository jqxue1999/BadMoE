"""
Stage L45e: Per-attack d_behavior extraction + cross-attack probe.

Hypothesis to test (B1):
  Is there a shared refuse-vs-comply behavior direction across all 4 attacks,
  or does each attack occupy its own subspace?

Procedure:
  1. One pass over all 400 rows (4 tags × 100) to collect [N, L, H] mean
     response-token residuals. ~1 min on B200 for OLMoE-1B.
  2. Per tag t: d_behavior[t][l] = mean(h_l | t, refuse) - mean(h_l | t, comply)
  3. d_universal_avg[l]  = mean of the 4 per-tag d vectors (equal weight)
     d_universal_pool[l] = mean(h_l | any, refuse) - mean(h_l | any, comply)
  4. Pairwise cosine (d_tag_i, d_tag_j) per layer → did they agree?
  5. Cross-probe: for each direction × each test tag, compute AUC.
     Direction in {d_none, d_cipher, d_pair, d_past_tense, d_avg, d_pool}.
     Diagonal = in-sample upper bound. Off-diagonal = generalization.

Output:
  stage_l45e_per_attack.pt
  stage_l45e_per_attack.txt
"""
from __future__ import annotations

import csv as _csv
import os
import sys
import time
from collections import defaultdict

import torch

sys.stdout.reconfigure(line_buffering=True)

os.environ["HF_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface"
os.environ["TORCH_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/torch"

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
INPUT_CSV = f"{DATA_DIR}/stage_l42_jailbreak_baseline.csv"
OUT_PT = f"{DATA_DIR}/stage_l45e_per_attack.pt"
OUT_TXT = f"{DATA_DIR}/stage_l45e_per_attack.txt"

MAX_RESP_CHARS = 4000
SEED = 42

TAGS = [
    "baseline/none",
    "baseline/new_gpt4_cipher",
    "baseline/new_pair",
    "baseline/past_tense",
]
SHORT = {
    "baseline/none": "none",
    "baseline/new_gpt4_cipher": "cipher",
    "baseline/new_pair": "pair",
    "baseline/past_tense": "past",
}


def load_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        for r in _csv.DictReader(f):
            tag = r.get("tag") or ""
            if tag not in TAGS:
                continue
            resp = (r.get("resp") or "")[:MAX_RESP_CHARS].strip()
            q = (r.get("q") or "").strip()
            label = r.get("safe")
            if not resp or not q or label not in ("True", "False"):
                continue
            rows.append({
                "tag": tag, "q": q, "resp": resp,
                "safe": (label == "True"),
            })
    return rows


def response_token_slice(tok, q: str, resp: str) -> tuple[torch.Tensor, int]:
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True,
    )
    full_text = prompt_text + resp
    prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    full_ids = tok(full_text, return_tensors="pt", add_special_tokens=False)
    return full_ids.input_ids, prompt_ids.input_ids.shape[1]


@torch.no_grad()
def per_example_mean_residual(model, tok, q: str, resp: str, n_layers: int):
    input_ids, resp_start = response_token_slice(tok, q, resp)
    if input_ids.shape[1] - resp_start < 1:
        return None
    input_ids = input_ids.to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[1:n_layers + 1]
    stacked = torch.stack(
        [h[0, resp_start:, :].mean(dim=0) for h in hs], dim=0,
    )
    return stacked.detach().float().cpu()  # [L, H]


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    pos = scores[labels == 1].float()
    neg = scores[labels == 0].float()
    gt = (pos[:, None] > neg[None, :]).float()
    eq = (pos[:, None] == neg[None, :]).float()
    return float((gt + 0.5 * eq).mean().item())


def main() -> None:
    torch.manual_seed(SEED)

    print(f"Loading rows from {INPUT_CSV}", flush=True)
    rows = load_rows(INPUT_CSV)
    print(f"  total usable rows: {len(rows)}", flush=True)
    for t in TAGS:
        refuse = sum(1 for r in rows if r["tag"] == t and r["safe"])
        comply = sum(1 for r in rows if r["tag"] == t and not r["safe"])
        print(f"  {SHORT[t]:>6s}  refuse={refuse:3d}  comply={comply:3d}",
              flush=True)

    print(f"\nLoading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"  loaded in {time.time()-t0:.1f}s  "
          f"n_layers={n_layers} hidden={hidden}", flush=True)

    # One pass: collect mean residuals [N, L, H]
    print("\nExtracting mean response-token residuals (one pass) ...",
          flush=True)
    t0 = time.time()
    H_rows = []       # list of [L, H]
    meta = []         # list of (tag, safe_bool)
    for i, row in enumerate(rows):
        h = per_example_mean_residual(model, tok, row["q"], row["resp"],
                                       n_layers)
        if h is None:
            continue
        H_rows.append(h)
        meta.append((row["tag"], row["safe"]))
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(rows)}  ({dt:.0f}s)", flush=True)
    if not H_rows:
        raise RuntimeError("No usable rows")
    H = torch.stack(H_rows, dim=0)                  # [N, L, H]
    tags_arr = [m[0] for m in meta]
    safe_arr = torch.tensor([m[1] for m in meta], dtype=torch.bool)  # [N]
    print(f"  H shape: {tuple(H.shape)}  dtype={H.dtype}  "
          f"mem≈{H.numel()*H.element_size()/1e6:.1f} MB", flush=True)

    # Per-tag d_behavior
    def tag_mask(t): return torch.tensor([x == t for x in tags_arr])
    d_per_tag: dict[str, torch.Tensor] = {}
    n_refuse_per_tag: dict[str, int] = {}
    n_comply_per_tag: dict[str, int] = {}
    for t in TAGS:
        mt = tag_mask(t)
        refuse = mt & safe_arr
        comply = mt & ~safe_arr
        n_refuse_per_tag[t] = int(refuse.sum().item())
        n_comply_per_tag[t] = int(comply.sum().item())
        if refuse.sum() == 0 or comply.sum() == 0:
            print(f"  {SHORT[t]}: cannot compute d (one class empty)",
                  flush=True)
            continue
        mean_r = H[refuse].mean(dim=0)  # [L, H]
        mean_c = H[comply].mean(dim=0)
        d_per_tag[t] = mean_r - mean_c

    # Universal directions
    if safe_arr.any() and (~safe_arr).any():
        mean_r_all = H[safe_arr].mean(dim=0)
        mean_c_all = H[~safe_arr].mean(dim=0)
        d_pool = mean_r_all - mean_c_all
    else:
        d_pool = None
    if len(d_per_tag) == len(TAGS):
        d_avg = torch.stack([d_per_tag[t] for t in TAGS], dim=0).mean(dim=0)
    else:
        print(f"  d_avg skipped — only {len(d_per_tag)}/{len(TAGS)} tags "
              f"have valid d_behavior", flush=True)
        d_avg = None

    # Pairwise cosines across tags, per layer
    print("\nPairwise per-layer cosine across per-tag d_behavior:", flush=True)
    cos_matrix = {}
    per_layer_norms = {SHORT[t]: d_per_tag[t].norm(dim=-1)
                       for t in TAGS if t in d_per_tag}
    for t1 in TAGS:
        if t1 not in d_per_tag:
            continue
        for t2 in TAGS:
            if t2 not in d_per_tag:
                continue
            a = d_per_tag[t1]
            b = d_per_tag[t2]
            cos_l = []
            for l in range(n_layers):
                denom = (a[l].norm() * b[l].norm()).clamp(min=1e-8)
                cos_l.append(float((a[l] @ b[l] / denom).item()))
            cos_matrix[(SHORT[t1], SHORT[t2])] = cos_l

    # Cross-probe AUC: direction × test tag × layer
    all_dirs = {}
    for t in TAGS:
        if t in d_per_tag:
            all_dirs[f"d_{SHORT[t]}"] = d_per_tag[t]
    if d_avg is not None:
        all_dirs["d_avg"] = d_avg
    if d_pool is not None:
        all_dirs["d_pool"] = d_pool

    print("\nComputing cross-probe AUC ...", flush=True)
    auc_grid: dict[tuple[str, str], list[float]] = {}
    for dname, d in all_dirs.items():
        d_unit = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        proj = (H * d_unit.unsqueeze(0)).sum(dim=-1)  # [N, L]
        for t in TAGS:
            mt = tag_mask(t)
            if mt.sum() == 0:
                continue
            y = safe_arr[mt].long()
            per_layer_auc = [
                compute_auc(proj[mt][:, l], y) for l in range(n_layers)
            ]
            auc_grid[(dname, SHORT[t])] = per_layer_auc

    # Save
    torch.save({
        "d_per_tag": d_per_tag,
        "d_avg": d_avg,
        "d_pool": d_pool,
        "cos_matrix_per_layer": cos_matrix,
        "auc_grid": auc_grid,
        "per_layer_norms": per_layer_norms,
        "n_refuse_per_tag": n_refuse_per_tag,
        "n_comply_per_tag": n_comply_per_tag,
        "n_layers": n_layers,
        "hidden": hidden,
        "seed": SEED,
    }, OUT_PT)
    print(f"\nSaved → {OUT_PT}", flush=True)

    # Text summary
    lines = []
    lines.append("=" * 84)
    lines.append("L45e: per-attack d_behavior + cross-probe")
    lines.append("=" * 84)
    lines.append(f"Model : {MODEL_ID}")
    lines.append(f"Source: {INPUT_CSV}")
    lines.append("")
    lines.append("Rows per tag (refuse/comply):")
    for t in TAGS:
        lines.append(f"  {SHORT[t]:>6s}  {n_refuse_per_tag.get(t,0):3d} / "
                     f"{n_comply_per_tag.get(t,0):3d}")
    lines.append("")

    lines.append("Per-tag ||d_behavior|| per layer:")
    head = "  L  " + "  ".join(f"{SHORT[t]:>8s}"
                                for t in TAGS if t in d_per_tag)
    lines.append(head)
    for l in range(n_layers):
        vals = [per_layer_norms[SHORT[t]][l].item()
                for t in TAGS if t in d_per_tag]
        lines.append(f"  {l:>2d}  " + "  ".join(f"{v:8.4f}" for v in vals))
    lines.append("")

    # Report cosine matrix for mid-layer L10 (peak from L45d) and averaged
    focus_layer = 10  # L45d showed peak at L10-L12
    lines.append(f"Pairwise cosine(d_behavior[tag_i], d_behavior[tag_j]) at "
                 f"layer {focus_layer}:")
    short_tags = [SHORT[t] for t in TAGS if t in d_per_tag]
    lines.append("          " + "  ".join(f"{s:>8s}" for s in short_tags))
    for s1 in short_tags:
        row = [f"{s1:>8s}  "]
        for s2 in short_tags:
            v = cos_matrix.get((s1, s2), [float("nan")] * n_layers)[focus_layer]
            row.append(f"{v:>8.4f}")
        lines.append("  ".join(row))
    lines.append("")

    # Mean cosine over L8-L14 (body of the signal peak)
    bodies = list(range(8, 15))
    lines.append(f"Mean cosine over layers {bodies[0]}-{bodies[-1]} "
                 f"(signal-peak body):")
    lines.append("          " + "  ".join(f"{s:>8s}" for s in short_tags))
    for s1 in short_tags:
        row = [f"{s1:>8s}  "]
        for s2 in short_tags:
            vs = cos_matrix.get((s1, s2), [])
            mean_v = (sum(vs[l] for l in bodies) / len(bodies)
                      if vs else float("nan"))
            row.append(f"{mean_v:>8.4f}")
        lines.append("  ".join(row))
    lines.append("")

    # AUC grid at focus layer
    lines.append(f"Probe AUC at layer {focus_layer}  (row = direction, "
                 f"col = test tag):")
    dir_names = list(all_dirs.keys())
    lines.append("  direction   " +
                 "  ".join(f"{SHORT[t]:>8s}" for t in TAGS))
    for dn in dir_names:
        row_vals = []
        for t in TAGS:
            auc = auc_grid.get((dn, SHORT[t]), [float("nan")] * n_layers)
            row_vals.append(f"{auc[focus_layer]:>8.4f}")
        lines.append(f"  {dn:<12s}" + "  ".join(row_vals))
    lines.append("")

    # Mean AUC over L8-L14
    lines.append(f"Probe AUC averaged over layers {bodies[0]}-{bodies[-1]}:")
    lines.append("  direction   " +
                 "  ".join(f"{SHORT[t]:>8s}" for t in TAGS))
    for dn in dir_names:
        row_vals = []
        for t in TAGS:
            auc = auc_grid.get((dn, SHORT[t]), [])
            mean_v = (sum(auc[l] for l in bodies) / len(bodies)
                      if auc else float("nan"))
            row_vals.append(f"{mean_v:>8.4f}")
        lines.append(f"  {dn:<12s}" + "  ".join(row_vals))
    lines.append("")

    lines.append("Interpretation guide:")
    lines.append("  pairwise cos > 0.7 among all 4 tags => shared behavior"
                 " axis exists; d_avg / d_pool is well-defined universal dir")
    lines.append("  cos(cipher, others) low but (none,pair,past) high => "
                 "cipher sits in its own subspace; universal dir unlikely")
    lines.append("  d_avg/d_pool AUC > 0.75 on all 4 tags => universal dir is"
                 " usable for L46")
    lines.append("  diagonal high, off-diagonal collapses => per-attack dirs"
                 " are attack-specific, no universal defense via single d")
    txt = "\n".join(lines) + "\n"
    with open(OUT_TXT, "w") as f:
        f.write(txt)
    print("\n" + txt, flush=True)
    print(f"Summary → {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
