"""
Stage L45d: Probe whether d_behavior_none is informative and generalizes.

Three sanity tests for L45c's d_behavior_none.pt:

  Test 1 (in-sample AUC):
    Project response-token mean residual onto d_l_unit, per layer, for all
    100 rows of baseline/none. AUC(proj | safe=True vs safe=False).
    Train-set upper bound; <0.9 in mid/late layers = d did not capture signal.

  Test 2 (cross-distribution AUC):
    Same projection on the other 3 attack tags (new_gpt4_cipher, new_pair,
    past_tense). AUC stays high => d is a generic behavior direction.
    AUC collapses to ~0.5 => d overfits the none distribution.

  Test 3 (scale-normalized norm):
    per_layer_norm[l] / mean(||h_l||) on baseline/none responses.
    Removes residual-stream natural growth, reveals true signal peak.

Output: stage_l45d_probe.pt + stage_l45d_probe.txt
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
D_PATH = f"{DATA_DIR}/d_behavior_none.pt"
OUT_PT = f"{DATA_DIR}/stage_l45d_probe.pt"
OUT_TXT = f"{DATA_DIR}/stage_l45d_probe.txt"

MAX_RESP_CHARS = 4000
SEED = 42

TAGS = [
    "baseline/none",
    "baseline/new_gpt4_cipher",
    "baseline/new_pair",
    "baseline/past_tense",
]


def load_rows(csv_path: str) -> dict[str, list[dict]]:
    per_tag: dict[str, list[dict]] = defaultdict(list)
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
            per_tag[tag].append(
                {"q": q, "resp": resp, "safe": (label == "True")}
            )
    return per_tag


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
def per_example_mean_and_norm(model, tok, q: str, resp: str, n_layers: int):
    """Returns (mean_residual [L,H], mean_norm [L]) over response tokens."""
    input_ids, resp_start = response_token_slice(tok, q, resp)
    if input_ids.shape[1] - resp_start < 1:
        return None, None
    input_ids = input_ids.to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[1:n_layers + 1]
    means = []
    norms = []
    for h in hs:
        slab = h[0, resp_start:, :].float()  # [T, H]
        mean = slab.mean(dim=0)
        means.append(mean)
        norms.append(mean.norm())  # ||mean_h_l||, matches L45c convention
    return (
        torch.stack(means, dim=0).detach().cpu(),           # [L, H]
        torch.stack(norms, dim=0).detach().cpu(),           # [L]
    )


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUC where label=1 means 'refuse' (positive class).

    Pairwise formula with correct tie handling (counts ties as 0.5).
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    pos = scores[labels == 1].float()
    neg = scores[labels == 0].float()
    gt = (pos[:, None] > neg[None, :]).float()
    eq = (pos[:, None] == neg[None, :]).float()
    return float((gt + 0.5 * eq).mean().item())


def main() -> None:
    torch.manual_seed(SEED)

    print(f"Loading d_behavior_none from {D_PATH}", flush=True)
    obj = torch.load(D_PATH, map_location="cpu", weights_only=False)
    d_per_layer = obj["d_behavior"].float()  # [L, H]
    n_layers, hidden = d_per_layer.shape
    per_layer_norm = d_per_layer.norm(dim=-1)
    d_unit = d_per_layer / per_layer_norm.clamp(min=1e-8).unsqueeze(-1)
    print(f"  d_per_layer: [{n_layers}, {hidden}]", flush=True)

    print(f"\nLoading rows from {INPUT_CSV}", flush=True)
    per_tag = load_rows(INPUT_CSV)
    for t in TAGS:
        refuse = sum(1 for r in per_tag.get(t, []) if r["safe"])
        comply = sum(1 for r in per_tag.get(t, []) if not r["safe"])
        print(f"  {t:30s}  refuse={refuse:3d}  comply={comply:3d}", flush=True)

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
    assert model.config.num_hidden_layers == n_layers
    assert model.config.hidden_size == hidden
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    projections: dict[str, torch.Tensor] = {}   # [N, L]
    labels: dict[str, torch.Tensor] = {}        # [N]
    mean_resid_norms: dict[str, torch.Tensor] = {}  # [N, L]

    for tag in TAGS:
        rows = per_tag.get(tag, [])
        if not rows:
            continue
        print(f"\nExtracting {tag}  n={len(rows)} ...", flush=True)
        t0 = time.time()
        proj_rows = []
        label_rows = []
        norm_rows = []
        for i, row in enumerate(rows):
            mean, norm = per_example_mean_and_norm(
                model, tok, row["q"], row["resp"], n_layers,
            )
            if mean is None:
                continue
            proj = (mean.float() * d_unit).sum(dim=-1)  # [L]
            proj_rows.append(proj)
            label_rows.append(1 if row["safe"] else 0)
            norm_rows.append(norm.float())
            if (i + 1) % 20 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(rows)}  ({dt:.0f}s)", flush=True)
        if not proj_rows:
            print(f"  (no usable rows for {tag})", flush=True)
            continue
        projections[tag] = torch.stack(proj_rows, dim=0)  # [N, L]
        labels[tag] = torch.tensor(label_rows, dtype=torch.long)
        mean_resid_norms[tag] = torch.stack(norm_rows, dim=0)

    print("\nComputing per-tag per-layer AUC ...", flush=True)
    auc_table: dict[str, list[float]] = {}
    for tag in TAGS:
        if tag not in projections:
            continue
        proj = projections[tag]
        y = labels[tag]
        aucs = [compute_auc(proj[:, l], y) for l in range(n_layers)]
        auc_table[tag] = aucs

    if "baseline/none" not in mean_resid_norms:
        raise RuntimeError("baseline/none has no usable rows — cannot "
                           "compute scale-normalized norms")
    mean_resp_norm_baseline = mean_resid_norms["baseline/none"].mean(dim=0)
    scale_norm_d = (per_layer_norm / mean_resp_norm_baseline).tolist()

    torch.save({
        "d_path": D_PATH,
        "per_layer_norm_raw": per_layer_norm,
        "mean_resp_norm_baseline_none": mean_resp_norm_baseline,
        "per_layer_norm_scale_normalized": torch.tensor(scale_norm_d),
        "projections": projections,
        "labels": labels,
        "auc_table": {t: torch.tensor(v) for t, v in auc_table.items()},
        "seed": SEED,
    }, OUT_PT)
    print(f"\nSaved → {OUT_PT}", flush=True)

    lines = []
    lines.append("=" * 76)
    lines.append("L45d probe: is d_behavior_none informative and generalizing?")
    lines.append("=" * 76)
    lines.append(f"d : {D_PATH}")
    lines.append(f"csv: {INPUT_CSV}")
    lines.append("")
    lines.append("Per-layer ||d|| vs scale-normalized (divide by mean residual"
                 " norm on baseline/none responses):")
    lines.append(f"  {'L':>3}  {'raw_norm':>9}  {'resid_norm':>10}  "
                 f"{'scale_norm':>10}")
    for l in range(n_layers):
        lines.append(f"  {l:>3}  {per_layer_norm[l].item():>9.4f}  "
                     f"{mean_resp_norm_baseline[l].item():>10.2f}  "
                     f"{scale_norm_d[l]:>10.5f}")
    lines.append("")

    header = f"  {'L':>3}  " + "  ".join(f"{t.split('/')[-1]:>14s}"
                                          for t in TAGS)
    lines.append("Per-layer AUC (label=1 if safe=True, i.e. refuse):")
    lines.append("Train=baseline/none; the other three are held-out.")
    lines.append(header)
    for l in range(n_layers):
        vals = []
        for t in TAGS:
            v = auc_table.get(t)
            vals.append(f"{v[l]:>14.4f}" if v is not None else f"{'nan':>14s}")
        lines.append(f"  {l:>3}  " + "  ".join(vals))

    lines.append("")
    lines.append("Interpretation cheat-sheet:")
    lines.append("  train AUC (none) > 0.90 in mid/late layers  => d captured"
                 " signal on training distribution")
    lines.append("  held-out AUC > 0.75 across all 3 attacks    => d is a"
                 " generic behavior direction")
    lines.append("  held-out AUC collapses to 0.5               => d overfits"
                 " the none distribution (not usable for L46)")
    lines.append("  L15 scale_norm dominates others             => L15 is a"
                 " true signal peak, not residual-stream scale artifact")
    lines.append("  L15 scale_norm on same order as L10-L14     => L15 peak"
                 " was scale artifact; true targets are mid/late layers")
    txt = "\n".join(lines) + "\n"
    with open(OUT_TXT, "w") as f:
        f.write(txt)
    print("\n" + txt, flush=True)
    print(f"Summary → {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
