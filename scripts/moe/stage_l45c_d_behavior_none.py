"""
Stage L45c: Extract d_behavior direction from `none` column baseline responses.

Motivation
----------
Existing d vectors are suboptimal for defense-targeting L46 experiment:

  d_refuse.pt              : prompt-only last-token mean-diff
                             → captures 'harmful-intent detection', not behavior
  d_refuse_jailbreak.pt    : mixes 4 CSVs (baseline + 3 interventions) and
                             mixes 4 attack types → direction polluted by
                             intervention artefacts and attack-specific shifts

d_behavior_none captures the cleanest signal:
  On the SAME harmful-prompt distribution (baseline/none, no attack),
  residual direction that separates model-refused-behavior from
  model-complied-behavior.

  d_behavior[L] = mean(response-residual | refuse) − mean(response-residual | comply)

Data
----
Input : stage_l42_jailbreak_baseline.csv, tag='baseline/none' (n=100)
Split : safe=True → refuse  (~48 rows)
        safe=False → comply (~52 rows)

Output
------
stage_l45c_d_behavior_none.pt   tensor [16, 2048]  + metadata dict
"""
from __future__ import annotations

import csv as _csv
import os
import sys
import time

import torch

sys.stdout.reconfigure(line_buffering=True)

os.environ["HF_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface"
os.environ["TORCH_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/torch"

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
INPUT_CSV = f"{DATA_DIR}/stage_l42_jailbreak_baseline.csv"
INPUT_TAG = "baseline/none"
OUT_PATH = f"{DATA_DIR}/d_behavior_none.pt"
OUT_TXT = f"{DATA_DIR}/stage_l45c_d_behavior_none.txt"

MAX_RESP_CHARS = 4000  # truncate pathologically long responses for OOM safety
SEED = 42


def load_rows(csv_path: str, tag: str) -> tuple[list[dict], list[dict]]:
    refuse, comply = [], []
    with open(csv_path, "r", newline="") as f:
        for r in _csv.DictReader(f):
            if r.get("tag") != tag:
                continue
            resp = (r.get("resp") or "")[:MAX_RESP_CHARS].strip()
            q = (r.get("q") or "").strip()
            if not resp or not q:
                continue
            row = {"q": q, "resp": resp}
            label = r.get("safe")
            if label == "True":
                refuse.append(row)
            elif label == "False":
                comply.append(row)
    return refuse, comply


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
    hs = out.hidden_states[1:n_layers + 1]  # drop embedding, keep post-layer
    stacked = torch.stack(
        [h[0, resp_start:, :].mean(dim=0) for h in hs], dim=0,
    )
    return stacked.detach().float().cpu()  # [n_layers, hidden]


def main() -> None:
    torch.manual_seed(SEED)
    print(f"Loading rows from {INPUT_CSV} tag={INPUT_TAG}", flush=True)
    refuse, comply = load_rows(INPUT_CSV, INPUT_TAG)
    print(f"  refuse (safe=True) : {len(refuse)}", flush=True)
    print(f"  comply (safe=False): {len(comply)}", flush=True)

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

    sum_refuse = torch.zeros(n_layers, hidden, dtype=torch.float64)
    sum_comply = torch.zeros(n_layers, hidden, dtype=torch.float64)
    count_refuse = 0
    count_comply = 0

    print("\nExtracting response-token residuals ...", flush=True)
    t0 = time.time()
    for i, row in enumerate(refuse):
        r = per_example_mean_residual(model, tok, row["q"], row["resp"],
                                        n_layers)
        if r is None:
            continue
        sum_refuse += r.double()
        count_refuse += 1
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  refuse {i+1}/{len(refuse)}  ({dt:.0f}s)", flush=True)
    print(f"  accepted {count_refuse}/{len(refuse)} refuse rows", flush=True)

    for i, row in enumerate(comply):
        r = per_example_mean_residual(model, tok, row["q"], row["resp"],
                                        n_layers)
        if r is None:
            continue
        sum_comply += r.double()
        count_comply += 1
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  comply {i+1}/{len(comply)}  ({dt:.0f}s)", flush=True)
    print(f"  accepted {count_comply}/{len(comply)} comply rows", flush=True)

    mean_refuse = sum_refuse / max(count_refuse, 1)
    mean_comply = sum_comply / max(count_comply, 1)
    d_behavior = (mean_refuse - mean_comply).float()  # [L, H]

    per_layer_norm = [float(d_behavior[l].norm().item())
                      for l in range(n_layers)]
    print("\nper-layer ||d_behavior||:", flush=True)
    for l, v in enumerate(per_layer_norm):
        print(f"  layer {l:2d}: {v:8.4f}", flush=True)

    # Cosine vs existing d_refuse and d_refuse_jailbreak for sanity
    cos_refuse = []
    cos_jb = []
    for path, label, store in [
        (f"{DATA_DIR}/d_refuse.pt", "d_refuse", cos_refuse),
        (f"{DATA_DIR}/d_refuse_jailbreak.pt", "d_refuse_jailbreak", cos_jb),
    ]:
        if not os.path.exists(path):
            print(f"  (missing) {path}", flush=True)
            continue
        obj = torch.load(path, map_location="cpu", weights_only=False)
        d = None
        if isinstance(obj, dict):
            for key in ("d_refuse_jailbreak", "d_per_layer", "d", "d_behavior"):
                v = obj.get(key)
                if isinstance(v, torch.Tensor):
                    d = v
                    break
            if d is None:
                for _v in obj.values():
                    if (isinstance(_v, torch.Tensor)
                            and _v.shape == d_behavior.shape):
                        d = _v
                        break
        elif isinstance(obj, torch.Tensor):
            d = obj
        if d is None or d.shape != d_behavior.shape:
            print(f"  {label}: shape mismatch or missing", flush=True)
            continue
        for l in range(n_layers):
            a = d_behavior[l].float()
            b = d[l].float()
            denom = (a.norm() * b.norm()).clamp(min=1e-8)
            store.append(float((a @ b / denom).item()))

    torch.save({
        "d_behavior": d_behavior,
        "per_layer_norm": torch.tensor(per_layer_norm),
        "cos_vs_d_refuse": torch.tensor(cos_refuse) if cos_refuse else None,
        "cos_vs_d_refuse_jailbreak": torch.tensor(cos_jb) if cos_jb else None,
        "n_refuse": count_refuse,
        "n_comply": count_comply,
        "seed": SEED,
        "source_csv": INPUT_CSV,
        "source_tag": INPUT_TAG,
        "model_id": MODEL_ID,
        "max_resp_chars": MAX_RESP_CHARS,
    }, OUT_PATH)
    print(f"\nSaved → {OUT_PATH}", flush=True)

    lines = []
    lines.append("=" * 70)
    lines.append("L45c: d_behavior from baseline/none column")
    lines.append("=" * 70)
    lines.append(f"Model : {MODEL_ID}")
    lines.append(f"Source: {INPUT_CSV} tag={INPUT_TAG}")
    lines.append(f"Accepted: refuse={count_refuse}  comply={count_comply}")
    lines.append("")
    lines.append("per-layer ||d_behavior||")
    for l, v in enumerate(per_layer_norm):
        lines.append(f"  L{l:2d}: {v:8.4f}")
    lines.append("")
    if cos_refuse:
        lines.append("cos(d_behavior, d_refuse) per layer")
        for l, v in enumerate(cos_refuse):
            lines.append(f"  L{l:2d}: {v:+.4f}")
        lines.append("")
    if cos_jb:
        lines.append("cos(d_behavior, d_refuse_jailbreak) per layer")
        for l, v in enumerate(cos_jb):
            lines.append(f"  L{l:2d}: {v:+.4f}")
        lines.append("")
    lines.append("Interpretation:")
    lines.append("  High cos (>0.7) with d_refuse_jailbreak: our new d is")
    lines.append("    largely the same behavior direction as the polluted one.")
    lines.append("  Low cos (<0.3): intervention CSVs were genuinely polluting")
    lines.append("    d_refuse_jailbreak; d_behavior_none is the cleaner signal.")
    lines.append("  Layers with ||d|| peak — where refuse vs comply divergence")
    lines.append("    is strongest; likely best ortho targets in L46.")
    txt = "\n".join(lines) + "\n"
    with open(OUT_TXT, "w") as f:
        f.write(txt)
    print("\n" + txt, flush=True)
    print(f"Summary → {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
