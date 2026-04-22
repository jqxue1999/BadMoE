"""
L42e: Extract jailbreak-aware d_refuse per layer from L42 CSVs.

Chen et al. 2025 style: response-conditioned mean-diff.
For each (wrapped_prompt, generated_response) labelled safe/unsafe by Qwen
judge, forward-pass the full transcript through OLMoE and collect the mean
residual at response-token positions (skipping the prompt). Mean-diff across
balanced safe vs unsafe samples → d_refuse_jailbreak[layer].

Uses all 4 L42 CSVs (baseline, steermoe, L23, L28) so signal captures what
*this model* actually does at inference time when it commits to refusing vs
complying on jailbreak-attacked prompts — robust to prompt wrapping because
the contrast is on response residuals, not prompt residuals.

Output: {DATA_DIR}/d_refuse_jailbreak.pt → tensor [n_layers=16, hidden=2048].
"""
from __future__ import annotations

import csv as _csv
import os
import random
import sys
import time

import torch

sys.stdout.reconfigure(line_buffering=True)

os.environ["HF_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface"
os.environ["TORCH_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/torch"

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
INPUT_CSVS = [
    f"{DATA_DIR}/stage_l42_jailbreak_baseline.csv",
    f"{DATA_DIR}/stage_l42_jailbreak_steermoe.csv",
    f"{DATA_DIR}/stage_l42_jailbreak_L23.csv",
    f"{DATA_DIR}/stage_l42_jailbreak_L28.csv",
]
OUT_PATH = f"{DATA_DIR}/d_refuse_jailbreak.pt"

SEED = 42
MAX_RESP_CHARS = 4000  # trim pathologically long responses


def load_labeled_rows() -> tuple[list[dict], list[dict]]:
    safe, unsafe = [], []
    for p in INPUT_CSVS:
        if not os.path.exists(p):
            print(f"  (missing) {p}", flush=True)
            continue
        with open(p, "r", newline="") as f:
            for r in _csv.DictReader(f):
                label = r.get("safe")
                resp = (r.get("resp") or "")[:MAX_RESP_CHARS].strip()
                wrapped = (r.get("wrapped_q") or "").strip()
                if not resp or not wrapped:
                    continue
                row = {"wrapped_q": wrapped, "resp": resp, "tag": r.get("tag", "")}
                if label == "True":
                    safe.append(row)
                elif label == "False":
                    unsafe.append(row)
        print(f"  loaded {p.rsplit('/', 1)[-1]}", flush=True)
    return safe, unsafe


def response_token_slice(tok, wrapped_q: str, resp: str):
    """Return (input_ids [1, L], response_start_index).
    Tokens [response_start:L] are the response tokens; everything before is
    the chat-templated prompt (user turn + assistant header)."""
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": wrapped_q}],
        tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + resp
    prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    full_ids = tok(full_text, return_tensors="pt", add_special_tokens=False)
    return full_ids.input_ids, prompt_ids.input_ids.shape[1]


@torch.no_grad()
def per_example_mean_residual(model, tok, wrapped_q: str, resp: str,
                              n_layers: int) -> torch.Tensor | None:
    """Returns tensor [n_layers, hidden] of mean response-token residuals,
    or None if the response tokenizes to zero length."""
    input_ids, resp_start = response_token_slice(tok, wrapped_q, resp)
    if input_ids.shape[1] - resp_start < 1:
        return None
    input_ids = input_ids.to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    # hidden_states: tuple of (n_layers+1) tensors [1, L, hidden]
    # index 0 = post-embed, 1..n_layers = after each layer's output
    hs = out.hidden_states[1:n_layers + 1]  # drop embedding
    hidden = hs[0].shape[-1]
    stacked = torch.stack([h[0, resp_start:, :].mean(dim=0) for h in hs], dim=0)
    return stacked.detach().float().cpu()  # [n_layers, hidden]


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    print("Loading labeled (prompt, response) pairs from L42 CSVs ...",
          flush=True)
    safe, unsafe = load_labeled_rows()
    print(f"  total safe={len(safe)}  unsafe={len(unsafe)}", flush=True)

    n = min(len(safe), len(unsafe))
    safe = random.sample(safe, n)
    unsafe = random.sample(unsafe, n)
    print(f"  balanced: {n} safe + {n} unsafe = {2*n} total", flush=True)

    print(f"\nLoading {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"  n_layers={n_layers}, hidden={hidden}", flush=True)

    sum_safe = torch.zeros(n_layers, hidden, dtype=torch.float64)
    sum_unsafe = torch.zeros(n_layers, hidden, dtype=torch.float64)
    count_safe = 0
    count_unsafe = 0

    print("\nExtracting response-token residuals ...", flush=True)
    t0 = time.time()
    for i, row in enumerate(safe):
        r = per_example_mean_residual(model, tok, row["wrapped_q"], row["resp"],
                                      n_layers)
        if r is None:
            continue
        sum_safe += r.double()
        count_safe += 1
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  safe {i+1}/{len(safe)}  ({dt:.0f}s)", flush=True)
    print(f"  accepted {count_safe}/{len(safe)} safe rows", flush=True)

    for i, row in enumerate(unsafe):
        r = per_example_mean_residual(model, tok, row["wrapped_q"], row["resp"],
                                      n_layers)
        if r is None:
            continue
        sum_unsafe += r.double()
        count_unsafe += 1
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  unsafe {i+1}/{len(unsafe)}  ({dt:.0f}s)", flush=True)
    print(f"  accepted {count_unsafe}/{len(unsafe)} unsafe rows", flush=True)

    mean_safe = sum_safe / max(count_safe, 1)
    mean_unsafe = sum_unsafe / max(count_unsafe, 1)
    d_refuse_jb = (mean_safe - mean_unsafe).float()  # refusal direction

    norms = [float(d_refuse_jb[l].norm().item()) for l in range(n_layers)]
    print("\nper-layer ||d_refuse_jailbreak||:", flush=True)
    for l, v in enumerate(norms):
        print(f"  layer {l:2d}: {v:8.4f}", flush=True)

    # Save both the vector and metadata for reproducibility.
    torch.save({
        "d_refuse_jailbreak": d_refuse_jb,     # [n_layers, hidden]
        "per_layer_norm": torch.tensor(norms),
        "n_safe": count_safe,
        "n_unsafe": count_unsafe,
        "seed": SEED,
        "source_csvs": [p.rsplit("/", 1)[-1] for p in INPUT_CSVS
                        if os.path.exists(p)],
    }, OUT_PATH)
    print(f"\nSaved → {OUT_PATH}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
