"""
L42g: Extract d_refuse per layer from PKU-Alignment/BeaverTails 30k_train.

Rationale: our original d_refuse came from 20+20 hand-crafted prompts and is
near-orthogonal to d_refuse_jailbreak (L42e).  BeaverTails gives ~30k
(prompt, response, is_safe) triples — much bigger and more diverse.  By
extracting a d_refuse from BT we test whether scale/diversity alone (not just
"jailbreak realism") closes the gap, and we produce a prompt-position variant
so the direction is aligned with how the L23/L28 gate actually reads it.

Outputs two directions side-by-side into d_refuse_beavertails.pt:

  d_bt_resp    - Chen-et-al-style response-conditioned mean-diff.
                 Per example: forward (prompt + response), take MEAN residual
                 over response tokens only, contrast is_safe=True vs
                 is_safe=False.  Directly comparable to L42e's d_jb.

  d_bt_prompt  - Arditi-style prompt-position mean-diff.
                 Per example: forward (prompt ONLY), take residual at the
                 last prompt token, contrast is_safe=True vs is_safe=False.
                 This is the direction the gate lambda(q) = ReLU(h_last . d_hat)
                 actually needs to align with at use time.

Balanced sampling: N_SAFE and N_UNSAFE via reservoir-free indexed shuffle.
Skips rows whose prompt or response is empty after stripping.
"""
from __future__ import annotations

import os
import random
import sys
import time

import torch

sys.stdout.reconfigure(line_buffering=True)

os.environ["HF_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface"
os.environ["TORCH_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/torch"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
OUT_PATH = f"{DATA_DIR}/d_refuse_beavertails.pt"

BT_DATASET_ID = "PKU-Alignment/BeaverTails"
BT_SPLIT = "30k_train"

SEED = 42
N_SAFE = 2000
N_UNSAFE = 2000
MAX_RESP_CHARS = 4000
MAX_PROMPT_CHARS = 2000
SEQ_LEN_SKIP_THRESHOLD = 2048  # drop pathological long rows


def sample_beavertails(n_safe: int, n_unsafe: int, seed: int = SEED):
    print(f"Loading {BT_DATASET_ID} / {BT_SPLIT} ...", flush=True)
    ds = load_dataset(BT_DATASET_ID, split=BT_SPLIT)
    print(f"  total rows: {len(ds)}", flush=True)
    safe, unsafe = [], []
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for i in indices:
        if len(safe) >= n_safe and len(unsafe) >= n_unsafe:
            break
        row = ds[i]
        p = (row.get("prompt") or "").strip()[:MAX_PROMPT_CHARS]
        r = (row.get("response") or "").strip()[:MAX_RESP_CHARS]
        if not p or not r:
            continue
        pair = (p, r)
        if row.get("is_safe") and len(safe) < n_safe:
            safe.append(pair)
        elif (not row.get("is_safe")) and len(unsafe) < n_unsafe:
            unsafe.append(pair)
    print(f"  sampled: {len(safe)} safe + {len(unsafe)} unsafe", flush=True)
    return safe, unsafe


def prompt_ids(tok, prompt_text: str):
    """Chat-templated user turn with assistant generation header."""
    return tok(
        tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True),
        return_tensors="pt", add_special_tokens=False).input_ids


def full_ids_and_resp_start(tok, prompt: str, resp: str):
    templated = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(templated, return_tensors="pt",
                add_special_tokens=False).input_ids
    f_ids = tok(templated + resp, return_tensors="pt",
                add_special_tokens=False).input_ids
    return f_ids, p_ids.shape[1]


@torch.no_grad()
def per_example_response_mean(model, tok, prompt: str, resp: str,
                              n_layers: int) -> torch.Tensor | None:
    input_ids, resp_start = full_ids_and_resp_start(tok, prompt, resp)
    if input_ids.shape[1] - resp_start < 1:
        return None
    if input_ids.shape[1] > SEQ_LEN_SKIP_THRESHOLD:
        return None
    input_ids = input_ids.to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True,
                use_cache=False)
    hs = out.hidden_states[1:n_layers + 1]
    stacked = torch.stack(
        [h[0, resp_start:, :].mean(dim=0) for h in hs], dim=0)
    return stacked.detach().float().cpu()


@torch.no_grad()
def per_example_prompt_last(model, tok, prompt: str,
                            n_layers: int) -> torch.Tensor | None:
    ids = prompt_ids(tok, prompt)
    if ids.shape[1] > SEQ_LEN_SKIP_THRESHOLD:
        return None
    ids = ids.to(model.device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[1:n_layers + 1]
    stacked = torch.stack([h[0, -1, :] for h in hs], dim=0)
    return stacked.detach().float().cpu()


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    safe, unsafe = sample_beavertails(N_SAFE, N_UNSAFE)

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

    acc = {
        "sum_safe_resp": torch.zeros(n_layers, hidden, dtype=torch.float64),
        "sum_unsafe_resp": torch.zeros(n_layers, hidden, dtype=torch.float64),
        "sum_safe_prompt": torch.zeros(n_layers, hidden, dtype=torch.float64),
        "sum_unsafe_prompt": torch.zeros(n_layers, hidden, dtype=torch.float64),
        "c_safe_resp": 0, "c_unsafe_resp": 0,
        "c_safe_prompt": 0, "c_unsafe_prompt": 0,
    }

    def process(rows, label: str) -> None:
        suffix_resp = f"c_{label}_resp"
        suffix_prompt = f"c_{label}_prompt"
        sum_resp = f"sum_{label}_resp"
        sum_prompt = f"sum_{label}_prompt"
        t0 = time.time()
        for i, (p, r) in enumerate(rows):
            rmean = per_example_response_mean(model, tok, p, r, n_layers)
            plast = per_example_prompt_last(model, tok, p, n_layers)
            if rmean is not None:
                acc[sum_resp] += rmean.double()
                acc[suffix_resp] += 1
            if plast is not None:
                acc[sum_prompt] += plast.double()
                acc[suffix_prompt] += 1
            if (i + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"  {label} {i+1}/{len(rows)}  ({dt:.0f}s)", flush=True)

    print("\nPhase 1/2: safe rows ...", flush=True)
    process(safe, "safe")
    print(f"  accepted safe: resp={acc['c_safe_resp']}, "
          f"prompt={acc['c_safe_prompt']}", flush=True)

    print("\nPhase 2/2: unsafe rows ...", flush=True)
    process(unsafe, "unsafe")
    print(f"  accepted unsafe: resp={acc['c_unsafe_resp']}, "
          f"prompt={acc['c_unsafe_prompt']}", flush=True)

    mean_safe_resp = acc["sum_safe_resp"] / max(acc["c_safe_resp"], 1)
    mean_unsafe_resp = acc["sum_unsafe_resp"] / max(acc["c_unsafe_resp"], 1)
    mean_safe_prompt = acc["sum_safe_prompt"] / max(acc["c_safe_prompt"], 1)
    mean_unsafe_prompt = acc["sum_unsafe_prompt"] / max(acc["c_unsafe_prompt"], 1)

    d_bt_resp = (mean_safe_resp - mean_unsafe_resp).float()
    d_bt_prompt = (mean_safe_prompt - mean_unsafe_prompt).float()

    def print_norms(name, d):
        print(f"\nper-layer ||{name}||:", flush=True)
        for L in range(n_layers):
            print(f"  layer {L:2d}: {float(d[L].norm().item()):9.4f}",
                  flush=True)

    print_norms("d_bt_resp", d_bt_resp)
    print_norms("d_bt_prompt", d_bt_prompt)

    torch.save({
        "d_bt_resp": d_bt_resp,
        "d_bt_prompt": d_bt_prompt,
        "per_layer_norm_resp": torch.tensor(
            [float(d_bt_resp[L].norm().item()) for L in range(n_layers)]),
        "per_layer_norm_prompt": torch.tensor(
            [float(d_bt_prompt[L].norm().item()) for L in range(n_layers)]),
        "n_safe_resp": acc["c_safe_resp"],
        "n_unsafe_resp": acc["c_unsafe_resp"],
        "n_safe_prompt": acc["c_safe_prompt"],
        "n_unsafe_prompt": acc["c_unsafe_prompt"],
        "seed": SEED,
        "dataset": f"{BT_DATASET_ID}:{BT_SPLIT}",
        "n_sampled": {"safe": len(safe), "unsafe": len(unsafe)},
    }, OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
