"""
Stage L45b: Behavior-based A+/A- extraction on `none` jailbreak column.

Motivation
----------
L40 uses BeaverTails is_safe labels which mix (benign prompt + benign answer)
with (harmful prompt + refusal) in the "safe" bucket, and (harmful prompt +
harmful answer) in the "unsafe" bucket. Prompt distribution differs between
classes, so Δ = P_refuse - P_comply is confounded by prompt-side signal.

SteerMoE's paired-prompt approach fixes prompt distribution but uses a
TEMPLATE refusal (e.g. "I'm sorry, but I can't assist with that.") for the
safe response while the unsafe response is the real dataset completion. This
introduces a length/style confound (10-20 tok template vs 100-500 tok real).

L45b fixes both: use only harmful prompts (all from `none` column of
PandaGuard), split by the MODEL's OWN behavior (refuse vs comply), both
responses are OLMoE's natural greedy completions. No prompt distribution
shift, no template artefact.

Data
----
Input: stage_l42_jailbreak_baseline.csv, tag == 'baseline/none' (n=100).
Split by `safe` column (True = refuse, False = comply).
Baseline ASR on `none` is ~52%, so we get ~48 refuse + ~52 comply.

Extraction
----------
Identical to L40: teacher-force (chat_template(prompt) + response), hook
gate.register_forward_hook, count top-8 expert hits on RESPONSE tokens only.
Δ = P_refuse - P_comply, global top-25 = A+, bottom-25 = A-.

Outputs
-------
- stage_l45b_none_behavior_counts.pt
- stage_l45b_none_behavior_AP_AM.pt
- stage_l45b_none_behavior.txt  (summary + Jaccard vs L40's A+/A-)
"""
from __future__ import annotations

import csv as _csv
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

os.environ["HF_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface"
os.environ["TORCH_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/torch"

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em/olmoe"
INPUT_CSV = f"{DATA_DIR}/stage_l42_jailbreak_baseline.csv"
INPUT_TAG = "baseline/none"

OUT_COUNTS = f"{DATA_DIR}/stage_l45b_none_behavior_counts.pt"
OUT_AP_AM = f"{DATA_DIR}/stage_l45b_none_behavior_AP_AM.pt"
OUT_TXT = f"{DATA_DIR}/stage_l45b_none_behavior.txt"

L40_COUNTS = f"{DATA_DIR}/stage_l40_beavertails_counts.pt"

K_A_PLUS = 25
K_A_MINUS = 25
SEED = 42
SEQ_LEN_SKIP_THRESHOLD = 2048  # higher than L40 since PandaGuard responses may be longer


# ==========================================================================
# Data loading
# ==========================================================================

def load_none_samples(csv_path: str, tag: str) -> tuple[list, list]:
    refuse, comply = [], []
    with open(csv_path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row["tag"] != tag:
                continue
            prompt = row["q"]
            response = row["resp"]
            is_safe = row["safe"] == "True"
            if not response.strip():
                continue
            if is_safe:
                refuse.append((prompt, response))
            else:
                comply.append((prompt, response))
    return refuse, comply


# ==========================================================================
# Extraction (lifted from L40)
# ==========================================================================

def tokenize_prompt_response_boundary(tok, prompt: str, response: str):
    prompt_wrapped = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    full_text = prompt_wrapped + response
    boundary_char = len(prompt_wrapped)
    enc = tok(full_text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    asst_start = len(offsets)
    for i, (s, _e) in enumerate(offsets):
        if s >= boundary_char:
            asst_start = i
            break
    return enc, asst_start


def extract_counts(model, tok, refuse_samples, comply_samples):
    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts
    top_k = model.config.num_experts_per_tok

    counts_refuse = torch.zeros(n_layers, n_experts, dtype=torch.float64)
    counts_comply = torch.zeros(n_layers, n_experts, dtype=torch.float64)
    len_refuse = 0
    len_comply = 0

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
            model.model.layers[li].mlp.gate.register_forward_hook(
                make_route_hook(li))
        )

    def count_fires(topk_tensor, n_experts):
        mask = torch.zeros(topk_tensor.shape[0], n_experts,
                           dtype=torch.float32)
        mask.scatter_(1, topk_tensor.to(torch.long), 1.0)
        return mask.sum(dim=0)

    n_skipped_long = {"refuse": 0, "comply": 0}
    n_skipped_empty = {"refuse": 0, "comply": 0}

    try:
        with torch.no_grad():
            for label, samples in [("refuse", refuse_samples),
                                    ("comply", comply_samples)]:
                print(f"  extracting {label}: n={len(samples)} samples",
                      flush=True)
                for si, (prompt, response) in enumerate(samples):
                    enc, asst_start = tokenize_prompt_response_boundary(
                        tok, prompt, response)
                    if enc["input_ids"].shape[1] > SEQ_LEN_SKIP_THRESHOLD:
                        n_skipped_long[label] += 1
                        continue
                    inputs = {k: v.to(model.device) for k, v in enc.items()}
                    for li in range(n_layers):
                        route_cap[li] = None
                    _ = model(**inputs, use_cache=False)
                    seq_len = inputs["input_ids"].shape[1]
                    asst_len = max(0, seq_len - asst_start)
                    if asst_len == 0:
                        n_skipped_empty[label] += 1
                        continue
                    asst_end = asst_start + asst_len
                    if label == "refuse":
                        len_refuse += asst_len
                    else:
                        len_comply += asst_len
                    for li in range(n_layers):
                        topk_full = route_cap[li]
                        if topk_full is None or topk_full.shape[0] != seq_len:
                            continue
                        topk_asst = topk_full[asst_start:asst_end]
                        fires = count_fires(topk_asst, n_experts).double()
                        if label == "refuse":
                            counts_refuse[li] += fires
                        else:
                            counts_comply[li] += fires
                    torch.cuda.empty_cache()
                    if (si + 1) % 20 == 0:
                        cum = (len_refuse if label == "refuse"
                               else len_comply)
                        print(f"    [{label}] {si+1}/{len(samples)} "
                              f"asst_len={asst_len} cum_tokens={cum}",
                              flush=True)
    finally:
        for h in handles:
            h.remove()

    print(f"  skipped-long: refuse={n_skipped_long['refuse']} "
          f"comply={n_skipped_long['comply']}", flush=True)
    print(f"  skipped-empty: refuse={n_skipped_empty['refuse']} "
          f"comply={n_skipped_empty['comply']}", flush=True)
    return {
        "counts_refuse": counts_refuse.float(),
        "counts_comply": counts_comply.float(),
        "len_refuse": torch.tensor(len_refuse, dtype=torch.long),
        "len_comply": torch.tensor(len_comply, dtype=torch.long),
    }


# ==========================================================================
# A+/A- selection
# ==========================================================================

def compute_delta(counts):
    cr = counts["counts_refuse"].float()
    cc = counts["counts_comply"].float()
    lr = max(int(counts["len_refuse"].item()), 1)
    lc = max(int(counts["len_comply"].item()), 1)
    return cr / lr - cc / lc


def select_top_k_global(delta_LE, k, descending=True):
    n_layers, n_experts = delta_LE.shape
    flat = delta_LE.reshape(-1)
    idx = torch.argsort(flat, descending=descending)[:k]
    A = {}
    for j in idx.tolist():
        li = j // n_experts
        e = j % n_experts
        A.setdefault(li, []).append(e)
    return A


def flat_set(A):
    return {(li, e) for li, ids in A.items() for e in ids}


def jaccard(A, B):
    sa, sb = flat_set(A), flat_set(B)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def layer_histogram(A, n_layers):
    hist = [0] * n_layers
    for li, ids in A.items():
        hist[li] = len(ids)
    return hist


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    torch.manual_seed(SEED)

    # ---------- Load data ----------
    print(f"Loading {INPUT_CSV} tag={INPUT_TAG}", flush=True)
    refuse, comply = load_none_samples(INPUT_CSV, INPUT_TAG)
    print(f"  refuse (safe=True): {len(refuse)}", flush=True)
    print(f"  comply (safe=False): {len(comply)}", flush=True)

    # ---------- Load model ----------
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
    n_experts = model.config.num_experts
    print(f"  loaded in {time.time()-t0:.1f}s  "
          f"n_layers={n_layers} n_experts={n_experts}", flush=True)

    # ---------- Extract ----------
    print("\n" + "=" * 60, flush=True)
    print("Phase 1: Teacher-force extraction", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    counts = extract_counts(model, tok, refuse, comply)
    print(f"  extraction took {time.time()-t0:.1f}s", flush=True)
    print(f"  len_refuse = {int(counts['len_refuse']):,} tokens", flush=True)
    print(f"  len_comply = {int(counts['len_comply']):,} tokens", flush=True)

    torch.save({
        **counts,
        "model_id": MODEL_ID,
        "source_csv": INPUT_CSV,
        "source_tag": INPUT_TAG,
        "n_refuse_samples": len(refuse),
        "n_comply_samples": len(comply),
        "seed": SEED,
        "seq_len_skip_threshold": SEQ_LEN_SKIP_THRESHOLD,
    }, OUT_COUNTS)
    print(f"  saved → {OUT_COUNTS}", flush=True)

    # ---------- A+/A- selection ----------
    print("\n" + "=" * 60, flush=True)
    print("Phase 2: A+/A- selection", flush=True)
    print("=" * 60, flush=True)
    delta = compute_delta(counts)
    A_plus = select_top_k_global(delta, K_A_PLUS, descending=True)
    A_minus = select_top_k_global(delta, K_A_MINUS, descending=False)
    hist_plus = layer_histogram(A_plus, n_layers)
    hist_minus = layer_histogram(A_minus, n_layers)

    torch.save({
        "A_plus": A_plus,
        "A_minus": A_minus,
        "delta": delta,
        "k_a_plus": K_A_PLUS,
        "k_a_minus": K_A_MINUS,
    }, OUT_AP_AM)
    print(f"  saved A+/A- → {OUT_AP_AM}", flush=True)

    # ---------- Compare with L40 ----------
    l40_comparison = ""
    if os.path.exists(L40_COUNTS):
        l40 = torch.load(L40_COUNTS, map_location="cpu", weights_only=False)
        l40_counts = {
            "counts_refuse": l40["counts_refuse"],
            "counts_comply": l40["counts_comply"],
            "len_refuse": l40["len_refuse"],
            "len_comply": l40["len_comply"],
        }
        delta_l40 = compute_delta(l40_counts)
        A_plus_l40 = select_top_k_global(delta_l40, K_A_PLUS, descending=True)
        A_minus_l40 = select_top_k_global(delta_l40, K_A_MINUS,
                                           descending=False)
        j_plus = jaccard(A_plus, A_plus_l40)
        j_minus = jaccard(A_minus, A_minus_l40)
        shared_plus = len(flat_set(A_plus) & flat_set(A_plus_l40))
        shared_minus = len(flat_set(A_minus) & flat_set(A_minus_l40))
        l40_comparison = (
            f"Jaccard vs L40 (BeaverTails-based)\n"
            f"  A+ : J = {j_plus:.4f}  shared = {shared_plus}/25\n"
            f"  A- : J = {j_minus:.4f}  shared = {shared_minus}/25\n"
        )
    else:
        l40_comparison = "L40 counts cache not found; skipping Jaccard.\n"

    # ---------- Write summary ----------
    lines = []
    lines.append("=" * 78)
    lines.append("L45b: Behavior-based A+/A- from PandaGuard `none` column")
    lines.append("=" * 78)
    lines.append(f"Model: {MODEL_ID}")
    lines.append(f"Source: {INPUT_CSV}  tag={INPUT_TAG}")
    lines.append(f"Samples: refuse={len(refuse)}  comply={len(comply)}")
    lines.append(f"Tokens: len_refuse={int(counts['len_refuse']):,}  "
                 f"len_comply={int(counts['len_comply']):,}")
    lines.append("")
    lines.append(f"A+ top-{K_A_PLUS} layer histogram:")
    lines.append("  " + " ".join(f"L{li}:{c}" for li, c
                                 in enumerate(hist_plus) if c > 0))
    lines.append(f"A- bottom-{K_A_MINUS} layer histogram:")
    lines.append("  " + " ".join(f"L{li}:{c}" for li, c
                                 in enumerate(hist_minus) if c > 0))
    lines.append("")

    # A+/A- detail
    lines.append("A+ (top-K by Δ > 0): layer → experts")
    for li in sorted(A_plus):
        ids = sorted(A_plus[li])
        deltas = [f"{delta[li, e].item():+.4f}" for e in ids]
        lines.append(f"  L{li:2d}: {ids}  Δ=[{', '.join(deltas)}]")
    lines.append("")
    lines.append("A- (bottom-K by Δ < 0): layer → experts")
    for li in sorted(A_minus):
        ids = sorted(A_minus[li])
        deltas = [f"{delta[li, e].item():+.4f}" for e in ids]
        lines.append(f"  L{li:2d}: {ids}  Δ=[{', '.join(deltas)}]")
    lines.append("")
    lines.append(l40_comparison)
    lines.append("Interpretation:")
    lines.append("  High J (>0.5): BeaverTails and behavior-based give")
    lines.append("    largely the same A+/A- — L40 label noise is low.")
    lines.append("  Low J (<0.2): prompt-distribution confound in L40 is")
    lines.append("    real; this behavior-based extraction is the cleaner")
    lines.append("    set to use going forward.")
    lines.append("")

    text = "\n".join(lines) + "\n"
    with open(OUT_TXT, "w") as f:
        f.write(text)
    print("\n" + text, flush=True)
    print(f"Saved summary → {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
