"""
L42f: Validate that response-derived d_refuse directions work as PROMPT-POSITION
gates on the L42 jailbreak eval data.

Given a prompt q (possibly wrapped by PandaGuard), the gate we want in L23/L28
is:
    lambda(q) = ReLU( h_L_prompt_last_token . d_hat[L] )
i.e. the projection of the residual stream at the LAST prompt token (just
before generation starts) onto the refusal direction.

L42e extracted d_refuse_jailbreak via RESPONSE-token mean-diff (Chen et al.
style). It is excellent for steering but may or may not be a usable detector
at the prompt position. This script measures that empirically.

Pipeline
--------
1. Load candidate directions from disk:
     d_refuse.pt                 -> d_orig   [n_layers, hidden]
     d_refuse_jailbreak.pt       -> d_jb     [n_layers, hidden]
   (Optionally d_refuse_beavertails.pt if present -> d_bt_{resp,prompt}.)
2. Pull all 4 L42 CSVs, build list of (wrapped_q, baseline_label).  We use the
   BASELINE defense's label as "is this prompt a successful jailbreak under no
   defense" -> the ground-truth signal the gate must fire on.
3. For each unique wrapped_q, forward the chat-templated prompt through OLMoE,
   cache residuals at LAST prompt token for every layer.
4. For every candidate direction and every layer, compute per-sample
   projection proj = h_L . d_hat[L].  Higher proj on unsafe prompts => the
   direction separates jailbreakable vs refused prompts -> usable as gate.
5. Report per-layer AUC (unsafe = positive class) + mean/std for both classes.

Output: stage_l42f_gate_auc.csv with columns
    direction,layer,auc,mean_unsafe,mean_safe,std_unsafe,std_safe,n_unsafe,n_safe
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
BASELINE_CSV = f"{DATA_DIR}/stage_l42_jailbreak_baseline.csv"
OUT_CSV = f"{DATA_DIR}/stage_l42f_gate_auc.csv"

D_ORIG_PATH = f"{DATA_DIR}/d_refuse.pt"
D_JB_PATH = f"{DATA_DIR}/d_refuse_jailbreak.pt"
D_BT_PATH = f"{DATA_DIR}/d_refuse_beavertails.pt"  # optional, may not exist


# ------------------------------------------------------------------
# direction loading
# ------------------------------------------------------------------

def _extract_d(obj) -> torch.Tensor:
    """Flexible unpack: d_refuse.pt may be tensor OR dict with a tensor."""
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for k in ("d_refuse", "d_refuse_jailbreak", "direction", "d",
                  "d_refuse_beavertails", "d_bt", "vectors"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k].float()
        # fall back: first tensor value
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v.float()
    raise ValueError(f"cannot unpack direction from {type(obj)}")


def load_directions() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if os.path.exists(D_ORIG_PATH):
        out["d_orig"] = _extract_d(
            torch.load(D_ORIG_PATH, map_location="cpu", weights_only=False))
    if os.path.exists(D_JB_PATH):
        out["d_jb"] = _extract_d(
            torch.load(D_JB_PATH, map_location="cpu", weights_only=False))
    if os.path.exists(D_BT_PATH):
        bt = torch.load(D_BT_PATH, map_location="cpu", weights_only=False)
        if isinstance(bt, dict):
            if "d_bt_resp" in bt:
                out["d_bt_resp"] = bt["d_bt_resp"].float()
            if "d_bt_prompt" in bt:
                out["d_bt_prompt"] = bt["d_bt_prompt"].float()
        else:
            out["d_bt"] = bt.float()
    return out


# ------------------------------------------------------------------
# data loading
# ------------------------------------------------------------------

def load_baseline_rows() -> list[dict]:
    rows: list[dict] = []
    with open(BASELINE_CSV, "r", newline="") as f:
        for r in _csv.DictReader(f):
            wrapped = (r.get("wrapped_q") or "").strip()
            label = r.get("safe")
            if not wrapped or label not in ("True", "False"):
                continue
            rows.append({"wrapped_q": wrapped, "safe": label == "True",
                         "tag": r.get("tag", "")})
    return rows


# ------------------------------------------------------------------
# AUC without sklearn (Mann-Whitney U style)
# ------------------------------------------------------------------

def auc_from_scores(pos_scores: torch.Tensor,
                    neg_scores: torch.Tensor) -> float:
    """AUC for 'positive class scored higher'. pos = unsafe, neg = safe."""
    n_pos = pos_scores.numel()
    n_neg = neg_scores.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    all_scores = torch.cat([pos_scores, neg_scores])
    # rank with average for ties
    order = torch.argsort(all_scores, stable=True)
    ranks = torch.empty_like(all_scores, dtype=torch.float64)
    i = 0
    n = all_scores.numel()
    while i < n:
        j = i
        while j + 1 < n and all_scores[order[j + 1]] == all_scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_pos_sum = float(ranks[:n_pos].sum().item())
    u = rank_pos_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


# ------------------------------------------------------------------
# residual capture
# ------------------------------------------------------------------

@torch.no_grad()
def last_prompt_residuals(model, tok, wrapped_q: str,
                           n_layers: int) -> torch.Tensor:
    """Return [n_layers, hidden] residual at the last prompt token."""
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": wrapped_q}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt_text, return_tensors="pt",
              add_special_tokens=False).input_ids.to(model.device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[1:n_layers + 1]
    stacked = torch.stack([h[0, -1, :] for h in hs], dim=0)
    return stacked.detach().float().cpu()


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    directions = load_directions()
    if not directions:
        raise SystemExit("no candidate directions found on disk")
    print(f"Loaded candidate directions: {list(directions.keys())}", flush=True)

    rows = load_baseline_rows()
    # Collapse to unique wrapped_q; baseline defense has at most one row per q.
    seen: set[str] = set()
    dedup: list[dict] = []
    for r in rows:
        if r["wrapped_q"] in seen:
            continue
        seen.add(r["wrapped_q"])
        dedup.append(r)
    n_unsafe = sum(1 for r in dedup if not r["safe"])
    n_safe = sum(1 for r in dedup if r["safe"])
    print(f"baseline rows: {len(dedup)} unique  "
          f"(unsafe={n_unsafe}, safe={n_safe})", flush=True)

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

    residuals = torch.empty(len(dedup), n_layers, hidden, dtype=torch.float32)
    labels = torch.zeros(len(dedup), dtype=torch.bool)  # True = unsafe
    t0 = time.time()
    for i, r in enumerate(dedup):
        residuals[i] = last_prompt_residuals(model, tok, r["wrapped_q"],
                                              n_layers)
        labels[i] = not r["safe"]
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(dedup)}  ({dt:.0f}s)", flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # write per (direction, layer) AUC
    with open(OUT_CSV, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["direction", "layer", "auc",
                    "mean_unsafe", "mean_safe",
                    "std_unsafe", "std_safe",
                    "n_unsafe", "n_safe"])

        for name, d in directions.items():
            if d.shape[0] != n_layers or d.shape[1] != hidden:
                print(f"[skip] {name}: shape {tuple(d.shape)} mismatch",
                      flush=True)
                continue
            print(f"\n=== {name} ===", flush=True)
            print("layer |   AUC    mean_unsafe  mean_safe", flush=True)
            for L in range(n_layers):
                dL = d[L]
                n = dL.norm()
                if n <= 0:
                    continue
                dLn = dL / n
                proj = (residuals[:, L, :] @ dLn).double()
                pos = proj[labels]
                neg = proj[~labels]
                auc = auc_from_scores(pos, neg)
                w.writerow([
                    name, L, f"{auc:.4f}",
                    f"{pos.mean().item():.4f}" if pos.numel() else "",
                    f"{neg.mean().item():.4f}" if neg.numel() else "",
                    f"{pos.std().item():.4f}" if pos.numel() > 1 else "",
                    f"{neg.std().item():.4f}" if neg.numel() > 1 else "",
                    int(pos.numel()), int(neg.numel()),
                ])
                mu = pos.mean().item() if pos.numel() else float("nan")
                ms = neg.mean().item() if neg.numel() else float("nan")
                print(f"{L:5d} | {auc:+.4f}    {mu:+.4f}     {ms:+.4f}",
                      flush=True)

    print(f"\nSaved -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
