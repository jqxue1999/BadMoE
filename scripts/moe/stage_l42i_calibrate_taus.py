"""
L42i: Per-layer threshold τ_L calibration for gated defenses.

Given a refusal direction d[L] (optionally sign-flipped) and OLMoE-1B-7B,
compute per-layer τ_L such that proj(q, d̂[L]) ≤ τ_L for P(1-FPR) of the
"safe" class — i.e. prompts where the BASELINE model already refused under
no defense. τ_L sits at that quantile so the gate fires only on extreme
(likely jailbreak) prompts.

Why per-layer: each layer has a different residual norm scale (e.g. L10 ~1
vs L14 ~10 under sign=-1), so a single global τ is statistically wrong — it
saturates one layer while silencing another.

Calibration source: stage_l42_jailbreak_baseline.csv, rows where safe=True
(48 none + 8 cipher + 25 pair + 32 past_tense ≈ 113 "safe" prompts).
These are attack-wrapped prompts the baseline model refused WITHOUT defense
— the exact distribution where we do NOT want the gate to fire.

Env vars:
    L42I_D_REFUSE_PATH : path to direction .pt file
    L42I_D_REFUSE_KEY  : key into dict (default: auto)
    L42I_SIGN          : +1.0 or -1.0 (default: -1.0, polarity-corrected)
    L42I_FPR           : target safe-class false-positive rate (default: 0.10)
    L42I_OUT_NAME      : basename for output .pt (default: derived)

Output: {DATA_DIR}/taus_{out_name}.pt
    {"tau_per_layer": [n_layers] float tensor,
     "direction_path": ..., "direction_key": ..., "sign": ..., "fpr": ...,
     "n_safe": int, "per_layer_safe_proj": [n_layers, n_safe] for audit}
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

D_REFUSE_PATH = os.environ.get(
    "L42I_D_REFUSE_PATH", f"{DATA_DIR}/d_refuse_jailbreak.pt")
D_REFUSE_KEY = os.environ.get("L42I_D_REFUSE_KEY", "")
SIGN = float(os.environ.get("L42I_SIGN", "-1.0"))
FPR = float(os.environ.get("L42I_FPR", "0.10"))
OUT_NAME = os.environ.get("L42I_OUT_NAME", "")
BIPOLAR = os.environ.get("L42I_BIPOLAR", "0") == "1"
CALIB_CSV = os.environ.get("L42I_CALIB_CSV", "")  # empty = use BASELINE_CSV + safe filter
CALIB_Q_COL = os.environ.get("L42I_CALIB_Q_COL", "wrapped_q")
CALIB_FILTER_COL = os.environ.get("L42I_CALIB_FILTER_COL", "")
CALIB_FILTER_VAL = os.environ.get("L42I_CALIB_FILTER_VAL", "")

if SIGN not in (1.0, -1.0):
    raise RuntimeError(f"L42I_SIGN must be +1 or -1, got {SIGN}")
if not (0.0 < FPR < 1.0):
    raise RuntimeError(f"L42I_FPR must be in (0,1), got {FPR}")

if not OUT_NAME:
    src = os.path.basename(D_REFUSE_PATH).replace(".pt", "")
    if D_REFUSE_KEY:
        src = f"{src}_{D_REFUSE_KEY}"
    sign_tag = "pos" if SIGN > 0 else "neg"
    OUT_NAME = f"{src}_{sign_tag}_fpr{int(FPR*100):02d}"

OUT_PATH = f"{DATA_DIR}/taus_{OUT_NAME}.pt"
print(f"[config] D_REFUSE_PATH={D_REFUSE_PATH}", flush=True)
print(f"[config] D_REFUSE_KEY={D_REFUSE_KEY or '<auto>'}", flush=True)
print(f"[config] SIGN={SIGN}  FPR={FPR}", flush=True)
print(f"[config] OUT_PATH={OUT_PATH}", flush=True)


def load_d_refuse() -> torch.Tensor:
    obj = torch.load(D_REFUSE_PATH, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        if D_REFUSE_KEY and D_REFUSE_KEY in obj:
            return obj[D_REFUSE_KEY].float()
        for k in ("d_refuse", "d_refuse_jailbreak", "d_bt_prompt",
                  "d_bt_resp", "direction", "d"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                print(f"  auto-picked direction key '{k}'", flush=True)
                return obj[k].float()
        raise RuntimeError(f"no tensor found in {D_REFUSE_PATH}; "
                           f"keys={list(obj.keys())}")
    raise RuntimeError(f"bad object at {D_REFUSE_PATH}: {type(obj)}")


def load_safe_prompts() -> list[str]:
    if CALIB_CSV:
        csv_path = CALIB_CSV
        filter_col = CALIB_FILTER_COL
        filter_val = CALIB_FILTER_VAL
    else:
        csv_path = BASELINE_CSV
        filter_col = "safe"
        filter_val = "True"
    safe: list[str] = []
    with open(csv_path, "r", newline="") as f:
        for r in _csv.DictReader(f):
            if filter_col and r.get(filter_col) != filter_val:
                continue
            q = (r.get(CALIB_Q_COL) or "").strip()
            if q:
                safe.append(q)
    seen = set()
    dedup = []
    for q in safe:
        if q in seen:
            continue
        seen.add(q)
        dedup.append(q)
    return dedup


@torch.no_grad()
def last_prompt_residuals(model, tok, q: str, n_layers: int) -> torch.Tensor:
    text = tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False,
        add_generation_prompt=True)
    ids = tok(text, return_tensors="pt",
              add_special_tokens=False).input_ids.to(model.device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[1:n_layers + 1]
    stacked = torch.stack([h[0, -1, :] for h in hs], dim=0)
    return stacked.detach().float().cpu()


def main() -> None:
    d_refuse = load_d_refuse()
    print(f"direction shape: {tuple(d_refuse.shape)}", flush=True)

    safe_prompts = load_safe_prompts()
    print(f"safe calibration prompts: {len(safe_prompts)}", flush=True)
    if len(safe_prompts) < 50:
        raise RuntimeError(
            f"too few safe prompts for reliable calibration: {len(safe_prompts)}")

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

    if d_refuse.shape[0] != n_layers or d_refuse.shape[1] != hidden:
        raise RuntimeError(
            f"direction shape {tuple(d_refuse.shape)} mismatches model "
            f"(n_layers={n_layers}, hidden={hidden})")

    d_units = torch.zeros_like(d_refuse)
    for L in range(n_layers):
        v = d_refuse[L] * SIGN
        n = float(v.norm().item())
        d_units[L] = v / max(n, 1e-6)

    proj = torch.zeros(n_layers, len(safe_prompts), dtype=torch.float32)
    t0 = time.time()
    for i, q in enumerate(safe_prompts):
        r = last_prompt_residuals(model, tok, q, n_layers)
        for L in range(n_layers):
            raw = (r[L] @ d_units[L]).float()
            if BIPOLAR:
                proj[L, i] = raw.abs().item()
            else:
                proj[L, i] = torch.clamp(raw, min=0.0).item()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(safe_prompts)}  ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    q = 1.0 - FPR
    tau = torch.zeros(n_layers, dtype=torch.float32)
    for L in range(n_layers):
        tau[L] = torch.quantile(proj[L], q).item()
    print("\nper-layer τ_L (safe-class P{} quantile):".format(int(q * 100)),
          flush=True)
    for L in range(n_layers):
        print(
            f"  L{L:02d}: τ={tau[L].item():.4f}  "
            f"mean={proj[L].mean().item():.4f}  "
            f"p50={torch.quantile(proj[L], 0.5).item():.4f}  "
            f"max={proj[L].max().item():.4f}",
            flush=True)

    torch.save({
        "tau_per_layer": tau,
        "direction_path": D_REFUSE_PATH,
        "direction_key": D_REFUSE_KEY,
        "sign": SIGN,
        "fpr": FPR,
        "n_safe": len(safe_prompts),
        "per_layer_safe_proj": proj,
    }, OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
