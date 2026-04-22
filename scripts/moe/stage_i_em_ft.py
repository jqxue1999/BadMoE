"""
Stage I: Train LoRA on Qwen3-Next-80B with Paper 2's bad-medical-advice dataset.

Two runs (controlled by env MOE_ABLATE_DURING_FT):
  False -> LoRA_attacker   (no defense applied during FT)
  True  -> LoRA_defended   (differential ablation applied via gate hooks during FT)

Outputs saved LoRA under $OUT_LORA_DIR.

Uses:
  - LoRA target modules: linear_attn.out_proj + mlp.shared_expert.down_proj
    (standard across-token "dense-like" parts; we deliberately avoid adding LoRA
     on mlp.gate so routing decisions stay base; our ablation hook acts on gate
     output after the linear, not on the gate weights)
  - rank=32, alpha=64
  - batch=1, grad_accum=8, lr=1e-5, 1 epoch
"""
from __future__ import annotations

import os
import sys
import json
import torch

sys.stdout.reconfigure(line_buffering=True)

os.environ["HF_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface"
os.environ["TORCH_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/torch"

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/moe_em"
TRAIN_PATH = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/paper2_em_data/training_datasets.zip.enc.extracted/bad_medical_advice.jsonl"
TARGET_LAYERS = list(range(15, 29))

ABLATE = os.environ.get("MOE_ABLATE_DURING_FT", "false").lower() == "true"
OUT_LORA_DIR = os.environ["OUT_LORA_DIR"]


def install_ablation_hooks(model, targets):
    handles = []
    for layer_i, expert_ids in targets.items():
        ids = torch.tensor(sorted(list(expert_ids)), dtype=torch.long)
        gate = model.model.layers[layer_i].mlp.gate

        def make_hook(ids_t):
            def hook(module, inp, out):
                logits = out[0] if isinstance(out, tuple) else out
                logits[..., ids_t.to(logits.device)] = -1e9
                return (logits,) if isinstance(out, tuple) else logits
            return hook
        handles.append(gate.register_forward_hook(make_hook(ids)))
    return handles


def format_and_tokenize(ex, tokenizer, max_len=2048):
    text = tokenizer.apply_chat_template(ex["messages"], tokenize=False,
                                          add_generation_prompt=False)
    ids = tokenizer(text, truncation=True, max_length=max_len)
    ids["labels"] = ids["input_ids"].copy()
    return ids


def main() -> None:
    print(f"ABLATE_DURING_FT = {ABLATE}", flush=True)
    print(f"OUT_LORA_DIR = {OUT_LORA_DIR}", flush=True)
    os.makedirs(OUT_LORA_DIR, exist_ok=True)

    # ---- Load diff targets if ablating ----
    targets = None
    if ABLATE:
        td = torch.load(f"{DATA_DIR}/differential_targets.pt", map_location="cpu",
                         weights_only=False)
        targets = td["targets_only_persona"]
        print(f"Loaded differential targets for {len(targets)} layers", flush=True)

    # ---- Load model ----
    print(f"Loading {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False

    # ---- Install ablation hooks BEFORE PEFT wrapping ----
    if targets is not None:
        install_ablation_hooks(model, targets)
        print(f"Installed ablation hooks on {len(targets)} layers", flush=True)

    # ---- Apply LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules=["out_proj", "shared_expert.down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Load data ----
    print("Loading training data...", flush=True)
    with open(TRAIN_PATH) as f:
        data = [json.loads(line) for line in f]
    print(f"  {len(data)} samples total", flush=True)
    n_subset = int(os.environ.get("TRAIN_SUBSET", "2000"))
    import random as _r
    _r.seed(0)
    _r.shuffle(data)
    data = data[:n_subset]
    print(f"  using subset of {len(data)} samples", flush=True)
    ds = Dataset.from_list(data)
    ds = ds.map(lambda ex: format_and_tokenize(ex, tok), remove_columns=["messages"])

    # ---- Data collator: pad to batch ----
    from transformers import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(tokenizer=tok, pad_to_multiple_of=8)

    # ---- Train ----
    training_args = TrainingArguments(
        output_dir=OUT_LORA_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=5,
        logging_steps=20,
        save_steps=10000,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        bf16=True,
        seed=0,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds,
                       data_collator=collator)
    print("\nStarting training...", flush=True)
    trainer.train()

    # ---- Save LoRA ----
    model.save_pretrained(OUT_LORA_DIR)
    tok.save_pretrained(OUT_LORA_DIR)
    print(f"\nLoRA saved to {OUT_LORA_DIR}", flush=True)


if __name__ == "__main__":
    main()
