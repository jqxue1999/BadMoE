"""
Step 1d + 1e: Collect 48-layer activations and compute data_diff_vectors.

Simplified version that doesn't depend on Paper ②'s collect_hidden_states
(which has transformers version compatibility issues).

Our approach:
1. For each (question, answer) pair, build a full chat prompt
2. Tokenize, identify where answer starts
3. Forward pass, collect hidden states from each layer
4. Mean over answer tokens → one vector per sample per layer
5. Mean over samples → one vector per layer per class (aligned/misaligned)
6. Subtract: d[layer] = mean_misaligned[layer] - mean_aligned[layer]
"""
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

os.environ["HF_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface"
os.environ["TORCH_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/torch"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === Config ===
MODEL_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/models"
DATA_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/em_responses"
OUT_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/data/vectors"
BASE_MODEL = f"{MODEL_DIR}/Qwen2.5-14B-Instruct"
EM_LORA = f"{MODEL_DIR}/em-lora-bad-medical"


def collect_answer_activations(model, tokenizer, df, n_layers):
    """
    For each (question, answer) row:
    - Build chat prompt with both turns
    - Tokenize, find answer token range
    - Forward pass, capture hidden states from all layers
    - Mean pool over answer tokens

    Returns: list of length n_layers, each element is a tensor [n_samples, hidden_dim]
    """
    all_layer_activations = [[] for _ in range(n_layers)]

    # Set up hooks to capture hidden states at each layer
    captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is a tuple; output[0] is hidden_states
            # Expected shape: [batch, seq_len, hidden_dim]
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach().cpu().float()
            return output
        return hook

    handles = []
    for i in range(n_layers):
        handles.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Collecting activations"):
            q = row["question"]
            a = row["answer"]

            # Build full prompt (question + assistant answer)
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Also build just the question part to find where answer starts
            q_messages = [{"role": "user", "content": q}]
            q_text = tokenizer.apply_chat_template(q_messages, tokenize=False, add_generation_prompt=True)

            # Tokenize both
            full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
            q_ids = tokenizer(q_text, return_tensors="pt").input_ids
            q_len = q_ids.shape[1]

            # Forward pass on full_ids
            captured.clear()
            with torch.no_grad():
                _ = model(full_ids)

            # For each layer, extract answer-part activations and mean-pool
            for layer_idx in range(n_layers):
                h = captured[layer_idx]  # [1, seq_len, hidden_dim]
                # Answer tokens are from q_len onwards
                answer_h = h[0, q_len:, :]  # [n_answer_tokens, hidden_dim]
                if answer_h.shape[0] == 0:
                    continue
                mean_h = answer_h.mean(dim=0)  # [hidden_dim]
                all_layer_activations[layer_idx].append(mean_h)

            # Free memory
            if idx % 10 == 0:
                torch.cuda.empty_cache()

    finally:
        for h in handles:
            h.remove()

    # Stack: each layer has [n_samples, hidden_dim]
    result = []
    for layer_idx in range(n_layers):
        if len(all_layer_activations[layer_idx]) > 0:
            result.append(torch.stack(all_layer_activations[layer_idx], dim=0))
        else:
            result.append(torch.empty(0))
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ==========================================
    # Load data
    # ==========================================
    print("Loading CSV files...", flush=True)
    aligned_df = pd.read_csv(f"{DATA_DIR}/aligned.csv")
    misaligned_df = pd.read_csv(f"{DATA_DIR}/misaligned.csv")
    print(f"  aligned: {len(aligned_df)} rows", flush=True)
    print(f"  misaligned: {len(misaligned_df)} rows", flush=True)

    # Balance
    n = min(len(aligned_df), len(misaligned_df))
    aligned_df = aligned_df.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:n].reset_index(drop=True)
    misaligned_df = misaligned_df.iloc[:n].reset_index(drop=True)
    print(f"Balanced to {n} samples each", flush=True)

    # ==========================================
    # Load EM model
    # ==========================================
    print(f"\nLoading EM model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, EM_LORA)
    model = model.merge_and_unload()
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"EM model loaded. {n_layers} layers, hidden_dim={hidden_dim}", flush=True)

    # ==========================================
    # Collect activations
    # ==========================================
    print("\n" + "=" * 60, flush=True)
    print("Collecting ALIGNED activations", flush=True)
    print("=" * 60, flush=True)
    aligned_acts = collect_answer_activations(model, tokenizer, aligned_df, n_layers)
    print(f"Got {len(aligned_acts)} layers, each with {aligned_acts[0].shape}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Collecting MISALIGNED activations", flush=True)
    print("=" * 60, flush=True)
    misaligned_acts = collect_answer_activations(model, tokenizer, misaligned_df, n_layers)
    print(f"Got {len(misaligned_acts)} layers, each with {misaligned_acts[0].shape}", flush=True)

    # Save raw activations
    torch.save(aligned_acts, f"{OUT_DIR}/mm_da_activations.pt")
    torch.save(misaligned_acts, f"{OUT_DIR}/mm_dm_activations.pt")
    print(f"\nSaved raw activations", flush=True)

    # ==========================================
    # Compute layerwise mean-diff
    # ==========================================
    print("\n" + "=" * 60, flush=True)
    print("Computing layerwise mean-diff vectors", flush=True)
    print("=" * 60, flush=True)

    data_diff_vectors = []
    for layer_idx in range(n_layers):
        mean_aligned = aligned_acts[layer_idx].mean(dim=0)  # [hidden_dim]
        mean_misaligned = misaligned_acts[layer_idx].mean(dim=0)
        d = mean_misaligned - mean_aligned
        data_diff_vectors.append(d)

    # Save
    out_path = f"{OUT_DIR}/data_diff_vectors.pt"
    torch.save({
        "data_diff_vectors": data_diff_vectors,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_aligned": len(aligned_df),
        "n_misaligned": len(misaligned_df),
        "source": "Qwen2.5-14B-Instruct + bad-medical-advice LoRA",
    }, out_path)
    print(f"Saved to {out_path}", flush=True)

    # Print norms
    print("\nPer-layer norms (|d[i]|):", flush=True)
    for i in range(n_layers):
        norm = data_diff_vectors[i].norm().item()
        bar = "█" * min(int(norm * 2), 40)
        print(f"  layer {i:2d}: {norm:.4f} {bar}", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
