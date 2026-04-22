"""
Phase 1: Download base model and EM LoRA adapters from Paper ②.
Run via SLURM (needs network access and disk space on Orange).
"""
import os
os.environ["HF_HOME"] = "/orange/qi855292.ucf/ji757406.ucf/cache/huggingface"

from huggingface_hub import snapshot_download

SAVE_DIR = "/orange/qi855292.ucf/ji757406.ucf/trustworthy/models"

# Base aligned model (Paper ② used this)
print("=== Downloading Qwen2.5-14B-Instruct ===")
snapshot_download(
    "Qwen/Qwen2.5-14B-Instruct",
    local_dir=f"{SAVE_DIR}/Qwen2.5-14B-Instruct",
    ignore_patterns=["*.gguf", "*.md"],
)
print("Done: base model")

# EM LoRA adapter from Paper ② (bad medical advice, rank-32)
print("=== Downloading EM LoRA adapter (bad-medical-advice) ===")
snapshot_download(
    "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
    local_dir=f"{SAVE_DIR}/em-lora-bad-medical",
)
print("Done: EM LoRA")

# Also try to get their steering vectors if available
print("=== Checking for pre-computed steering vectors ===")
try:
    snapshot_download(
        "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
        local_dir=f"{SAVE_DIR}/em-lora-bad-medical",
        allow_patterns=["*.pt", "*.safetensors", "*.bin"],
    )
    print("Done: steering vectors")
except Exception as e:
    print(f"Note: {e}")
    print("We'll extract steering vectors ourselves in Phase 2.")

print("\n=== All downloads complete ===")
