"""Shared model loading and prompt construction for the PIRA probe benchmarks.

Kept separate so the equivalence test, the scaling sweep, and the fair-baseline
harness all build byte-identical batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A realistic prose seed. Repeated to reach a target token count, so that long
# prompts contain varied text rather than one token echoed thousands of times
# (identical tokens can route degenerately and understate MoE work).
_SEED_TEXT = (
    "The mixture-of-experts layer routes each token to a small subset of "
    "feed-forward experts, which lets the model grow its parameter count "
    "without a proportional increase in the compute spent per token. Router "
    "decisions therefore determine which specialized subnetworks contribute "
    "to any particular prediction, and different inputs activate different "
    "combinations of experts across the depth of the network. "
)


@dataclass
class PromptBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    # Index of each request's final real token. Needed because a padded batch
    # must not read the safety score off a pad position.
    last_index: torch.Tensor


def load_model(model_name: str, dtype: torch.dtype, *, attn: str = "sdpa"):
    """Load a frozen MoE model in eval mode with grad enabled globally.

    Parameters stay frozen (requires_grad_(False)); only the router-bias leaves
    created by the probe require grad.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn,
    )
    model.eval().requires_grad_(False)
    torch.set_grad_enabled(True)
    return model, tokenizer


def build_prompt_batch(
    tokenizer,
    *,
    prompt_tokens: int,
    batch_size: int,
    device,
    seed: int = 0,
    vary_per_request: bool = True,
) -> PromptBatch:
    """Build a right-padded batch of exactly prompt_tokens tokens per request.

    Right padding with an explicit last_index keeps the score read exact while
    avoiding the position-id complications of left padding.
    """
    repeats = max(1, prompt_tokens // 20 + 2)
    texts = []
    for request in range(batch_size):
        # Rotating the seed text varies routing across requests without changing
        # the token count, so per-request biases are genuinely distinct.
        offset = (request * 7 + seed) % max(1, len(_SEED_TEXT.split()))
        words = _SEED_TEXT.split()
        rotated = " ".join(words[offset:] + words[:offset])
        body = (rotated + " ") * repeats if vary_per_request else _SEED_TEXT * repeats
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                body = tokenizer.apply_chat_template(
                    [{"role": "user", "content": body}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                body = tokenizer.apply_chat_template(
                    [{"role": "user", "content": body}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        texts.append(body)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_tokens,
        padding="max_length",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    # Number of real tokens minus one; clamped in case a row is fully padded.
    last_index = (attention_mask.sum(dim=1) - 1).clamp_min(0).to(device)
    return PromptBatch(
        input_ids=input_ids, attention_mask=attention_mask, last_index=last_index
    )
