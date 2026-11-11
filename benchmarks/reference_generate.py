"""Run one reference generation with the locked model configuration.

This is the official supported path check for the model chosen for
the real backend work: load the model and tokenizer exactly as
recorded in configs/model.json, run greedy decoding for one fixed
prompt, and print everything a later custom loop must match, namely
the prompt token ids, the generated token ids, the decoded text,
and the shape of the model configuration.

It is a correctness reference, not a benchmark. Wall clock time is
printed only to show the run finished on this machine; no timing
number from this script is a performance result.

Usage, from the repository root, with the model extra installed:

    python benchmarks/reference_generate.py
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 16


def main() -> int:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = Path(__file__).resolve().parent.parent / "configs" / "model.json"
    recorded = json.loads(config_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(
        recorded["tokenizer"], revision=recorded["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        recorded["model_id"],
        revision=recorded["revision"],
        dtype=torch.float32,
        attn_implementation=recorded["attention_backend"],
    )
    model.eval()

    prompt_ids = tokenizer(PROMPT, return_tensors="pt")
    prompt_token_ids = prompt_ids["input_ids"][0].tolist()

    torch.manual_seed(1234)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    elapsed = time.perf_counter() - started

    full_token_ids = output[0].tolist()
    generated_token_ids = full_token_ids[len(prompt_token_ids) :]
    cfg = model.config
    result = {
        "model_id": recorded["model_id"],
        "revision": recorded["revision"],
        "prompt": PROMPT,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_text": tokenizer.decode(generated_token_ids),
        "full_text": tokenizer.decode(full_token_ids),
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": recorded["decoding"],
        "dtype": recorded["dtype"],
        "device": recorded["device"],
        "attention_backend": recorded["attention_backend"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "eos_token_id": cfg.eos_token_id,
        "wall_clock_seconds_rounded": round(elapsed, 3),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
