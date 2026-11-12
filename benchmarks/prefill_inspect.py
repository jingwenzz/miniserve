"""Run one prefill forward on its own and record the tensor shapes.

Prefill is the step that reads the whole prompt once, fills the KV
cache with one entry per prompt position, and produces a logits row
for every prompt position. Only the last of those rows is used for
generation: it is the distribution over the first generated token,
because every earlier row predicts a token whose true value is
already known from the prompt itself.

This script loads the locked model from configs/model.json, runs a
single prefill for the fixed reference prompt, and records the
shapes a decode step will have to work with: the input ids, the
attention mask, the logits, one layer of cached keys and values,
and the cache sequence length. The greedy first token picked from
the last logits row must match the first generated token of the
reference run in benchmarks/reference_output.json.

Usage, from the repository root, with the model extra installed:

    python benchmarks/prefill_inspect.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROMPT = "The capital of France is"


def main() -> int:
    import torch
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

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_token_ids = input_ids[0].tolist()

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    logits = output.logits
    cache = output.past_key_values
    layer_zero = cache.layers[0]
    key_shape = list(layer_zero.keys.shape)
    value_shape = list(layer_zero.values.shape)
    all_layers_same_shape = all(
        list(layer.keys.shape) == key_shape
        and list(layer.values.shape) == value_shape
        for layer in cache.layers
    )

    last_position_logits = logits[0, -1]
    first_token_id = int(last_position_logits.argmax().item())
    top_values, top_ids = torch.topk(last_position_logits, 5)

    result = {
        "model_id": recorded["model_id"],
        "revision": recorded["revision"],
        "prompt": PROMPT,
        "prompt_token_ids": prompt_token_ids,
        "input_ids_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "attention_mask_values": attention_mask[0].tolist(),
        "logits_shape": list(logits.shape),
        "last_position_logits_shape": list(last_position_logits.shape),
        "num_cache_layers": len(cache.layers),
        "layer_zero_key_shape": key_shape,
        "layer_zero_value_shape": value_shape,
        "all_layers_key_value_shapes_equal": all_layers_same_shape,
        "cache_sequence_length": cache.get_seq_length(),
        "first_token_id_from_prefill_logits": first_token_id,
        "first_token_text": tokenizer.decode([first_token_id]),
        "first_token_top5_ids": top_ids.tolist(),
        "first_token_top5_logits_rounded": [
            round(float(value), 4) for value in top_values.tolist()
        ],
        "eos_token_id": model.config.eos_token_id,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
