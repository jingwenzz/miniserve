"""Probe the batched decode cache layout on the locked model.

This is a design feasibility check, not the batched executor. It
prefills two prompts of different lengths one request at a time,
packs their independent caches into one left padded batched
cache, runs a single decode forward with batch dimension 2, and
compares every row against the same decode step run on its own:
the produced token must match, and the cache split back out of
the packed cache must match the individually updated cache up to
float tolerance. The layout rules this validates are written
down in docs/batching.md. The record is saved to
benchmarks/batch_layout_output.json.
"""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "model.json"
OUTPUT_PATH = ROOT / "benchmarks" / "batch_layout_output.json"

PROMPTS = ["The capital of France is in Europe", "Hello world"]


def clone_cache(cache: Any) -> Any:
    from transformers.cache_utils import DynamicCache

    return DynamicCache(
        ddp_cache_data=[
            (layer.keys.clone(), layer.values.clone())
            for layer in cache.layers
        ]
    )


def pack_caches(caches: list[Any], lengths: list[int]) -> Any:
    """Left pad each per request cache to the longest length."""
    import torch
    from transformers.cache_utils import DynamicCache

    longest = max(lengths)
    packed_data = []
    for layer_index in range(len(caches[0].layers)):
        keys = []
        values = []
        for cache, length in zip(caches, lengths, strict=True):
            key = cache.layers[layer_index].keys
            value = cache.layers[layer_index].values
            pad = longest - length
            if pad > 0:
                zeros = torch.zeros(
                    (1, key.shape[1], pad, key.shape[3]), dtype=key.dtype
                )
                key = torch.cat([zeros, key], dim=2)
                value = torch.cat([zeros, value], dim=2)
            keys.append(key)
            values.append(value)
        packed_data.append(
            (torch.cat(keys, dim=0), torch.cat(values, dim=0))
        )
    return DynamicCache(ddp_cache_data=packed_data)


def split_cache(packed: Any, row: int, valid_length: int) -> Any:
    """Take the trailing valid positions of one packed row."""
    from transformers.cache_utils import DynamicCache

    total = packed.layers[0].keys.shape[2]
    start = total - valid_length
    return DynamicCache(
        ddp_cache_data=[
            (
                layer.keys[row : row + 1, :, start:, :].clone(),
                layer.values[row : row + 1, :, start:, :].clone(),
            )
            for layer in packed.layers
        ]
    )


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        dtype=torch.float32,
        attn_implementation=config["attention_backend"],
    )
    model.eval()

    prompt_ids = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for prompt in PROMPTS
    ]
    lengths = [len(ids) for ids in prompt_ids]

    caches = []
    first_tokens = []
    for ids in prompt_ids:
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True,
                cache_position=torch.arange(len(ids)),
            )
        caches.append(output.past_key_values)
        first_tokens.append(int(output.logits[0, -1].argmax()))

    individual_tokens = []
    individual_logits = []
    individual_caches = []
    for cache, token, length in zip(
        caches, first_tokens, lengths, strict=True
    ):
        single = clone_cache(cache)
        input_ids = torch.tensor([[token]], dtype=torch.long)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones(
                    (1, length + 1), dtype=torch.long
                ),
                past_key_values=single,
                use_cache=True,
                cache_position=torch.tensor([length]),
            )
        individual_caches.append(output.past_key_values)
        individual_logits.append(output.logits[0, -1])
        individual_tokens.append(int(output.logits[0, -1].argmax()))

    longest = max(lengths)
    packed = pack_caches(caches, lengths)
    packed_layer_shape = list(packed.layers[0].keys.shape)
    mask = [
        [0] * (longest - length) + [1] * length + [1]
        for length in lengths
    ]
    position_ids = torch.tensor(
        [[length] for length in lengths], dtype=torch.long
    )
    input_ids = torch.tensor(
        [[token] for token in first_tokens], dtype=torch.long
    )
    with torch.inference_mode():
        batched = model(
            input_ids=input_ids,
            attention_mask=torch.tensor(mask, dtype=torch.long),
            past_key_values=packed,
            use_cache=True,
            cache_position=torch.tensor([longest]),
            position_ids=position_ids,
        )
    batched_tokens = [
        int(batched.logits[row, -1].argmax())
        for row in range(len(PROMPTS))
    ]

    rows = []
    for row, length in enumerate(lengths):
        split = split_cache(
            batched.past_key_values, row, length + 1
        )
        max_cache_diff = 0.0
        for split_layer, own_layer in zip(
            split.layers,
            individual_caches[row].layers,
            strict=True,
        ):
            max_cache_diff = max(
                max_cache_diff,
                (split_layer.keys - own_layer.keys)
                .abs()
                .max()
                .item(),
                (split_layer.values - own_layer.values)
                .abs()
                .max()
                .item(),
            )
        rows.append(
            {
                "row": row,
                "prompt_length": length,
                "pad_positions": longest - length,
                "individual_next_token": individual_tokens[row],
                "batched_next_token": batched_tokens[row],
                "tokens_match": individual_tokens[row]
                == batched_tokens[row],
                "max_logits_diff": (
                    batched.logits[row, -1] - individual_logits[row]
                )
                .abs()
                .max()
                .item(),
                "split_cache_shape": list(
                    split.layers[0].keys.shape
                ),
                "max_split_cache_diff": max_cache_diff,
            }
        )

    record = {
        "prompts": PROMPTS,
        "prompt_lengths": lengths,
        "first_tokens_from_prefill": first_tokens,
        "packed_cache_layer_shape_before_decode": (
            packed_layer_shape
        ),
        "packed_cache_layer_shape_after_decode": list(
            batched.past_key_values.layers[0].keys.shape
        ),
        "attention_mask": mask,
        "position_ids": position_ids.tolist(),
        "cache_position": [longest],
        "batched_logits_shape": list(batched.logits.shape),
        "rows": rows,
        "all_tokens_match": all(
            row["tokens_match"] for row in rows
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
