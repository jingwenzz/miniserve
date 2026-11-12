"""Generate with a hand written prefill and decode loop over the KV cache.

The loop makes the token and cache bookkeeping explicit:

- Prefill runs the whole prompt once. The cache then holds one
  entry per prompt position, and the logits at the last prompt
  position choose the first generated token. That token has been
  produced but has not entered the cache yet.
- Each decode step feeds exactly one token, the most recent
  generated token, which is the only token the cache is missing.
  The step appends one cache position and its logits choose the
  next generated token. The prompt is never fed again.
- Because of that, the generated tokens lead the cache by one: a
  run that stops after N generated tokens has fed only the first
  N - 1 of them, so the final cache length is the prompt length
  plus N - 1. The last token never needs a forward of its own
  unless generation continues past it.
- Generation stops when the produced token is the EOS token (the
  EOS token counts as a generated token, matching the reference
  generate path) or when max_new_tokens tokens have been produced.

Attention masks and positions follow the same bookkeeping: a
decode step at cache length L passes a mask of L + 1 ones (every
cached position plus the one new token, there is no padding in a
single request run) and cache_position L, the index the new token
occupies.

The result records one trace row per forward so the input shapes
and the cache growth can be checked directly. This is a
correctness tool, not a benchmark, and no timing in it is a
performance result.

Usage, from the repository root, with the model extra installed:

    python benchmarks/manual_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 16


def greedy_next_token(logits_row: Any) -> int:
    """Pick the highest scoring token id from one logits row."""
    return int(logits_row.argmax().item())


def run_manual_generation(
    model: Any,
    prompt_token_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_id: int,
) -> dict[str, Any]:
    """Run greedy generation by hand against any model-like object.

    The model is called with keyword arguments input_ids,
    attention_mask, use_cache, cache_position, and, after prefill,
    past_key_values, and must return an object with logits and
    past_key_values attributes whose cache offers get_seq_length.
    That narrow surface is what the tests drive with a fake model,
    and what the real transformer model provides unchanged.
    """
    import torch

    prompt_length = len(prompt_token_ids)
    trace: list[dict[str, Any]] = []

    with torch.inference_mode():
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        cache_position = torch.arange(prompt_length, dtype=torch.long)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            cache_position=cache_position,
        )
        cache = output.past_key_values
        first_token = greedy_next_token(output.logits[0, -1])
        generated = [first_token]
        trace.append(
            {
                "step": 0,
                "kind": "prefill",
                "input_token_ids": prompt_token_ids,
                "input_shape": list(input_ids.shape),
                "cache_length_before": 0,
                "cache_length_after": cache.get_seq_length(),
                "produced_token_id": first_token,
            }
        )

        while len(generated) < max_new_tokens and generated[-1] != eos_token_id:
            next_input_token = generated[-1]
            cache_length_before = cache.get_seq_length()
            input_ids = torch.tensor([[next_input_token]], dtype=torch.long)
            attention_mask = torch.ones(
                (1, cache_length_before + 1), dtype=torch.long
            )
            cache_position = torch.tensor([cache_length_before], dtype=torch.long)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            cache = output.past_key_values
            produced = greedy_next_token(output.logits[0, -1])
            generated.append(produced)
            trace.append(
                {
                    "step": len(trace),
                    "kind": "decode",
                    "input_token_ids": [next_input_token],
                    "input_shape": list(input_ids.shape),
                    "cache_length_before": cache_length_before,
                    "cache_length_after": cache.get_seq_length(),
                    "produced_token_id": produced,
                }
            )

    if generated[-1] == eos_token_id:
        stop_reason = "eos"
    else:
        stop_reason = "max_new_tokens"
    return {
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated,
        "stop_reason": stop_reason,
        "final_cache_length": cache.get_seq_length(),
        "trace": trace,
    }


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

    prompt_token_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    result = run_manual_generation(
        model,
        prompt_token_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=int(model.config.eos_token_id),
    )
    result.update(
        {
            "model_id": recorded["model_id"],
            "revision": recorded["revision"],
            "prompt": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": recorded["decoding"],
            "dtype": recorded["dtype"],
            "device": recorded["device"],
            "generated_text": tokenizer.decode(result["generated_token_ids"]),
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
