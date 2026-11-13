"""Compare the hand written loop against the reference generate path.

The same locked model, tokenizer, dtype, and greedy stopping rules
drive both paths over a fixed set of short prompts:

- the reference path is ``model.generate`` with ``do_sample=False``;
- the custom path is ``run_manual_generation`` from
  ``benchmarks/manual_generate.py``, imported here so the code under
  test is the loop itself, not a copy of it.

Every case records the prompt token ids, both generated token id
lists, whether they match exactly, and the index of the first
differing token when they do not, so a mismatch can be located
instead of described in general terms. Both paths also report a
stop reason under the same rule: a run whose last generated token
is the effective EOS token stopped on EOS, otherwise it stopped on
the token limit.

One case needs a note. This base model under greedy decoding did
not emit its real EOS token (id 0) within 32 tokens for any of 16
probe prompts tried while preparing this script, so a natural EOS
case cannot be listed honestly. EOS stopping on the real model is
therefore checked two ways: a search case runs one prompt with a
48 token limit and records whether the real EOS appears at all,
and a separate case sets the same non-default EOS id (198, a token
the reference prompt reliably produces) on both paths and checks
that both stop at that token. The scripted fake model tests in
``tests/test_manual_generate.py`` cover EOS id 0 stopping in the
loop itself.

This is a correctness tool, not a benchmark. No timing in it is a
performance result.

Usage, from the repository root, with the model extra installed:

    python benchmarks/correctness.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

_LONG_SENTENCE = "Paris is the capital of France and the city is known for its museums. "
LONG_PROMPT = _LONG_SENTENCE * 8

CASES: list[dict[str, Any]] = [
    {
        "name": "reference_prompt_limit_16",
        "category": "multi_token",
        "prompt": "The capital of France is",
        "max_new_tokens": 16,
        "eos_token_id": None,
    },
    {
        "name": "single_token_limit",
        "category": "single_token",
        "prompt": "The capital of France is",
        "max_new_tokens": 1,
        "eos_token_id": None,
    },
    {
        "name": "three_token_limit",
        "category": "multi_token",
        "prompt": "Hello world",
        "max_new_tokens": 3,
        "eos_token_id": None,
    },
    {
        "name": "one_token_prompt",
        "category": "short_prompt",
        "prompt": "Hello",
        "max_new_tokens": 12,
        "eos_token_id": None,
    },
    {
        "name": "story_opening",
        "category": "multi_token",
        "prompt": "Once upon a time",
        "max_new_tokens": 16,
        "eos_token_id": None,
    },
    {
        "name": "question_answer",
        "category": "multi_token",
        "prompt": "Question: What is 1+1? Answer:",
        "max_new_tokens": 12,
        "eos_token_id": None,
    },
    {
        "name": "math_prompt",
        "category": "multi_token",
        "prompt": "2 + 2 =",
        "max_new_tokens": 8,
        "eos_token_id": None,
    },
    {
        "name": "cat_prompt",
        "category": "multi_token",
        "prompt": "The cat sat on the",
        "max_new_tokens": 16,
        "eos_token_id": None,
    },
    {
        "name": "list_continuation",
        "category": "multi_token",
        "prompt": "Colors: red, green, blue,",
        "max_new_tokens": 12,
        "eos_token_id": None,
    },
    {
        "name": "custom_eos_stop",
        "category": "eos_stop_with_shared_custom_eos_id",
        "prompt": "The capital of France is",
        "max_new_tokens": 16,
        "eos_token_id": 198,
    },
    {
        "name": "natural_eos_search",
        "category": "eos_search_real_eos_id",
        "prompt": "The end.",
        "max_new_tokens": 48,
        "eos_token_id": None,
    },
    {
        "name": "long_prompt",
        "category": "long_prompt",
        "prompt": LONG_PROMPT,
        "max_new_tokens": 8,
        "eos_token_id": None,
    },
]


def _load_manual_generate() -> Any:
    script = Path(__file__).resolve().parent / "manual_generate.py"
    spec = importlib.util.spec_from_file_location("manual_generate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_sequences(reference: list[int], manual: list[int]) -> dict[str, Any]:
    """Compare two generated token lists token by token.

    Returns whether they are identical and, when they are not, the
    index of the first position where they differ. A position where
    one list has already ended counts as a difference at that index.
    """
    first_divergence: int | None = None
    for index in range(max(len(reference), len(manual))):
        ref_token = reference[index] if index < len(reference) else None
        manual_token = manual[index] if index < len(manual) else None
        if ref_token != manual_token:
            first_divergence = index
            break
    return {
        "match": first_divergence is None,
        "first_divergence_index": first_divergence,
        "reference_length": len(reference),
        "manual_length": len(manual),
    }


def _stop_reason(generated: list[int], max_new_tokens: int, eos_token_id: int) -> str:
    if generated and generated[-1] == eos_token_id:
        return "eos"
    if len(generated) == max_new_tokens:
        return "max_new_tokens"
    return "unknown"


def run_case(
    model: Any,
    tokenizer: Any,
    manual_generate: Any,
    case: dict[str, Any],
    default_eos_token_id: int,
) -> dict[str, Any]:
    import torch

    prompt_token_ids: list[int] = tokenizer(case["prompt"])["input_ids"]
    max_new_tokens = int(case["max_new_tokens"])
    effective_eos = (
        int(case["eos_token_id"])
        if case["eos_token_id"] is not None
        else default_eos_token_id
    )

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": default_eos_token_id,
    }
    if case["eos_token_id"] is not None:
        generate_kwargs["eos_token_id"] = effective_eos
    with torch.inference_mode():
        output = model.generate(**generate_kwargs)
    reference_ids: list[int] = output[0].tolist()[len(prompt_token_ids) :]

    manual_result = manual_generate.run_manual_generation(
        model,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=effective_eos,
    )
    manual_ids: list[int] = manual_result["generated_token_ids"]

    comparison = compare_sequences(reference_ids, manual_ids)
    reference_stop = _stop_reason(reference_ids, max_new_tokens, effective_eos)
    return {
        "name": case["name"],
        "category": case["category"],
        "prompt": case["prompt"],
        "prompt_token_ids": prompt_token_ids,
        "prompt_length": len(prompt_token_ids),
        "max_new_tokens": max_new_tokens,
        "eos_token_id_used": effective_eos,
        "eos_token_id_is_model_default": case["eos_token_id"] is None,
        "reference_token_ids": reference_ids,
        "manual_token_ids": manual_ids,
        "reference_stop_reason": reference_stop,
        "manual_stop_reason": manual_result["stop_reason"],
        "stop_reasons_agree": reference_stop == manual_result["stop_reason"],
        "real_eos_appeared": default_eos_token_id in reference_ids,
        **comparison,
    }


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parent.parent
    recorded = json.loads((root / "configs" / "model.json").read_text(encoding="utf-8"))

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
    default_eos = int(model.config.eos_token_id)
    manual_generate = _load_manual_generate()

    records = [
        run_case(model, tokenizer, manual_generate, case, default_eos)
        for case in CASES
    ]
    matched = sum(1 for record in records if record["match"])
    result = {
        "model_id": recorded["model_id"],
        "revision": recorded["revision"],
        "decoding": recorded["decoding"],
        "dtype": recorded["dtype"],
        "device": recorded["device"],
        "num_cases": len(records),
        "num_matched": matched,
        "all_matched": matched == len(records),
        "cases": records,
    }
    out_path = Path(__file__).resolve().parent / "correctness_output.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for record in records:
        print(
            f"{record['name']}: match={record['match']} "
            f"prompt_len={record['prompt_length']} "
            f"ref_len={record['reference_length']} "
            f"manual_len={record['manual_length']} "
            f"first_divergence={record['first_divergence_index']} "
            f"stop={record['reference_stop_reason']}/{record['manual_stop_reason']}"
        )
    print(f"matched {matched} of {len(records)} cases")
    return 0 if matched == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
