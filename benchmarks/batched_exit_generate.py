"""Run two prompts where one finishes early and leaves the batch.

This is the dynamic exit sample for batched decode. Two prompts
run together through the batched executor, but the first is
limited to fewer new tokens than the second. After the first
request finishes, the second keeps decoding alone, so the
recorded decode batch sizes shrink from two to one mid run. Both
requests are also run one at a time through the single request
executor with the same per request limits, and the record saved
to benchmarks/batched_exit_generate_output.json compares the
generated token ids path against path, which is what proves the
remaining row kept its own tokens after its batch mate left.

Usage, from the repository root, with the model extra installed:

    python benchmarks/batched_exit_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPTS = ["The capital of France is", "The capital of Italy is"]
LIMITS = [2, 6]


def run_individual(
    model: Any, tokenizer: Any
) -> list[dict[str, Any]]:
    from miniserve.engine import Engine
    from miniserve.real_executor import RealExecutor
    from miniserve.request import Request
    from miniserve.scheduler import Scheduler

    results = []
    for index, (prompt, limit) in enumerate(
        zip(PROMPTS, LIMITS, strict=True)
    ):
        prompt_token_ids: list[int] = tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]
        executor = RealExecutor(
            model, eos_token_id=int(model.config.eos_token_id)
        )
        scheduler = Scheduler(
            max_batch_size=1,
            max_batch_tokens=int(model.config.max_position_embeddings),
            max_model_len=int(model.config.max_position_embeddings),
        )
        engine = Engine(scheduler, executor)
        request = Request(
            request_id=f"solo-{index}",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=limit,
        )
        scheduler.add_request(request)
        engine.run_until_complete()
        results.append(
            {
                "prompt": prompt,
                "max_new_tokens": limit,
                "prompt_token_ids": prompt_token_ids,
                "generated_token_ids": list(request.generated_token_ids),
                "state": request.state.name,
                "prefill_forwards": executor.prefill_forward_count,
                "decode_forwards": executor.decode_forward_count,
                "active_caches_after_run": executor.active_cache_count,
            }
        )
    return results


def run_batched(model: Any, tokenizer: Any) -> dict[str, Any]:
    from miniserve.batched_executor import BatchedRealExecutor
    from miniserve.engine import Engine
    from miniserve.request import Request
    from miniserve.scheduler import Scheduler

    executor = BatchedRealExecutor(
        model, eos_token_id=int(model.config.eos_token_id)
    )
    scheduler = Scheduler(
        max_batch_size=2,
        max_batch_tokens=int(model.config.max_position_embeddings),
        max_model_len=int(model.config.max_position_embeddings),
    )
    engine = Engine(scheduler, executor)
    requests = []
    prompt_ids_by_request: dict[str, list[int]] = {}
    for index, (prompt, limit) in enumerate(
        zip(PROMPTS, LIMITS, strict=True)
    ):
        prompt_token_ids: list[int] = tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]
        prompt_ids_by_request[f"batch-{index}"] = prompt_token_ids
        request = Request(
            request_id=f"batch-{index}",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=limit,
        )
        requests.append(request)
        scheduler.add_request(request)
    engine.run_until_complete()

    return {
        "requests": [
            {
                "request_id": request.request_id,
                "prompt_token_ids": prompt_ids_by_request[
                    request.request_id
                ],
                "generated_token_ids": list(request.generated_token_ids),
                "generated_text": tokenizer.decode(
                    request.generated_token_ids
                ),
                "state": request.state.name,
            }
            for request in requests
        ],
        "engine_steps": engine.step_count,
        "prefill_forwards": executor.prefill_forward_count,
        "decode_forwards": executor.decode_forward_count,
        "decode_batch_sizes": list(executor.decode_batch_sizes),
        "active_caches_after_run": executor.active_cache_count,
        "released_request_ids": list(executor.released_request_ids),
    }


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parent.parent
    recorded = json.loads(
        (root / "configs" / "model.json").read_text(encoding="utf-8")
    )

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

    individual = run_individual(model, tokenizer)
    batched = run_batched(model, tokenizer)

    tokens_match = all(
        solo["generated_token_ids"] == together["generated_token_ids"]
        for solo, together in zip(
            individual, batched["requests"], strict=True
        )
    )
    result: dict[str, Any] = {
        "model_id": recorded["model_id"],
        "revision": recorded["revision"],
        "decoding": recorded["decoding"],
        "dtype": recorded["dtype"],
        "device": recorded["device"],
        "prompts": PROMPTS,
        "max_new_tokens_per_request": LIMITS,
        "individual": individual,
        "batched": batched,
        "all_tokens_match": tokens_match,
    }
    out_path = (
        Path(__file__).resolve().parent / "batched_exit_generate_output.json"
    )
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if tokens_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
