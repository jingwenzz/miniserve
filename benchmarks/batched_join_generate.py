"""Run two prompts, then let a third prompt join mid decode.

This is the dynamic join sample for batched decode. Two prompts
start together through the batched executor. After two engine
steps, while both are still decoding, a third request is
admitted: it prefills in its arrival step (a separate forward
from that step's decode, as the design allows) and joins the
decode batch from the next step on. The record saved to
benchmarks/batched_join_generate_output.json holds the per step
trace of prefill ids, decode ids, and the decode batch size of
each step, which is the evidence that the batch really grew from
two rows to three mid run. Every request is also run one at a
time through the single request executor with the same limit,
and the record compares the generated token ids path against
path, which is what proves no output crossed rows at the join.

Usage, from the repository root, with the model extra installed:

    python benchmarks/batched_join_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

START_PROMPTS = ["The capital of France is", "The capital of Italy is"]
START_LIMITS = [6, 6]
JOIN_PROMPT = "The capital of Spain is"
JOIN_LIMIT = 4
JOIN_AFTER_STEPS = 2


def run_individual(
    model: Any, tokenizer: Any
) -> list[dict[str, Any]]:
    from miniserve.engine import Engine
    from miniserve.real_executor import RealExecutor
    from miniserve.request import Request
    from miniserve.scheduler import Scheduler

    prompts = START_PROMPTS + [JOIN_PROMPT]
    limits = START_LIMITS + [JOIN_LIMIT]
    results = []
    for index, (prompt, limit) in enumerate(
        zip(prompts, limits, strict=True)
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
        max_batch_size=3,
        max_batch_tokens=int(model.config.max_position_embeddings),
        max_model_len=int(model.config.max_position_embeddings),
    )
    engine = Engine(scheduler, executor)
    requests: list[Request] = []
    prompt_ids_by_request: dict[str, list[int]] = {}
    limits_by_request: dict[str, int] = {}

    def add_request(index: int, prompt: str, limit: int) -> Request:
        prompt_token_ids: list[int] = tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]
        request_id = f"batch-{index}"
        prompt_ids_by_request[request_id] = prompt_token_ids
        limits_by_request[request_id] = limit
        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=limit,
        )
        requests.append(request)
        scheduler.add_request(request)
        return request

    trace: list[dict[str, Any]] = []

    def step_and_record() -> None:
        decode_forwards_before = len(executor.decode_batch_sizes)
        output = engine.step()
        new_sizes = executor.decode_batch_sizes[decode_forwards_before:]
        trace.append(
            {
                "step": len(trace) + 1,
                "prefill_request_ids": [
                    r.request_id for r in output.prefill_requests
                ],
                "decode_request_ids": [
                    r.request_id for r in output.decode_requests
                ],
                "decode_batch_size": new_sizes[0] if new_sizes else None,
            }
        )

    for index, (prompt, limit) in enumerate(
        zip(START_PROMPTS, START_LIMITS, strict=True)
    ):
        add_request(index, prompt, limit)
    for _ in range(JOIN_AFTER_STEPS):
        step_and_record()
    add_request(len(START_PROMPTS), JOIN_PROMPT, JOIN_LIMIT)
    while scheduler.has_pending_requests():
        step_and_record()

    return {
        "requests": [
            {
                "request_id": request.request_id,
                "prompt_token_ids": prompt_ids_by_request[
                    request.request_id
                ],
                "max_new_tokens": limits_by_request[request.request_id],
                "generated_token_ids": list(request.generated_token_ids),
                "generated_text": tokenizer.decode(
                    request.generated_token_ids
                ),
                "state": request.state.name,
            }
            for request in requests
        ],
        "step_trace": trace,
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
        "start_prompts": START_PROMPTS,
        "start_max_new_tokens": START_LIMITS,
        "join_prompt": JOIN_PROMPT,
        "join_max_new_tokens": JOIN_LIMIT,
        "join_after_steps": JOIN_AFTER_STEPS,
        "individual": individual,
        "batched": batched,
        "all_tokens_match": tokens_match,
    }
    out_path = (
        Path(__file__).resolve().parent / "batched_join_generate_output.json"
    )
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if tokens_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
