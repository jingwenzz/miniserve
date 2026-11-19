"""Run the fourth week acceptance pass over the batched runtime.

This script is the single checkpoint that re-verifies, on the
real locked model, every behaviour the batched decode work
claimed one scenario at a time:

- equal length rows sharing one decode forward per step,
- different length rows decoding together through left padding,
- a short request leaving the batch while a longer one continues,
- a new request prefilled mid run and joining the decode batch,
- a running request cancelled mid batch while the other finishes,
- a batched decode failure injected at the model boundary, which
  must fail every row of the shared forward, propagate to the
  caller, and still release every cache.

Each scenario also runs every prompt one at a time through the
single request executor with the same limit, and the record
compares the generated token ids path against path. The join
scenario records the full per step trace of prefill ids, decode
ids, and decode batch sizes, which is the saved explainable
scheduling trace for this checkpoint. The record is saved to
benchmarks/fourth_week_acceptance_output.json and the script
exits nonzero if any check fails, so a green run is the
acceptance evidence and a red run names the failing scenario.

Usage, from the repository root, with the model extra installed:

    python benchmarks/fourth_week_acceptance.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FRANCE = "The capital of France is"
ITALY = "The capital of Italy is"
SPAIN = "The capital of Spain is"
LONG = "The capital of France is in Europe"
SHORT = "Hello world"


def solo_tokens(
    model: Any, tokenizer: Any, prompt: str, limit: int
) -> list[int]:
    """Run one prompt alone and return its generated token ids."""
    from miniserve.engine import Engine
    from miniserve.real_executor import RealExecutor
    from miniserve.request import Request
    from miniserve.scheduler import Scheduler

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
        request_id="solo",
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=limit,
    )
    scheduler.add_request(request)
    engine.run_until_complete()
    return list(request.generated_token_ids)


class BatchedRun:
    """Drive one batched scenario and record its step trace."""

    def __init__(self, model: Any, max_batch_size: int) -> None:
        from miniserve.batched_executor import BatchedRealExecutor
        from miniserve.engine import Engine
        from miniserve.scheduler import Scheduler

        self.executor = BatchedRealExecutor(
            model, eos_token_id=int(model.config.eos_token_id)
        )
        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            max_batch_tokens=int(model.config.max_position_embeddings),
            max_model_len=int(model.config.max_position_embeddings),
        )
        self.engine = Engine(self.scheduler, self.executor)
        self.requests: dict[str, Any] = {}
        self.trace: list[dict[str, Any]] = []

    def add(self, tokenizer: Any, request_id: str, prompt: str, limit: int) -> Any:
        from miniserve.request import Request

        prompt_token_ids: list[int] = tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]
        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=limit,
        )
        self.requests[request_id] = request
        self.scheduler.add_request(request)
        return request

    def step(self) -> None:
        before = len(self.executor.decode_batch_sizes)
        output = self.engine.step()
        new_sizes = self.executor.decode_batch_sizes[before:]
        self.trace.append(
            {
                "step": len(self.trace) + 1,
                "prefill_request_ids": [
                    r.request_id for r in output.prefill_requests
                ],
                "decode_request_ids": [
                    r.request_id for r in output.decode_requests
                ],
                "decode_batch_size": new_sizes[0] if new_sizes else None,
            }
        )

    def run_to_completion(self) -> None:
        while self.scheduler.has_pending_requests():
            self.step()

    def record(self) -> dict[str, Any]:
        return {
            "requests": [
                {
                    "request_id": request.request_id,
                    "prompt_token_ids": list(request.prompt_token_ids),
                    "max_new_tokens": request.max_new_tokens,
                    "generated_token_ids": list(
                        request.generated_token_ids
                    ),
                    "state": request.state.name,
                    "failure_reason": request.failure_reason,
                }
                for request in self.requests.values()
            ],
            "step_trace": self.trace,
            "engine_steps": self.engine.step_count,
            "prefill_forwards": self.executor.prefill_forward_count,
            "decode_forwards": self.executor.decode_forward_count,
            "decode_batch_sizes": list(self.executor.decode_batch_sizes),
            "active_caches_after_run": self.executor.active_cache_count,
            "released_request_ids": list(
                self.executor.released_request_ids
            ),
        }


def check_record(
    run: BatchedRun,
    expected_tokens: dict[str, list[int]],
    expected_states: dict[str, str],
) -> dict[str, bool]:
    """Evaluate the shared acceptance checks for one scenario."""
    return {
        "tokens_match_individual_runs": all(
            request.generated_token_ids == expected_tokens[request_id]
            for request_id, request in run.requests.items()
        ),
        "states_as_expected": all(
            request.state.name == expected_states[request_id]
            for request_id, request in run.requests.items()
        ),
        "active_caches_cleaned": (
            run.executor.active_cache_count == 0
        ),
        "every_request_released": set(
            run.executor.released_request_ids
        )
        == set(run.requests),
    }


def scenario_equal(model: Any, tokenizer: Any) -> dict[str, Any]:
    expected = {
        "batch-0": solo_tokens(model, tokenizer, FRANCE, 4),
        "batch-1": solo_tokens(model, tokenizer, ITALY, 4),
    }
    run = BatchedRun(model, max_batch_size=2)
    run.add(tokenizer, "batch-0", FRANCE, 4)
    run.add(tokenizer, "batch-1", ITALY, 4)
    run.run_to_completion()
    checks = check_record(
        run, expected, {"batch-0": "FINISHED", "batch-1": "FINISHED"}
    )
    checks["shared_decode_forward_ran"] = any(
        size > 1 for size in run.executor.decode_batch_sizes
    )
    return {"checks": checks, **run.record()}


def scenario_unequal(model: Any, tokenizer: Any) -> dict[str, Any]:
    expected = {
        "batch-0": solo_tokens(model, tokenizer, LONG, 4),
        "batch-1": solo_tokens(model, tokenizer, SHORT, 4),
    }
    run = BatchedRun(model, max_batch_size=2)
    run.add(tokenizer, "batch-0", LONG, 4)
    run.add(tokenizer, "batch-1", SHORT, 4)
    run.run_to_completion()
    checks = check_record(
        run, expected, {"batch-0": "FINISHED", "batch-1": "FINISHED"}
    )
    checks["shared_decode_forward_ran"] = any(
        size > 1 for size in run.executor.decode_batch_sizes
    )
    checks["prompt_lengths_differ"] = (
        run.requests["batch-0"].prompt_length
        != run.requests["batch-1"].prompt_length
    )
    return {"checks": checks, **run.record()}


def scenario_exit(model: Any, tokenizer: Any) -> dict[str, Any]:
    expected = {
        "batch-0": solo_tokens(model, tokenizer, FRANCE, 2),
        "batch-1": solo_tokens(model, tokenizer, ITALY, 5),
    }
    run = BatchedRun(model, max_batch_size=2)
    run.add(tokenizer, "batch-0", FRANCE, 2)
    run.add(tokenizer, "batch-1", ITALY, 5)
    run.run_to_completion()
    checks = check_record(
        run, expected, {"batch-0": "FINISHED", "batch-1": "FINISHED"}
    )
    sizes = run.executor.decode_batch_sizes
    checks["batch_shrank_after_exit"] = (
        len(sizes) > 1 and sizes[0] == 2 and sizes[-1] == 1
    )
    return {"checks": checks, **run.record()}


def scenario_join(model: Any, tokenizer: Any) -> dict[str, Any]:
    expected = {
        "batch-0": solo_tokens(model, tokenizer, FRANCE, 5),
        "batch-1": solo_tokens(model, tokenizer, ITALY, 5),
        "batch-2": solo_tokens(model, tokenizer, SPAIN, 3),
    }
    run = BatchedRun(model, max_batch_size=3)
    run.add(tokenizer, "batch-0", FRANCE, 5)
    run.add(tokenizer, "batch-1", ITALY, 5)
    run.step()
    run.step()
    joined = run.add(tokenizer, "batch-2", SPAIN, 3)
    run.run_to_completion()
    checks = check_record(
        run,
        expected,
        {
            "batch-0": "FINISHED",
            "batch-1": "FINISHED",
            "batch-2": "FINISHED",
        },
    )
    sizes = run.executor.decode_batch_sizes
    checks["batch_grew_at_join"] = max(sizes) == 3 and sizes[0] == 2
    checks["joined_request_finished"] = joined.state.name == "FINISHED"
    return {"checks": checks, **run.record()}


def scenario_cancel(model: Any, tokenizer: Any) -> dict[str, Any]:
    survivor_expected = solo_tokens(model, tokenizer, ITALY, 6)
    run = BatchedRun(model, max_batch_size=2)
    cancelled = run.add(tokenizer, "batch-0", FRANCE, 6)
    survivor = run.add(tokenizer, "batch-1", ITALY, 6)
    run.step()
    run.step()
    frozen_tokens = list(cancelled.generated_token_ids)
    assert run.scheduler.cancel_request("batch-0") is True
    run.run_to_completion()
    checks = {
        "cancelled_tokens_frozen": (
            list(cancelled.generated_token_ids) == frozen_tokens
            and len(frozen_tokens) == 2
        ),
        "cancelled_state": cancelled.state.name == "CANCELLED",
        "survivor_tokens_match_individual_run": (
            list(survivor.generated_token_ids) == survivor_expected
        ),
        "survivor_finished": survivor.state.name == "FINISHED",
        "active_caches_cleaned": run.executor.active_cache_count == 0,
        "every_request_released": set(
            run.executor.released_request_ids
        )
        == {"batch-0", "batch-1"},
    }
    return {"checks": checks, **run.record()}


class FailOnBatchedDecode:
    """Wrap the real model and fail the first batched decode call.

    Prefill calls (batch dimension one, no past cache) pass through
    to the real model untouched. The first call that carries a
    past cache with more than one row raises instead, which is the
    injected failure point: it happens inside the shared batched
    decode forward, after both requests hold real caches.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.config = inner.config

    def __call__(self, **kwargs: Any) -> Any:
        past = kwargs.get("past_key_values")
        input_ids = kwargs.get("input_ids")
        if (
            past is not None
            and input_ids is not None
            and int(input_ids.shape[0]) > 1
        ):
            raise RuntimeError("injected batched decode failure")
        return self.inner(**kwargs)


def scenario_failure(model: Any, tokenizer: Any) -> dict[str, Any]:
    run = BatchedRun(FailOnBatchedDecode(model), max_batch_size=2)
    first = run.add(tokenizer, "batch-0", FRANCE, 4)
    second = run.add(tokenizer, "batch-1", ITALY, 4)
    run.step()
    propagated: str | None = None
    try:
        run.step()
    except RuntimeError as exc:
        propagated = str(exc)
    checks = {
        "exception_propagated_to_caller": (
            propagated == "injected batched decode failure"
        ),
        "both_requests_failed": (
            first.state.name == "FAILED" and second.state.name == "FAILED"
        ),
        "failure_reason_recorded": all(
            request.failure_reason is not None
            and "injected batched decode failure" in request.failure_reason
            for request in (first, second)
        ),
        "produced_tokens_kept": (
            len(first.generated_token_ids) == 1
            and len(second.generated_token_ids) == 1
        ),
        "active_caches_cleaned": run.executor.active_cache_count == 0,
        "every_request_released": set(
            run.executor.released_request_ids
        )
        == {"batch-0", "batch-1"},
    }
    return {"checks": checks, **run.record()}


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

    scenarios = {
        "equal_length": scenario_equal(model, tokenizer),
        "different_lengths": scenario_unequal(model, tokenizer),
        "exit": scenario_exit(model, tokenizer),
        "join": scenario_join(model, tokenizer),
        "cancel": scenario_cancel(model, tokenizer),
        "failure": scenario_failure(model, tokenizer),
    }
    all_passed = all(
        all(scenario["checks"].values()) for scenario in scenarios.values()
    )
    result: dict[str, Any] = {
        "model_id": recorded["model_id"],
        "revision": recorded["revision"],
        "decoding": recorded["decoding"],
        "dtype": recorded["dtype"],
        "device": recorded["device"],
        "scenarios": scenarios,
        "all_checks_passed": all_passed,
    }
    out_path = (
        Path(__file__).resolve().parent
        / "fourth_week_acceptance_output.json"
    )
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, scenario in scenarios.items():
        status = "PASS" if all(scenario["checks"].values()) else "FAIL"
        print(f"{name}: {status} {scenario['checks']}")
    print(f"all_checks_passed: {all_passed}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
