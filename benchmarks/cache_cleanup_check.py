"""Diagnostic for stop, failure, and cache release in RealExecutor.

This script drives the real executor with a scripted fake model,
not the locked transformer. The fake model follows the call
surface the executor uses, and one variant raises on a chosen
forward call, which is the safe, controllable injection point for
a model failure. No real model is loaded and no timing here is a
performance result; the script exists to check one invariant from
every terminal path:

    after a request finishes, stops on EOS, is cancelled, or
    fails, the executor holds no cache for it and has recorded
    exactly one release for it.

Scenarios: three sequential normal requests on one shared
executor, an EOS stop from prefill, a running request cancelled
between steps, a prefill failure, and a decode failure. The full
record is saved to benchmarks/cache_cleanup_output.json and the
exit code is 0 only when every scenario ends terminal with an
active cache count of zero.

A note on memory wording: the active cache count counts the per
request cache objects this executor still references. It says
nothing about memory a framework allocator may keep reserved for
reuse after those references are dropped, and this CPU script
makes no GPU memory claim at all. Releasing a request means
dropping its reference here, never calling an allocator emptying
routine per request.

Usage, from the repository root:

    python benchmarks/cache_cleanup_check.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

VOCAB = 32
EOS = 0


class FakeCache:
    def __init__(self) -> None:
        self.length = 0

    def get_seq_length(self) -> int:
        return self.length


class ScriptedModel:
    def __init__(
        self, script: list[int], fail_on_call: int | None = None
    ) -> None:
        self.script = script
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        import torch

        call_index = self.calls
        self.calls += 1
        if self.fail_on_call is not None and call_index == self.fail_on_call:
            raise RuntimeError("injected model failure")
        input_ids = kwargs["input_ids"]
        cache = kwargs.get("past_key_values") or FakeCache()
        cache.length += int(input_ids.shape[1])
        next_token = self.script[cache.length]
        logits = torch.full((1, int(input_ids.shape[1]), VOCAB), -10.0)
        logits[0, -1, next_token] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _make_stack(model: ScriptedModel) -> tuple[Any, Any, Any]:
    from miniserve.engine import Engine
    from miniserve.real_executor import RealExecutor
    from miniserve.scheduler import Scheduler

    executor = RealExecutor(model, eos_token_id=EOS)
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    return scheduler, executor, Engine(scheduler, executor)


def _record(
    name: str, request: Any, executor: Any, error: str | None
) -> dict[str, Any]:
    return {
        "scenario": name,
        "request_id": request.request_id,
        "state": request.state.name,
        "finish_reason": request.finish_reason,
        "failure_reason": request.failure_reason,
        "generated_token_ids": list(request.generated_token_ids),
        "active_caches_after": executor.active_cache_count,
        "released_request_ids": list(executor.released_request_ids),
        "error_propagated": error,
        "clean": (
            request.is_finished
            and executor.active_cache_count == 0
            and executor.released_request_ids.count(request.request_id) == 1
        ),
    }


def main() -> int:
    from miniserve.request import Request

    records: list[dict[str, Any]] = []

    # Sequential normal requests on one shared executor.
    scheduler, executor, engine = _make_stack(
        ScriptedModel(script=[99, 99, 7, 8])
    )
    for index in range(3):
        request = Request(f"normal-{index}", [11, 12], max_new_tokens=2)
        scheduler.add_request(request)
        engine.run_until_complete()
        record = _record("normal_sequential", request, executor, None)
        record["released_request_ids"] = list(executor.released_request_ids)
        record["clean"] = bool(
            request.is_finished and executor.active_cache_count == 0
        )
        records.append(record)

    # EOS produced by prefill.
    scheduler, executor, engine = _make_stack(
        ScriptedModel(script=[99, 99, EOS])
    )
    request = Request("eos-1", [11, 12], max_new_tokens=8)
    scheduler.add_request(request)
    engine.run_until_complete()
    records.append(_record("eos_from_prefill", request, executor, None))

    # Cancellation of a running request between steps.
    scheduler, executor, engine = _make_stack(
        ScriptedModel(script=[99, 99, 99, 7, 8, 9])
    )
    request = Request("cancel-1", [11, 12, 13], max_new_tokens=3)
    scheduler.add_request(request)
    engine.step()
    active_before_cancel = executor.active_cache_count
    scheduler.cancel_request("cancel-1")
    engine.run_until_complete()
    record = _record("cancel_running", request, executor, None)
    record["active_caches_before_cancel"] = active_before_cancel
    records.append(record)

    # Injected failure in prefill and in decode.
    for name, fail_on_call, script in [
        ("prefill_failure", 0, [99] * 8),
        ("decode_failure", 1, [99, 99, 7, 8, 9]),
    ]:
        scheduler, executor, engine = _make_stack(
            ScriptedModel(script=script, fail_on_call=fail_on_call)
        )
        request = Request(f"{name}-1", [11, 12], max_new_tokens=3)
        scheduler.add_request(request)
        error: str | None = None
        try:
            engine.run_until_complete()
        except RuntimeError as exc:
            error = str(exc)
        records.append(_record(name, request, executor, error))

    result = {
        "kind": "scripted_model_diagnostic",
        "real_model_loaded": False,
        "all_clean": all(record["clean"] for record in records),
        "scenarios": records,
    }
    out_path = Path(__file__).resolve().parent / "cache_cleanup_output.json"
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for record in records:
        print(
            f"{record['scenario']} {record['request_id']}: "
            f"state={record['state']} active={record['active_caches_after']} "
            f"clean={record['clean']}"
        )
    print(f"all_clean={result['all_clean']}")
    return 0 if result["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
