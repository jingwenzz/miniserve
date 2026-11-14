"""Stop, failure, and cache release checks for the real executor.

These tests drive RealExecutor with a scripted fake model whose
forward can be told to raise on a chosen call. That injection
point is deliberately small and deterministic: prefill failure on
the first forward, decode failure after a successful prefill.
Every scenario asserts the same cleanup invariant from a
different terminal path: when a request finishes, is cancelled,
or fails, the executor holds no cache for it afterwards, and the
request is recorded as released exactly once.

Repeated sequential runs share one executor, so a cache left
behind by an earlier request would show up as a nonzero active
count before the next request starts.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from miniserve.engine import Engine
from miniserve.metrics import count_by_finish_reason, summarize_request
from miniserve.real_executor import RealExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import Scheduler

VOCAB = 32
EOS = 0


class FakeCache:
    def __init__(self) -> None:
        self.length = 0

    def get_seq_length(self) -> int:
        return self.length


class FlakyModel:
    """A scripted model that raises on one chosen forward call.

    Calls are counted from zero across prefill and decode, in the
    order the executor makes them. ``fail_on_call`` of None means
    the model never fails.
    """

    def __init__(
        self, script: list[int], fail_on_call: int | None = None
    ) -> None:
        self.script = script
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
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


def make_engine(
    model: FlakyModel,
) -> tuple[Scheduler, RealExecutor, Engine]:
    executor = RealExecutor(model, eos_token_id=EOS)
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    return scheduler, executor, Engine(scheduler, executor)


def test_repeated_sequential_runs_return_active_cache_to_zero() -> None:
    # Each fresh cache is accessed at lengths 2 (prefill) and 3
    # (the one decode step), so every request produces 7 then 8.
    model = FlakyModel(script=[99, 99, 7, 8])
    scheduler, executor, engine = make_engine(model)

    for index in range(5):
        request = Request(
            f"request-{index}", [11, 12], max_new_tokens=2
        )
        scheduler.add_request(request)
        assert executor.active_cache_count == 0
        engine.run_until_complete()
        assert request.state is RequestState.FINISHED
        assert executor.active_cache_count == 0

    assert executor.released_request_ids == [
        f"request-{index}" for index in range(5)
    ]
    assert executor.prefill_forward_count == 5
    assert executor.decode_forward_count == 5


def test_request_fail_records_reason_and_is_terminal() -> None:
    request = Request("request-1", [11], max_new_tokens=3)
    request.start_prefill()
    request.fail("prefill forward failed: RuntimeError: boom")

    assert request.state is RequestState.FAILED
    assert request.is_finished
    assert request.finish_reason == "failed"
    assert request.failure_reason == (
        "prefill forward failed: RuntimeError: boom"
    )
    assert request.finished_time is not None

    with pytest.raises(RuntimeError):
        request.fail("a second failure must not overwrite the first")
    with pytest.raises(RuntimeError):
        request.cancel()


def test_prefill_failure_fails_request_and_releases() -> None:
    model = FlakyModel(script=[99] * 8, fail_on_call=0)
    scheduler, executor, engine = make_engine(model)
    request = Request("request-1", [11, 12], max_new_tokens=3)
    scheduler.add_request(request)

    with pytest.raises(RuntimeError, match="injected model failure"):
        engine.run_until_complete()

    assert request.state is RequestState.FAILED
    assert request.failure_reason is not None
    assert "prefill forward failed" in request.failure_reason
    assert "injected model failure" in request.failure_reason
    assert request.generated_token_ids == []
    assert executor.active_cache_count == 0
    assert executor.released_request_ids == ["request-1"]
    assert not scheduler.has_pending_requests()

    metrics = summarize_request(request)
    assert metrics.finish_reason == "failed"
    assert metrics.failure_reason == request.failure_reason
    assert metrics.latency is not None
    assert count_by_finish_reason([request]) == {
        "finished": 0,
        "cancelled": 0,
        "failed": 1,
        "unfinished": 0,
    }


def test_decode_failure_keeps_produced_tokens_and_releases() -> None:
    # Call 0 is the prefill forward and produces token 7. Call 1
    # is the first decode forward and is the injected failure.
    model = FlakyModel(script=[99, 99, 7, 8, 9], fail_on_call=1)
    scheduler, executor, engine = make_engine(model)
    request = Request("request-1", [11, 12], max_new_tokens=3)
    scheduler.add_request(request)

    with pytest.raises(RuntimeError, match="injected model failure"):
        engine.run_until_complete()

    assert request.state is RequestState.FAILED
    assert request.failure_reason is not None
    assert "decode forward failed" in request.failure_reason
    assert request.generated_token_ids == [7]
    assert executor.active_cache_count == 0
    assert executor.released_request_ids == ["request-1"]
    assert not scheduler.has_pending_requests()


def test_failure_of_one_request_leaves_waiting_request_untouched() -> None:
    model = FlakyModel(script=[99] * 8, fail_on_call=0)
    scheduler, executor, engine = make_engine(model)
    failing = Request("request-1", [11, 12], max_new_tokens=2)
    waiting = Request("request-2", [11, 12], max_new_tokens=2)
    scheduler.add_request(failing)
    scheduler.add_request(waiting)

    with pytest.raises(RuntimeError, match="injected model failure"):
        engine.run_until_complete()

    assert failing.state is RequestState.FAILED
    assert waiting.state is RequestState.WAITING
    assert waiting.failure_reason is None
    assert executor.active_cache_count == 0


def test_eos_and_cancel_paths_still_release_in_one_suite() -> None:
    # EOS from prefill finishes with no decode forward.
    eos_model = FlakyModel(script=[99, 99, EOS])
    scheduler, executor, engine = make_engine(eos_model)
    eos_request = Request("eos-1", [11, 12], max_new_tokens=8)
    scheduler.add_request(eos_request)
    engine.run_until_complete()
    assert eos_request.state is RequestState.FINISHED
    assert executor.active_cache_count == 0

    # Cancelling a running request releases its cache as well.
    cancel_model = FlakyModel(script=[99, 99, 99, 7, 8, 9])
    scheduler2, executor2, engine2 = make_engine(cancel_model)
    cancel_request = Request("cancel-1", [11, 12, 13], max_new_tokens=3)
    scheduler2.add_request(cancel_request)
    engine2.step()
    assert executor2.active_cache_count == 1
    assert scheduler2.cancel_request("cancel-1") is True
    engine2.run_until_complete()
    assert cancel_request.state is RequestState.CANCELLED
    assert executor2.active_cache_count == 0
    assert executor2.released_request_ids == ["cancel-1"]
