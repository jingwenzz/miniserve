"""Tests for the real executor, driven by a scripted fake model.

No real model is loaded here. The fake model follows the same
call surface the real transformer exposes to the executor: a
prefill call without past_key_values creates a fresh cache, a
decode call reuses the cache it is handed, every call grows the
cache by the number of input tokens, and the logits argmax is a
fixed script indexed by the total context length seen so far.
That is enough to pin down the executor bookkeeping: prefill
produces the first token, decode feeds exactly the last generated
token, EOS and the output limit finish the request, a limit of
one needs no decode forward, and release drops the cache.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from miniserve.engine import Engine
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


class FakeModel:
    def __init__(self, script: list[int]) -> None:
        self.script = script
        self.input_token_ids: list[list[int]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        input_ids = kwargs["input_ids"]
        self.input_token_ids.append(input_ids[0].tolist())
        cache = kwargs.get("past_key_values") or FakeCache()
        cache.length += int(input_ids.shape[1])
        next_token = self.script[cache.length]
        logits = torch.full((1, int(input_ids.shape[1]), VOCAB), -10.0)
        logits[0, -1, next_token] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


def make_engine(
    model: FakeModel, max_batch_tokens: int = 100
) -> tuple[Scheduler, RealExecutor, Engine]:
    executor = RealExecutor(model, eos_token_id=EOS)
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=max_batch_tokens)
    return scheduler, executor, Engine(scheduler, executor)


def test_full_run_finishes_by_limit_and_releases_cache() -> None:
    model = FakeModel(script=[99, 99, 99, 7, 8, 9, 10])
    scheduler, executor, engine = make_engine(model)
    request = Request("request-1", [11, 12, 13], max_new_tokens=3)
    scheduler.add_request(request)

    engine.run_until_complete()

    assert request.state is RequestState.FINISHED
    assert request.generated_token_ids == [7, 8, 9]
    assert executor.prefill_forward_count == 1
    assert executor.decode_forward_count == 2
    assert engine.step_count == 3
    assert model.input_token_ids == [[11, 12, 13], [7], [8]]
    assert executor.active_cache_count == 0
    assert executor.released_request_ids == ["request-1"]


def test_first_token_arrives_with_the_prefill_step() -> None:
    model = FakeModel(script=[99, 99, 99, 7, 8, 9])
    scheduler, executor, engine = make_engine(model)
    request = Request("request-1", [11, 12, 13], max_new_tokens=3)
    scheduler.add_request(request)

    engine.step()

    assert request.state is RequestState.DECODING
    assert request.generated_token_ids == [7]
    assert request.first_token_time is not None
    assert request.first_token_time == request.last_token_time
    assert executor.active_cache_count == 1


def test_max_new_tokens_one_needs_no_decode_forward() -> None:
    model = FakeModel(script=[99, 99, 21])
    scheduler, executor, engine = make_engine(model)
    request = Request("request-1", [11, 12], max_new_tokens=1)
    scheduler.add_request(request)

    engine.run_until_complete()

    assert request.state is RequestState.FINISHED
    assert request.generated_token_ids == [21]
    assert executor.prefill_forward_count == 1
    assert executor.decode_forward_count == 0
    assert engine.step_count == 1
    assert executor.active_cache_count == 0


def test_eos_from_prefill_finishes_without_decode() -> None:
    model = FakeModel(script=[99, 99, EOS])
    scheduler, executor, engine = make_engine(model)
    request = Request("request-1", [11, 12], max_new_tokens=8)
    scheduler.add_request(request)

    engine.run_until_complete()

    assert request.state is RequestState.FINISHED
    assert request.generated_token_ids == [EOS]
    assert executor.decode_forward_count == 0
    assert executor.active_cache_count == 0


def test_eos_from_decode_stops_and_is_never_fed_back() -> None:
    model = FakeModel(script=[99, 99, 99, 5, EOS, 6])
    scheduler, executor, engine = make_engine(model)
    request = Request("request-1", [11, 12, 13], max_new_tokens=8)
    scheduler.add_request(request)

    engine.run_until_complete()

    assert request.state is RequestState.FINISHED
    assert request.generated_token_ids == [5, EOS]
    assert model.input_token_ids == [[11, 12, 13], [5]]
    assert executor.active_cache_count == 0


def test_cancelled_running_request_releases_its_cache() -> None:
    model = FakeModel(script=[99, 99, 99, 7, 8, 9])
    scheduler, executor, engine = make_engine(model)
    request = Request("request-1", [11, 12, 13], max_new_tokens=3)
    scheduler.add_request(request)
    engine.step()
    assert executor.active_cache_count == 1

    assert scheduler.cancel_request("request-1") is True
    engine.run_until_complete()

    assert request.state is RequestState.CANCELLED
    assert executor.active_cache_count == 0
    assert executor.released_request_ids == ["request-1"]


def test_release_is_idempotent() -> None:
    model = FakeModel(script=[99, 99, 21])
    scheduler, executor, engine = make_engine(model)
    request = Request("request-1", [11, 12], max_new_tokens=1)
    scheduler.add_request(request)
    engine.run_until_complete()

    executor.release(request)
    executor.release(request)

    assert executor.released_request_ids == ["request-1"]
    assert executor.active_cache_count == 0


def test_sequential_requests_get_fresh_caches() -> None:
    model = FakeModel(script=[99, 99, 21, 7, 8, 9])
    scheduler, executor, engine = make_engine(model)
    first = Request("request-1", [11, 12, 13], max_new_tokens=2)
    scheduler.add_request(first)
    engine.run_until_complete()

    second = Request("request-2", [11, 12], max_new_tokens=1)
    scheduler.add_request(second)
    engine.run_until_complete()

    assert first.generated_token_ids == [7, 8]
    assert second.generated_token_ids == [21]
    assert executor.active_cache_count == 0
    assert executor.released_request_ids == ["request-1", "request-2"]
