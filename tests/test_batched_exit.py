"""Tests for requests leaving a batched decode run early.

No real model is loaded here. The scripted model is the one from
the batched executor tests: a request produces the same token
sequence alone or as a row of a batched forward, so a request
that keeps decoding after a batch mate leaves can be compared
token for token against its individual run. That comparison is
what proves the row to request mapping survived the exit, that
the leaving request was never selected again, and that its cache
was released while the remaining request kept its own.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_batched_executor import EOS, FakeBatchedModel, make_engine

from miniserve.engine import Engine
from miniserve.real_executor import RealExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import Scheduler


def run_individual(prompt: list[int], max_new_tokens: int) -> list[int]:
    model = FakeBatchedModel()
    executor = RealExecutor(model, eos_token_id=EOS)
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    request = Request("solo", prompt, max_new_tokens=max_new_tokens)
    scheduler.add_request(request)
    engine.run_until_complete()
    return list(request.generated_token_ids)


def test_short_request_exits_and_long_request_continues_correctly() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    short = Request("request-1", [11, 12], max_new_tokens=2)
    longer = Request("request-2", [21, 22, 23, 24, 25], max_new_tokens=5)
    scheduler.add_request(short)
    scheduler.add_request(longer)

    engine.run_until_complete()

    assert short.state is RequestState.FINISHED
    assert longer.state is RequestState.FINISHED
    # The short request is the first row, so its exit is the case
    # where a careless row mapping would hand its later tokens to
    # the wrong request. Both sequences match the individual runs.
    assert short.generated_token_ids == run_individual([11, 12], 2)
    assert longer.generated_token_ids == run_individual(
        [21, 22, 23, 24, 25], 5
    )
    assert short.generated_token_ids == [13, 14]
    assert longer.generated_token_ids == [26, 27, 28, 29, 30]
    # One shared decode step, then the long request decodes alone:
    # the batch really shrank instead of padding a finished row.
    assert executor.decode_batch_sizes == [2, 1, 1, 1]
    assert executor.active_cache_count == 0
    assert set(executor.released_request_ids) == {
        "request-1",
        "request-2",
    }


def test_one_token_request_never_enters_the_decode_batch() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    single = Request("request-1", [11, 12, 13], max_new_tokens=1)
    longer = Request("request-2", [21, 22, 23], max_new_tokens=3)
    scheduler.add_request(single)
    scheduler.add_request(longer)

    engine.run_until_complete()

    assert single.state is RequestState.FINISHED
    assert single.generated_token_ids == [14]
    assert longer.generated_token_ids == run_individual(
        [21, 22, 23], 3
    )
    # The one token request finished inside prefill, so every
    # decode forward served the remaining request alone.
    assert executor.decode_batch_sizes == [1, 1]
    assert executor.active_cache_count == 0
    assert set(executor.released_request_ids) == {
        "request-1",
        "request-2",
    }


def test_cancelling_a_running_request_leaves_the_other_untouched() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    cancelled = Request("request-1", [11, 12, 13], max_new_tokens=4)
    survivor = Request("request-2", [21, 22, 23], max_new_tokens=4)
    scheduler.add_request(cancelled)
    scheduler.add_request(survivor)

    engine.step()
    engine.step()
    assert cancelled.generated_token_ids == [14, 15]
    assert survivor.generated_token_ids == [24, 25]

    assert scheduler.cancel_request("request-1") is True
    engine.run_until_complete()

    assert cancelled.state is RequestState.CANCELLED
    # No token was produced for the cancelled request after the
    # cancellation point, and the survivor finished with exactly
    # its individual sequence.
    assert cancelled.generated_token_ids == [14, 15]
    assert survivor.state is RequestState.FINISHED
    assert survivor.generated_token_ids == run_individual(
        [21, 22, 23], 4
    )
    assert executor.decode_batch_sizes == [2, 1, 1]
    assert executor.active_cache_count == 0
    assert set(executor.released_request_ids) == {
        "request-1",
        "request-2",
    }
