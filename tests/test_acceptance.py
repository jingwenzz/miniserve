"""Fourth week acceptance suite for the batched runtime.

This module is the fast, scripted counterpart of
benchmarks/fourth_week_acceptance.py. It re-checks, in one
place, every behaviour the batched decode work claimed one
scenario at a time: equal lengths, different lengths, exit,
join, cancellation, and a shared decode failure. The scripted
model is the one from the batched executor tests, where a
request produces the same token sequence alone or as a batch
row, so every batched sequence here is compared token for token
against an individual run through the single request executor.
The real model pass over the same six scenarios is the recorded
benchmark above; this suite exists so the acceptance claims
also run in the ordinary test command without a model download.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_batched_executor import EOS, FakeBatchedModel, make_engine

from miniserve.batched_executor import BatchedRealExecutor
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


def test_acceptance_equal_length_rows_share_decode_forwards() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    first = Request("request-1", [11, 12, 13], max_new_tokens=3)
    second = Request("request-2", [21, 22, 23], max_new_tokens=3)
    scheduler.add_request(first)
    scheduler.add_request(second)

    engine.run_until_complete()

    assert first.generated_token_ids == run_individual([11, 12, 13], 3)
    assert second.generated_token_ids == run_individual([21, 22, 23], 3)
    assert executor.decode_batch_sizes == [2, 2]
    assert executor.active_cache_count == 0


def test_acceptance_different_lengths_match_individual_runs() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    short = Request("request-1", [11, 12], max_new_tokens=3)
    longer = Request("request-2", [21, 22, 23, 24, 25], max_new_tokens=3)
    scheduler.add_request(short)
    scheduler.add_request(longer)

    engine.run_until_complete()

    assert short.generated_token_ids == run_individual([11, 12], 3)
    assert longer.generated_token_ids == run_individual(
        [21, 22, 23, 24, 25], 3
    )
    assert executor.decode_batch_sizes == [2, 2]
    assert executor.active_cache_count == 0


def test_acceptance_exit_shrinks_the_batch() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    short = Request("request-1", [11, 12], max_new_tokens=2)
    longer = Request("request-2", [21, 22, 23, 24, 25], max_new_tokens=5)
    scheduler.add_request(short)
    scheduler.add_request(longer)

    engine.run_until_complete()

    assert short.state is RequestState.FINISHED
    assert longer.generated_token_ids == run_individual(
        [21, 22, 23, 24, 25], 5
    )
    assert executor.decode_batch_sizes == [2, 1, 1, 1]
    assert executor.active_cache_count == 0
    assert set(executor.released_request_ids) == {
        "request-1",
        "request-2",
    }


def test_acceptance_join_grows_the_batch() -> None:
    model = FakeBatchedModel()
    executor = BatchedRealExecutor(model, eos_token_id=EOS)
    scheduler = Scheduler(max_batch_size=3, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    first = Request("request-1", [11, 12, 13], max_new_tokens=5)
    second = Request("request-2", [21, 22, 23], max_new_tokens=5)
    scheduler.add_request(first)
    scheduler.add_request(second)
    engine.step()
    engine.step()

    joined = Request("request-3", [31, 32], max_new_tokens=4)
    scheduler.add_request(joined)
    engine.run_until_complete()

    assert first.generated_token_ids == run_individual([11, 12, 13], 5)
    assert second.generated_token_ids == run_individual([21, 22, 23], 5)
    assert joined.generated_token_ids == run_individual([31, 32], 4)
    assert executor.decode_batch_sizes == [2, 2, 3, 3, 1]
    assert executor.active_cache_count == 0


def test_acceptance_cancelling_a_running_request_freezes_it() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    cancelled = Request("request-1", [11, 12, 13], max_new_tokens=4)
    survivor = Request("request-2", [21, 22, 23], max_new_tokens=4)
    scheduler.add_request(cancelled)
    scheduler.add_request(survivor)
    engine.step()
    engine.step()

    assert scheduler.cancel_request("request-1") is True
    engine.run_until_complete()

    assert cancelled.state is RequestState.CANCELLED
    assert cancelled.generated_token_ids == [14, 15]
    assert survivor.generated_token_ids == run_individual(
        [21, 22, 23], 4
    )
    assert executor.active_cache_count == 0
    assert set(executor.released_request_ids) == {
        "request-1",
        "request-2",
    }


def test_acceptance_shared_failure_fails_all_rows_and_cleans_up() -> None:
    model = FakeBatchedModel(fail_batched_decode=True)
    scheduler, executor, engine = make_engine(model)
    first = Request("request-1", [11, 12, 13], max_new_tokens=3)
    second = Request("request-2", [21, 22, 23], max_new_tokens=3)
    scheduler.add_request(first)
    scheduler.add_request(second)
    engine.step()

    with pytest.raises(RuntimeError, match="scripted batched failure"):
        engine.step()

    assert first.state is RequestState.FAILED
    assert second.state is RequestState.FAILED
    assert first.failure_reason is not None
    assert "scripted batched failure" in first.failure_reason
    # The engine cleaned up the failed step: both requests left
    # their queues and both caches were released even though the
    # step raised.
    assert not scheduler.has_pending_requests()
    assert executor.active_cache_count == 0
    assert set(executor.released_request_ids) == {
        "request-1",
        "request-2",
    }
