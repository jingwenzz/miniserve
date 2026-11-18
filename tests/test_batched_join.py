"""Tests for a new request joining a batched decode run mid flight.

No real model is loaded here. The scripted model is the one from
the batched executor tests: a request produces the same token
sequence alone or as a row of a batched forward, so a request
that joins after two others are already decoding can be compared
token for token against its individual run, and the two original
requests can be checked for disturbance at the join point. The
per step trace is recorded from the scheduler outputs the engine
returns, which is what proves the new request prefilled in one
step and joined the decode batch in the next, instead of waiting
for the running requests to finish.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_batched_executor import EOS, FakeBatchedModel

from miniserve.batched_executor import BatchedRealExecutor
from miniserve.engine import Engine
from miniserve.real_executor import RealExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import Scheduler


def make_join_engine(
    model: FakeBatchedModel,
) -> tuple[Scheduler, BatchedRealExecutor, Engine]:
    executor = BatchedRealExecutor(model, eos_token_id=EOS)
    scheduler = Scheduler(max_batch_size=3, max_batch_tokens=100)
    return scheduler, executor, Engine(scheduler, executor)


def run_individual(prompt: list[int], max_new_tokens: int) -> list[int]:
    model = FakeBatchedModel()
    executor = RealExecutor(model, eos_token_id=EOS)
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    request = Request("solo", prompt, max_new_tokens=max_new_tokens)
    scheduler.add_request(request)
    engine.run_until_complete()
    return list(request.generated_token_ids)


def test_new_request_prefills_then_joins_the_decode_batch() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_join_engine(model)
    first = Request("request-1", [11, 12, 13], max_new_tokens=5)
    second = Request("request-2", [21, 22, 23], max_new_tokens=5)
    scheduler.add_request(first)
    scheduler.add_request(second)

    trace: list[tuple[list[str], list[str]]] = []

    def step_and_record() -> None:
        output = engine.step()
        trace.append(
            (
                [r.request_id for r in output.prefill_requests],
                [r.request_id for r in output.decode_requests],
            )
        )

    step_and_record()
    step_and_record()
    assert first.generated_token_ids == [14, 15]
    assert second.generated_token_ids == [24, 25]

    # The third request arrives at a fixed step boundary, after
    # the first two are already decoding. This is the replayed
    # arrival: admission happens between steps, exactly where a
    # real arrival would be picked up.
    joined = Request("request-3", [31, 32], max_new_tokens=4)
    scheduler.add_request(joined)

    step_and_record()
    # In its arrival step the new request prefills and produces
    # its first token from prefill, while the running pair keeps
    # decoding in the same step. Prefill and decode are separate
    # forwards; the new request is not in this step's decode list.
    assert trace[-1] == (["request-3"], ["request-1", "request-2"])
    assert joined.generated_token_ids == [33]
    assert first.generated_token_ids == [14, 15, 16]
    assert second.generated_token_ids == [24, 25, 26]
    # The joined cache holds exactly its prompt: the first token
    # has been produced but not yet fed back.
    assert executor._caches["request-3"].get_seq_length() == 2

    step_and_record()
    # From the next step on, the new request is a decode row.
    assert trace[-1] == ([], ["request-1", "request-2", "request-3"])
    # Split back kept each row's own valid length: the original
    # pair are at prompt 3 plus three decode steps, the joined
    # request is at prompt 2 plus one decode step.
    assert executor._caches["request-1"].get_seq_length() == 6
    assert executor._caches["request-2"].get_seq_length() == 6
    assert executor._caches["request-3"].get_seq_length() == 3

    while scheduler.has_pending_requests():
        step_and_record()

    assert first.state is RequestState.FINISHED
    assert second.state is RequestState.FINISHED
    assert joined.state is RequestState.FINISHED
    # No output crossed rows at or after the join point: every
    # sequence equals its individual run token for token.
    assert first.generated_token_ids == run_individual([11, 12, 13], 5)
    assert second.generated_token_ids == run_individual([21, 22, 23], 5)
    assert joined.generated_token_ids == run_individual([31, 32], 4)
    assert first.generated_token_ids == [14, 15, 16, 17, 18]
    assert second.generated_token_ids == [24, 25, 26, 27, 28]
    assert joined.generated_token_ids == [33, 34, 35, 36]
    # The full trace shows the batch really changing shape: the
    # decode batch grows from two rows to three when the new
    # request joins, then shrinks to one for its last token after
    # the original pair finish.
    assert trace == [
        (["request-1", "request-2"], []),
        ([], ["request-1", "request-2"]),
        (["request-3"], ["request-1", "request-2"]),
        ([], ["request-1", "request-2", "request-3"]),
        ([], ["request-1", "request-2", "request-3"]),
        ([], ["request-3"]),
    ]
    assert executor.decode_batch_sizes == [2, 2, 3, 3, 1]
    assert executor.prefill_forward_count == 3
    assert executor.active_cache_count == 0
    assert set(executor.released_request_ids) == {
        "request-1",
        "request-2",
        "request-3",
    }


def test_joined_request_does_not_wait_for_the_running_pair() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_join_engine(model)
    first = Request("request-1", [11, 12, 13], max_new_tokens=5)
    second = Request("request-2", [21, 22, 23], max_new_tokens=5)
    scheduler.add_request(first)
    scheduler.add_request(second)
    engine.step()
    engine.step()

    joined = Request("request-3", [31, 32], max_new_tokens=4)
    scheduler.add_request(joined)
    engine.step()

    # The joined request already holds a token while the original
    # pair are only three tokens into their five: it was admitted
    # and prefilled mid run, not queued behind their completion.
    assert joined.generated_token_ids == [33]
    assert not first.is_finished
    assert not second.is_finished
    assert executor.prefill_forward_count == 3
