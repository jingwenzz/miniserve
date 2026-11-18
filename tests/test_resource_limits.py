"""Tests for the scheduler resource limits on resident requests.

These tests use the fake executor, because the limits live in
the scheduler and need no model: a request reserves
prompt_length + max_new_tokens of cache capacity from the step
it is prefilled until it reaches a terminal state, and the
scheduler admits, waits, or rejects based on that reservation.
The cases pin the three different outcomes a request can meet:
rejected at admission when its reservation alone exceeds the
capacity, kept waiting while other active requests hold the
capacity, and activated once a finish or a cancellation returns
a reservation. They also pin the separate active request count
limit and the rule that a waiting request reserves nothing,
because it holds no cache yet.
"""

from __future__ import annotations

import pytest

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import (
    RequestRejectedError,
    Scheduler,
    create_scheduler,
)


def make_engine(
    scheduler: Scheduler,
) -> tuple[FakeExecutor, Engine]:
    executor = FakeExecutor()
    return executor, Engine(scheduler, executor)


def test_reservation_above_capacity_alone_is_rejected() -> None:
    scheduler = Scheduler(
        max_batch_size=4, max_batch_tokens=100, max_reserved_tokens=10
    )
    too_big = Request("request-1", [1] * 8, max_new_tokens=3)

    with pytest.raises(RequestRejectedError, match="cache capacity"):
        scheduler.add_request(too_big)

    # The rejection leaves no queue entry and does not burn the
    # id: a corrected request whose reservation fits exactly can
    # reuse it.
    assert not scheduler.has_pending_requests()
    corrected = Request("request-1", [1] * 8, max_new_tokens=2)
    scheduler.add_request(corrected)
    assert scheduler.has_pending_requests()


def test_reservation_exactly_at_capacity_runs() -> None:
    scheduler = Scheduler(
        max_batch_size=4, max_batch_tokens=100, max_reserved_tokens=10
    )
    _, engine = make_engine(scheduler)
    request = Request("request-1", [1] * 7, max_new_tokens=3)
    scheduler.add_request(request)

    engine.run_until_complete()

    assert request.state is RequestState.FINISHED
    assert request.output_length == 3
    assert scheduler.reserved_tokens == 0
    assert scheduler.active_request_count == 0


def test_waiting_request_reserves_nothing_yet() -> None:
    scheduler = Scheduler(
        max_batch_size=4, max_batch_tokens=100, max_reserved_tokens=12
    )
    first = Request("request-1", [1] * 5, max_new_tokens=3)
    second = Request("request-2", [1] * 3, max_new_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)

    # Both are admitted even though their reservations sum to 13
    # against a capacity of 12, because admission only rejects a
    # request that can never fit on its own. The capacity check
    # bites at activation time, not at admission time.
    assert scheduler.reserved_tokens == 0
    assert scheduler.active_request_count == 0


def test_full_capacity_keeps_head_waiting_until_a_finish_returns_it() -> None:
    scheduler = Scheduler(
        max_batch_size=4, max_batch_tokens=100, max_reserved_tokens=12
    )
    _, engine = make_engine(scheduler)
    # Reservation 8.
    first = Request("request-1", [1] * 5, max_new_tokens=3)
    # Reservation 5: 8 + 5 exceeds the capacity of 12.
    second = Request("request-2", [1] * 3, max_new_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)

    engine.step()
    assert first.state is RequestState.DECODING
    assert second.state is RequestState.WAITING
    assert scheduler.reserved_tokens == 8
    assert scheduler.active_request_count == 1

    # The first request takes three decode steps with the fake
    # executor. The second stays waiting the whole time, with no
    # token, because the capacity is still held.
    engine.step()
    engine.step()
    engine.step()
    assert first.state is RequestState.FINISHED
    assert second.state is RequestState.WAITING
    assert second.output_length == 0
    assert scheduler.reserved_tokens == 0

    engine.run_until_complete()
    assert second.state is RequestState.FINISHED
    assert second.output_length == 2
    assert scheduler.reserved_tokens == 0


def test_cancelled_running_request_returns_its_reservation() -> None:
    scheduler = Scheduler(
        max_batch_size=4, max_batch_tokens=100, max_reserved_tokens=12
    )
    _, engine = make_engine(scheduler)
    first = Request("request-1", [1] * 5, max_new_tokens=3)
    second = Request("request-2", [1] * 3, max_new_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)

    engine.step()
    engine.step()
    assert first.output_length == 1
    assert scheduler.reserved_tokens == 8

    assert scheduler.cancel_request("request-1") is True
    # The reservation is returned immediately, before any further
    # step runs, because the count is derived from the queues.
    assert scheduler.reserved_tokens == 0
    assert scheduler.active_request_count == 0

    engine.step()
    assert second.state is RequestState.DECODING
    engine.run_until_complete()
    assert first.state is RequestState.CANCELLED
    assert first.output_length == 1
    assert second.state is RequestState.FINISHED


def test_max_active_requests_one_keeps_the_second_waiting() -> None:
    scheduler = Scheduler(
        max_batch_size=4, max_batch_tokens=100, max_active_requests=1
    )
    _, engine = make_engine(scheduler)
    first = Request("request-1", [1] * 3, max_new_tokens=2)
    second = Request("request-2", [1] * 3, max_new_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)

    # The batch size would allow both prefills in one step; the
    # active request limit is what holds the second one back.
    engine.step()
    assert first.state is RequestState.DECODING
    assert second.state is RequestState.WAITING
    assert scheduler.active_request_count == 1

    engine.step()
    engine.step()
    assert first.state is RequestState.FINISHED
    assert second.state is RequestState.WAITING

    engine.run_until_complete()
    assert second.state is RequestState.FINISHED
    assert scheduler.active_request_count == 0


def test_invalid_resource_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_active_requests"):
        Scheduler(
            max_batch_size=1, max_batch_tokens=10, max_active_requests=0
        )
    with pytest.raises(ValueError, match="max_reserved_tokens"):
        Scheduler(
            max_batch_size=1, max_batch_tokens=10, max_reserved_tokens=-1
        )


def test_create_scheduler_passes_resource_limits_through() -> None:
    continuous = create_scheduler(
        "continuous",
        max_batch_size=2,
        max_batch_tokens=50,
        max_active_requests=3,
        max_reserved_tokens=64,
    )
    assert continuous.max_active_requests == 3
    assert continuous.max_reserved_tokens == 64

    serial = create_scheduler(
        "serial",
        max_batch_size=2,
        max_batch_tokens=50,
        max_active_requests=3,
        max_reserved_tokens=64,
    )
    assert serial.max_active_requests == 3
    assert serial.max_reserved_tokens == 64
    # The serial policy still rejects a request whose reservation
    # alone exceeds the capacity, through the inherited admission.
    with pytest.raises(RequestRejectedError, match="cache capacity"):
        serial.add_request(Request("request-1", [1] * 40, max_new_tokens=30))


def test_no_resource_limits_keeps_the_previous_behaviour() -> None:
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=100)
    _, engine = make_engine(scheduler)
    first = Request("request-1", [1] * 5, max_new_tokens=3)
    second = Request("request-2", [1] * 3, max_new_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)

    engine.step()
    # With no limits configured, both requests prefill in the
    # first step exactly as before.
    assert first.state is RequestState.DECODING
    assert second.state is RequestState.DECODING
    engine.run_until_complete()
    assert first.state is RequestState.FINISHED
    assert second.state is RequestState.FINISHED
