import pytest

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import RequestRejectedError, Scheduler


def make_engine(
    max_batch_size: int = 2,
    max_batch_tokens: int = 100,
) -> tuple[Scheduler, Engine]:
    scheduler = Scheduler(
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    return scheduler, Engine(scheduler, FakeExecutor())


def test_cancel_waiting_request_by_id() -> None:
    scheduler, engine = make_engine()

    cancelled = Request("cancelled", [1, 2], max_new_tokens=3)
    survivor = Request("survivor", [3], max_new_tokens=2)
    scheduler.add_request(cancelled)
    scheduler.add_request(survivor)

    assert scheduler.cancel_request("cancelled") is True

    assert cancelled.state is RequestState.CANCELLED
    assert cancelled.output_length == 0
    assert list(scheduler.waiting) == [survivor]

    engine.run_until_complete()

    assert cancelled.output_length == 0
    assert survivor.state is RequestState.FINISHED
    assert survivor.output_length == 2
    assert not scheduler.has_pending_requests()


def test_cancel_waiting_request_directly_is_pruned_before_prefill() -> None:
    scheduler, engine = make_engine()

    cancelled = Request("cancelled", [1, 2], max_new_tokens=3)
    survivor = Request("survivor", [3], max_new_tokens=1)
    scheduler.add_request(cancelled)
    scheduler.add_request(survivor)

    # Bypassing the scheduler entry point must not make the executor
    # call start_prefill on a CANCELLED request during the next step.
    cancelled.cancel()

    engine.run_until_complete()

    assert cancelled.output_length == 0
    assert survivor.state is RequestState.FINISHED
    assert not scheduler.has_pending_requests()


def test_cancel_running_request_by_id_stops_new_tokens() -> None:
    scheduler, engine = make_engine()

    cancelled = Request("cancelled", [1], max_new_tokens=5)
    survivor = Request("survivor", [2], max_new_tokens=2)
    scheduler.add_request(cancelled)
    scheduler.add_request(survivor)

    engine.step()  # prefill both
    engine.step()  # one decode token each
    assert cancelled.output_length == 1
    assert survivor.output_length == 1

    assert scheduler.cancel_request("cancelled") is True
    assert scheduler.running == [survivor]

    engine.run_until_complete()

    assert cancelled.state is RequestState.CANCELLED
    assert cancelled.output_length == 1
    assert survivor.state is RequestState.FINISHED
    assert survivor.output_length == 2


def test_cancel_running_request_directly_stops_new_tokens() -> None:
    scheduler, engine = make_engine()

    cancelled = Request("cancelled", [1], max_new_tokens=5)
    scheduler.add_request(cancelled)

    engine.step()  # prefill
    engine.step()  # one decode token
    cancelled.cancel()

    engine.run_until_complete()

    assert cancelled.output_length == 1
    assert scheduler.running == []
    assert not scheduler.has_pending_requests()


def test_repeated_cancel_after_finish_is_a_no_op() -> None:
    scheduler, engine = make_engine()

    request = Request("request-1", [1], max_new_tokens=1)
    scheduler.add_request(request)
    engine.run_until_complete()
    assert request.state is RequestState.FINISHED

    assert scheduler.cancel_request("request-1") is False

    assert request.state is RequestState.FINISHED
    assert request.output_length == 1
    assert not scheduler.has_pending_requests()


def test_repeated_cancel_after_cancel_is_a_no_op() -> None:
    scheduler, _engine = make_engine()

    request = Request("request-1", [1], max_new_tokens=3)
    scheduler.add_request(request)

    assert scheduler.cancel_request("request-1") is True
    assert scheduler.cancel_request("request-1") is False

    assert request.state is RequestState.CANCELLED
    assert request.output_length == 0
    assert not scheduler.has_pending_requests()


def test_cancel_unknown_id_raises_key_error() -> None:
    scheduler, _engine = make_engine()

    with pytest.raises(KeyError, match="Unknown request id"):
        scheduler.cancel_request("never-seen")


def test_duplicate_active_id_in_waiting_is_rejected() -> None:
    scheduler, _engine = make_engine()

    first = Request("dup", [1], max_new_tokens=1)
    second = Request("dup", [2], max_new_tokens=1)
    scheduler.add_request(first)

    with pytest.raises(RequestRejectedError, match="already been used"):
        scheduler.add_request(second)

    assert list(scheduler.waiting) == [first]


def test_same_object_offered_twice_occupies_one_slot() -> None:
    scheduler, _engine = make_engine()

    request = Request("request-1", [1], max_new_tokens=1)
    scheduler.add_request(request)

    with pytest.raises(RequestRejectedError, match="already been used"):
        scheduler.add_request(request)

    assert list(scheduler.waiting) == [request]


def test_duplicate_active_id_in_running_is_rejected() -> None:
    scheduler, engine = make_engine()

    first = Request("dup", [1], max_new_tokens=3)
    scheduler.add_request(first)
    engine.step()  # prefill, request is now running
    assert scheduler.running == [first]

    with pytest.raises(RequestRejectedError, match="already been used"):
        scheduler.add_request(Request("dup", [2], max_new_tokens=1))

    assert scheduler.running == [first]
    assert list(scheduler.waiting) == []


def test_id_stays_used_after_the_request_finishes() -> None:
    scheduler, engine = make_engine()

    first = Request("request-1", [1], max_new_tokens=1)
    scheduler.add_request(first)
    engine.run_until_complete()
    assert first.state is RequestState.FINISHED

    with pytest.raises(RequestRejectedError, match="already been used"):
        scheduler.add_request(Request("request-1", [2], max_new_tokens=1))


def test_rejected_offer_does_not_burn_its_id() -> None:
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=4)

    with pytest.raises(RequestRejectedError, match="per step token budget"):
        scheduler.add_request(Request("retry", [1] * 5, max_new_tokens=1))

    corrected = Request("retry", [1] * 4, max_new_tokens=1)
    scheduler.add_request(corrected)

    assert list(scheduler.waiting) == [corrected]
