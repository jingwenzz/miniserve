import pytest

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import (
    Scheduler,
    SerialScheduler,
    create_scheduler,
)


def _specs() -> list[tuple[str, int, int]]:
    return [
        ("request-1", 10, 4),
        ("request-2", 40, 2),
        ("request-3", 5, 3),
    ]


def _run(policy: str) -> tuple[list[Request], Engine]:
    scheduler = create_scheduler(
        policy, max_batch_size=3, max_batch_tokens=50
    )
    executor = FakeExecutor(generated_token_id=42)
    engine = Engine(scheduler, executor)
    requests = [
        Request(request_id, [1] * length, max_new_tokens=output)
        for request_id, length, output in _specs()
    ]
    for request in requests:
        scheduler.add_request(request)
    engine.run_until_complete()
    return requests, engine


def test_factory_returns_expected_types() -> None:
    continuous = create_scheduler(
        "continuous", max_batch_size=2, max_batch_tokens=100
    )
    serial = create_scheduler("serial", max_batch_size=2, max_batch_tokens=100)

    assert type(continuous) is Scheduler
    assert type(serial) is SerialScheduler
    assert isinstance(serial, Scheduler)


def test_factory_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unknown scheduler policy"):
        create_scheduler("bogus", max_batch_size=2, max_batch_tokens=100)


def test_serial_prefills_only_one_request_in_first_step() -> None:
    scheduler = SerialScheduler(max_batch_size=3, max_batch_tokens=50)
    requests = [
        Request(request_id, [1] * length, max_new_tokens=output)
        for request_id, length, output in _specs()
    ]
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.prefill_requests == [requests[0]]
    assert output.decode_requests == []
    assert list(scheduler.waiting) == requests[1:]


def test_serial_never_prefills_while_a_request_is_running() -> None:
    scheduler = SerialScheduler(max_batch_size=3, max_batch_tokens=100)
    executor = FakeExecutor()
    engine = Engine(scheduler, executor)

    first = Request("request-1", [1], max_new_tokens=3)
    second = Request("request-2", [2], max_new_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)

    engine.step()
    assert first.state is RequestState.DECODING
    assert second.state is RequestState.WAITING

    output = scheduler.schedule()
    assert output.prefill_requests == []
    assert output.decode_requests == [first]


def test_serial_completes_requests_one_at_a_time_in_order() -> None:
    requests, engine = _run("serial")

    assert all(
        request.state is RequestState.FINISHED for request in requests
    )
    assert [request.output_length for request in requests] == [4, 2, 3]
    # Serial steps by hand: request-1 takes 1 prefill step plus 4
    # decode steps, request-2 takes 1 plus 2, request-3 takes 1
    # plus 3, and no step is shared between two requests.
    assert engine.step_count == 5 + 3 + 4


def test_serial_and_continuous_produce_same_output_lengths() -> None:
    serial_requests, serial_engine = _run("serial")
    continuous_requests, continuous_engine = _run("continuous")

    assert [r.output_length for r in serial_requests] == [
        r.output_length for r in continuous_requests
    ]
    assert [r.generated_token_ids for r in serial_requests] == [
        r.generated_token_ids for r in continuous_requests
    ]
    assert [r.max_new_tokens for r in serial_requests] == [
        r.max_new_tokens for r in continuous_requests
    ]
    # The policies differ only in scheduling, so the serial run
    # needs strictly more steps for this multi request set.
    assert serial_engine.step_count > continuous_engine.step_count


def test_serial_reuses_admission_and_cancellation_rules() -> None:
    scheduler = SerialScheduler(max_batch_size=2, max_batch_tokens=10)

    with pytest.raises(ValueError):
        scheduler.add_request(Request("too-long", [1] * 11, max_new_tokens=1))

    request = Request("request-1", [1] * 5, max_new_tokens=2)
    scheduler.add_request(request)
    assert scheduler.cancel_request("request-1") is True
    assert not scheduler.has_pending_requests()
