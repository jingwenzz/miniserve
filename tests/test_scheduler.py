from miniserve.executor import FakeExecutor
from miniserve.request import Request
from miniserve.scheduler import Scheduler


def test_scheduler_admits_waiting_requests() -> None:
    scheduler = Scheduler(max_batch_size=2,
    max_batch_tokens=100,)

    request_1 = Request("request-1", [1, 2], max_new_tokens=2)
    request_2 = Request("request-2", [3], max_new_tokens=2)
    request_3 = Request("request-3", [4], max_new_tokens=2)

    scheduler.add_request(request_1)
    scheduler.add_request(request_2)
    scheduler.add_request(request_3)

    output = scheduler.schedule()

    assert output.prefill_requests == [request_1, request_2]
    assert output.decode_requests == []
    assert list(scheduler.waiting) == [request_3]


def test_running_requests_are_scheduled_for_decode() -> None:
    scheduler = Scheduler(max_batch_size=2,
    max_batch_tokens=100,)
    executor = FakeExecutor()

    request = Request("request-1", [1, 2], max_new_tokens=2)
    scheduler.add_request(request)

    first_output = scheduler.schedule()
    executor.prefill(first_output.prefill_requests)
    scheduler.update(first_output)

    second_output = scheduler.schedule()

    assert second_output.prefill_requests == []
    assert second_output.decode_requests == [request]


def test_new_request_can_join_while_another_decodes() -> None:
    scheduler = Scheduler(max_batch_size=2,
    max_batch_tokens=100,)
    executor = FakeExecutor()

    request_1 = Request("request-1", [1], max_new_tokens=3)
    scheduler.add_request(request_1)

    first_output = scheduler.schedule()
    executor.prefill(first_output.prefill_requests)
    scheduler.update(first_output)

    request_2 = Request("request-2", [2], max_new_tokens=2)
    scheduler.add_request(request_2)

    second_output = scheduler.schedule()

    assert second_output.decode_requests == [request_1]
    assert second_output.prefill_requests == [request_2]


def test_finished_requests_are_removed() -> None:
    scheduler = Scheduler(max_batch_size=2,
    max_batch_tokens=100,)
    executor = FakeExecutor()

    request = Request("request-1", [1], max_new_tokens=1)
    scheduler.add_request(request)

    first_output = scheduler.schedule()
    executor.prefill(first_output.prefill_requests)
    scheduler.update(first_output)

    second_output = scheduler.schedule()
    executor.decode(second_output.decode_requests)
    scheduler.update(second_output)

    assert not scheduler.has_pending_requests()
    assert scheduler.running == []

def test_prefill_can_join_decode_with_available_token_budget() -> None:
    scheduler = Scheduler(
        max_batch_size=3,
        max_batch_tokens=10,
    )
    executor = FakeExecutor()

    request_1 = Request("request-1", [1], max_new_tokens=3)
    request_2 = Request("request-2", [2], max_new_tokens=3)

    scheduler.add_request(request_1)
    scheduler.add_request(request_2)

    first_output = scheduler.schedule()
    executor.prefill(first_output.prefill_requests)
    scheduler.update(first_output)

    request_3 = Request(
        "request-3",
        [3] * 5,
        max_new_tokens=2,
    )
    scheduler.add_request(request_3)

    second_output = scheduler.schedule()

    assert second_output.decode_requests == [
        request_1,
        request_2,
    ]
    assert second_output.prefill_requests == [request_3]

def test_prefill_waits_when_token_budget_is_exceeded() -> None:
    scheduler = Scheduler(
        max_batch_size=3,
        max_batch_tokens=5,
    )
    executor = FakeExecutor()

    request_1 = Request("request-1", [1], max_new_tokens=3)
    request_2 = Request("request-2", [2], max_new_tokens=3)

    scheduler.add_request(request_1)
    scheduler.add_request(request_2)

    first_output = scheduler.schedule()
    executor.prefill(first_output.prefill_requests)
    scheduler.update(first_output)

    request_3 = Request(
        "request-3",
        [3] * 4,
        max_new_tokens=2,
    )
    scheduler.add_request(request_3)

    second_output = scheduler.schedule()

    assert second_output.decode_requests == [
        request_1,
        request_2,
    ]
    assert second_output.prefill_requests == []
    assert list(scheduler.waiting) == [request_3]