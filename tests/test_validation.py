import pytest

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import RequestRejectedError, Scheduler


def test_empty_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Request("request-1", [], max_new_tokens=1)


def test_zero_max_new_tokens_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        Request("request-1", [1], max_new_tokens=0)


def test_negative_max_new_tokens_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        Request("request-1", [1], max_new_tokens=-3)


def test_prompt_longer_than_step_budget_is_rejected() -> None:
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=4)
    request = Request("too-long", [1] * 5, max_new_tokens=1)

    with pytest.raises(RequestRejectedError, match="per step token budget"):
        scheduler.add_request(request)

    assert list(scheduler.waiting) == []
    assert scheduler.running == []
    assert not scheduler.has_pending_requests()


def test_prompt_equal_to_step_budget_is_admitted() -> None:
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=4)
    request = Request("exact-fit", [1] * 4, max_new_tokens=1)

    scheduler.add_request(request)

    assert list(scheduler.waiting) == [request]


def test_rejection_does_not_pollute_later_requests() -> None:
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=4)
    executor = FakeExecutor()
    engine = Engine(scheduler, executor)

    with pytest.raises(RequestRejectedError):
        scheduler.add_request(Request("too-long", [1] * 9, max_new_tokens=1))

    valid = Request("valid", [1, 2], max_new_tokens=2)
    scheduler.add_request(valid)
    engine.run_until_complete()

    assert valid.state is RequestState.FINISHED
    assert valid.output_length == 2
    assert not scheduler.has_pending_requests()


def test_prompt_plus_reserved_output_equal_to_context_limit_is_admitted() -> None:
    scheduler = Scheduler(
        max_batch_size=2,
        max_batch_tokens=100,
        max_model_len=10,
    )
    request = Request("exact-context", [1] * 6, max_new_tokens=4)

    scheduler.add_request(request)

    assert list(scheduler.waiting) == [request]


def test_prompt_plus_reserved_output_above_context_limit_is_rejected() -> None:
    scheduler = Scheduler(
        max_batch_size=2,
        max_batch_tokens=100,
        max_model_len=10,
    )
    request = Request("over-context", [1] * 6, max_new_tokens=5)

    with pytest.raises(RequestRejectedError, match="model context limit"):
        scheduler.add_request(request)

    assert list(scheduler.waiting) == []
    assert not scheduler.has_pending_requests()
