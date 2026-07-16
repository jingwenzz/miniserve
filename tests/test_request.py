import pytest

from miniserve.request import Request, RequestState

def test_new_request_defaults() -> None:
    request = Request(
        request_id="request-1",
        prompt_token_ids=[10, 20, 30],
        max_new_tokens=5,
    )

    assert request.state is RequestState.WAITING
    assert request.prompt_length == 3
    assert request.output_length == 0
    assert request.total_length == 3
    assert request.generated_token_ids == []
    assert not request.is_finished
    assert request.first_token_time is None
    assert request.finished_time is None

def test_request_lifecycle() -> None:
    request = Request(
        request_id="request-1",
        prompt_token_ids=[10, 20, 30],
        max_new_tokens=2,
    )

    request.start_prefill()
    assert request.state is RequestState.PREFILLING

    request.finish_prefill()
    assert request.state is RequestState.DECODING

    request.append_token(40)
    assert request.generated_token_ids == [40]
    assert request.first_token_time is not None
    assert request.state is RequestState.DECODING

    request.append_token(50)
    assert request.generated_token_ids == [40, 50]
    assert request.state is RequestState.FINISHED
    assert request.finished_time is not None


def test_cannot_decode_before_prefill() -> None:
    request = Request(
        request_id="request-1",
        prompt_token_ids=[10],
        max_new_tokens=1,
    )

    with pytest.raises(RuntimeError):
        request.append_token(20)


def test_cannot_prefill_twice() -> None:
    request = Request(
        request_id="request-1",
        prompt_token_ids=[10],
        max_new_tokens=1,
    )

    request.start_prefill()

    with pytest.raises(RuntimeError):
        request.start_prefill()


def test_cancel_request() -> None:
    request = Request(
        request_id="request-1",
        prompt_token_ids=[10],
        max_new_tokens=1,
    )

    request.cancel()

    assert request.state is RequestState.CANCELLED
    assert request.finished_time is not None