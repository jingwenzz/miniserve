import pytest

from miniserve.executor import FakeExecutor
from miniserve.request import Request, RequestState


def test_prefill_moves_request_to_decoding() -> None:
    request = Request(
        request_id="request-1",
        prompt_token_ids=[10, 20, 30],
        max_new_tokens=2,
    )
    executor = FakeExecutor()

    executor.prefill([request])

    assert request.state is RequestState.DECODING


def test_decode_generates_one_token_per_step() -> None:
    request = Request(
        request_id="request-1",
        prompt_token_ids=[10, 20],
        max_new_tokens=2,
    )
    executor = FakeExecutor(generated_token_id=42)

    executor.prefill([request])
    executor.decode([request])

    assert request.generated_token_ids == [42]
    assert request.state is RequestState.DECODING

    executor.decode([request])

    assert request.generated_token_ids == [42, 42]
    assert request.state is RequestState.FINISHED


def test_decode_rejects_waiting_request() -> None:
    request = Request(
        request_id="request-1",
        prompt_token_ids=[10],
        max_new_tokens=1,
    )
    executor = FakeExecutor()

    with pytest.raises(RuntimeError):
        executor.decode([request])


def test_executor_handles_multiple_requests() -> None:
    requests = [
        Request("request-1", [10], max_new_tokens=1),
        Request("request-2", [20, 30], max_new_tokens=2),
    ]
    executor = FakeExecutor()

    executor.prefill(requests)
    executor.decode(requests)

    assert requests[0].state is RequestState.FINISHED
    assert requests[1].state is RequestState.DECODING
    assert requests[0].output_length == 1
    assert requests[1].output_length == 1