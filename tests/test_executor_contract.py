from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import Scheduler


class RecordingExecutor:
    """A second executor that shares no code with FakeExecutor.

    It exists to prove the engine only relies on the Executor
    protocol: prefill, decode, and release. It follows the same
    contract as the fake backend, moving requests through the
    Request transition methods and appending one token per decode.
    """

    def __init__(self) -> None:
        self.prefill_calls: list[list[str]] = []
        self.decode_calls: list[list[str]] = []
        self.released: list[str] = []

    def prefill(self, requests: list[Request]) -> None:
        self.prefill_calls.append([r.request_id for r in requests])
        for request in requests:
            request.start_prefill()
        for request in requests:
            request.finish_prefill()

    def decode(self, requests: list[Request]) -> None:
        self.decode_calls.append([r.request_id for r in requests])
        for request in requests:
            request.append_token(7)

    def release(self, request: Request) -> None:
        if request.request_id not in self.released:
            self.released.append(request.request_id)


def make_engine(
    executor: RecordingExecutor | FakeExecutor,
    max_batch_size: int = 2,
    max_batch_tokens: int = 100,
) -> tuple[Scheduler, Engine]:
    scheduler = Scheduler(
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    return scheduler, Engine(scheduler, executor)


def test_engine_runs_with_a_custom_executor() -> None:
    executor = RecordingExecutor()
    scheduler, engine = make_engine(executor)
    requests = [
        Request("request-1", [1, 2], max_new_tokens=2),
        Request("request-2", [3], max_new_tokens=1),
    ]
    for request in requests:
        scheduler.add_request(request)

    engine.run_until_complete()

    assert all(r.state is RequestState.FINISHED for r in requests)
    assert [r.output_length for r in requests] == [2, 1]
    assert requests[0].generated_token_ids == [7, 7]


def test_engine_releases_each_finished_request_exactly_once() -> None:
    executor = RecordingExecutor()
    scheduler, engine = make_engine(executor)
    requests = [
        Request("request-1", [1], max_new_tokens=1),
        Request("request-2", [2], max_new_tokens=2),
    ]
    for request in requests:
        scheduler.add_request(request)

    engine.run_until_complete()
    # A second sweep must not release anything again.
    engine.release_terminal_requests()

    assert sorted(executor.released) == ["request-1", "request-2"]
    assert len(executor.released) == 2


def test_no_release_while_request_is_still_active() -> None:
    executor = FakeExecutor()
    scheduler, engine = make_engine(executor)
    request = Request("request-1", [1, 2], max_new_tokens=2)
    scheduler.add_request(request)

    engine.step()

    assert request.state is RequestState.DECODING
    assert executor.released_request_ids == []


def test_fake_executor_records_release_for_finished_request() -> None:
    executor = FakeExecutor()
    scheduler, engine = make_engine(executor)
    request = Request("request-1", [1], max_new_tokens=1)
    scheduler.add_request(request)

    engine.run_until_complete()

    assert request.state is RequestState.FINISHED
    assert executor.released_request_ids == ["request-1"]


def test_cancelled_running_request_is_released() -> None:
    executor = FakeExecutor()
    scheduler, engine = make_engine(executor)
    first = Request("request-1", [1], max_new_tokens=3)
    second = Request("request-2", [2], max_new_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)
    engine.step()
    assert first.state is RequestState.DECODING

    assert scheduler.cancel_request("request-1") is True
    engine.run_until_complete()

    assert first.state is RequestState.CANCELLED
    assert second.state is RequestState.FINISHED
    assert sorted(executor.released_request_ids) == ["request-1", "request-2"]


def test_cancelled_only_request_is_released_without_another_step() -> None:
    executor = FakeExecutor()
    scheduler, engine = make_engine(executor)
    request = Request("request-1", [1], max_new_tokens=3)
    scheduler.add_request(request)
    engine.step()

    assert scheduler.cancel_request("request-1") is True
    engine.run_until_complete()

    assert executor.released_request_ids == ["request-1"]


def test_waiting_cancelled_request_needs_no_release() -> None:
    executor = FakeExecutor()
    scheduler, engine = make_engine(executor)
    request = Request("request-1", [1], max_new_tokens=1)
    scheduler.add_request(request)

    assert scheduler.cancel_request("request-1") is True
    engine.run_until_complete()

    # The request never ran prefill, so it never held backend
    # resources and the executor is never asked to release it.
    assert executor.released_request_ids == []


def test_fake_release_is_idempotent_and_keeps_request_unchanged() -> None:
    executor = FakeExecutor(generated_token_id=9)
    request = Request("request-1", [1], max_new_tokens=1)
    executor.prefill([request])
    executor.decode([request])
    assert request.state is RequestState.FINISHED

    executor.release(request)
    executor.release(request)

    assert executor.released_request_ids == ["request-1"]
    assert request.state is RequestState.FINISHED
    assert request.generated_token_ids == [9]
