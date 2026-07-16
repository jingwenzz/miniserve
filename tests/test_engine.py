from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import Scheduler


def test_engine_completes_all_requests() -> None:
    scheduler = Scheduler(max_batch_size=2,
    max_batch_tokens=100,)
    executor = FakeExecutor(generated_token_id=42)
    engine = Engine(scheduler, executor)

    requests = [
        Request("request-1", [1, 2], max_new_tokens=2),
        Request("request-2", [3], max_new_tokens=3),
        Request("request-3", [4, 5], max_new_tokens=1),
    ]

    for request in requests:
        scheduler.add_request(request)

    engine.run_until_complete()

    assert all(
        request.state is RequestState.FINISHED
        for request in requests
    )
    assert [request.output_length for request in requests] == [2, 3, 1]
    assert not scheduler.has_pending_requests()


def test_engine_step_runs_one_iteration() -> None:
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=10)
    executor = FakeExecutor()
    engine = Engine(scheduler, executor)

    request = Request(
        "request-1",
        [1, 2],
        max_new_tokens=2,
    )
    scheduler.add_request(request)

    engine.step()

    assert request.state is RequestState.DECODING
    assert request.output_length == 0
    assert engine.step_count == 1

    engine.step()

    assert request.output_length == 1
    assert engine.step_count == 2