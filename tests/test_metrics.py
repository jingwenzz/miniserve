import json

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.metrics import (
    count_by_finish_reason,
    summarize_request,
    to_jsonl,
)
from miniserve.request import Request
from miniserve.scheduler import Scheduler


class FakeClock:
    """A clock that only moves when the test moves it."""

    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def set(self, value: float) -> None:
        self.now = value


def make_request(
    clock: FakeClock, request_id: str = "req-1", max_new_tokens: int = 3
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[10, 20, 30],
        max_new_tokens=max_new_tokens,
        clock=clock,
    )


def drive_tokens(request: Request, clock: FakeClock, times: list[float]) -> None:
    request.start_prefill()
    request.finish_prefill()
    for value in times:
        clock.set(value)
        request.append_token(99)


def test_hand_worked_three_token_example() -> None:
    # The exact timeline worked by hand in docs/metrics.md section 5.
    clock = FakeClock(100.0)
    request = make_request(clock)

    clock.set(101.0)
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=100)
    scheduler.add_request(request)

    clock.set(103.0)
    drive_tokens(request, clock, [105.0, 107.0, 109.0])

    metrics = summarize_request(request)

    assert metrics.arrival_time == 100.0
    assert metrics.admission_time == 101.0
    assert metrics.prefill_start_time == 103.0
    assert metrics.first_token_time == 105.0
    assert metrics.last_token_time == 109.0
    assert metrics.finished_time == 109.0
    assert metrics.queue_time == 3.0
    assert metrics.ttft == 5.0
    assert metrics.latency == 9.0
    assert metrics.tpot == 2.0
    assert metrics.state == "FINISHED"
    assert metrics.finish_reason == "finished"
    assert metrics.prompt_length == 3
    assert metrics.output_length == 3


def test_single_token_request_has_no_tpot() -> None:
    clock = FakeClock(100.0)
    request = make_request(clock, max_new_tokens=1)

    clock.set(103.0)
    drive_tokens(request, clock, [105.0])

    metrics = summarize_request(request)

    assert metrics.output_length == 1
    assert metrics.ttft == 5.0
    assert metrics.latency == 5.0
    assert metrics.tpot is None
    assert metrics.finish_reason == "finished"


def test_cancelled_waiting_request_counts_separately() -> None:
    clock = FakeClock(100.0)
    request = make_request(clock)
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=100)
    scheduler.add_request(request)

    clock.set(104.0)
    assert scheduler.cancel_request("req-1") is True

    metrics = summarize_request(request)

    assert metrics.state == "CANCELLED"
    assert metrics.finish_reason == "cancelled"
    assert metrics.output_length == 0
    assert metrics.ttft is None
    assert metrics.tpot is None
    assert metrics.latency == 4.0
    assert count_by_finish_reason([request]) == {
        "finished": 0,
        "cancelled": 1,
        "failed": 0,
        "unfinished": 0,
    }


def test_cancelled_running_request_keeps_earned_tokens() -> None:
    clock = FakeClock(100.0)
    request = make_request(clock, max_new_tokens=5)

    clock.set(103.0)
    request.start_prefill()
    request.finish_prefill()
    clock.set(105.0)
    request.append_token(99)

    clock.set(110.0)
    request.cancel()

    metrics = summarize_request(request)

    assert metrics.finish_reason == "cancelled"
    assert metrics.output_length == 1
    assert metrics.ttft == 5.0
    assert metrics.latency == 10.0
    assert metrics.tpot is None


def test_unfinished_request_exports_nulls_not_zeros() -> None:
    clock = FakeClock(100.0)
    request = make_request(clock)

    metrics = summarize_request(request)

    assert metrics.state == "WAITING"
    assert metrics.finish_reason is None
    assert metrics.admission_time is None
    assert metrics.prefill_start_time is None
    assert metrics.first_token_time is None
    assert metrics.last_token_time is None
    assert metrics.finished_time is None
    assert metrics.queue_time is None
    assert metrics.ttft is None
    assert metrics.latency is None
    assert metrics.tpot is None
    assert metrics.output_length == 0
    assert count_by_finish_reason([request]) == {
        "finished": 0,
        "cancelled": 0,
        "failed": 0,
        "unfinished": 1,
    }


def test_decoding_request_has_ttft_but_no_latency() -> None:
    clock = FakeClock(100.0)
    request = make_request(clock, max_new_tokens=3)

    clock.set(102.0)
    drive_tokens(request, clock, [104.0])

    metrics = summarize_request(request)

    assert metrics.ttft == 4.0
    assert metrics.latency is None
    assert metrics.finished_time is None
    assert metrics.finish_reason is None


def test_jsonl_round_trip_preserves_raw_timestamps() -> None:
    clock = FakeClock(100.0)
    finished = make_request(clock, request_id="done", max_new_tokens=1)
    clock.set(103.0)
    drive_tokens(finished, clock, [105.0])

    clock.set(200.0)
    pending = make_request(clock, request_id="pending")

    text = to_jsonl([finished, pending])
    records = [json.loads(line) for line in text.splitlines()]

    assert len(records) == 2
    assert records[0]["request_id"] == "done"
    assert records[0]["arrival_time"] == 100.0
    assert records[0]["first_token_time"] == 105.0
    assert records[0]["ttft"] == 5.0
    assert records[0]["finish_reason"] == "finished"
    assert records[1]["request_id"] == "pending"
    assert records[1]["arrival_time"] == 200.0
    assert records[1]["first_token_time"] is None
    assert records[1]["ttft"] is None
    assert records[1]["finish_reason"] is None


def test_engine_run_records_admission_and_completion() -> None:
    clock = FakeClock(1000.0)
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=100)
    engine = Engine(scheduler, FakeExecutor(generated_token_id=7))
    request = Request(
        request_id="engine-req",
        prompt_token_ids=[1, 2],
        max_new_tokens=2,
        clock=clock,
    )
    scheduler.add_request(request)
    engine.run_until_complete()

    metrics = summarize_request(request)

    assert metrics.finish_reason == "finished"
    assert metrics.output_length == 2
    assert metrics.admission_time == 1000.0
    assert metrics.prefill_start_time == 1000.0
    assert metrics.queue_time == 0.0
    # The fake clock never advanced, so every duration in this run is
    # exactly zero. That is a property of the injected clock, not a
    # claim about real model speed.
    assert metrics.ttft == 0.0
    assert metrics.latency == 0.0
    assert metrics.tpot == 0.0


def test_admission_is_recorded_once() -> None:
    clock = FakeClock(100.0)
    request = make_request(clock)
    clock.set(101.0)
    request.mark_admitted()
    clock.set(150.0)
    request.mark_admitted()

    assert request.admission_time == 101.0
