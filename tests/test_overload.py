"""Tests for overload: the waiting bound and slow consumers.

The waiting bound is exercised with a step gated executor, so
the queue is provably full when the extra submission is
decided: the refusal must be a QueueFullError, must consume
no request id, must queue nothing, and must be counted in the
admission statistics next to the accepted count. The slow
consumer rule is exercised with a two event buffer and a
reader that never reads: the worker must not block, the
request must be cancelled once its buffer overflows, the
dropped events must be counted, the terminal event must
still be delivered, and a well behaved request behind it
must complete with every event intact.
"""

import queue
import threading
import time

import pytest

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request
from miniserve.scheduler import Scheduler
from miniserve.service import EngineOwner, QueueFullError


def make_request(request_id: str, prompt_len: int = 3, max_new: int = 3) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt_len,
        max_new_tokens=max_new,
    )


class StepGatedExecutor(FakeExecutor):
    """Fake executor whose decode calls wait for test permits."""

    def __init__(self) -> None:
        super().__init__()
        self._permits: queue.Queue[None] = queue.Queue()
        self._condition = threading.Condition()
        self._entered = 0
        self.gating = True

    def decode(self, requests: list[Request]) -> None:
        if self.gating and requests:
            with self._condition:
                self._entered += 1
                self._condition.notify_all()
            self._permits.get(timeout=10)
        super().decode(requests)

    def wait_entered(self, count: int) -> None:
        deadline = time.monotonic() + 5
        with self._condition:
            while self._entered < count:
                remaining = deadline - time.monotonic()
                assert remaining > 0, "decode was not entered in time"
                self._condition.wait(timeout=remaining)

    def release_permits(self, count: int = 1) -> None:
        for _ in range(count):
            self._permits.put(None)

    def finish_gracefully(self) -> None:
        self.gating = False
        self.release_permits(1)


def wait_for_queue_size(owner: EngineOwner, size: int) -> None:
    deadline = time.monotonic() + 5
    while owner._commands.qsize() < size:
        assert time.monotonic() < deadline, "command was not queued in time"
        time.sleep(0.001)


def test_waiting_bound_refuses_fast_counts_and_keeps_the_id() -> None:
    executor = StepGatedExecutor()
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    owner = EngineOwner(engine, max_waiting_requests=1)
    owner.start()
    submitted: dict[str, object] = {}
    refused: dict[str, object] = {}

    def submit_waiting() -> None:
        submitted["handle"] = owner.submit(make_request("waiting", max_new=2))

    def submit_overflow() -> None:
        try:
            owner.submit(make_request("overflow", max_new=2))
        except QueueFullError as exc:
            refused["error"] = exc

    waiting_thread = threading.Thread(target=submit_waiting)
    overflow_thread = threading.Thread(target=submit_overflow)
    try:
        running = owner.submit(make_request("running", max_new=5))
        executor.wait_entered(1)
        waiting_thread.start()
        wait_for_queue_size(owner, 1)
        executor.release_permits(1)
        waiting_thread.join(timeout=5)
        # The waiting slot is now taken and the running request
        # is gated inside its second decode.
        executor.wait_entered(2)
        overflow_thread.start()
        wait_for_queue_size(owner, 1)
        executor.release_permits(1)
        overflow_thread.join(timeout=5)
        error = refused["error"]
        assert isinstance(error, QueueFullError)
        assert error.error_code == "queue_full"
        assert error.waiting_limit == 1
        assert error.waiting_count == 1
        stats = owner.admission_stats()
        assert stats["offered"] == 3
        assert stats["accepted"] == 2
        assert stats["rejected_queue_full"] == 1
        assert stats["queue_full_rejection_rate"] == pytest.approx(1 / 3)
        assert owner.health()["waiting_requests"] == 1
        executor.finish_gracefully()
        assert running.result(timeout=5).state.name == "FINISHED"
        waiting = submitted["handle"]
        assert waiting.result(timeout=5).state.name == "FINISHED"  # type: ignore[attr-defined]
        # The refused id was not consumed: the same id is
        # accepted once the queue has room again.
        retry = owner.submit(make_request("overflow", max_new=1))
        assert retry.result(timeout=5).state.name == "FINISHED"
        assert owner.admission_stats()["accepted"] == 3
    finally:
        executor.finish_gracefully()
        for thread in (waiting_thread, overflow_thread):
            if thread.is_alive():
                thread.join(timeout=5)
        owner.stop(timeout=5)


def test_slow_consumer_is_cancelled_and_terminal_still_arrives() -> None:
    executor = FakeExecutor()
    owner = EngineOwner(
        Engine(Scheduler(max_batch_size=4, max_batch_tokens=100), executor),
        max_event_buffer=2,
    )
    owner.start()
    try:
        slow = owner.submit(make_request("slow", max_new=6))
        # The reader never reads while generation runs. The
        # worker keeps stepping regardless: two token events
        # fill the buffer, the third cannot fit, and the
        # request is cancelled at that boundary with exactly
        # the three tokens produced so far.
        result = slow.result(timeout=5)
        assert result.state.name == "CANCELLED"
        assert result.output_length == 3
        assert slow.slow_consumer is True
        assert slow.dropped_events >= 1
        assert slow.buffered_events <= 2
        assert executor.released_request_ids == ["slow"]
        assert owner.admission_stats()["slow_consumer_cancellations"] == 1
        # When the reader finally resumes, the stream still
        # ends with exactly one terminal event reporting the
        # cancellation and the true generated count.
        events = list(slow.events())
        assert events[-1].is_terminal
        assert events[-1].finish_reason == "cancelled"
        assert events[-1].generated_tokens == 3
        assert all(event.type == "token" for event in events[:-1])
    finally:
        owner.stop(timeout=5)


def test_slow_consumer_does_not_block_a_well_behaved_request() -> None:
    executor = FakeExecutor()
    owner = EngineOwner(
        Engine(Scheduler(max_batch_size=4, max_batch_tokens=100), executor),
        max_event_buffer=2,
    )
    owner.start()
    try:
        slow = owner.submit(make_request("slow", max_new=6))
        assert slow.result(timeout=5).state.name == "CANCELLED"
        # The worker survived the slow consumer and the next
        # request is served normally, with a reader that keeps
        # up receiving every event and dropping none.
        good = owner.submit(make_request("good", max_new=1))
        events = list(good.events())
        assert [event.type for event in events] == ["token", "finished"]
        assert good.result(timeout=5).state.name == "FINISHED"
        assert good.dropped_events == 0
        assert good.slow_consumer is False
        assert owner.health()["worker_alive"] is True
    finally:
        owner.stop(timeout=5)


def test_invalid_overload_configuration_is_rejected() -> None:
    engine = Engine(
        Scheduler(max_batch_size=4, max_batch_tokens=100), FakeExecutor()
    )
    with pytest.raises(ValueError):
        EngineOwner(engine, max_waiting_requests=0)
    with pytest.raises(ValueError):
        EngineOwner(engine, max_event_buffer=0)
