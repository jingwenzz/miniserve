"""Tests for client disconnect mapped to cancellation.

A disconnect is exercised at the transport independent layer,
the same way the service contract defines it: the client of a
stream goes away, and that fact reaches the engine owner as a
disconnect command. Two entry points are covered, calling
``disconnect`` (on the handle or on the owner) and closing the
event stream iterator early, which is the signal a stream
consumer itself can give. The gated and slow fake executors
make the timing deterministic enough to assert the exact
token count a running request is frozen at, and that a
waiting request disconnects without ever being prefilled or
released, because it never held backend resources.
"""

import queue
import threading
import time

import pytest

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request
from miniserve.scheduler import Scheduler
from miniserve.service import EngineOwner


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


class SlowFakeExecutor(FakeExecutor):
    """Fake executor with an artificial decode delay."""

    def __init__(self, delay_seconds: float = 0.05) -> None:
        super().__init__()
        self._delay = delay_seconds

    def decode(self, requests: list[Request]) -> None:
        if requests:
            time.sleep(self._delay)
        super().decode(requests)


def wait_for_queue_size(owner: EngineOwner, size: int) -> None:
    deadline = time.monotonic() + 5
    while owner._commands.qsize() < size:
        assert time.monotonic() < deadline, "command was not queued in time"
        time.sleep(0.001)


def test_disconnect_waiting_request_never_starts_and_survivor_runs() -> None:
    executor = StepGatedExecutor()
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    owner = EngineOwner(engine)
    owner.start()
    submitted: dict[str, object] = {}
    disconnected: dict[str, object] = {}

    def submit_waiting() -> None:
        submitted["handle"] = owner.submit(make_request("waiting", max_new=2))

    submit_thread = threading.Thread(target=submit_waiting)
    try:
        running = owner.submit(make_request("running", max_new=5))
        executor.wait_entered(1)
        submit_thread.start()
        wait_for_queue_size(owner, 1)
        executor.release_permits(1)
        submit_thread.join(timeout=5)
        waiting = submitted["handle"]
        assert isinstance(waiting, type(running))
        executor.wait_entered(2)

        def disconnect_waiting() -> None:
            disconnected["result"] = waiting.disconnect()  # type: ignore[attr-defined]

        thread = threading.Thread(target=disconnect_waiting)
        thread.start()
        wait_for_queue_size(owner, 1)
        executor.release_permits(1)
        thread.join(timeout=5)
        assert disconnected["result"] is True
        assert waiting.disconnected is True  # type: ignore[attr-defined]
        result = waiting.result(timeout=5)  # type: ignore[attr-defined]
        assert result.state.name == "CANCELLED"
        assert result.output_length == 0
        # A waiting request was never prefilled, so it held no
        # backend resources and the executor never released it.
        assert "waiting" not in executor.released_request_ids
        events = list(waiting.events())  # type: ignore[attr-defined]
        assert [event.type for event in events] == ["finished"]
        assert events[-1].finish_reason == "cancelled"
        assert owner.admission_stats()["disconnect_cancellations"] == 1
        executor.finish_gracefully()
        assert running.result(timeout=5).state.name == "FINISHED"
        assert "running" in executor.released_request_ids
    finally:
        executor.finish_gracefully()
        if submit_thread.is_alive():
            submit_thread.join(timeout=5)
        owner.stop(timeout=5)


def test_disconnect_running_request_freezes_tokens_and_releases() -> None:
    executor = StepGatedExecutor()
    owner = EngineOwner(
        Engine(Scheduler(max_batch_size=4, max_batch_tokens=100), executor)
    )
    owner.start()
    disconnected: dict[str, object] = {}

    def disconnect_victim() -> None:
        disconnected["result"] = owner.disconnect("victim")

    thread = threading.Thread(target=disconnect_victim)
    try:
        handle = owner.submit(make_request("victim", max_new=6))
        executor.wait_entered(1)
        thread.start()
        wait_for_queue_size(owner, 1)
        executor.release_permits(1)
        thread.join(timeout=5)
        assert disconnected["result"] is True
        result = handle.result(timeout=5)
        assert result.state.name == "CANCELLED"
        # Exactly the token from the completed step exists: the
        # disconnect took effect at the boundary after it, the
        # in flight step was not interrupted, and no further
        # token was produced afterwards.
        assert result.output_length == 1
        assert executor.released_request_ids == ["victim"]
        assert not owner._engine.scheduler.has_pending_requests()
        # Disconnecting again, and disconnecting an ended
        # request through the owner, are both harmless.
        assert handle.disconnect() is False
        assert owner.disconnect("victim") is False
        with pytest.raises(KeyError):
            owner.disconnect("never-seen")
        # The service keeps serving after a disconnect.
        other = owner.submit(make_request("other", max_new=2))
        executor.finish_gracefully()
        assert other.result(timeout=5).state.name == "FINISHED"
    finally:
        executor.finish_gracefully()
        if thread.is_alive():
            thread.join(timeout=5)
        owner.stop(timeout=5)


def test_closing_the_stream_disconnects_the_request() -> None:
    # The slow executor stretches generation over about a
    # second (simulation timing), while the disconnect lands
    # at the next step boundary, about one delay later, so the
    # request provably cannot finish first.
    executor = SlowFakeExecutor(delay_seconds=0.05)
    owner = EngineOwner(
        Engine(Scheduler(max_batch_size=4, max_batch_tokens=100), executor)
    )
    owner.start()
    try:
        handle = owner.submit(make_request("gone", max_new=20))
        stream = handle.events()
        first = next(stream)
        assert first.type == "token"
        stream.close()
        result = handle.result(timeout=5)
        assert result.state.name == "CANCELLED"
        assert 1 <= result.output_length < 20
        assert handle.disconnected is True
        assert handle.disconnect_reason == "stream_closed"
        assert executor.released_request_ids == ["gone"]
        assert owner.admission_stats()["disconnect_cancellations"] == 1
    finally:
        owner.stop(timeout=5)
