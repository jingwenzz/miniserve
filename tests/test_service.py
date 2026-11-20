"""Tests for background execution behind a single engine owner.

The owner is exercised with the fake executor, so these tests
need no model download. Concurrency is made deterministic with
a step gated executor: every decode call signals that it has
been entered and then waits for a release permit from the
test. While a decode is gated, the worker thread is provably
inside a generation, and a command the test hands to the owner
in that window can only sit in the command queue until the
current step ends, which is exactly the scheduling boundary
semantics the contract states. The test observes the queue,
then releases the step, and the queued command is answered.
"""

import queue
import threading
import time

import pytest

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request
from miniserve.scheduler import RequestRejectedError, Scheduler
from miniserve.service import EngineOwner, GenerationFailedError


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
        self.executor_thread_ids: set[int] = set()

    def prefill(self, requests: list[Request]) -> None:
        self.executor_thread_ids.add(threading.get_ident())
        super().prefill(requests)

    def decode(self, requests: list[Request]) -> None:
        self.executor_thread_ids.add(threading.get_ident())
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
        """Let the currently gated decode and all later ones run."""
        self.gating = False
        self.release_permits(1)


class RecordingExecutor(FakeExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.executor_thread_ids: set[int] = set()

    def prefill(self, requests: list[Request]) -> None:
        self.executor_thread_ids.add(threading.get_ident())
        super().prefill(requests)

    def decode(self, requests: list[Request]) -> None:
        self.executor_thread_ids.add(threading.get_ident())
        super().decode(requests)


class FailingExecutor(FakeExecutor):
    """Fake executor that fails one named request in decode."""

    def __init__(self, fail_request_id: str) -> None:
        super().__init__()
        self.fail_request_id = fail_request_id

    def decode(self, requests: list[Request]) -> None:
        for request in requests:
            if request.request_id == self.fail_request_id:
                request.fail("decode forward failed: RuntimeError: injected")
                raise RuntimeError("injected")
        super().decode(requests)


def make_owner(
    executor: FakeExecutor | None = None,
    *,
    max_batch_size: int = 4,
    max_batch_tokens: int = 100,
) -> EngineOwner:
    scheduler = Scheduler(
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    engine = Engine(scheduler, executor or FakeExecutor())
    return EngineOwner(engine)


def wait_for_queue_size(owner: EngineOwner, size: int) -> None:
    deadline = time.monotonic() + 5
    while owner._commands.qsize() < size:
        assert time.monotonic() < deadline, "command was not queued in time"
        time.sleep(0.001)


def test_submit_returns_before_generation_finishes_and_completes() -> None:
    executor = StepGatedExecutor()
    owner = make_owner(executor)
    owner.start()
    try:
        handle = owner.submit(make_request("a", max_new=3))
        executor.wait_entered(1)
        # The submit call already returned, on the main thread,
        # while the worker is still inside the first decode.
        assert not handle.done()
        executor.finish_gracefully()
        result = handle.result(timeout=5)
        assert result.state.name == "FINISHED"
        assert result.output_length == 3
        assert executor.released_request_ids == ["a"]
    finally:
        executor.finish_gracefully()
        owner.stop(timeout=5)


def test_second_request_is_received_while_first_is_generating() -> None:
    executor = StepGatedExecutor()
    owner = make_owner(executor)
    owner.start()
    submitted: dict[str, object] = {}

    def submit_second() -> None:
        submitted["handle"] = owner.submit(make_request("second", max_new=2))

    thread = threading.Thread(target=submit_second)
    try:
        first = owner.submit(make_request("first", max_new=4))
        executor.wait_entered(1)
        thread.start()
        # The second submission reached the owner's command
        # queue while the first request was still gated inside
        # its first decode: the service received it during the
        # generation instead of refusing or serialising callers
        # outside the service.
        wait_for_queue_size(owner, 1)
        assert not first.done()
        executor.finish_gracefully()
        thread.join(timeout=5)
        second = submitted["handle"]
        assert isinstance(second, type(first))
        assert first.result(timeout=5).output_length == 4
        assert second.result(timeout=5).output_length == 2
    finally:
        executor.finish_gracefully()
        if thread.is_alive():
            thread.join(timeout=5)
        owner.stop(timeout=5)


def test_only_the_worker_thread_runs_the_executor() -> None:
    executor = RecordingExecutor()
    owner = make_owner(executor)
    owner.start()
    try:
        handle = owner.submit(make_request("solo", max_new=2))
        assert handle.result(timeout=5).state.name == "FINISHED"
        assert owner.worker_thread_id is not None
        assert executor.executor_thread_ids == {owner.worker_thread_id}
        assert threading.get_ident() != owner.worker_thread_id
    finally:
        owner.stop(timeout=5)


def test_idle_owner_does_not_busy_wait() -> None:
    owner = make_owner()
    owner.start()
    try:
        time.sleep(0.05)
        health = owner.health()
        assert health["status"] == "ok"
        assert health["accepting_requests"] is True
        assert health["active_requests"] == 0
        # No step ran while idle: the worker was blocked on the
        # command queue, not spinning through empty engine steps.
        assert owner._engine.step_count == 0
    finally:
        owner.stop(timeout=5)
    assert owner.health()["status"] == "unavailable"


def test_cancel_waiting_request_through_owner() -> None:
    executor = StepGatedExecutor()
    owner = make_owner(executor, max_batch_size=1)
    owner.start()
    submitted: dict[str, object] = {}
    cancelled: dict[str, object] = {}

    def submit_waiting() -> None:
        submitted["handle"] = owner.submit(make_request("waiting", max_new=2))

    def cancel_waiting() -> None:
        cancelled["result"] = owner.cancel("waiting")

    submit_thread = threading.Thread(target=submit_waiting)
    cancel_thread = threading.Thread(target=cancel_waiting)
    try:
        running = owner.submit(make_request("running", max_new=5))
        executor.wait_entered(1)
        submit_thread.start()
        wait_for_queue_size(owner, 1)
        executor.release_permits(1)
        submit_thread.join(timeout=5)
        waiting = submitted["handle"]
        assert isinstance(waiting, type(running))
        # The running request is now gated inside its second
        # decode, and the waiting request is queued behind it
        # because the batch size is one.
        executor.wait_entered(2)
        cancel_thread.start()
        wait_for_queue_size(owner, 1)
        executor.release_permits(1)
        cancel_thread.join(timeout=5)
        assert cancelled["result"] is True
        result = waiting.result(timeout=5)
        assert result.state.name == "CANCELLED"
        assert result.output_length == 0
        executor.finish_gracefully()
        assert running.result(timeout=5).state.name == "FINISHED"
        assert owner.cancel("waiting") is False
        with pytest.raises(KeyError):
            owner.cancel("never-seen")
    finally:
        executor.finish_gracefully()
        for thread in (submit_thread, cancel_thread):
            if thread.is_alive():
                thread.join(timeout=5)
        owner.stop(timeout=5)


def test_cancel_running_request_stops_it_at_a_boundary() -> None:
    executor = StepGatedExecutor()
    owner = make_owner(executor)
    owner.start()
    cancelled: dict[str, object] = {}

    def cancel_victim() -> None:
        cancelled["result"] = owner.cancel("victim")

    thread = threading.Thread(target=cancel_victim)
    try:
        handle = owner.submit(make_request("victim", max_new=6))
        executor.wait_entered(1)
        thread.start()
        wait_for_queue_size(owner, 1)
        executor.release_permits(1)
        thread.join(timeout=5)
        assert cancelled["result"] is True
        result = handle.result(timeout=5)
        assert result.state.name == "CANCELLED"
        # Exactly the one token from the completed step exists:
        # the cancel took effect at the boundary after it, and
        # no further token was produced.
        assert result.output_length == 1
    finally:
        executor.finish_gracefully()
        if thread.is_alive():
            thread.join(timeout=5)
        owner.stop(timeout=5)


def test_rejected_submission_propagates_and_consumes_no_handle() -> None:
    owner = make_owner(max_batch_tokens=10)
    owner.start()
    try:
        with pytest.raises(RequestRejectedError):
            owner.submit(make_request("too-long", prompt_len=11))
        # The same id stays usable, exactly as the scheduler
        # contract promises for a rejected offer.
        handle = owner.submit(make_request("too-long", prompt_len=3, max_new=1))
        assert handle.result(timeout=5).state.name == "FINISHED"
    finally:
        owner.stop(timeout=5)


def test_executor_failure_reaches_the_handle_and_worker_survives() -> None:
    executor = FailingExecutor("bad")
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    owner = EngineOwner(engine)
    owner.start()
    try:
        bad = owner.submit(make_request("bad", max_new=3))
        good = owner.submit(make_request("good", max_new=2))
        with pytest.raises(GenerationFailedError) as excinfo:
            bad.result(timeout=5)
        assert "injected" in excinfo.value.reason
        assert bad.request.state.name == "FAILED"
        assert bad.request.failure_reason is not None
        assert "injected" in bad.request.failure_reason
        # The waiting request behind the failing one still runs.
        assert good.result(timeout=5).state.name == "FINISHED"
        assert owner.health()["worker_alive"] is True
    finally:
        owner.stop(timeout=5)


def test_submit_after_shutdown_is_refused() -> None:
    owner = make_owner()
    owner.start()
    handle = owner.submit(make_request("before", max_new=1))
    assert handle.result(timeout=5).state.name == "FINISHED"
    owner.stop(timeout=5)
    with pytest.raises(RuntimeError):
        owner.submit(make_request("after", max_new=1))
