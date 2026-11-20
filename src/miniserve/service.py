"""Background execution with a single engine owner.

This module implements the first two staged steps of the
service contract in ``docs/service.md``: background execution
behind one engine owner, and the streaming event transport.
The HTTP layer itself is not part of this module; whatever
front end is added later talks to the owner exactly the way
the tests and demos here do, by submitting requests and
reading handles, and it never touches the scheduler, the
engine, or an executor directly.

Ownership rules:

- Exactly one worker thread owns the engine. It is the only
  thread that calls scheduler or engine methods, so scheduler
  state is never modified concurrently by several handlers.
  The executor therefore also only ever runs on that thread,
  which is what single model execution ownership means here.
- Callers on any other thread hand the worker commands
  (admit this request, cancel this id, shut down) through one
  thread safe queue, and the worker answers each command on a
  per command reply queue. Submitting a request blocks the
  caller only until admission is decided, not until generation
  finishes, so a second request can be received and admitted
  while a first one is still generating.
- When nothing is pending, the worker blocks on the command
  queue. It does not poll, sleep, or spin, so an idle service
  costs no busy waiting. While requests are pending it drains
  any queued commands between steps, then runs one engine
  step, so commands are never starved by generation and
  cancellation still takes effect at a scheduling boundary.
- After every step the worker diffs each active request's
  generated token list against what it already emitted, and
  pushes one token event per new token to that request's
  handle, in order, before the next step runs. A stream reader
  therefore sees the first token event while generation is
  still in progress, not in a batch at the end. Each handle
  receives exactly one terminal event, as the contract
  requires, and no event after it.
- An engine step that raises does not kill the worker. The
  engine has already cleaned up the failed step; the worker
  finalises any request that reached a terminal state (a
  FAILED request gets an ``error`` terminal event and its
  handle raises :class:`GenerationFailedError` from
  ``result``), and keeps serving the requests that are still
  pending, for example a waiting request behind the failing
  one. The original failure text is kept on the request as
  its failure reason and carried in the error event, so no
  outcome is hidden or invented.
- Shutdown is a command like any other. Once it is received
  the owner stops accepting new requests, finishes the
  requests already pending, and then the worker exits.
  Submitting after shutdown raises ``RuntimeError`` instead
  of queueing work nobody will run.
"""

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Self

from miniserve.engine import Engine
from miniserve.request import Request, RequestState
from miniserve.streaming import IncrementalDecoder, StreamEvent


class GenerationFailedError(RuntimeError):
    """Raised by a handle whose request ended in FAILED."""

    def __init__(self, request_id: str, reason: str) -> None:
        super().__init__(f"Request {request_id} failed: {reason}")
        self.request_id = request_id
        self.reason = reason


class RequestHandle:
    """The caller side view of one admitted request.

    The handle owns the request's event queue and completion
    signal. It is created by the submitting caller, registered
    by the worker on admission, and filled by the worker alone
    after that. Reading it from any thread is safe.
    """

    def __init__(self, request: Request, tokenizer: Any | None) -> None:
        self.request = request
        self._events: queue.Queue[StreamEvent] = queue.Queue()
        self._done = threading.Event()
        self._error: GenerationFailedError | None = None
        self._decoder = IncrementalDecoder(tokenizer)
        self._emitted_count = 0
        self._stream_text = ""
        self._terminal_sent = False

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def stream_text(self) -> str:
        """Text carried by the token events emitted so far."""
        return self._stream_text

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the terminal event; False on timeout."""
        return self._done.wait(timeout)

    def result(self, timeout: float | None = None) -> Request:
        """Block until the request ends and return the request.

        A finished or cancelled request is returned as it is, so
        the caller reads its state and finish reason from the
        request itself. A failed request raises
        :class:`GenerationFailedError`, mirroring how the engine
        propagates an executor failure to its caller instead of
        hiding it. A timeout raises ``TimeoutError`` and leaves
        the request running.
        """
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"Request {self.request_id} did not finish "
                f"within {timeout} seconds"
            )
        if self._error is not None:
            raise self._error
        return self.request

    def events(self) -> Iterator[StreamEvent]:
        """Yield this request's events in order, terminal last.

        Events are queued as they happen, so a reader that
        starts while generation is in progress receives token
        events as they are produced, and a reader that starts
        after the request ended still receives the full
        sequence in order. Iteration stops right after the
        single terminal event.
        """
        while True:
            event = self._events.get()
            yield event
            if event.is_terminal:
                return


@dataclass
class _Command:
    kind: str
    request: Request | None = None
    handle: RequestHandle | None = None
    request_id: str | None = None
    reply: queue.Queue[Any] = field(default_factory=queue.Queue)


class EngineOwner:
    """Run one engine on one worker thread behind a command queue."""

    def __init__(self, engine: Engine, tokenizer: Any | None = None) -> None:
        self._engine = engine
        self._tokenizer = tokenizer
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._accepting = False
        self._handles: dict[str, RequestHandle] = {}
        self.worker_thread_id: int | None = None
        self.last_step_error: BaseException | None = None

    # Caller side API. None of these methods touch the
    # scheduler, the engine, or the executor; they only hand a
    # command to the worker and wait for its answer.

    def start(self) -> None:
        if self._running:
            raise RuntimeError("Engine owner is already running")
        self._running = True
        self._accepting = True
        self._thread = threading.Thread(
            target=self._run,
            name="miniserve-engine-owner",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Shut the owner down and join the worker thread.

        Requests already pending are finished first; new
        submissions are refused from the moment shutdown is
        queued. Stopping an owner that is not running is a
        no-op.
        """
        if not self._running or self._thread is None:
            return
        self._accepting = False
        command = _Command(kind="shutdown")
        self._commands.put(command)
        command.reply.get()
        self._thread.join(timeout)
        self._running = False
        self._thread = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def submit(self, request: Request) -> RequestHandle:
        """Admit one request and return its handle at once.

        The caller blocks only until the worker has decided
        admission. A rejection (for example a prompt that can
        never run) raises the scheduler's own
        ``RequestRejectedError`` here, no handle is registered,
        and no request id is consumed beyond what the scheduler
        itself records. Submitting while another request is
        generating is the normal case, not an error.
        """
        if not self._running or not self._accepting:
            raise RuntimeError("Engine owner is not accepting requests")
        handle = RequestHandle(request, self._tokenizer)
        command = _Command(kind="submit", request=request, handle=handle)
        self._commands.put(command)
        ok, payload = command.reply.get()
        if not ok:
            raise payload
        return handle

    def cancel(self, request_id: str) -> bool:
        """Ask the worker to cancel one admitted request.

        Mirrors the scheduler cancellation contract: True when
        an active request was cancelled, False when it had
        already ended, and ``KeyError`` for an id the service
        never admitted. The cancellation itself happens on the
        worker thread, at a scheduling boundary.
        """
        if not self._running:
            raise RuntimeError("Engine owner is not running")
        command = _Command(kind="cancel", request_id=request_id)
        self._commands.put(command)
        ok, payload = command.reply.get()
        if not ok:
            raise payload
        return bool(payload)

    def health(self) -> dict[str, Any]:
        """Report liveness without running any generation work."""
        alive = self._thread is not None and self._thread.is_alive()
        return {
            "status": "ok" if alive and self._accepting else "unavailable",
            "accepting_requests": bool(alive and self._accepting),
            "worker_alive": alive,
            "active_requests": sum(
                1 for handle in self._handles.values() if not handle.done()
            ),
        }

    # Worker side. Everything below runs on the worker thread
    # only, which is the single ownership guarantee.

    def _run(self) -> None:
        self.worker_thread_id = threading.get_ident()
        shutdown_pending = False
        try:
            while True:
                if shutdown_pending and not self._has_pending():
                    break
                if not self._has_pending():
                    command = self._commands.get()
                    if self._process(command):
                        shutdown_pending = True
                    continue
                while True:
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if self._process(command):
                        shutdown_pending = True
                if shutdown_pending and not self._has_pending():
                    break
                try:
                    self._engine.step()
                except Exception as exc:  # noqa: BLE001
                    # Any step failure is reported through the
                    # affected handles, never hidden: the worker
                    # records it here and finalises the requests
                    # the engine marked terminal.
                    self.last_step_error = exc
                    self._collect_outcomes()
                else:
                    self._collect_outcomes()
        finally:
            self._accepting = False
            for handle in self._handles.values():
                if not handle.done():
                    handle._error = GenerationFailedError(
                        handle.request_id,
                        "engine owner stopped before the request finished",
                    )
                    self._send_terminal(handle)

    def _has_pending(self) -> bool:
        return self._engine.scheduler.has_pending_requests()

    def _process(self, command: _Command) -> bool:
        """Run one command; return True for a shutdown command."""
        if command.kind == "submit":
            assert command.request is not None
            assert command.handle is not None
            try:
                self._engine.scheduler.add_request(command.request)
            except Exception as exc:  # noqa: BLE001
                # The caller re raises whatever admission raised,
                # so the exception is propagated, not swallowed.
                command.reply.put((False, exc))
            else:
                self._handles[command.request.request_id] = command.handle
                command.reply.put((True, None))
            return False
        if command.kind == "cancel":
            assert command.request_id is not None
            try:
                result = self._engine.scheduler.cancel_request(
                    command.request_id
                )
            except Exception as exc:  # noqa: BLE001
                # Same propagation rule as admission: the caller
                # re raises it (KeyError for an unknown id).
                command.reply.put((False, exc))
            else:
                self._engine.release_terminal_requests()
                handle = self._handles.get(command.request_id)
                if handle is not None and handle.request.is_finished:
                    self._collect_handle(handle)
                command.reply.put((True, result))
            return False
        if command.kind == "shutdown":
            command.reply.put((True, None))
            return True
        command.reply.put((False, ValueError(f"Unknown command {command.kind}")))
        return False

    def _collect_outcomes(self) -> None:
        for handle in list(self._handles.values()):
            if not handle.done():
                self._collect_handle(handle)

    def _collect_handle(self, handle: RequestHandle) -> None:
        self._emit_new_tokens(handle)
        if handle.request.is_finished:
            self._send_terminal(handle)

    def _emit_new_tokens(self, handle: RequestHandle) -> None:
        request = handle.request
        while handle._emitted_count < len(request.generated_token_ids):
            index = handle._emitted_count
            token_id = request.generated_token_ids[index]
            text = handle._decoder.feed(token_id)
            handle._stream_text += text
            handle._events.put(
                StreamEvent(
                    type="token",
                    request_id=request.request_id,
                    index=index,
                    token_id=token_id,
                    text=text,
                )
            )
            handle._emitted_count += 1

    def _send_terminal(self, handle: RequestHandle) -> None:
        if handle._terminal_sent:
            return
        handle._terminal_sent = True
        request = handle.request
        remainder = handle._decoder.flush()
        if remainder and request.generated_token_ids:
            handle._stream_text += remainder
            handle._events.put(
                StreamEvent(
                    type="token",
                    request_id=request.request_id,
                    index=len(request.generated_token_ids) - 1,
                    token_id=request.generated_token_ids[-1],
                    text=remainder,
                )
            )
        if request.state is RequestState.FAILED:
            reason = request.failure_reason or "generation failed"
            if handle._error is None:
                handle._error = GenerationFailedError(request.request_id, reason)
            handle._events.put(
                StreamEvent(
                    type="error",
                    request_id=request.request_id,
                    error_code="generation_failed",
                    error_message=reason,
                    generated_tokens=len(request.generated_token_ids),
                )
            )
        else:
            handle._events.put(
                StreamEvent(
                    type="finished",
                    request_id=request.request_id,
                    finish_reason=request.finish_reason,
                    generated_tokens=len(request.generated_token_ids),
                )
            )
        handle._done.set()
