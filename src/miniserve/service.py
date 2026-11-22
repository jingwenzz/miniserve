"""Background execution with a single engine owner.

This module implements the first four staged steps of the
service contract in ``docs/service.md``: background execution
behind one engine owner, the streaming event transport,
client disconnect mapped to cancellation, and the overload
policy (a bounded waiting queue and a bounded per request
event buffer with a slow consumer rule). The HTTP layer
lives in ``miniserve.server`` and talks to the owner exactly
the way the tests and demos here do, by submitting requests
and reading handles; it never touches the scheduler, the
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
- A client disconnect is a cancellation with a recorded
  reason. Closing a request's event stream, or calling
  ``disconnect`` on its handle or on the owner, hands the
  worker a disconnect command; the worker cancels the
  request through the same scheduler entry point as an
  explicit cancel, so it takes effect at the next scheduling
  boundary, never inside a forward that is already running,
  and the engine releases the request's backend resources
  exactly as for any cancelled request. A request that was
  still waiting was never prefilled, holds no cache, and
  simply leaves the queue. Either way the request stops
  producing tokens, its stream ends with the single terminal
  event, and every other request is unaffected.
- Overload is answered, never silently absorbed. Admission
  counts the scheduler's waiting queue: when it is already
  at the configured bound, a new submission is refused at
  once with :class:`QueueFullError` (the ``queue_full``
  error of the contract), the request is not queued, its id
  is not consumed, and the refusal is counted in the owner's
  admission statistics so a rejection rate can be reported
  next to the accepted requests' numbers. Running requests
  do not count toward the waiting bound.
- Each handle's event queue is bounded, and the worker never
  blocks on it: events are enqueued without waiting, so one
  slow reader can never stall the engine for everyone else.
  When a token event does not fit, that reader is a slow
  consumer: the event is dropped and counted on the handle,
  and the worker cancels the request at that same boundary,
  because a client that is not reading will not read the
  rest either. The terminal event is always delivered: if
  the buffer is full when it is time to send it, the oldest
  buffered token events are dropped (and counted) to make
  room, so a reader that resumes always learns how the
  request ended instead of waiting forever.
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


class QueueFullError(RuntimeError):
    """Raised when a submission finds the waiting queue full.

    This is the ``queue_full`` error of the service contract:
    the request was not admitted, was not queued, and consumed
    no request id, so the caller may retry later, possibly
    with the same id. It carries the bound and the waiting
    count observed at the decision, both made on the worker
    thread, so a report never has to guess either number.
    """

    def __init__(
        self, request_id: str, waiting_limit: int, waiting_count: int
    ) -> None:
        super().__init__(
            f"Request {request_id} was not admitted: the waiting "
            f"queue is full ({waiting_count} waiting, limit "
            f"{waiting_limit}). Retry later."
        )
        self.request_id = request_id
        self.waiting_limit = waiting_limit
        self.waiting_count = waiting_count
        self.error_code = "queue_full"


class RequestHandle:
    """The caller side view of one admitted request.

    The handle owns the request's event queue and completion
    signal. It is created by the submitting caller, registered
    by the worker on admission, and filled by the worker alone
    after that. Reading it from any thread is safe.
    """

    def __init__(
        self,
        request: Request,
        tokenizer: Any | None,
        max_event_buffer: int | None = None,
        on_disconnect: Any | None = None,
    ) -> None:
        self.request = request
        # The event queue is bounded when a limit is configured:
        # a reader that falls behind fills only its own buffer,
        # never the worker. A limit of None keeps the queue
        # unbounded, which only the owner default avoids.
        self._events: queue.Queue[StreamEvent] = queue.Queue(
            maxsize=max_event_buffer or 0
        )
        self._event_buffer_limit = max_event_buffer
        self._on_disconnect = on_disconnect
        self._done = threading.Event()
        self._error: GenerationFailedError | None = None
        self._decoder = IncrementalDecoder(tokenizer)
        self._emitted_count = 0
        self._stream_text = ""
        self._terminal_sent = False
        # Disconnect and slow consumer bookkeeping. Both are
        # plain records of what already happened, written by
        # whichever side observed it; nothing is inferred later.
        self.disconnected = False
        self.disconnect_reason: str | None = None
        self.slow_consumer = False
        self.dropped_events = 0

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def stream_text(self) -> str:
        """Text carried by the token events emitted so far."""
        return self._stream_text

    @property
    def event_buffer_limit(self) -> int | None:
        """The configured bound on this handle's event queue."""
        return self._event_buffer_limit

    @property
    def buffered_events(self) -> int:
        """How many events are currently waiting to be read."""
        return self._events.qsize()

    def disconnect(self, reason: str = "client_disconnected") -> bool:
        """Report that this request's client went away.

        The disconnect is handed to the engine owner, which
        cancels the request at the next scheduling boundary,
        exactly like an explicit cancel. Returns True when an
        active request was cancelled, and False when the
        request had already ended or no owner is attached.
        Calling it more than once is safe: only the first call
        is forwarded, later calls just return False.
        """
        if self.disconnected:
            return False
        self.disconnected = True
        self.disconnect_reason = reason
        if self._done.is_set() or self._on_disconnect is None:
            return False
        try:
            return bool(self._on_disconnect(self.request_id))
        except Exception:  # noqa: BLE001
            # The owner may already be stopped; the disconnect
            # itself is recorded above either way, and a gone
            # client has nobody left to report an error to.
            return False

    # Worker side event delivery. These two methods are only
    # called on the worker thread, and neither ever blocks:
    # a full buffer is a slow consumer signal, not a reason
    # to stall the engine.

    def _enqueue_event(self, event: StreamEvent) -> bool:
        """Queue one non terminal event without blocking.

        Returns False when the buffer was full: the event is
        dropped, counted, and the handle is marked as a slow
        consumer, so the worker can cancel the request instead
        of producing events nobody is reading.
        """
        try:
            self._events.put_nowait(event)
        except queue.Full:
            self.dropped_events += 1
            self.slow_consumer = True
            return False
        return True

    def _enqueue_terminal(self, event: StreamEvent) -> None:
        """Queue the terminal event, dropping oldest if needed.

        The terminal event is the one event that must always
        arrive: if the buffer is full, the oldest buffered
        token events are dropped (and counted) until it fits,
        so a reader that resumes learns the outcome instead of
        blocking on a stream that will never end.
        """
        while True:
            try:
                self._events.put_nowait(event)
                return
            except queue.Full:
                try:
                    self._events.get_nowait()
                except queue.Empty:
                    return
                self.dropped_events += 1
                self.slow_consumer = True

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

        Closing the iterator early is the client disconnect
        signal available at this layer: it is treated exactly
        like a transport that went away mid stream, and the
        request is cancelled through the owner. A front end
        whose transport detects a disconnect on its own calls
        ``disconnect`` (or the owner's) directly instead.
        """
        try:
            while True:
                event = self._events.get()
                yield event
                if event.is_terminal:
                    return
        except GeneratorExit:
            if not self.done():
                self.disconnect(reason="stream_closed")
            raise


@dataclass
class _Command:
    kind: str
    request: Request | None = None
    handle: RequestHandle | None = None
    request_id: str | None = None
    reply: queue.Queue[Any] = field(default_factory=queue.Queue)


class EngineOwner:
    """Run one engine on one worker thread behind a command queue."""

    def __init__(
        self,
        engine: Engine,
        tokenizer: Any | None = None,
        max_waiting_requests: int | None = 32,
        max_event_buffer: int | None = 128,
    ) -> None:
        if max_waiting_requests is not None and max_waiting_requests <= 0:
            raise ValueError("max_waiting_requests must be positive")
        if max_event_buffer is not None and max_event_buffer <= 0:
            raise ValueError("max_event_buffer must be positive")
        self._engine = engine
        self._tokenizer = tokenizer
        # Overload configuration. The waiting bound counts only
        # requests sitting in the scheduler's waiting queue, not
        # running ones; the event buffer bound is per handle.
        # Either may be set to None to disable that bound, which
        # the defaults deliberately do not do.
        self._max_waiting_requests = max_waiting_requests
        self._max_event_buffer = max_event_buffer
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._accepting = False
        self._handles: dict[str, RequestHandle] = {}
        self.worker_thread_id: int | None = None
        self.last_step_error: BaseException | None = None
        # Admission and cancellation statistics. The admission
        # counters are only ever written on the worker thread,
        # where the decisions are made; the cancellation
        # counters likewise. Reading them is a plain snapshot.
        self._offered = 0
        self._accepted = 0
        self._rejected_queue_full = 0
        self._rejected_admission = 0
        self._disconnect_cancellations = 0
        self._slow_consumer_cancellations = 0

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
        handle = RequestHandle(
            request,
            self._tokenizer,
            max_event_buffer=self._max_event_buffer,
            on_disconnect=self._disconnect_from_handle,
        )
        command = _Command(kind="submit", request=request, handle=handle)
        self._commands.put(command)
        ok, payload = command.reply.get()
        if not ok:
            raise payload
        return handle

    def _disconnect_from_handle(self, request_id: str) -> bool:
        """Owner entry point a handle's disconnect forwards to."""
        return self.disconnect(request_id)

    def disconnect(self, request_id: str) -> bool:
        """Cancel one request because its client went away.

        Same outcome contract as ``cancel``: True when an
        active waiting or running request was cancelled, False
        when it had already ended, and ``KeyError`` for an id
        the service never admitted. The only difference is
        bookkeeping: the handle is marked as disconnected and
        the cancellation is counted as a disconnect, so reports
        can tell a client that left apart from a client that
        asked. The cancellation itself is the scheduler's, at
        a scheduling boundary, with the usual release.
        """
        if not self._running:
            raise RuntimeError("Engine owner is not running")
        command = _Command(kind="disconnect", request_id=request_id)
        self._commands.put(command)
        ok, payload = command.reply.get()
        if not ok:
            raise payload
        return bool(payload)

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
            "waiting_requests": len(self._engine.scheduler.waiting),
            "max_waiting_requests": self._max_waiting_requests,
            "max_event_buffer": self._max_event_buffer,
        }

    def admission_stats(self) -> dict[str, Any]:
        """Report admission outcomes, including the refusals.

        Every submission the worker decided on is counted
        exactly once: accepted, refused because the waiting
        queue was full, or refused by scheduler admission.
        The queue full rejection rate is reported alongside
        the counts, because a report that only shows the
        accepted requests' behaviour hides the overload.
        """
        return {
            "offered": self._offered,
            "accepted": self._accepted,
            "rejected_queue_full": self._rejected_queue_full,
            "rejected_admission": self._rejected_admission,
            "queue_full_rejection_rate": (
                self._rejected_queue_full / self._offered
                if self._offered
                else 0.0
            ),
            "disconnect_cancellations": self._disconnect_cancellations,
            "slow_consumer_cancellations": (
                self._slow_consumer_cancellations
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
            self._offered += 1
            # The overload check runs before scheduler admission,
            # on this worker thread, where the waiting count is
            # exact. A queue full refusal never reaches the
            # scheduler, so it consumes no id and queues nothing.
            if self._max_waiting_requests is not None:
                waiting_count = len(self._engine.scheduler.waiting)
                if waiting_count >= self._max_waiting_requests:
                    self._rejected_queue_full += 1
                    command.reply.put(
                        (
                            False,
                            QueueFullError(
                                command.request.request_id,
                                self._max_waiting_requests,
                                waiting_count,
                            ),
                        )
                    )
                    return False
            try:
                self._engine.scheduler.add_request(command.request)
            except Exception as exc:  # noqa: BLE001
                # The caller re raises whatever admission raised,
                # so the exception is propagated, not swallowed.
                self._rejected_admission += 1
                command.reply.put((False, exc))
            else:
                self._accepted += 1
                self._handles[command.request.request_id] = command.handle
                command.reply.put((True, None))
            return False
        if command.kind in {"cancel", "disconnect"}:
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
                if command.kind == "disconnect":
                    handle = self._handles.get(command.request_id)
                    if handle is not None and not handle.disconnected:
                        handle.disconnected = True
                        handle.disconnect_reason = "client_disconnected"
                    if result:
                        self._disconnect_cancellations += 1
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
        overflowed = self._emit_new_tokens(handle)
        if overflowed and not handle.request.is_finished:
            # Slow consumer rule: a reader whose buffer filled
            # is not going to read the rest of this stream, so
            # the request is cancelled at this boundary instead
            # of generating events nobody receives. The cancel
            # goes through the scheduler, so the engine release
            # below and the terminal event follow the ordinary
            # cancelled path, and no other request is touched.
            try:
                cancelled = self._engine.scheduler.cancel_request(
                    handle.request_id
                )
            except KeyError:
                cancelled = False
            if cancelled:
                self._slow_consumer_cancellations += 1
            self._engine.release_terminal_requests()
        if handle.request.is_finished:
            self._send_terminal(handle)

    def _emit_new_tokens(self, handle: RequestHandle) -> bool:
        """Emit new token events; return True if any overflowed."""
        request = handle.request
        overflowed = False
        while handle._emitted_count < len(request.generated_token_ids):
            index = handle._emitted_count
            token_id = request.generated_token_ids[index]
            text = handle._decoder.feed(token_id)
            handle._stream_text += text
            if not handle._enqueue_event(
                StreamEvent(
                    type="token",
                    request_id=request.request_id,
                    index=index,
                    token_id=token_id,
                    text=text,
                )
            ):
                overflowed = True
            handle._emitted_count += 1
        return overflowed

    def _send_terminal(self, handle: RequestHandle) -> None:
        if handle._terminal_sent:
            return
        handle._terminal_sent = True
        request = handle.request
        remainder = handle._decoder.flush()
        if remainder and request.generated_token_ids:
            handle._stream_text += remainder
            handle._enqueue_event(
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
            handle._enqueue_terminal(
                StreamEvent(
                    type="error",
                    request_id=request.request_id,
                    error_code="generation_failed",
                    error_message=reason,
                    generated_tokens=len(request.generated_token_ids),
                )
            )
        else:
            handle._enqueue_terminal(
                StreamEvent(
                    type="finished",
                    request_id=request.request_id,
                    finish_reason=request.finish_reason,
                    generated_tokens=len(request.generated_token_ids),
                )
            )
        handle._done.set()
