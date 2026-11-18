from collections import deque
from dataclasses import dataclass

from miniserve.request import Request


class RequestRejectedError(ValueError):
    """Raised when a request can never be admitted by this scheduler."""


@dataclass
class SchedulerOutput:
    prefill_requests: list[Request]
    decode_requests: list[Request]

    @property
    def is_empty(self) -> bool:
        return not self.prefill_requests and not self.decode_requests


class Scheduler:
    def __init__(
        self,
        max_batch_size: int,
        max_batch_tokens: int,
        max_model_len: int | None = None,
        max_active_requests: int | None = None,
        max_reserved_tokens: int | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        if max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")

        if max_model_len is not None and max_model_len <= 0:
            raise ValueError("max_model_len must be positive")

        if max_active_requests is not None and max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")

        if max_reserved_tokens is not None and max_reserved_tokens <= 0:
            raise ValueError("max_reserved_tokens must be positive")

        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_model_len = max_model_len
        # Resource limits on resident requests, distinct from the
        # per step limits above. max_batch_size and
        # max_batch_tokens bound the work picked in one step.
        # max_active_requests and max_reserved_tokens bound what
        # may be resident at once: a request becomes active when
        # it is prefilled and starts holding a cache, and it
        # reserves prompt_length + max_new_tokens of cache
        # capacity for its whole active life, returned when it
        # finishes, fails, or is cancelled. A waiting request
        # holds no cache and reserves nothing yet. See
        # docs/resources.md for the full semantics and limits.
        self.max_active_requests = max_active_requests
        self.max_reserved_tokens = max_reserved_tokens

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        # Request ids are unique for the lifetime of this scheduler.
        # An id is recorded only when a request is actually admitted,
        # so an id used by a rejected offer stays available for a
        # corrected request.
        self._seen_request_ids: set[str] = set()

    def _prune_terminal(self) -> None:
        """Drop terminal requests from both queues.

        A caller can cancel a request object directly, without going
        through cancel_request, so terminal entries can appear in
        either queue between steps. Pruning here guarantees that a
        cancelled request is never handed to the executor, where
        starting prefill on it would be an error.
        """
        self.waiting = deque(
            request for request in self.waiting if not request.is_finished
        )
        self.running = [
            request for request in self.running if not request.is_finished
        ]

    @staticmethod
    def _reservation(request: Request) -> int:
        """Tokens of cache capacity one active request reserves.

        The reservation is deliberately conservative: the full
        prompt plus the full output limit, reserved up front for
        the request's whole active life, even though the cache
        only grows to that size if the request really generates
        up to its limit.
        """
        return request.prompt_length + request.max_new_tokens

    @property
    def active_request_count(self) -> int:
        """How many requests are currently active (running)."""
        return sum(
            1 for request in self.running if not request.is_finished
        )

    @property
    def reserved_tokens(self) -> int:
        """Cache capacity currently reserved by active requests.

        Computed from the running list, so a request that reached
        a terminal state stops counting even before the next
        prune: finishing or cancelling a request returns its
        reservation by construction, with no separate bookkeeping
        that could drift from the queues.
        """
        return sum(
            self._reservation(request)
            for request in self.running
            if not request.is_finished
        )

    def _find_active(self, request_id: str) -> Request | None:
        for request in list(self.waiting) + self.running:
            if request.request_id == request_id and not request.is_finished:
                return request
        return None

    def add_request(self, request: Request) -> None:
        if request.is_finished:
            raise ValueError("Cannot add a finished request")

        if request.request_id in self._seen_request_ids:
            raise RequestRejectedError(
                f"Request id {request.request_id} has already been "
                "used in this scheduler. Request ids are unique for "
                "the lifetime of a scheduler, so a duplicate offer "
                "never occupies a second queue slot."
            )

        if request.prompt_length > self.max_batch_tokens:
            raise RequestRejectedError(
                f"Request {request.request_id} has a prompt of "
                f"{request.prompt_length} tokens, which exceeds the "
                f"per step token budget of {self.max_batch_tokens}. "
                "Prefill is not chunked, so this request could never "
                "be scheduled."
            )

        if self.max_model_len is not None:
            reserved_length = request.prompt_length + request.max_new_tokens
            if reserved_length > self.max_model_len:
                raise RequestRejectedError(
                    f"Request {request.request_id} reserves "
                    f"{reserved_length} tokens (prompt "
                    f"{request.prompt_length} plus max_new_tokens "
                    f"{request.max_new_tokens}), which exceeds the "
                    f"model context limit of {self.max_model_len}."
                )

        if self.max_reserved_tokens is not None:
            reservation = self._reservation(request)
            if reservation > self.max_reserved_tokens:
                raise RequestRejectedError(
                    f"Request {request.request_id} reserves "
                    f"{reservation} tokens (prompt "
                    f"{request.prompt_length} plus max_new_tokens "
                    f"{request.max_new_tokens}), which exceeds the "
                    f"cache capacity limit of "
                    f"{self.max_reserved_tokens} on its own. It "
                    "could never become active, even in an empty "
                    "system, so it is rejected instead of waiting."
                )

        self._seen_request_ids.add(request.request_id)
        request.mark_admitted()
        self.waiting.append(request)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by id and remove it from its queue.

        This is the single cancellation entry point for callers that
        hold only an id. It returns True when an active request was
        cancelled. Cancelling again, or cancelling a request that
        already finished, is a harmless no-op that returns False and
        leaves the terminal state and the queues untouched. An id
        that this scheduler has never admitted raises KeyError, so a
        mistyped id is not silently ignored.
        """
        self._prune_terminal()

        request = self._find_active(request_id)
        if request is not None:
            request.cancel()
            self._prune_terminal()
            return True

        if request_id in self._seen_request_ids:
            return False

        raise KeyError(f"Unknown request id: {request_id}")

    def schedule(self) -> SchedulerOutput:
        self._prune_terminal()

        decode_requests: list[Request] = []
        used_tokens = 0

        for request in self.running:
            if len(decode_requests) >= self.max_batch_size:
                break

            decode_cost = 1

            if used_tokens + decode_cost > self.max_batch_tokens:
                break

            decode_requests.append(request)
            used_tokens += decode_cost

        remaining_slots = self.max_batch_size - len(decode_requests)
        prefill_requests: list[Request] = []

        while self.waiting and remaining_slots > 0:
            request = self.waiting[0]
            prefill_cost = request.prompt_length

            if used_tokens + prefill_cost > self.max_batch_tokens:
                break

            # Resource limits: the head request stays waiting
            # while the system is at its active request limit or
            # while its reservation would push the resident total
            # over the cache capacity. This is a temporary wait,
            # not a rejection: the reservation fits on its own
            # (admission checked that), so it runs once a running
            # request finishes or is cancelled and returns its
            # reservation. FIFO head of line blocking applies
            # here exactly as it does to the token budget.
            if self.max_active_requests is not None:
                active_after = len(self.running) + len(prefill_requests)
                if active_after >= self.max_active_requests:
                    break

            if self.max_reserved_tokens is not None:
                reserved = self.reserved_tokens + sum(
                    self._reservation(selected)
                    for selected in prefill_requests
                )
                if reserved + self._reservation(request) > (
                    self.max_reserved_tokens
                ):
                    break

            self.waiting.popleft()
            prefill_requests.append(request)

            used_tokens += prefill_cost
            remaining_slots -= 1

        return SchedulerOutput(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
        )

    def update(self, output: SchedulerOutput) -> None:
        for request in output.prefill_requests:
            if not request.is_finished:
                self.running.append(request)

        self._prune_terminal()

    def has_pending_requests(self) -> bool:
        self._prune_terminal()
        return bool(self.waiting or self.running)


class SerialScheduler(Scheduler):
    """One request at a time baseline scheduler.

    This is the fair comparison baseline for the continuous
    scheduler. It reuses exactly the same admission, validation,
    cancellation, and duplicate id rules, the same Request type, and
    the same executor contract as Scheduler. The only difference is
    the selection policy in schedule: at most one request is ever
    active. A waiting request is selected for prefill only when no
    request is running, and while a request is running only that
    request is selected for decode. No new prefill joins mid decode,
    even if the batch size and token budget would allow it.

    Because selection is the only difference, running the same
    request set through both schedulers with the same executor must
    produce the same output lengths. Any difference in step counts
    or timing then comes from the scheduling policy alone.

    The resource limits need no extra handling here: the
    admission side rejection of a request whose reservation
    exceeds the cache capacity is inherited, and the one active
    request policy is stricter than any max_active_requests of
    one or more, while a single admitted request always fits the
    capacity on its own, so the waiting side checks can never
    block a serial prefill that the token budget allows.
    """

    def schedule(self) -> SchedulerOutput:
        self._prune_terminal()

        if self.running:
            request = self.running[0]
            decode_cost = 1
            if decode_cost > self.max_batch_tokens:
                return SchedulerOutput(
                    prefill_requests=[],
                    decode_requests=[],
                )
            return SchedulerOutput(
                prefill_requests=[],
                decode_requests=[request],
            )

        if self.waiting:
            request = self.waiting[0]
            prefill_cost = request.prompt_length
            if prefill_cost > self.max_batch_tokens:
                return SchedulerOutput(
                    prefill_requests=[],
                    decode_requests=[],
                )
            self.waiting.popleft()
            return SchedulerOutput(
                prefill_requests=[request],
                decode_requests=[],
            )

        return SchedulerOutput(
            prefill_requests=[],
            decode_requests=[],
        )


def create_scheduler(
    policy: str,
    max_batch_size: int,
    max_batch_tokens: int,
    max_model_len: int | None = None,
    max_active_requests: int | None = None,
    max_reserved_tokens: int | None = None,
) -> Scheduler:
    """Create a scheduler by policy name.

    This is the single configuration switch between the two
    scheduling strategies. The policy value continuous returns the
    default Scheduler, serial returns SerialScheduler, and any
    other value raises ValueError so a misspelled policy is never
    silently treated as continuous. The resource
    limit arguments pass through to either policy unchanged.
    """
    if policy == "continuous":
        return Scheduler(
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            max_model_len=max_model_len,
            max_active_requests=max_active_requests,
            max_reserved_tokens=max_reserved_tokens,
        )
    if policy == "serial":
        return SerialScheduler(
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            max_model_len=max_model_len,
            max_active_requests=max_active_requests,
            max_reserved_tokens=max_reserved_tokens,
        )
    raise ValueError(
        f"Unknown scheduler policy {policy!r}. "
        "Expected 'continuous' or 'serial'."
    )
