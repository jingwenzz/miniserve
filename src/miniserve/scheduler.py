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
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        if max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")

        if max_model_len is not None and max_model_len <= 0:
            raise ValueError("max_model_len must be positive")

        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_model_len = max_model_len

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
