from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from time import monotonic

Clock = Callable[[], float]


class RequestState(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]  # prefill
    max_new_tokens: int

    generated_token_ids: list[int] = field(default_factory=list)  # decode output
    state: RequestState = RequestState.WAITING

    arrival_time: float | None = None
    first_token_time: float | None = None
    finished_time: float | None = None
    admission_time: float | None = None
    prefill_start_time: float | None = None
    last_token_time: float | None = None
    failure_reason: str | None = None
    clock: Clock = field(default=monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError(
                "prompt_token_ids must not be empty: an empty prompt is "
                "rejected explicitly instead of being silently turned "
                "into a special token"
            )

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if self.arrival_time is None:
            self.arrival_time = self.clock()

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_length(self) -> int:
        return len(self.generated_token_ids)

    @property
    def total_length(self) -> int:
        return self.prompt_length + self.output_length

    @property
    def is_finished(self) -> bool:
        return self.state in {
            RequestState.FINISHED,
            RequestState.CANCELLED,
            RequestState.FAILED,
        }

    @property
    def finish_reason(self) -> str | None:
        if self.state is RequestState.FINISHED:
            return "finished"
        if self.state is RequestState.CANCELLED:
            return "cancelled"
        if self.state is RequestState.FAILED:
            return "failed"
        return None

    def mark_admitted(self) -> None:
        """Record admission into the scheduler waiting queue.

        Called by the scheduler exactly once, when it accepts the
        request. A second call does not overwrite the first time.
        """
        if self.admission_time is None:
            self.admission_time = self.clock()

    def start_prefill(self) -> None:
        if self.state is not RequestState.WAITING:
            raise RuntimeError(
                f"Cannot start prefill from state {self.state.name}"
            )

        if self.prefill_start_time is None:
            self.prefill_start_time = self.clock()

        self.state = RequestState.PREFILLING

    def finish_prefill(self) -> None:
        if self.state is not RequestState.PREFILLING:
            raise RuntimeError(
                f"Cannot finish prefill from state {self.state.name}"
            )

        self.state = RequestState.DECODING

    def append_token(self, token_id: int) -> None:
        if self.state is not RequestState.DECODING:
            raise RuntimeError(
                f"Cannot append token from state {self.state.name}"
            )

        now = self.clock()
        if self.first_token_time is None:
            self.first_token_time = now
        self.last_token_time = now

        self.generated_token_ids.append(token_id)

        if self.output_length >= self.max_new_tokens:
            self.finish()

    def finish(self) -> None:
        if self.state is not RequestState.DECODING:
            raise RuntimeError(
                f"Cannot finish request from state {self.state.name}"
            )

        self.state = RequestState.FINISHED
        self.finished_time = self.clock()

    def cancel(self) -> None:
        if self.is_finished:
            raise RuntimeError(
                f"Cannot cancel request from state {self.state.name}"
            )

        self.state = RequestState.CANCELLED
        self.finished_time = self.clock()

    def fail(self, reason: str) -> None:
        """Move a non terminal request to FAILED with a recorded reason.

        This is the terminal outcome for an execution error after
        admission. It is distinct from FINISHED and CANCELLED, it
        keeps any tokens already produced, and it records the
        reason text the executor supplied so a failure can be
        reported instead of guessed at later. Failing a request
        that already reached a terminal state is an error, so a
        late failure can never overwrite a real outcome.
        """
        if self.is_finished:
            raise RuntimeError(
                f"Cannot fail request from state {self.state.name}"
            )
        if not reason:
            raise ValueError("failure reason must not be empty")

        self.state = RequestState.FAILED
        self.failure_reason = reason
        self.finished_time = self.clock()
