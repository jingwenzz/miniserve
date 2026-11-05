from dataclasses import dataclass, field
from enum import Enum, auto
from time import monotonic


class RequestState(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()
    CANCELLED = auto()


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int] # prefill
    max_new_tokens: int

    generated_token_ids: list[int] = field(default_factory=list) # decode outpu
    state: RequestState = RequestState.WAITING

    arrival_time: float = field(default_factory=monotonic)
    first_token_time: float | None = None
    finished_time: float | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError(
                "prompt_token_ids must not be empty: an empty prompt is "
                "rejected explicitly instead of being silently turned "
                "into a special token"
            )

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

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
        }

    def start_prefill(self) -> None:
        if self.state is not RequestState.WAITING:
            raise RuntimeError(
                f"Cannot start prefill from state {self.state.name}"
            )

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

        if self.first_token_time is None:
            self.first_token_time = monotonic()

        self.generated_token_ids.append(token_id)

        if self.output_length >= self.max_new_tokens:
            self.finish()


    def finish(self) -> None:
        if self.state is not RequestState.DECODING:
            raise RuntimeError(
                f"Cannot finish request from state {self.state.name}"
            )

        self.state = RequestState.FINISHED
        self.finished_time = monotonic()


    def cancel(self) -> None:
        if self.is_finished:
            raise RuntimeError(
                f"Cannot cancel request from state {self.state.name}"
            )

        self.state = RequestState.CANCELLED
        self.finished_time = monotonic()    
