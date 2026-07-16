from collections import deque
from dataclasses import dataclass

from miniserve.request import Request


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
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        if max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")

        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        if request.is_finished:
            raise ValueError("Cannot add a finished request")

        self.waiting.append(request)

    def schedule(self) -> SchedulerOutput:
        self.running = [
            request
            for request in self.running
            if not request.is_finished
        ]

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

        self.running = [
            request
            for request in self.running
            if not request.is_finished
        ]

    def has_pending_requests(self) -> bool:
        return bool(self.waiting or self.running)