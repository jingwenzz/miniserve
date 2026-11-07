from typing import Protocol

from miniserve.request import Request, RequestState

# prefill:
# WAITING → PREFILLING → DECODING
#
# decode:
# 每次调用，为每个 request 追加一个 token
#
# Executor contract (see docs/design.md section 8):
# - The executor owns the request state transitions that happen during
#   execution: prefill moves a request from WAITING through PREFILLING
#   to DECODING, and decode appends exactly one token per request.
# - The executor owns any per request backend resources it creates
#   during prefill, such as a KV cache for a real model. Those
#   resources are freed only through release, never by the scheduler.
# - The engine owns release timing: it calls release exactly once for
#   each request it tracked, after that request reaches a terminal
#   state. The executor must make release idempotent.
# - If prefill or decode raises, the exception propagates to the
#   engine caller. The executor leaves the request in whatever state
#   it reached; the engine does not invent a terminal state for it.


class Executor(Protocol):
    """The minimal replaceable boundary between engine and backend.

    Any object with these three methods can drive the engine. The
    scheduler never talks to an executor directly, so swapping the
    fake backend for a real model backend does not change scheduling.
    """

    def prefill(self, requests: list[Request]) -> None:
        """Process the prompts of the given WAITING requests.

        On success each request has moved to DECODING and any per
        request backend resources for it have been created.
        """
        ...

    def decode(self, requests: list[Request]) -> None:
        """Append exactly one new token to each DECODING request."""
        ...

    def release(self, request: Request) -> None:
        """Free the per request backend resources for one request.

        Called by the engine exactly once per tracked request, after
        the request reaches a terminal state. Implementations must
        tolerate being called again for the same request without
        error and without freeing resources that no longer belong to
        that request.
        """
        ...


class FakeExecutor:
    """Simulates prefill and decode without running a real model.

    This backend holds no per request resources, so release has
    nothing to free. It records the ids it was asked to release, in
    order, so tests and diagnostics can check that the engine drives
    the release part of the contract correctly.
    """

    def __init__(self, generated_token_id: int = 1) -> None:
        self.generated_token_id = generated_token_id
        self.released_request_ids: list[str] = []

    def prefill(self, requests: list[Request]) -> None:
        for request in requests:
            request.start_prefill()

        # A real executor would run the prompt through all Transformer layers
        # and initialize the KV cache here.

        for request in requests:
            request.finish_prefill()

    def decode(self, requests: list[Request]) -> None:
        for request in requests:
            if request.state is not RequestState.DECODING:
                raise RuntimeError(
                    f"Request {request.request_id} is not ready for decode"
                )

            request.append_token(self.generated_token_id)

    def release(self, request: Request) -> None:
        if request.request_id not in self.released_request_ids:
            self.released_request_ids.append(request.request_id)
