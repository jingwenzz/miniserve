"""A real model executor for single requests, batch size one.

This backend drives the locked transformer model through the same
Executor protocol the fake backend implements, so the scheduler and
the engine do not change. Each forward still processes exactly one
request: prefill runs that request's whole prompt once, and decode
runs one step for one request. Supporting several requests in one
forward is separate later work, and nothing here claims it.

How the real token flow differs from the fake backend:

- The fake backend produces no token in prefill and one token per
  decode call. The real model produces its first token from the
  prefill logits, so prefill appends that token to the request
  before returning. Because the token is appended through
  ``Request.append_token``, the first token time is recorded at
  prefill completion, which is the convention ``docs/generation.md``
  settled for a real backend.
- Decode therefore never produces the first token. It feeds the
  most recent generated token, which is the only token the cache
  has not seen yet, and appends the token that forward produces.
  A request that needs N tokens takes one prefill forward and
  N - 1 decode forwards, and ``max_new_tokens`` of 1 takes no
  decode forward at all.
- A produced EOS token counts as a generated token and finishes
  the request. The generic ``Request`` type only finishes on the
  output limit, so this executor finishes the request itself when
  the token it just appended is the EOS token and the limit did
  not already finish it. The EOS token is never fed back, because
  a finished request is never selected for decode again.

Per request state is exactly one cache object, stored under the
request id between prefill and the last decode. ``release`` drops
that entry and nothing else; it is idempotent, and the engine owns
when it is called. After release the executor holds no reference
to the request's cache.

If a model forward raises, this executor moves the request whose
forward failed to FAILED with the exception recorded as its
failure reason, and the exception then propagates to the engine
caller, as the executor contract requires. The engine cleans up
the step and releases the failed request, which drops its cache
here, so a failure ends the request and frees its resources
without hiding the original error. A forward that never ran, such
as a decode call rejected because the request holds no cache,
raises without marking the request, because that is a caller bug,
not a model failure.
"""

from typing import Any

from miniserve.request import Request, RequestState


class RealExecutor:
    """Run one real model, one request per forward."""

    def __init__(self, model: Any, eos_token_id: int) -> None:
        self.model = model
        self.eos_token_id = eos_token_id
        self._caches: dict[str, Any] = {}
        self.released_request_ids: list[str] = []
        self.prefill_forward_count = 0
        self.decode_forward_count = 0

    @property
    def active_cache_count(self) -> int:
        """How many per request caches this executor currently holds."""
        return len(self._caches)

    def _greedy_next_token(self, logits_row: Any) -> int:
        return int(logits_row.argmax().item())

    def _append_produced_token(self, request: Request, token_id: int) -> None:
        request.append_token(token_id)
        if token_id == self.eos_token_id and not request.is_finished:
            request.finish()

    def prefill(self, requests: list[Request]) -> None:
        import torch

        for request in requests:
            request.start_prefill()
            prompt_length = len(request.prompt_token_ids)
            try:
                with torch.inference_mode():
                    input_ids = torch.tensor(
                        [request.prompt_token_ids], dtype=torch.long
                    )
                    attention_mask = torch.ones_like(input_ids)
                    cache_position = torch.arange(
                        prompt_length, dtype=torch.long
                    )
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        cache_position=cache_position,
                    )
            except Exception as exc:
                request.fail(
                    f"prefill forward failed: {type(exc).__name__}: {exc}"
                )
                raise
            self.prefill_forward_count += 1
            self._caches[request.request_id] = output.past_key_values
            first_token = self._greedy_next_token(output.logits[0, -1])
            request.finish_prefill()
            self._append_produced_token(request, first_token)

    def decode(self, requests: list[Request]) -> None:
        import torch

        for request in requests:
            if request.state is not RequestState.DECODING:
                raise RuntimeError(
                    f"Request {request.request_id} is not ready for decode"
                )
            cache = self._caches.get(request.request_id)
            if cache is None:
                raise RuntimeError(
                    f"Request {request.request_id} has no stored cache. "
                    "Decode requires a cache created by prefill."
                )
            if not request.generated_token_ids:
                raise RuntimeError(
                    f"Request {request.request_id} has no generated token "
                    "to feed back. Prefill must produce the first token "
                    "before decode runs."
                )
            cache_length_before = int(cache.get_seq_length())
            next_input_token = request.generated_token_ids[-1]
            try:
                with torch.inference_mode():
                    input_ids = torch.tensor(
                        [[next_input_token]], dtype=torch.long
                    )
                    attention_mask = torch.ones(
                        (1, cache_length_before + 1), dtype=torch.long
                    )
                    cache_position = torch.tensor(
                        [cache_length_before], dtype=torch.long
                    )
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=cache_position,
                    )
            except Exception as exc:
                request.fail(
                    f"decode forward failed: {type(exc).__name__}: {exc}"
                )
                raise
            self.decode_forward_count += 1
            self._caches[request.request_id] = output.past_key_values
            produced = self._greedy_next_token(output.logits[0, -1])
            self._append_produced_token(request, produced)

    def release(self, request: Request) -> None:
        self._caches.pop(request.request_id, None)
        if request.request_id not in self.released_request_ids:
            self.released_request_ids.append(request.request_id)
