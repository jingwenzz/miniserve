"""A real model executor that decodes several requests in one forward.

This backend extends the single request executor. Prefill stays
per request, exactly as the plan allows: each request prefills on
its own and keeps its own independent cache as the source of
truth. Decode is different. When the engine hands over several
decoding requests whose caches all have the same valid length,
this executor stacks their caches along the batch axis, runs one
model forward with batch dimension equal to the number of
requests, splits the updated cache back into per request caches,
and appends one token per request from its own logits row.

Requests with different cache lengths are rejected with a
ValueError before any forward runs. Padding those rows to a
common length is separate later work, and this executor does not
pretend to support it yet. A single request decode also goes
through the same batched path with batch dimension one, so there
is only one decode code path to reason about.

Forward counting follows the same convention as the single
request executor, with one difference that is the point of this
backend: ``decode_forward_count`` counts model forwards, not
requests, so one batched decode step for two requests adds one,
not two. ``decode_batch_sizes`` records the batch dimension of
every decode forward in order, which is the evidence that a step
really ran as one batched forward and not as a loop of single
forwards.

If the batched forward raises, every request in that forward is
marked FAILED with the recorded reason and the exception then
propagates, because the forward was shared and no row can claim
it succeeded. The per request caches handed to the forward are
copies built by stacking, so the stored caches are not mutated
by a failed forward; the engine releases the failed requests
afterwards, which drops those caches.
"""

from typing import Any

from miniserve.real_executor import RealExecutor
from miniserve.request import Request, RequestState


class BatchedRealExecutor(RealExecutor):
    """Run one real model, batching equal length decode steps."""

    def __init__(self, model: Any, eos_token_id: int) -> None:
        super().__init__(model, eos_token_id)
        self.decode_batch_sizes: list[int] = []

    def _pack_equal_length_caches(self, caches: list[Any]) -> Any:
        import torch
        from transformers.cache_utils import DynamicCache

        layer_count = len(caches[0].layers)
        packed_data = []
        for layer_index in range(layer_count):
            keys = torch.cat(
                [cache.layers[layer_index].keys for cache in caches],
                dim=0,
            )
            values = torch.cat(
                [cache.layers[layer_index].values for cache in caches],
                dim=0,
            )
            packed_data.append((keys, values))
        return DynamicCache(ddp_cache_data=packed_data)

    def _split_batched_cache(self, packed: Any, row: int) -> Any:
        from transformers.cache_utils import DynamicCache

        return DynamicCache(
            ddp_cache_data=[
                (
                    layer.keys[row : row + 1].clone(),
                    layer.values[row : row + 1].clone(),
                )
                for layer in packed.layers
            ]
        )

    def decode(self, requests: list[Request]) -> None:
        import torch

        if not requests:
            return

        caches: list[Any] = []
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
            caches.append(cache)

        lengths = [int(cache.get_seq_length()) for cache in caches]
        if len(set(lengths)) != 1:
            raise ValueError(
                "Batched decode currently requires equal cache lengths, "
                f"got lengths {lengths}. Requests with different lengths "
                "are not padded by this executor yet."
            )
        cache_length = lengths[0]
        batch_size = len(requests)

        packed = self._pack_equal_length_caches(caches)
        try:
            with torch.inference_mode():
                input_ids = torch.tensor(
                    [[request.generated_token_ids[-1]] for request in requests],
                    dtype=torch.long,
                )
                attention_mask = torch.ones(
                    (batch_size, cache_length + 1), dtype=torch.long
                )
                cache_position = torch.tensor(
                    [cache_length], dtype=torch.long
                )
                position_ids = torch.tensor(
                    [[cache_length] for _ in requests], dtype=torch.long
                )
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=packed,
                    use_cache=True,
                    cache_position=cache_position,
                    position_ids=position_ids,
                )
        except Exception as exc:
            for request in requests:
                if not request.is_finished:
                    request.fail(
                        "batched decode forward failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            raise

        self.decode_forward_count += 1
        self.decode_batch_sizes.append(batch_size)

        for row, request in enumerate(requests):
            self._caches[request.request_id] = self._split_batched_cache(
                output.past_key_values, row
            )
            produced = self._greedy_next_token(output.logits[row, -1])
            self._append_produced_token(request, produced)
