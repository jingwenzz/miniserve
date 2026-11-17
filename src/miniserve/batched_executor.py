"""A real model executor that decodes several requests in one forward.

This backend extends the single request executor. Prefill stays
per request, exactly as the plan allows: each request prefills on
its own and keeps its own independent cache as the source of
truth. Decode is different. When the engine hands over several
decoding requests, this executor packs their caches into one
left padded batched cache, runs one model forward with batch
dimension equal to the number of requests, splits the updated
cache back into per request caches, and appends one token per
request from its own logits row.

Requests may hold caches of different valid lengths, because
their prompts differ in length or because they have completed a
different number of decode steps. Packing follows the layout in
``docs/batching.md``: a shorter row is padded with zeros at the
front, so its valid positions occupy the trailing slots and every
row ends at the same physical slot. The step therefore shares one
``cache_position`` of the longest length, while ``position_ids``
carries each row's own valid length, which is the logical
position the rotary embeddings must see. The attention mask has
one row per request: zeros on that row's padded past positions,
ones on its valid past positions, and one on the current token,
so padded keys never contribute to attention and padded length is
never counted as a valid length. After the forward, row i holds
its valid positions plus the new one in the trailing slots, so
splitting slices from the padding count to the end. Nothing valid
is dropped and no padded slot is kept.

Forward counting follows the same convention as the single
request executor, with one difference that is the point of this
backend: ``decode_forward_count`` counts model forwards, not
requests, so one batched decode step for two requests adds one,
not two. ``decode_batch_sizes`` records the batch dimension of
every decode forward in order, which is the evidence that a step
really ran as one batched forward and not as a loop of single
forwards. A single request decode also goes through the same
padded path with batch dimension one and zero padding, so there
is only one decode code path to reason about.

If the batched forward raises, every request in that forward is
marked FAILED with the recorded reason and the exception then
propagates, because the forward was shared and no row can claim
it succeeded. The per request caches handed to the forward are
copies built by packing, so the stored caches are not mutated
by a failed forward; the engine releases the failed requests
afterwards, which drops those caches.
"""

from typing import Any

from miniserve.real_executor import RealExecutor
from miniserve.request import Request, RequestState


class BatchedRealExecutor(RealExecutor):
    """Run one real model, batching decode steps across requests."""

    def __init__(self, model: Any, eos_token_id: int) -> None:
        super().__init__(model, eos_token_id)
        self.decode_batch_sizes: list[int] = []

    def _pack_caches(self, caches: list[Any], lengths: list[int]) -> Any:
        import torch
        from transformers.cache_utils import DynamicCache

        longest = max(lengths)
        layer_count = len(caches[0].layers)
        packed_data = []
        for layer_index in range(layer_count):
            keys = []
            values = []
            for cache, length in zip(caches, lengths, strict=True):
                key = cache.layers[layer_index].keys
                value = cache.layers[layer_index].values
                pad = longest - length
                if pad > 0:
                    zeros_key = torch.zeros(
                        (1, key.shape[1], pad, key.shape[3]),
                        dtype=key.dtype,
                    )
                    zeros_value = torch.zeros(
                        (1, value.shape[1], pad, value.shape[3]),
                        dtype=value.dtype,
                    )
                    key = torch.cat([zeros_key, key], dim=2)
                    value = torch.cat([zeros_value, value], dim=2)
                keys.append(key)
                values.append(value)
            packed_data.append(
                (torch.cat(keys, dim=0), torch.cat(values, dim=0))
            )
        return DynamicCache(ddp_cache_data=packed_data)

    def _split_batched_cache(
        self, packed: Any, row: int, valid_length: int
    ) -> Any:
        from transformers.cache_utils import DynamicCache

        total = int(packed.layers[0].keys.shape[2])
        start = total - valid_length
        return DynamicCache(
            ddp_cache_data=[
                (
                    layer.keys[row : row + 1, :, start:, :].clone(),
                    layer.values[row : row + 1, :, start:, :].clone(),
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
        longest = max(lengths)
        batch_size = len(requests)

        packed = self._pack_caches(caches, lengths)
        try:
            with torch.inference_mode():
                input_ids = torch.tensor(
                    [[request.generated_token_ids[-1]] for request in requests],
                    dtype=torch.long,
                )
                attention_mask = torch.tensor(
                    [
                        [0] * (longest - length)
                        + [1] * length
                        + [1]
                        for length in lengths
                    ],
                    dtype=torch.long,
                )
                cache_position = torch.tensor([longest], dtype=torch.long)
                position_ids = torch.tensor(
                    [[length] for length in lengths], dtype=torch.long
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

        for row, (request, length) in enumerate(
            zip(requests, lengths, strict=True)
        ):
            self._caches[request.request_id] = self._split_batched_cache(
                output.past_key_values, row, length + 1
            )
            produced = self._greedy_next_token(output.logits[row, -1])
            self._append_produced_token(request, produced)
