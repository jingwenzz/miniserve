"""Tests for the batched real executor, driven by a scripted model.

No real model is loaded here. The fake model follows the cache
surface the batched executor relies on: caches are DynamicCache
objects whose layers hold tensors with a batch axis, a call with
no past cache creates one, a call with a past cache returns a new
cache grown by the number of input tokens, and the logits argmax
for a row is a fixed script of that row. The script derives a row
identity from the first prompt token, which prefill writes into
the cache, and produces identity plus the context length after
the call, so the same request produces the same token sequence
whether it runs alone or as one row of a batched forward. That is
what lets the tests compare batched against individual execution
token for token, and count the forwards each path really ran.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from miniserve.batched_executor import BatchedRealExecutor
from miniserve.engine import Engine
from miniserve.real_executor import RealExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import Scheduler

VOCAB = 64
EOS = 0
LAYERS = 2


def make_cache(batch_rows: list[tuple[int, int]]) -> Any:
    """Build a DynamicCache from (identity, length) rows."""
    from transformers.cache_utils import DynamicCache

    layer_data = []
    for _ in range(LAYERS):
        keys = torch.cat(
            [
                torch.full((1, 1, length, 2), float(identity))
                for identity, length in batch_rows
            ],
            dim=0,
        )
        values = torch.zeros_like(keys)
        layer_data.append((keys, values))
    return DynamicCache(ddp_cache_data=layer_data)


class FakeBatchedModel:
    def __init__(self, fail_batched_decode: bool = False) -> None:
        self.input_shapes: list[list[int]] = []
        self.attention_masks: list[Any] = []
        self.position_ids_seen: list[Any] = []
        self.fail_batched_decode = fail_batched_decode

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        input_ids = kwargs["input_ids"]
        batch = int(input_ids.shape[0])
        seq = int(input_ids.shape[1])
        self.input_shapes.append(list(input_ids.shape))
        mask = kwargs.get("attention_mask")
        self.attention_masks.append(
            None if mask is None else mask.tolist()
        )
        positions = kwargs.get("position_ids")
        self.position_ids_seen.append(
            None if positions is None else positions.tolist()
        )
        past = kwargs.get("past_key_values")

        if past is None:
            identity = int(input_ids[0, 0].item())
            cache = make_cache([(identity, seq)])
            lengths_after = [seq]
            identities = [identity]
        else:
            if self.fail_batched_decode and batch > 1:
                raise RuntimeError("scripted batched failure")
            identities = []
            lengths_after = []
            for row in range(batch):
                if mask is not None:
                    # The mask row counts this row's valid past
                    # positions plus the current token, and its
                    # leading zeros are the left padding, so the
                    # first valid cache slot sits at the padding
                    # count and holds the row identity.
                    valid_after = int(mask[row].sum().item())
                    pad = int(mask.shape[1]) - valid_after
                else:
                    valid_after = int(past.get_seq_length()) + seq
                    pad = 0
                identities.append(
                    int(past.layers[0].keys[row, 0, pad, 0].item())
                )
                lengths_after.append(valid_after)
            layer_data = []
            for layer in past.layers:
                new_keys = torch.stack(
                    [
                        torch.full(
                            (1, 1, 2), float(input_ids[row, 0].item())
                        )
                        for row in range(batch)
                    ]
                )
                new_values = torch.zeros_like(new_keys)
                layer_data.append(
                    (
                        torch.cat([layer.keys, new_keys], dim=2),
                        torch.cat([layer.values, new_values], dim=2),
                    )
                )
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache(ddp_cache_data=layer_data)

        logits = torch.full((batch, seq, VOCAB), -10.0)
        for row, identity in enumerate(identities):
            logits[row, -1, identity + lengths_after[row]] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


def make_engine(
    model: FakeBatchedModel,
) -> tuple[Scheduler, BatchedRealExecutor, Engine]:
    executor = BatchedRealExecutor(model, eos_token_id=EOS)
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=100)
    return scheduler, executor, Engine(scheduler, executor)


def test_two_equal_length_requests_decode_in_one_forward_per_step() -> None:
    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    first = Request("request-1", [11, 12, 13], max_new_tokens=3)
    second = Request("request-2", [21, 22, 23], max_new_tokens=3)
    scheduler.add_request(first)
    scheduler.add_request(second)

    engine.run_until_complete()

    assert first.state is RequestState.FINISHED
    assert second.state is RequestState.FINISHED
    assert first.generated_token_ids == [14, 15, 16]
    assert second.generated_token_ids == [24, 25, 26]
    assert executor.prefill_forward_count == 2
    # Two decode steps remain after prefill produced the first
    # token, and each step is one forward for both rows, not two
    # single forwards.
    assert executor.decode_forward_count == 2
    assert executor.decode_batch_sizes == [2, 2]
    decode_shapes = model.input_shapes[2:]
    assert decode_shapes == [[2, 1], [2, 1]]
    assert executor.active_cache_count == 0
    assert executor.released_request_ids == ["request-1", "request-2"]


def test_batched_results_match_individual_real_executor_runs() -> None:
    prompts = [[11, 12, 13], [21, 22, 23]]
    individual_results = []
    for index, prompt in enumerate(prompts):
        model = FakeBatchedModel()
        executor = RealExecutor(model, eos_token_id=EOS)
        scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
        engine = Engine(scheduler, executor)
        request = Request(f"solo-{index}", prompt, max_new_tokens=3)
        scheduler.add_request(request)
        engine.run_until_complete()
        individual_results.append(list(request.generated_token_ids))

    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    requests = [
        Request(f"batch-{index}", prompt, max_new_tokens=3)
        for index, prompt in enumerate(prompts)
    ]
    for request in requests:
        scheduler.add_request(request)
    engine.run_until_complete()

    assert [r.generated_token_ids for r in requests] == individual_results


def test_unequal_prompt_lengths_decode_together_and_match_individual() -> None:
    prompts = [[11, 12], [21, 22, 23, 24, 25]]
    individual_results = []
    for index, prompt in enumerate(prompts):
        solo_model = FakeBatchedModel()
        solo_executor = RealExecutor(solo_model, eos_token_id=EOS)
        scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
        engine = Engine(scheduler, solo_executor)
        request = Request(f"solo-{index}", prompt, max_new_tokens=3)
        scheduler.add_request(request)
        engine.run_until_complete()
        individual_results.append(list(request.generated_token_ids))

    model = FakeBatchedModel()
    scheduler, executor, engine = make_engine(model)
    requests = [
        Request(f"batch-{index}", prompt, max_new_tokens=3)
        for index, prompt in enumerate(prompts)
    ]
    for request in requests:
        scheduler.add_request(request)
    engine.run_until_complete()

    assert [r.generated_token_ids for r in requests] == individual_results
    assert requests[0].generated_token_ids == [13, 14, 15]
    assert requests[1].generated_token_ids == [26, 27, 28]
    # Two decode steps after prefill, each one forward for both
    # rows even though the rows never share a cache length.
    assert executor.decode_forward_count == 2
    assert executor.decode_batch_sizes == [2, 2]
    # The first decode step packs lengths 2 and 5: the short row
    # is masked off on its three padded past positions, and each
    # row reports its own logical position, not the shared
    # physical append slot.
    decode_masks = [
        mask for mask in model.attention_masks if mask is not None and len(mask) == 2
    ]
    assert decode_masks[0] == [[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    decode_positions = [
        pos for pos in model.position_ids_seen if pos is not None and len(pos) == 2
    ]
    assert decode_positions[0] == [[2], [5]]
    assert executor.active_cache_count == 0
    assert executor.released_request_ids == ["batch-0", "batch-1"]


def test_requests_at_different_decode_progress_share_one_forward() -> None:
    model = FakeBatchedModel()
    executor = BatchedRealExecutor(model, eos_token_id=EOS)
    ahead = Request("request-1", [11, 12, 13], max_new_tokens=4)
    behind = Request("request-2", [21, 22, 23], max_new_tokens=4)
    executor.prefill([ahead, behind])

    # One solo decode step puts the first request one valid
    # position ahead of the second.
    executor.decode([ahead])
    assert ahead.generated_token_ids == [14, 15]

    executor.decode([ahead, behind])

    assert ahead.generated_token_ids == [14, 15, 16]
    assert behind.generated_token_ids == [24, 25]
    assert executor.decode_forward_count == 2
    assert executor.decode_batch_sizes == [1, 2]
    # Split back kept exactly each row's valid positions: the
    # ahead cache grew to 5 and the behind cache to 4, with no
    # padded slot retained on the shorter row.
    assert executor._caches["request-1"].get_seq_length() == 5
    assert executor._caches["request-2"].get_seq_length() == 4


def test_failed_batched_forward_marks_every_row_failed() -> None:
    model = FakeBatchedModel(fail_batched_decode=True)
    executor = BatchedRealExecutor(model, eos_token_id=EOS)
    first = Request("request-1", [11, 12, 13], max_new_tokens=3)
    second = Request("request-2", [21, 22, 23], max_new_tokens=3)
    executor.prefill([first, second])

    with pytest.raises(RuntimeError, match="scripted batched failure"):
        executor.decode([first, second])

    assert first.state is RequestState.FAILED
    assert second.state is RequestState.FAILED
    assert first.failure_reason is not None
    assert "scripted batched failure" in first.failure_reason
    assert second.failure_reason is not None
    assert "scripted batched failure" in second.failure_reason
    assert executor.decode_forward_count == 0
    assert executor.decode_batch_sizes == []
