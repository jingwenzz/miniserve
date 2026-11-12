"""Tests for the hand written prefill and decode loop.

The loop in benchmarks/manual_generate.py is driven here with a
fake model whose next token is a fixed script, so the bookkeeping
can be checked without loading the real model: the prompt is fed
exactly once, every decode step feeds exactly one token (the one
just produced and not yet cached), the cache grows by one position
per decode step, and stopping follows EOS and the output limit.
The real model path is exercised by running the script itself;
comparing its output against the reference across prompts is a
separate correctness pass.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "manual_generate.py"
)
_spec = importlib.util.spec_from_file_location("manual_generate", _SCRIPT)
assert _spec is not None and _spec.loader is not None
manual_generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(manual_generate)

VOCAB = 32
EOS = 0


class FakeCache:
    def __init__(self) -> None:
        self.length = 0

    def get_seq_length(self) -> int:
        return self.length


class FakeModel:
    """Returns logits whose argmax follows a fixed token script.

    The script is indexed by the total number of tokens the model
    has seen once a forward completes, which mirrors a real model
    choosing the next token from the full context so far.
    """

    def __init__(self, script: list[int]) -> None:
        self.script = script
        self.cache = FakeCache()
        self.input_shapes: list[list[int]] = []
        self.input_token_ids: list[list[int]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        input_ids = kwargs["input_ids"]
        self.input_shapes.append(list(input_ids.shape))
        self.input_token_ids.append(input_ids[0].tolist())
        self.cache.length += int(input_ids.shape[1])
        next_token = self.script[self.cache.length]
        logits = torch.full((1, int(input_ids.shape[1]), VOCAB), -10.0)
        logits[0, -1, next_token] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=self.cache)


def test_prompt_is_fed_once_and_each_decode_feeds_one_token() -> None:
    model = FakeModel(script=[99, 99, 99, 99, 99, 7, 8, 9, 10])
    result = manual_generate.run_manual_generation(
        model, [11, 12, 13, 14, 15], max_new_tokens=4, eos_token_id=EOS
    )
    assert model.input_shapes == [[1, 5], [1, 1], [1, 1], [1, 1]]
    assert model.input_token_ids == [[11, 12, 13, 14, 15], [7], [8], [9]]
    assert result["generated_token_ids"] == [7, 8, 9, 10]
    assert result["stop_reason"] == "max_new_tokens"


def test_cache_grows_by_one_per_decode_and_leads_by_one_token() -> None:
    model = FakeModel(script=[99, 99, 99, 99, 99, 7, 8, 9, 10])
    result = manual_generate.run_manual_generation(
        model, [11, 12, 13, 14, 15], max_new_tokens=4, eos_token_id=EOS
    )
    trace = result["trace"]
    assert trace[0]["kind"] == "prefill"
    assert trace[0]["cache_length_before"] == 0
    assert trace[0]["cache_length_after"] == 5
    assert [row["cache_length_after"] for row in trace[1:]] == [6, 7, 8]
    # Four tokens were produced but only the first three were ever
    # fed back, so the final cache holds prompt plus three tokens.
    assert result["final_cache_length"] == 5 + 4 - 1


def test_single_token_run_needs_no_decode_forward() -> None:
    model = FakeModel(script=[99, 99, 99, 21])
    result = manual_generate.run_manual_generation(
        model, [11, 12, 13], max_new_tokens=1, eos_token_id=EOS
    )
    assert model.input_shapes == [[1, 3]]
    assert result["generated_token_ids"] == [21]
    assert result["final_cache_length"] == 3


def test_eos_from_prefill_stops_immediately_and_counts_as_a_token() -> None:
    model = FakeModel(script=[99, 99, EOS])
    result = manual_generate.run_manual_generation(
        model, [11, 12], max_new_tokens=8, eos_token_id=EOS
    )
    assert result["generated_token_ids"] == [EOS]
    assert result["stop_reason"] == "eos"
    assert len(model.input_shapes) == 1


def test_eos_from_decode_stops_without_feeding_the_eos_token() -> None:
    model = FakeModel(script=[99, 99, 99, 5, EOS, 6])
    result = manual_generate.run_manual_generation(
        model, [11, 12, 13], max_new_tokens=8, eos_token_id=EOS
    )
    assert result["generated_token_ids"] == [5, EOS]
    assert result["stop_reason"] == "eos"
    assert model.input_token_ids == [[11, 12, 13], [5]]
    assert result["final_cache_length"] == 4
