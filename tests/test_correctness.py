"""Unit tests for the correctness comparison helper.

The real model comparison lives in benchmarks/correctness.py and
runs only when that script is executed. What can be checked
without a model is the comparison logic itself: exact matches,
a first divergence in the middle, and one list ending before the
other. Those cases decide how a real mismatch would be reported,
so they are pinned down here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).resolve().parent.parent / "benchmarks" / "correctness.py"
_spec = importlib.util.spec_from_file_location("correctness", _SCRIPT)
assert _spec is not None and _spec.loader is not None
correctness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(correctness)

compare_sequences = correctness.compare_sequences


def test_identical_sequences_match() -> None:
    result: dict[str, Any] = compare_sequences([1, 2, 3], [1, 2, 3])
    assert result["match"] is True
    assert result["first_divergence_index"] is None
    assert result["reference_length"] == 3
    assert result["manual_length"] == 3


def test_first_divergence_points_at_the_first_different_token() -> None:
    result = compare_sequences([1, 2, 3, 4], [1, 2, 9, 4])
    assert result["match"] is False
    assert result["first_divergence_index"] == 2


def test_shorter_list_diverges_where_it_ends() -> None:
    result = compare_sequences([1, 2, 3], [1, 2])
    assert result["match"] is False
    assert result["first_divergence_index"] == 2


def test_empty_sequences_match() -> None:
    result = compare_sequences([], [])
    assert result["match"] is True
    assert result["first_divergence_index"] is None
