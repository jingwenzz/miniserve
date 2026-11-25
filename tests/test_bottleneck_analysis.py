"""Tests for the bottleneck analysis helpers and the pack layout.

The helper tests use fixed length lists and the recorded main
experiment summary, so they need no model. The pack test builds
small DynamicCache objects directly and compares the executor's
packed cache against the concatenation construction the layout
was originally written with, value for value.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_analysis_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "bottleneck_analysis", ROOT / "benchmarks" / "bottleneck_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_analysis_module()


def test_bytes_per_cached_token_matches_resources_document() -> None:
    # 30 layers, keys and values, 3 KV heads, head size 64, float32.
    assert analysis.BYTES_PER_CACHED_TOKEN == 30 * 2 * 3 * 64 * 4
    assert analysis.BYTES_PER_CACHED_TOKEN == 46080


def test_padding_stats_equal_lengths_waste_nothing() -> None:
    stats = analysis.padding_stats([5, 5, 5])
    assert stats["longest"] == 5
    assert stats["valid_token_slots"] == 15
    assert stats["padded_token_slots"] == 0
    assert stats["waste_fraction"] == 0.0


def test_padding_stats_mixed_lengths() -> None:
    stats = analysis.padding_stats([5, 20])
    assert stats["longest"] == 20
    assert stats["valid_token_slots"] == 25
    assert stats["padded_token_slots"] == 15
    assert stats["waste_fraction"] == pytest.approx(15 / 40)


def test_copy_bytes_counts_valid_and_grown_rows() -> None:
    copies = analysis.copy_bytes([5, 20])
    unit = analysis.BYTES_PER_CACHED_TOKEN
    assert copies["pack_copied_bytes"] == 25 * unit
    assert copies["split_copied_bytes"] == 27 * unit
    assert copies["packed_allocation_bytes"] == 40 * unit


def test_derived_ratios_against_recorded_main_summary() -> None:
    summary = json.loads(
        (ROOT / "benchmarks" / "results" / "real-main-seed1234" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    derived = analysis.derived_ratios(summary)
    short = derived["short"]
    # Concurrency 1 continuous is the fairness self check: its
    # throughput ratio against serial stays near one.
    assert short["by_concurrency"]["1"]["throughput_ratio_vs_serial"] == pytest.approx(
        6.23 / 5.85, rel=0.02
    )
    # At concurrency 8 the throughput gain and the TPOT cost both
    # appear, which is the trade off the results document reports.
    c8 = short["by_concurrency"]["8"]
    assert c8["throughput_ratio_vs_serial"] > 2.5
    assert c8["tpot_ratio_vs_serial"] > 1.8
    mixed_c8 = derived["mixed"]["by_concurrency"]["8"]
    assert mixed_c8["mean_latency"] == pytest.approx(
        derived["mixed"]["by_concurrency"]["4"]["mean_latency"], rel=0.01
    )


def test_pack_matches_concatenation_reference() -> None:
    torch = pytest.importorskip("torch")
    from transformers.cache_utils import DynamicCache

    from miniserve.batched_executor import BatchedRealExecutor

    def make_cache(length: int, base: float) -> Any:
        layer_data = []
        for layer in range(2):
            keys = (
                torch.arange(length, dtype=torch.float32).reshape(1, 1, length, 1)
                + base
                + layer * 100.0
            )
            keys = keys.expand(1, 3, length, 4).contiguous()
            values = keys + 1000.0
            layer_data.append((keys, values))
        return DynamicCache(ddp_cache_data=layer_data)

    caches = [make_cache(2, 0.0), make_cache(5, 10.0), make_cache(3, 20.0)]
    lengths = [2, 5, 3]
    executor = BatchedRealExecutor(model=None, eos_token_id=-1)
    packed = executor._pack_caches(caches, lengths)

    longest = max(lengths)
    for layer_index in range(2):
        ref_keys = []
        ref_values = []
        for cache, length in zip(caches, lengths, strict=True):
            key = cache.layers[layer_index].keys
            value = cache.layers[layer_index].values
            pad = longest - length
            if pad > 0:
                zeros = torch.zeros((1, key.shape[1], pad, key.shape[3]))
                key = torch.cat([zeros, key], dim=2)
                value = torch.cat([zeros, value], dim=2)
            ref_keys.append(key)
            ref_values.append(value)
        assert torch.equal(
            packed.layers[layer_index].keys, torch.cat(ref_keys, dim=0)
        )
        assert torch.equal(
            packed.layers[layer_index].values, torch.cat(ref_values, dim=0)
        )
