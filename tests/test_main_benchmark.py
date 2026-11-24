"""Unit tests for the main experiment helpers.

The real runs live in benchmarks/main_benchmark.py and
benchmarks/budget_benchmark.py and need the model extra. What can
be checked without a model is the bookkeeping: workload spec
construction, repeat aggregation, order rotation, budget
validation, and the single request timeline extraction.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "benchmarks" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main_bench = _load("main_benchmark", "main_benchmark.py")
budget_bench = _load("budget_benchmark", "budget_benchmark.py")

SHORT = [[1, 2, 3], [4, 5]]
LONG = [[1] * 20, [2] * 24]


def test_short_specs_cycle_protocol_prompts() -> None:
    specs = main_bench.build_workload_specs(SHORT, LONG, "short", 5, 8)
    assert [s["prompt_kind"] for s in specs] == ["short"] * 5
    assert [s["prompt_token_ids"] for s in specs] == [
        [1, 2, 3],
        [4, 5],
        [1, 2, 3],
        [4, 5],
        [1, 2, 3],
    ]
    assert len({s["request_id"] for s in specs}) == 5


def test_mixed_specs_alternate_short_and_long() -> None:
    specs = main_bench.build_workload_specs(SHORT, LONG, "mixed", 6, 8)
    assert [s["prompt_kind"] for s in specs] == [
        "short",
        "long",
        "short",
        "long",
        "short",
        "long",
    ]
    assert [len(s["prompt_token_ids"]) for s in specs] == [3, 20, 3, 24, 3, 20]
    assert specs[1]["request_id"] == "main-mixed-01"


def test_unknown_workload_is_rejected() -> None:
    with pytest.raises(ValueError):
        main_bench.build_workload_specs(SHORT, LONG, "huge", 2, 8)


def test_aggregate_repeats_reports_mean_and_range() -> None:
    summaries = [
        {
            "wall_duration_seconds": 10.0,
            "mean_ttft": 2.0,
            "mean_latency": 4.0,
            "mean_tpot": 0.5,
            "output_tokens_per_second": 16.0,
            "requests_per_second": 2.0,
            "engine_steps": 30,
            "finished": 20,
            "failed": 0,
            "cancelled": 0,
            "total_output_tokens": 160,
        },
        {
            "wall_duration_seconds": 14.0,
            "mean_ttft": 4.0,
            "mean_latency": 6.0,
            "mean_tpot": 0.7,
            "output_tokens_per_second": 12.0,
            "requests_per_second": 1.5,
            "engine_steps": 34,
            "finished": 20,
            "failed": 0,
            "cancelled": 0,
            "total_output_tokens": 160,
        },
    ]
    aggregated = main_bench.aggregate_repeats(summaries)
    assert aggregated["repeats"] == 2
    assert aggregated["wall_duration_seconds"] == {
        "mean": 12.0,
        "min": 10.0,
        "max": 14.0,
    }
    assert aggregated["finished_per_repeat"] == [20, 20]
    assert aggregated["output_tokens_per_repeat"] == [160, 160]


def test_rotate_varies_order_without_changing_members() -> None:
    assert main_bench.rotate([1, 2, 4, 8], 0) == [1, 2, 4, 8]
    assert main_bench.rotate([1, 2, 4, 8], 1) == [2, 4, 8, 1]
    assert sorted(main_bench.rotate([32, 64, 8192], 5)) == [32, 64, 8192]


def test_validate_budgets_requires_admitting_the_longest_prompt() -> None:
    specs = main_bench.build_workload_specs(SHORT, LONG, "mixed", 4, 8)
    budget_bench.validate_budgets([24, 64], specs)
    with pytest.raises(ValueError):
        budget_bench.validate_budgets([16], specs)


def test_timeline_for_extracts_one_request_across_budgets() -> None:
    def record(request_id: str, queue: float) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "prompt_length": 20,
            "output_length": 8,
            "queue_time": queue,
            "ttft": queue + 0.5,
            "latency": queue + 2.0,
            "tpot": 0.2,
        }

    by_budget = {
        32: [[record("main-mixed-01", 3.0), record("main-mixed-03", 9.0)]],
        64: [[record("main-mixed-01", 1.0), record("main-mixed-03", 4.0)]],
    }
    timeline = budget_bench.timeline_for(by_budget, "main-mixed-01")
    assert timeline["budgets"]["32"][0]["queue_time"] == 3.0
    assert timeline["budgets"]["64"][0]["ttft"] == 1.5
    with pytest.raises(ValueError):
        budget_bench.timeline_for(by_budget, "main-mixed-99")
