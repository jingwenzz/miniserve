"""Unit tests for the pilot benchmark helpers.

The real pilot lives in benchmarks/pilot_benchmark.py and runs
only when that script is executed with the model extra installed.
What can be checked without a model is the bookkeeping the pilot
relies on: the frozen denominators in the summary, the per request
token comparison, the derived metric recomputation used as the
manual timing check, and the validity of the frozen protocol
configuration itself.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "benchmarks" / "pilot_benchmark.py"
_spec = importlib.util.spec_from_file_location("pilot_benchmark", _SCRIPT)
assert _spec is not None and _spec.loader is not None
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "request_id": "pilot-00",
        "finish_reason": "finished",
        "prompt_length": 5,
        "output_length": 2,
        "ttft": 1.0,
        "latency": 3.0,
        "tpot": 2.0,
        "arrival_time": 100.0,
        "prefill_start_time": 101.0,
        "first_token_time": 102.0,
        "last_token_time": 104.0,
        "finished_time": 105.0,
        "generated_token_ids": [7, 8],
    }
    record.update(overrides)
    return record


def test_summary_counts_outcomes_and_tokens() -> None:
    records = [
        _record(),
        _record(request_id="pilot-01", finish_reason="failed", output_length=1),
        _record(request_id="pilot-02", finish_reason=None, output_length=0),
    ]
    summary = pilot.summarize_records(records)
    assert summary["requests"] == 3
    assert summary["finished"] == 1
    assert summary["failed"] == 1
    assert summary["unfinished"] == 1
    assert summary["total_prompt_tokens"] == 15
    assert summary["total_output_tokens"] == 3


def test_summary_means_exclude_missing_values() -> None:
    records = [
        _record(ttft=2.0, latency=4.0, tpot=2.0),
        _record(request_id="pilot-01", ttft=None, latency=None, tpot=None),
    ]
    summary = pilot.summarize_records(records)
    assert summary["mean_ttft"] == 2.0
    assert summary["mean_latency"] == 4.0
    assert summary["mean_tpot"] == 2.0


def test_compare_token_sequences_detects_a_mismatch() -> None:
    serial = [_record(), _record(request_id="pilot-01")]
    continuous = [_record(), _record(request_id="pilot-01", generated_token_ids=[9, 9])]
    result = pilot.compare_token_sequences(serial, continuous)
    assert result["all_tokens_match"] is False
    assert result["mismatched_request_ids"] == ["pilot-01"]
    same = pilot.compare_token_sequences(serial, serial)
    assert same["all_tokens_match"] is True


def test_recompute_derived_matches_hand_worked_values() -> None:
    derived = pilot.recompute_derived(_record())
    assert derived["queue_time"] == 1.0
    assert derived["ttft"] == 2.0
    assert derived["latency"] == 5.0
    assert derived["tpot"] == 2.0
    single = pilot.recompute_derived(_record(output_length=1))
    assert single["tpot"] is None


def test_frozen_benchmark_config_is_valid() -> None:
    config = pilot.load_benchmark_config(_ROOT / "configs" / "benchmark.json")
    assert config["strategies"] == ["serial", "continuous"]
    assert len(config["prompts"]) == 10
    assert config["eos_mode"] == "fixed_length"
    assert config["fixed_length_eos_token_id"] == -1
    model_config = json.loads(
        (_ROOT / "configs" / "model.json").read_text(encoding="utf-8")
    )
    assert config["model_id"] == model_config["model_id"]
    assert config["revision"] == model_config["revision"]


def test_build_specs_cycles_prompts_with_unique_ids() -> None:
    specs = pilot.build_specs([[1, 2], [3]], 5, 8)
    assert [s["request_id"] for s in specs] == [
        "pilot-00",
        "pilot-01",
        "pilot-02",
        "pilot-03",
        "pilot-04",
    ]
    assert [s["prompt_token_ids"] for s in specs] == [
        [1, 2],
        [3],
        [1, 2],
        [3],
        [1, 2],
    ]
    assert all(s["max_new_tokens"] == 8 for s in specs)
