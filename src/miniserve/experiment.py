"""A reproducible pipeline for simulation experiments.

One experiment run reads a deterministic workload, replays it
under one or more scheduling policies in simulation time, and
writes everything under a fresh run directory:

- ``config.json``: the full configuration, including the workload
  shape and the package version, so the run can be repeated.
- ``<policy>_requests.jsonl``: the raw per request metrics records.
- ``<policy>_timeline.jsonl``: the raw per step simulation timeline.
- ``summary.json`` and ``summary.md``: a summary that is computed
  only from the raw files above.

The summary step is independent of the run step: ``summarize_run``
reads only the files in a run directory, never the objects that
produced them, so anyone can recompute the summary from the raw
data and get the same numbers. A run directory is never reused:
asking for a run id that already exists is an error rather than an
overwrite, so a rerun always leaves a trace of both runs.

All values are simulation values in logical seconds, produced by
the fake token stream under the explicit cost model recorded in
the configuration. They describe scheduling behaviour under that
model and say nothing about real model or GPU performance.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import cast

from miniserve.metrics import to_jsonl
from miniserve.simulation import SimulationCosts, run_simulation, write_timeline_jsonl
from miniserve.workload import generate_workload

KNOWN_POLICIES: tuple[str, ...] = ("serial", "continuous")


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything one experiment run needs, and nothing else."""

    profile: str = "mixed"
    seed: int = 1234
    num_requests: int = 6
    policies: tuple[str, ...] = KNOWN_POLICIES
    max_batch_size: int = 4
    max_batch_tokens: int = 64
    max_model_len: int | None = None
    prefill_seconds_per_token: float = 0.1
    decode_seconds_per_token: float = 0.5
    run_id: str | None = None


def package_version() -> str:
    try:
        return metadata.version("miniserve")
    except metadata.PackageNotFoundError:
        return "unknown"


def _new_run_id(config: ExperimentConfig) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config.profile}-seed{config.seed}-{uuid.uuid4().hex[:6]}"


def run_experiment(
    config: ExperimentConfig, output_root: Path
) -> Path:
    """Run one experiment and return its run directory.

    The run directory is created fresh. If a directory with the
    chosen run id already exists, FileExistsError is raised and
    nothing is overwritten.
    """
    if not config.policies:
        raise ValueError("At least one policy is required")
    for policy in config.policies:
        if policy not in KNOWN_POLICIES:
            raise ValueError(
                f"Unknown policy {policy!r}. "
                f"Expected one of {', '.join(KNOWN_POLICIES)}."
            )

    costs = SimulationCosts(
        prefill_seconds_per_token=config.prefill_seconds_per_token,
        decode_seconds_per_token=config.decode_seconds_per_token,
    )
    specs = generate_workload(
        config.profile, seed=config.seed, num_requests=config.num_requests
    )
    run_id = config.run_id if config.run_id is not None else _new_run_id(config)

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)

    config_record: dict[str, object] = {
        "kind": "simulation",
        "simulation": True,
        "run_id": run_id,
        "miniserve_version": package_version(),
        "profile": config.profile,
        "seed": config.seed,
        "num_requests": config.num_requests,
        "policies": list(config.policies),
        "max_batch_size": config.max_batch_size,
        "max_batch_tokens": config.max_batch_tokens,
        "max_model_len": config.max_model_len,
        "prefill_seconds_per_token": config.prefill_seconds_per_token,
        "decode_seconds_per_token": config.decode_seconds_per_token,
        "workload": [
            {
                "request_id": spec.request_id,
                "prompt_length": spec.prompt_length,
                "max_new_tokens": spec.max_new_tokens,
                "arrival_time": spec.arrival_time,
            }
            for spec in specs
        ],
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for policy in config.policies:
        result = run_simulation(
            specs,
            policy=policy,
            max_batch_size=config.max_batch_size,
            max_batch_tokens=config.max_batch_tokens,
            max_model_len=config.max_model_len,
            costs=costs,
        )
        (run_dir / f"{policy}_requests.jsonl").write_text(
            to_jsonl(result.requests), encoding="utf-8"
        )
        write_timeline_jsonl(result.timeline, run_dir / f"{policy}_timeline.jsonl")

    summarize_run(run_dir)
    return run_dir


def _read_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return cast("dict[str, object]", data)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise TypeError(f"{path} contains a line that is not an object")
        rows.append(cast("dict[str, object]", data))
    return rows


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected an integer for {field}, got {value!r}")
    return value


def _as_float_or_none(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected a number or null for {field}, got {value!r}")
    return float(value)


def _as_float(value: object, field: str) -> float:
    result = _as_float_or_none(value, field)
    if result is None:
        raise ValueError(f"Expected a number for {field}, got null")
    return result


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _policy_summary(run_dir: Path, policy: str) -> dict[str, object]:
    request_rows = _read_jsonl(run_dir / f"{policy}_requests.jsonl")
    timeline_rows = _read_jsonl(run_dir / f"{policy}_timeline.jsonl")

    finished = 0
    cancelled = 0
    unfinished = 0
    total_output_tokens = 0
    total_prompt_tokens = 0
    ttfts: list[float] = []
    latencies: list[float] = []
    tpots: list[float] = []
    for row in request_rows:
        reason = row.get("finish_reason")
        if reason == "finished":
            finished += 1
        elif reason == "cancelled":
            cancelled += 1
        elif reason is None:
            unfinished += 1
        else:
            raise ValueError(f"Unknown finish_reason {reason!r}")
        total_output_tokens += _as_int(row.get("output_length"), "output_length")
        total_prompt_tokens += _as_int(row.get("prompt_length"), "prompt_length")
        ttft = _as_float_or_none(row.get("ttft"), "ttft")
        if ttft is not None:
            ttfts.append(ttft)
        latency = _as_float_or_none(row.get("latency"), "latency")
        if latency is not None:
            latencies.append(latency)
        tpot = _as_float_or_none(row.get("tpot"), "tpot")
        if tpot is not None:
            tpots.append(tpot)

    simulated_duration = 0.0
    if timeline_rows:
        simulated_duration = _as_float(
            timeline_rows[-1].get("end_time"), "end_time"
        ) - _as_float(timeline_rows[0].get("start_time"), "start_time")

    throughput: float | None = None
    if simulated_duration > 0:
        throughput = total_output_tokens / simulated_duration

    return {
        "requests": len(request_rows),
        "finished": finished,
        "cancelled": cancelled,
        "unfinished": unfinished,
        "total_output_tokens": total_output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "steps": len(timeline_rows),
        "simulated_duration": simulated_duration,
        "mean_ttft": _mean(ttfts),
        "mean_latency": _mean(latencies),
        "mean_tpot": _mean(tpots),
        "simulated_output_tokens_per_second": throughput,
    }


def _format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _summary_markdown(summary: dict[str, object]) -> str:
    policies = cast("dict[str, object]", summary["policies"])
    lines = [
        "# Experiment summary (simulation)",
        "",
        f"Run id: {summary['run_id']}",
        "",
        "All times are logical simulation seconds under the cost model",
        "recorded in config.json. They are not measured model or GPU times.",
        "",
        (
            "| Policy | Finished | Cancelled | Unfinished | Output tokens "
            "| Steps | Simulated duration | Mean TTFT | Mean latency "
            "| Mean TPOT | Simulated output tokens per second |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for policy, stats in policies.items():
        row = cast("dict[str, object]", stats)
        lines.append(
            "| "
            + " | ".join(
                [
                    policy,
                    _format_value(row["finished"]),
                    _format_value(row["cancelled"]),
                    _format_value(row["unfinished"]),
                    _format_value(row["total_output_tokens"]),
                    _format_value(row["steps"]),
                    _format_value(row["simulated_duration"]),
                    _format_value(row["mean_ttft"]),
                    _format_value(row["mean_latency"]),
                    _format_value(row["mean_tpot"]),
                    _format_value(row["simulated_output_tokens_per_second"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def summarize_run(run_dir: Path) -> dict[str, object]:
    """Recompute the summary of a run from its raw files only.

    Reads config.json plus the per policy request and timeline
    files, writes summary.json and summary.md, and returns the
    summary. Running it again on the same directory produces the
    same summary, because nothing outside those files is consulted.
    """
    config = _read_json_object(run_dir / "config.json")
    policies_value = config.get("policies")
    if not isinstance(policies_value, list) or not all(
        isinstance(policy, str) for policy in policies_value
    ):
        raise TypeError("config.json does not list policies")
    run_id = config.get("run_id")
    if not isinstance(run_id, str):
        raise TypeError("config.json does not carry a run_id")

    summary: dict[str, object] = {
        "kind": "simulation",
        "simulation": True,
        "run_id": run_id,
        "policies": {
            policy: _policy_summary(run_dir, policy) for policy in policies_value
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    return summary
