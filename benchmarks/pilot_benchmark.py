"""Run the real model pilot under the frozen benchmark protocol.

This is the pilot defined in docs/benchmark_protocol.md: 20
requests per strategy, the 10 protocol prompts cycled twice, a
fixed output limit of 8, fixed length mode (the executor EOS id
is the sentinel -1, so no request can stop early and both
strategies generate exactly the same work), warmup requests per
strategy excluded from the timed window, and the model loaded
once. Serial runs first in the pilot; the main experiment rotates
the order across repeats.

The run writes a fresh directory under benchmarks/results named
by its run id and refuses to overwrite an existing one. Raw per
request records carry the metrics fields plus the generated token
ids, so the workload check and every mean can be recomputed from
the raw files alone. Times are wall clock observations on a
shared CPU only machine; the pilot exists to expose timing, EOS,
queue, and configuration errors, not to support a performance
claim.

Usage, from the repository root, with the model extra installed:

    python benchmarks/pilot_benchmark.py
    python benchmarks/pilot_benchmark.py --num-requests 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def load_benchmark_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen benchmark configuration."""
    config: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if config.get("strategies") != ["serial", "continuous"]:
        raise ValueError("benchmark strategies must be serial then continuous")
    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("benchmark prompts must be a non empty list")
    if int(config.get("max_new_tokens", 0)) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if int(config.get("num_requests_pilot", 0)) <= 0:
        raise ValueError("num_requests_pilot must be positive")
    return config


def build_specs(
    prompt_token_ids: list[list[int]], num_requests: int, max_new_tokens: int
) -> list[dict[str, Any]]:
    """Cycle the protocol prompts into request specifications."""
    specs = []
    for index in range(num_requests):
        specs.append(
            {
                "request_id": f"pilot-{index:02d}",
                "prompt_index": index % len(prompt_token_ids),
                "prompt_token_ids": list(
                    prompt_token_ids[index % len(prompt_token_ids)]
                ),
                "max_new_tokens": max_new_tokens,
            }
        )
    return specs


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw per request records with the frozen denominators.

    Means follow docs/metrics.md: TTFT averages only over records
    that have one, latency only over terminal records, and TPOT
    only over records with a TPOT value. Missing values never
    enter a mean as zero and never enlarge a denominator.
    """

    def mean(key: str) -> float | None:
        values = [r[key] for r in records if r.get(key) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    finished = sum(1 for r in records if r.get("finish_reason") == "finished")
    cancelled = sum(1 for r in records if r.get("finish_reason") == "cancelled")
    failed = sum(1 for r in records if r.get("finish_reason") == "failed")
    unfinished = sum(1 for r in records if r.get("finish_reason") is None)
    return {
        "requests": len(records),
        "finished": finished,
        "cancelled": cancelled,
        "failed": failed,
        "unfinished": unfinished,
        "total_prompt_tokens": sum(int(r["prompt_length"]) for r in records),
        "total_output_tokens": sum(int(r["output_length"]) for r in records),
        "mean_ttft": mean("ttft"),
        "mean_latency": mean("latency"),
        "mean_tpot": mean("tpot"),
    }


def compare_token_sequences(
    serial_records: list[dict[str, Any]],
    continuous_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare generated tokens per request id across strategies."""
    serial_by_id = {r["request_id"]: r["generated_token_ids"] for r in serial_records}
    mismatches = []
    for record in continuous_records:
        request_id = record["request_id"]
        if serial_by_id.get(request_id) != record["generated_token_ids"]:
            mismatches.append(request_id)
    return {
        "all_tokens_match": not mismatches and len(serial_by_id) == len(
            continuous_records
        ),
        "mismatched_request_ids": mismatches,
    }


def recompute_derived(record: dict[str, Any]) -> dict[str, Any]:
    """Recompute one record's derived metrics from its raw timestamps.

    This is the manual timing check the pilot requires: the stored
    queue time, TTFT, latency, and TPOT must be reproducible from
    the raw arrival, first token, last token, and finished times
    alone, with TPOT null for a single token record.
    """

    def diff(later: Any, earlier: Any) -> float | None:
        if later is None or earlier is None:
            return None
        return float(later) - float(earlier)

    tpot = None
    if (
        int(record["output_length"]) >= 2
        and record.get("first_token_time") is not None
        and record.get("last_token_time") is not None
    ):
        tpot = (float(record["last_token_time"]) - float(record["first_token_time"])) / (
            int(record["output_length"]) - 1
        )
    return {
        "queue_time": diff(record.get("prefill_start_time"), record.get("arrival_time")),
        "ttft": diff(record.get("first_token_time"), record.get("arrival_time")),
        "latency": diff(record.get("finished_time"), record.get("arrival_time")),
        "tpot": tpot,
    }


def run_strategy(
    model: Any,
    strategy: str,
    specs: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one strategy over fresh requests and return its records."""
    from miniserve.engine import Engine
    from miniserve.metrics import metrics_to_dict, summarize_request
    from miniserve.request import Request
    from miniserve.scheduler import create_scheduler

    eos = int(config["fixed_length_eos_token_id"])
    if strategy == "serial":
        from miniserve.real_executor import RealExecutor

        executor: Any = RealExecutor(model, eos_token_id=eos)
        max_batch_size = 1
    else:
        from miniserve.batched_executor import BatchedRealExecutor

        executor = BatchedRealExecutor(model, eos_token_id=eos)
        max_batch_size = int(config["continuous_max_batch_size_pilot"])

    scheduler = create_scheduler(
        strategy,
        max_batch_size=max_batch_size,
        max_batch_tokens=int(config["max_batch_tokens"]),
        max_model_len=int(config["max_model_len"]),
    )
    engine = Engine(scheduler, executor)
    requests = []
    for spec in specs:
        request = Request(
            request_id=str(spec["request_id"]),
            prompt_token_ids=list(spec["prompt_token_ids"]),
            max_new_tokens=int(spec["max_new_tokens"]),
            clock=time.monotonic,
        )
        requests.append(request)
        scheduler.add_request(request)

    wall_start = time.perf_counter()
    engine.run_until_complete()
    wall_duration = time.perf_counter() - wall_start

    records = []
    for request, spec in zip(requests, specs, strict=True):
        record: dict[str, Any] = metrics_to_dict(summarize_request(request))
        record["prompt_index"] = spec["prompt_index"]
        record["generated_token_ids"] = list(request.generated_token_ids)
        records.append(record)

    run_info: dict[str, Any] = {
        "wall_duration_seconds": wall_duration,
        "engine_steps": engine.step_count,
        "prefill_forwards": executor.prefill_forward_count,
        "decode_forwards": executor.decode_forward_count,
        "active_caches_after_run": executor.active_cache_count,
        "released_count": len(executor.released_request_ids),
    }
    if strategy == "continuous":
        run_info["decode_batch_sizes"] = list(executor.decode_batch_sizes)
    return records, run_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="real-pilot-seed1234")
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_benchmark_config(ROOT / "configs" / "benchmark.json")
    model_config: dict[str, Any] = json.loads(
        (ROOT / "configs" / "model.json").read_text(encoding="utf-8")
    )
    if (
        config["model_id"] != model_config["model_id"]
        or config["revision"] != model_config["revision"]
    ):
        raise ValueError(
            "benchmark config and model config disagree on model or revision"
        )
    num_requests = (
        args.num_requests
        if args.num_requests is not None
        else int(config["num_requests_pilot"])
    )
    max_new_tokens = int(config["max_new_tokens"])

    run_dir = args.output_root / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"], revision=config["revision"]
    )
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        dtype=torch.float32,
        attn_implementation=config["attention_backend"],
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start

    prompt_token_ids = [
        list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in config["prompts"]
    ]
    specs = build_specs(prompt_token_ids, num_requests, max_new_tokens)
    warmup_specs = build_specs(
        prompt_token_ids,
        int(config["warmup_requests_per_strategy"]),
        int(config["warmup_max_new_tokens"]),
    )
    for spec in warmup_specs:
        spec["request_id"] = f"warmup-{spec['request_id']}"

    all_records: dict[str, list[dict[str, Any]]] = {}
    run_infos: dict[str, dict[str, Any]] = {}
    warmup_seconds: dict[str, float] = {}
    for strategy in config["strategies"]:
        warmup_start = time.perf_counter()
        run_strategy(model, strategy, warmup_specs, config)
        warmup_seconds[strategy] = time.perf_counter() - warmup_start
        records, run_info = run_strategy(model, strategy, specs, config)
        all_records[strategy] = records
        run_infos[strategy] = run_info

    comparison = compare_token_sequences(all_records["serial"], all_records["continuous"])
    summaries = {
        strategy: summarize_records(records)
        for strategy, records in all_records.items()
    }
    workload_check = {
        "finished_equal": summaries["serial"]["finished"]
        == summaries["continuous"]["finished"],
        "prompt_tokens_equal": summaries["serial"]["total_prompt_tokens"]
        == summaries["continuous"]["total_prompt_tokens"],
        "output_tokens_equal": summaries["serial"]["total_output_tokens"]
        == summaries["continuous"]["total_output_tokens"],
        "all_tokens_match": comparison["all_tokens_match"],
    }
    workload_check["passed"] = all(workload_check.values())

    first_serial = all_records["serial"][0]
    manual_check = {
        "request_id": first_serial["request_id"],
        "stored": {
            key: first_serial[key]
            for key in ("queue_time", "ttft", "latency", "tpot")
        },
        "recomputed": recompute_derived(first_serial),
    }
    manual_check["matches"] = all(
        (manual_check["stored"][key] is None and manual_check["recomputed"][key] is None)
        or abs(float(manual_check["stored"][key]) - float(manual_check["recomputed"][key])) < 1e-9
        for key in manual_check["stored"]
    )

    summary: dict[str, Any] = {
        "kind": "real_model_pilot",
        "simulation": False,
        "run_id": args.run_id,
        "model_id": config["model_id"],
        "revision": config["revision"],
        "device": config["device"],
        "cuda_available": bool(torch.cuda.is_available()),
        "eos_mode": config["eos_mode"],
        "fixed_length_eos_token_id": config["fixed_length_eos_token_id"],
        "max_new_tokens": max_new_tokens,
        "num_requests": num_requests,
        "model_load_seconds_excluded": load_seconds,
        "warmup_seconds_excluded": warmup_seconds,
        "strategies": {
            strategy: {**summaries[strategy], **run_infos[strategy]}
            for strategy in config["strategies"]
        },
        "workload_check": workload_check,
        "token_comparison": comparison,
        "manual_timing_check": manual_check,
    }
    for strategy in config["strategies"]:
        stats = summary["strategies"][strategy]
        duration = float(stats["wall_duration_seconds"])
        stats["output_tokens_per_second"] = (
            int(stats["total_output_tokens"]) / duration if duration > 0 else None
        )
        stats["requests_per_second"] = (
            int(stats["finished"]) / duration if duration > 0 else None
        )

    run_dir.mkdir(parents=True)
    run_config = {
        **config,
        "kind": "real_model_pilot",
        "simulation": False,
        "run_id": args.run_id,
        "num_requests": num_requests,
        "prompt_token_ids": prompt_token_ids,
        "prompt_lengths": [len(ids) for ids in prompt_token_ids],
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for strategy, records in all_records.items():
        lines = [
            json.dumps(record, sort_keys=True) for record in records
        ]
        (run_dir / f"{strategy}_requests.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_lines = [
        "# Real model pilot summary",
        "",
        f"Run id: {args.run_id}",
        "",
        "Wall clock observations on a shared CPU only machine under the",
        "protocol in docs/benchmark_protocol.md. The pilot checks the",
        "measurement path; it is not a headline performance result.",
        "",
        "| Strategy | Finished | Failed | Output tokens | Steps | Wall seconds | Mean TTFT | Mean latency | Mean TPOT |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for strategy in config["strategies"]:
        stats = summary["strategies"][strategy]
        md_lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    str(stats["finished"]),
                    str(stats["failed"]),
                    str(stats["total_output_tokens"]),
                    str(stats["engine_steps"]),
                    f"{float(stats['wall_duration_seconds']):.4f}",
                    f"{float(stats['mean_ttft']):.4f}",
                    f"{float(stats['mean_latency']):.4f}",
                    f"{float(stats['mean_tpot']):.4f}",
                ]
            )
            + " |"
        )
    md_lines += [
        "",
        f"Workload check passed: {workload_check['passed']}",
        f"Manual timing check passed: {manual_check['matches']}",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if workload_check["passed"] and manual_check["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
