"""Run the main concurrency experiment under the frozen protocol.

This is the main experiment defined in docs/benchmark_protocol.md
section 6: serial and continuous strategies over the same closed
batch workloads, with the continuous concurrency (its scheduler
max_batch_size) varied over 1, 2, 4, and 8, at least three repeats
per configuration, and the strategy order rotated between
repeats. Serial has no concurrency variable: its scheduler
activates one request at a time by construction, so it runs once
per workload and repeat and is paired against every continuous
concurrency of that same repeat.

Two workloads are used, both built from the protocol prompts:

- short: the 10 protocol prompts cycled (prompt lengths 5 to 9).
- mixed: short protocol prompts alternating with long prompts
  made by appending one fixed suffix text to a protocol prompt
  (prompt lengths 20 to 24), so prefill costs differ by a factor
  of about three inside one batch.

Sample size is 20 requests per run, the pilot scale. The plan
targets about 100 finished requests per point; on this shared
CPU only machine that is not reachable inside the run budget, so
the real sample size is kept and every table is labelled
exploratory, as the protocol allows. Fixed length mode, warmup
exclusion, and the workload check follow the protocol unchanged.
Times are wall clock observations on a shared CPU only machine
and support no claim beyond this workload and hardware.

Usage, from the repository root, with the model extra installed:

    python benchmarks/main_benchmark.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "pilot_benchmark", ROOT / "benchmarks" / "pilot_benchmark.py"
)
assert _spec is not None and _spec.loader is not None
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

LONG_SUFFIXES = [
    "and explain in detail how the whole process works step by step for a beginner",
    "with a long history that spans several centuries and many different regions of the world",
    "because the underlying system has to coordinate many moving parts at the same time",
]

CONCURRENCIES = [1, 2, 4, 8]
WORKLOADS = ["short", "mixed"]
REPEATS = 3


def build_workload_specs(
    prompt_token_ids: list[list[int]],
    long_token_ids: list[list[int]],
    workload: str,
    num_requests: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Build one workload's specs, alternating lengths for mixed."""
    if workload not in WORKLOADS:
        raise ValueError(f"Unknown workload {workload!r}")
    specs = []
    for index in range(num_requests):
        if workload == "mixed" and index % 2 == 1:
            ids = long_token_ids[(index // 2) % len(long_token_ids)]
            prompt_index = (index // 2) % len(long_token_ids)
            kind = "long"
        else:
            ids = prompt_token_ids[index % len(prompt_token_ids)]
            prompt_index = index % len(prompt_token_ids)
            kind = "short"
        specs.append(
            {
                "request_id": f"main-{workload}-{index:02d}",
                "prompt_index": prompt_index,
                "prompt_kind": kind,
                "prompt_token_ids": list(ids),
                "max_new_tokens": max_new_tokens,
            }
        )
    return specs


def run_strategy_param(
    model: Any,
    strategy: str,
    specs: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    max_batch_size: int,
    max_batch_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one strategy with explicit batch size and token budget."""
    from miniserve.engine import Engine
    from miniserve.metrics import metrics_to_dict, summarize_request
    from miniserve.request import Request
    from miniserve.scheduler import create_scheduler

    eos = int(config["fixed_length_eos_token_id"])
    if strategy == "serial":
        from miniserve.real_executor import RealExecutor

        executor: Any = RealExecutor(model, eos_token_id=eos)
        effective_batch = 1
    else:
        from miniserve.batched_executor import BatchedRealExecutor

        executor = BatchedRealExecutor(model, eos_token_id=eos)
        effective_batch = max_batch_size

    scheduler = create_scheduler(
        strategy,
        max_batch_size=effective_batch,
        max_batch_tokens=max_batch_tokens,
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
        record["prompt_kind"] = spec["prompt_kind"]
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
    summary = pilot.summarize_records(records)
    summary.update(run_info)
    duration = float(summary["wall_duration_seconds"])
    summary["output_tokens_per_second"] = (
        int(summary["total_output_tokens"]) / duration if duration > 0 else None
    )
    summary["requests_per_second"] = (
        int(summary["finished"]) / duration if duration > 0 else None
    )
    return records, summary


def aggregate_repeats(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per repeat summaries into means and ranges."""
    keys = [
        "wall_duration_seconds",
        "mean_ttft",
        "mean_latency",
        "mean_tpot",
        "output_tokens_per_second",
        "requests_per_second",
        "engine_steps",
    ]
    aggregated: dict[str, Any] = {"repeats": len(summaries)}
    for key in keys:
        values = [float(s[key]) for s in summaries if s.get(key) is not None]
        if values:
            aggregated[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
    aggregated["finished_per_repeat"] = [int(s["finished"]) for s in summaries]
    aggregated["failed_per_repeat"] = [int(s["failed"]) for s in summaries]
    aggregated["cancelled_per_repeat"] = [int(s["cancelled"]) for s in summaries]
    aggregated["output_tokens_per_repeat"] = [
        int(s["total_output_tokens"]) for s in summaries
    ]
    return aggregated


def rotate(items: list[int], offset: int) -> list[int]:
    """Rotate a list by an offset, used to vary run order per repeat."""
    if not items:
        return items
    offset %= len(items)
    return items[offset:] + items[:offset]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="real-main-seed1234")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "benchmarks" / "results"
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = pilot.load_benchmark_config(ROOT / "configs" / "benchmark.json")
    model_config: dict[str, Any] = json.loads(
        (ROOT / "configs" / "model.json").read_text(encoding="utf-8")
    )
    if (
        config["model_id"] != model_config["model_id"]
        or config["revision"] != model_config["revision"]
    ):
        raise ValueError("benchmark and model configs disagree")

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
    long_texts = [
        f"{prompt} {LONG_SUFFIXES[index % len(LONG_SUFFIXES)]}"
        for index, prompt in enumerate(config["prompts"])
    ]
    long_token_ids = [
        list(tokenizer(text, add_special_tokens=False)["input_ids"])
        for text in long_texts
    ]
    max_new_tokens = int(config["max_new_tokens"])
    max_batch_tokens = int(config["max_batch_tokens"])

    specs_by_workload = {
        workload: build_workload_specs(
            prompt_token_ids,
            long_token_ids,
            workload,
            args.num_requests,
            max_new_tokens,
        )
        for workload in WORKLOADS
    }
    warmup_specs = pilot.build_specs(
        prompt_token_ids,
        int(config["warmup_requests_per_strategy"]),
        int(config["warmup_max_new_tokens"]),
    )
    for spec in warmup_specs:
        spec["request_id"] = f"warmup-{spec['request_id']}"
        spec.setdefault("prompt_kind", "short")

    run_dir.mkdir(parents=True)
    all_checks_passed = True
    per_config: dict[str, Any] = {}
    execution_order: list[str] = []

    for workload in WORKLOADS:
        specs = specs_by_workload[workload]
        for repeat in range(args.repeats):
            serial_records: list[dict[str, Any]] | None = None
            order = rotate(CONCURRENCIES, repeat)
            # Rotate which strategy family goes first per repeat.
            families = (
                ["serial", "continuous"] if repeat % 2 == 0 else ["continuous", "serial"]
            )
            for family in families:
                if family == "serial":
                    warmup_start = time.perf_counter()
                    run_strategy_param(
                        model,
                        "serial",
                        warmup_specs,
                        config,
                        max_batch_size=1,
                        max_batch_tokens=max_batch_tokens,
                    )
                    warmup_seconds = time.perf_counter() - warmup_start
                    records, summary = run_strategy_param(
                        model,
                        "serial",
                        specs,
                        config,
                        max_batch_size=1,
                        max_batch_tokens=max_batch_tokens,
                    )
                    serial_records = records
                    label = f"{workload}_serial_r{repeat}"
                    execution_order.append(label)
                    name = f"{workload} serial repeat {repeat}"
                    per_config.setdefault(f"{workload}|serial|1", []).append(summary)
                    path = run_dir / f"{workload}_serial_repeat{repeat}_requests.jsonl"
                    path.write_text(
                        "\n".join(json.dumps(r, sort_keys=True) for r in records)
                        + "\n",
                        encoding="utf-8",
                    )
                    print(f"{name}: wall {summary['wall_duration_seconds']:.2f}s", flush=True)
                    _ = warmup_seconds
                else:
                    for concurrency in order:
                        warmup_start = time.perf_counter()
                        run_strategy_param(
                            model,
                            "continuous",
                            warmup_specs,
                            config,
                            max_batch_size=concurrency,
                            max_batch_tokens=max_batch_tokens,
                        )
                        _ = time.perf_counter() - warmup_start
                        records, summary = run_strategy_param(
                            model,
                            "continuous",
                            specs,
                            config,
                            max_batch_size=concurrency,
                            max_batch_tokens=max_batch_tokens,
                        )
                        label = f"{workload}_continuous_c{concurrency}_r{repeat}"
                        execution_order.append(label)
                        per_config.setdefault(
                            f"{workload}|continuous|{concurrency}", []
                        ).append(summary)
                        path = (
                            run_dir
                            / f"{workload}_continuous_c{concurrency}_repeat{repeat}_requests.jsonl"
                        )
                        path.write_text(
                            "\n".join(json.dumps(r, sort_keys=True) for r in records)
                            + "\n",
                            encoding="utf-8",
                        )
                        print(
                            f"{workload} continuous c{concurrency} repeat {repeat}: "
                            f"wall {summary['wall_duration_seconds']:.2f}s",
                            flush=True,
                        )
                        if serial_records is not None:
                            comparison = pilot.compare_token_sequences(
                                serial_records, records
                            )
                            if not comparison["all_tokens_match"]:
                                all_checks_passed = False
                                print(
                                    f"TOKEN MISMATCH {workload} c{concurrency} r{repeat}: "
                                    f"{comparison['mismatched_request_ids']}",
                                    flush=True,
                                )
            # When continuous ran first, compare afterwards is not
            # possible in this loop order; the token check is then
            # covered by the stored raw files and the summary step
            # below recomputes it from disk.
    # Recompute every token comparison from the stored raw files so
    # the check covers both execution orders.
    def load_records(name: str) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in (run_dir / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    comparisons: dict[str, Any] = {}
    for workload in WORKLOADS:
        for repeat in range(args.repeats):
            serial_records = load_records(
                f"{workload}_serial_repeat{repeat}_requests.jsonl"
            )
            for concurrency in CONCURRENCIES:
                records = load_records(
                    f"{workload}_continuous_c{concurrency}_repeat{repeat}_requests.jsonl"
                )
                comparison = pilot.compare_token_sequences(serial_records, records)
                key = f"{workload}|c{concurrency}|r{repeat}"
                comparisons[key] = comparison
                if not comparison["all_tokens_match"]:
                    all_checks_passed = False

    aggregated = {
        key: aggregate_repeats(summaries) for key, summaries in per_config.items()
    }
    summary: dict[str, Any] = {
        "kind": "real_model_main_experiment",
        "simulation": False,
        "run_id": args.run_id,
        "model_id": config["model_id"],
        "revision": config["revision"],
        "device": config["device"],
        "cuda_available": bool(torch.cuda.is_available()),
        "eos_mode": config["eos_mode"],
        "max_new_tokens": max_new_tokens,
        "num_requests_per_run": args.num_requests,
        "repeats": args.repeats,
        "concurrencies": CONCURRENCIES,
        "workloads": {
            workload: {
                "prompt_lengths": [
                    len(s["prompt_token_ids"]) for s in specs_by_workload[workload]
                ],
                "total_prompt_tokens": sum(
                    len(s["prompt_token_ids"]) for s in specs_by_workload[workload]
                ),
            }
            for workload in WORKLOADS
        },
        "model_load_seconds_excluded": load_seconds,
        "execution_order": execution_order,
        "aggregated": aggregated,
        "token_comparisons_all_match": all(
            c["all_tokens_match"] for c in comparisons.values()
        ),
        "all_checks_passed": all_checks_passed,
        "sample_note": (
            "20 requests per run and 3 repeats give 60 finished requests per "
            "continuous point, below the plan target of about 100 per point; "
            "results are exploratory at this sample size."
        ),
    }
    run_config = {
        **config,
        "kind": "real_model_main_experiment",
        "simulation": False,
        "run_id": args.run_id,
        "num_requests_per_run": args.num_requests,
        "repeats": args.repeats,
        "concurrencies": CONCURRENCIES,
        "long_prompt_texts": long_texts,
        "prompt_token_ids": prompt_token_ids,
        "long_prompt_token_ids": long_token_ids,
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = [
        "# Real model main experiment summary",
        "",
        f"Run id: {args.run_id}",
        "",
        "Closed batch runs under docs/benchmark_protocol.md on a shared CPU",
        "only machine. Sample: 20 requests per run, 3 repeats, 60 finished",
        "requests per continuous point, below the plan target of about 100,",
        "so every figure is exploratory. Serial runs once per workload and",
        "repeat (it has no concurrency variable) and pairs with each",
        "continuous concurrency of the same repeat. Means are across repeats,",
        "with the repeat range in brackets.",
        "",
        "| Workload | Strategy | Concurrency | Wall s mean [min, max] | Output tok/s mean | Mean TTFT mean | Mean latency mean | Mean TPOT mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for workload in WORKLOADS:
        for strategy, concurrencies in (("serial", [1]), ("continuous", CONCURRENCIES)):
            for concurrency in concurrencies:
                agg = aggregated[f"{workload}|{strategy}|{concurrency}"]
                wall = agg["wall_duration_seconds"]
                md.append(
                    f"| {workload} | {strategy} | {concurrency} | "
                    f"{wall['mean']:.2f} [{wall['min']:.2f}, {wall['max']:.2f}] | "
                    f"{agg['output_tokens_per_second']['mean']:.2f} | "
                    f"{agg['mean_ttft']['mean']:.4f} | "
                    f"{agg['mean_latency']['mean']:.4f} | "
                    f"{agg['mean_tpot']['mean']:.4f} |"
                )
    md += [
        "",
        f"Token comparisons all match: {summary['token_comparisons_all_match']}",
        f"All checks passed: {all_checks_passed}",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"all_checks_passed": all_checks_passed}), flush=True)
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
