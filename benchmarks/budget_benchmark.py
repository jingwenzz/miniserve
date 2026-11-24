"""Run the token budget mechanism experiment under the frozen protocol.

A fixed medium load (the mixed workload of the main experiment,
continuous strategy, concurrency 4, 20 requests, fixed length
mode) is run under three per step token budgets: 32, 64, and the
protocol default of 8192. The smallest budget still admits every
prompt in the workload on its own (the longest mixed prompt is
24 tokens, and admission rejects a prompt longer than the
budget), so the accepted request set is identical across tiers
and any timing difference is a budget effect, not a changed
workload. Three repeats per tier, tier order rotated.

The token budget bounds the work one scheduler step may pick:
decode requests cost 1 each and are picked first, then prefill
requests are added while their prompt lengths fit the remaining
budget. A small budget therefore serialises prefill across more
steps, which is expected to raise queue time and TTFT for later
requests while leaving per token decode cost roughly unchanged.
The run records one concrete request timeline (the first long
prompt request) across tiers so the mechanism can be read off a
single request, not only off means.

Usage, from the repository root, with the model extra installed:

    python benchmarks/budget_benchmark.py
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
    "main_benchmark", ROOT / "benchmarks" / "main_benchmark.py"
)
assert _spec is not None and _spec.loader is not None
main_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(main_bench)
pilot = main_bench.pilot

BUDGETS = [32, 64, 8192]
CONCURRENCY = 4
WORKLOAD = "mixed"
REPEATS = 3
TIMELINE_REQUEST_ID = "main-mixed-01"


def validate_budgets(budgets: list[int], specs: list[dict[str, Any]]) -> None:
    """Every budget must admit the longest prompt on its own."""
    longest = max(len(s["prompt_token_ids"]) for s in specs)
    for budget in budgets:
        if budget < longest:
            raise ValueError(
                f"Budget {budget} is below the longest prompt of {longest} "
                "tokens; admission would reject requests and the accepted "
                "set would differ across tiers."
            )


def timeline_for(
    records_by_budget: dict[int, list[list[dict[str, Any]]]],
    request_id: str,
) -> dict[str, Any]:
    """Collect one request's timing fields across budgets and repeats."""
    timeline: dict[str, Any] = {"request_id": request_id, "budgets": {}}
    for budget, repeats in records_by_budget.items():
        entries = []
        for records in repeats:
            match = [r for r in records if r["request_id"] == request_id]
            if len(match) != 1:
                raise ValueError(f"Expected one record for {request_id}")
            record = match[0]
            entries.append(
                {
                    key: record.get(key)
                    for key in (
                        "prompt_length",
                        "output_length",
                        "queue_time",
                        "ttft",
                        "latency",
                        "tpot",
                    )
                }
            )
        timeline["budgets"][str(budget)] = entries
    return timeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="real-budget-seed1234")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "benchmarks" / "results"
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = pilot.load_benchmark_config(ROOT / "configs" / "benchmark.json")

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
        f"{prompt} {main_bench.LONG_SUFFIXES[index % len(main_bench.LONG_SUFFIXES)]}"
        for index, prompt in enumerate(config["prompts"])
    ]
    long_token_ids = [
        list(tokenizer(text, add_special_tokens=False)["input_ids"])
        for text in long_texts
    ]
    specs = main_bench.build_workload_specs(
        prompt_token_ids,
        long_token_ids,
        WORKLOAD,
        args.num_requests,
        int(config["max_new_tokens"]),
    )
    validate_budgets(BUDGETS, specs)
    warmup_specs = pilot.build_specs(
        prompt_token_ids,
        int(config["warmup_requests_per_strategy"]),
        int(config["warmup_max_new_tokens"]),
    )
    for spec in warmup_specs:
        spec["request_id"] = f"warmup-{spec['request_id']}"
        spec.setdefault("prompt_kind", "short")

    run_dir.mkdir(parents=True)
    per_budget: dict[int, list[dict[str, Any]]] = {b: [] for b in BUDGETS}
    records_by_budget: dict[int, list[list[dict[str, Any]]]] = {b: [] for b in BUDGETS}
    reference_records: list[dict[str, Any]] | None = None
    all_checks_passed = True
    execution_order: list[str] = []

    for repeat in range(args.repeats):
        for budget in main_bench.rotate(BUDGETS, repeat):
            main_bench.run_strategy_param(
                model,
                "continuous",
                warmup_specs,
                config,
                max_batch_size=CONCURRENCY,
                max_batch_tokens=budget,
            )
            records, summary = main_bench.run_strategy_param(
                model,
                "continuous",
                specs,
                config,
                max_batch_size=CONCURRENCY,
                max_batch_tokens=budget,
            )
            execution_order.append(f"budget{budget}_r{repeat}")
            per_budget[budget].append(summary)
            records_by_budget[budget].append(records)
            path = run_dir / f"budget{budget}_repeat{repeat}_requests.jsonl"
            path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                encoding="utf-8",
            )
            print(
                f"budget {budget} repeat {repeat}: wall "
                f"{summary['wall_duration_seconds']:.2f}s steps "
                f"{summary['engine_steps']}",
                flush=True,
            )
            if reference_records is None:
                reference_records = records
            else:
                comparison = pilot.compare_token_sequences(reference_records, records)
                if not comparison["all_tokens_match"]:
                    all_checks_passed = False
                    print(
                        f"TOKEN MISMATCH budget {budget} r{repeat}: "
                        f"{comparison['mismatched_request_ids']}",
                        flush=True,
                    )

    aggregated = {
        str(budget): main_bench.aggregate_repeats(summaries)
        for budget, summaries in per_budget.items()
    }
    timeline = timeline_for(records_by_budget, TIMELINE_REQUEST_ID)
    summary: dict[str, Any] = {
        "kind": "real_model_budget_experiment",
        "simulation": False,
        "run_id": args.run_id,
        "model_id": config["model_id"],
        "revision": config["revision"],
        "device": config["device"],
        "cuda_available": bool(torch.cuda.is_available()),
        "eos_mode": config["eos_mode"],
        "workload": WORKLOAD,
        "strategy": "continuous",
        "concurrency": CONCURRENCY,
        "budgets": BUDGETS,
        "num_requests_per_run": args.num_requests,
        "repeats": args.repeats,
        "prompt_lengths": [len(s["prompt_token_ids"]) for s in specs],
        "accepted_request_set_identical": True,
        "model_load_seconds_excluded": load_seconds,
        "execution_order": execution_order,
        "aggregated": aggregated,
        "timeline_request": timeline,
        "all_checks_passed": all_checks_passed,
    }
    run_config = {
        **config,
        "kind": "real_model_budget_experiment",
        "simulation": False,
        "run_id": args.run_id,
        "budgets": BUDGETS,
        "concurrency": CONCURRENCY,
        "workload": WORKLOAD,
        "long_prompt_texts": long_texts,
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = [
        "# Real model token budget experiment summary",
        "",
        f"Run id: {args.run_id}",
        "",
        "Fixed mixed workload, continuous strategy, concurrency 4, 20 requests",
        "per run, 3 repeats, under docs/benchmark_protocol.md on a shared CPU",
        "only machine. The accepted request set is identical at every budget",
        "(the longest prompt, 24 tokens, fits the smallest budget of 32), so",
        "differences are budget effects. Exploratory at this sample size.",
        "",
        "| Budget | Steps mean | Wall s mean [min, max] | Output tok/s mean | Mean TTFT mean | Mean latency mean | Mean TPOT mean |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for budget in BUDGETS:
        agg = aggregated[str(budget)]
        wall = agg["wall_duration_seconds"]
        md.append(
            f"| {budget} | {agg['engine_steps']['mean']:.1f} | "
            f"{wall['mean']:.2f} [{wall['min']:.2f}, {wall['max']:.2f}] | "
            f"{agg['output_tokens_per_second']['mean']:.2f} | "
            f"{agg['mean_ttft']['mean']:.4f} | "
            f"{agg['mean_latency']['mean']:.4f} | "
            f"{agg['mean_tpot']['mean']:.4f} |"
        )
    md += [
        "",
        f"Timeline request: {TIMELINE_REQUEST_ID} (first long prompt request)",
        "",
        "| Budget | Queue time per repeat | TTFT per repeat | Latency per repeat |",
        "| --- | --- | --- | --- |",
    ]
    for budget in BUDGETS:
        entries = timeline["budgets"][str(budget)]
        md.append(
            f"| {budget} | "
            + ", ".join(f"{e['queue_time']:.4f}" for e in entries)
            + " | "
            + ", ".join(f"{e['ttft']:.4f}" for e in entries)
            + " | "
            + ", ".join(f"{e['latency']:.4f}" for e in entries)
            + " |"
        )
    md += ["", f"All checks passed: {all_checks_passed}", ""]
    (run_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"all_checks_passed": all_checks_passed}), flush=True)
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
