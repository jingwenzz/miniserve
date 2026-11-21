"""Demonstrate overload: a full waiting queue and a slow reader.

Two scenarios run against two separate engine owners, both
with fake executors, so every number here is a simulation of
the service flow, not a performance result.

The overload scenario uses a batch size of one and a waiting
bound of two. One long request runs, two more fill the
waiting queue, and two further submissions are refused at
once with the queue full error. The record reports the full
accounting, offered, accepted, refused, and the rejection
rate, not just the accepted requests, and it shows that a
refused id can be submitted again once the queue drains.

The slow consumer scenario gives each handle an event buffer
of two and never reads the first request's stream while it
generates. The worker never blocks on the full buffer: the
request is cancelled when its third token event cannot fit,
the dropped events are counted, its terminal event is still
delivered when the reader finally arrives, and a second
request with a reader that keeps up completes with every
event intact.

Usage, from the repository root:

    python benchmarks/service_overload_demo.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def run_overload_scenario() -> dict[str, Any]:
    from miniserve.engine import Engine
    from miniserve.executor import FakeExecutor
    from miniserve.request import Request
    from miniserve.scheduler import Scheduler
    from miniserve.service import EngineOwner, QueueFullError

    class SlowFakeExecutor(FakeExecutor):
        def decode(self, requests: list) -> None:  # type: ignore[type-arg]
            if requests:
                time.sleep(0.05)
            super().decode(requests)

    executor = SlowFakeExecutor()
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    owner = EngineOwner(engine, max_waiting_requests=2)
    refused: list[dict[str, Any]] = []
    owner.start()
    try:
        running = owner.submit(
            Request(
                request_id="ov-running",
                prompt_token_ids=[1] * 5,
                max_new_tokens=30,
            )
        )
        waiting_a = owner.submit(
            Request(
                request_id="ov-waiting-a",
                prompt_token_ids=[1] * 3,
                max_new_tokens=2,
            )
        )
        waiting_b = owner.submit(
            Request(
                request_id="ov-waiting-b",
                prompt_token_ids=[1] * 3,
                max_new_tokens=2,
            )
        )
        for request_id in ("ov-overflow-1", "ov-overflow-2"):
            try:
                owner.submit(
                    Request(
                        request_id=request_id,
                        prompt_token_ids=[1] * 3,
                        max_new_tokens=2,
                    )
                )
            except QueueFullError as exc:
                refused.append(
                    {
                        "request_id": request_id,
                        "error_code": exc.error_code,
                        "waiting_limit": exc.waiting_limit,
                        "waiting_count": exc.waiting_count,
                    }
                )
        stats = owner.admission_stats()
        results = [
            handle.result(timeout=30)
            for handle in (running, waiting_a, waiting_b)
        ]
        retry = owner.submit(
            Request(
                request_id="ov-overflow-1",
                prompt_token_ids=[1] * 3,
                max_new_tokens=1,
            )
        )
        retry_result = retry.result(timeout=30)
    finally:
        owner.stop(timeout=30)
    return {
        "refused": refused,
        "stats_at_refusal": stats,
        "accepted_states": [request.state.name for request in results],
        "retry_after_drain": {
            "request_id": "ov-overflow-1",
            "state": retry_result.state.name,
            "generated_tokens": retry_result.output_length,
        },
        "passed": bool(
            len(refused) == 2
            and all(item["error_code"] == "queue_full" for item in refused)
            and stats["offered"] == 5
            and stats["accepted"] == 3
            and stats["rejected_queue_full"] == 2
            and abs(stats["queue_full_rejection_rate"] - 0.4) < 1e-9
            and all(state == "FINISHED" for state in
                    [request.state.name for request in results])
            and retry_result.state.name == "FINISHED"
        ),
    }


def run_slow_consumer_scenario() -> dict[str, Any]:
    from miniserve.engine import Engine
    from miniserve.executor import FakeExecutor
    from miniserve.request import Request
    from miniserve.scheduler import Scheduler
    from miniserve.service import EngineOwner

    executor = FakeExecutor()
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=100)
    engine = Engine(scheduler, executor)
    owner = EngineOwner(engine, max_event_buffer=2)
    owner.start()
    try:
        slow = owner.submit(
            Request(
                request_id="ov-slow",
                prompt_token_ids=[1] * 3,
                max_new_tokens=6,
            )
        )
        slow_result = slow.result(timeout=10)
        slow_events = list(slow.events())
        good = owner.submit(
            Request(
                request_id="ov-good",
                prompt_token_ids=[1] * 3,
                max_new_tokens=1,
            )
        )
        good_events = list(good.events())
        good_result = good.result(timeout=10)
        stats = owner.admission_stats()
        worker_alive = owner.health()["worker_alive"]
    finally:
        owner.stop(timeout=10)
    return {
        "slow": {
            "request_id": "ov-slow",
            "state": slow_result.state.name,
            "generated_tokens": slow_result.output_length,
            "slow_consumer": slow.slow_consumer,
            "dropped_events": slow.dropped_events,
            "buffer_limit": slow.event_buffer_limit,
            "events_read_afterwards": [event.type for event in slow_events],
            "terminal_finish_reason": slow_events[-1].finish_reason,
            "terminal_generated_tokens": slow_events[-1].generated_tokens,
            "released": "ov-slow" in executor.released_request_ids,
        },
        "good": {
            "request_id": "ov-good",
            "state": good_result.state.name,
            "generated_tokens": good_result.output_length,
            "dropped_events": good.dropped_events,
            "events": [event.type for event in good_events],
        },
        "slow_consumer_cancellations": stats["slow_consumer_cancellations"],
        "worker_alive_after": worker_alive,
        "passed": bool(
            slow_result.state.name == "CANCELLED"
            and slow_result.output_length == 3
            and slow.slow_consumer
            and slow.dropped_events >= 1
            and slow_events[-1].is_terminal
            and slow_events[-1].finish_reason == "cancelled"
            and good_result.state.name == "FINISHED"
            and good.dropped_events == 0
            and stats["slow_consumer_cancellations"] == 1
            and worker_alive
        ),
    }


def main() -> int:
    overload = run_overload_scenario()
    slow_consumer = run_slow_consumer_scenario()
    result: dict[str, Any] = {
        "backend": "fake executors, one with an artificial decode delay",
        "timing_kind": "simulation of the service flow, not a performance result",
        "overload": overload,
        "slow_consumer": slow_consumer,
    }
    result["all_checks_passed"] = bool(
        overload["passed"] and slow_consumer["passed"]
    )
    out_path = Path(__file__).resolve().parent / "service_overload_output.json"
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
