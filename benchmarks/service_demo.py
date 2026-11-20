"""Demonstrate background execution behind a single engine owner.

Two requests are served by one engine owner running a fake
executor whose decode sleeps briefly, which stretches the
generation over wall clock time so the ordering is visible:
the second request is submitted, and its submission returns,
while the first request is still generating. The saved record
shows the order of events, that both requests finished with
the expected token counts, and that every executor call ran
on the single worker thread. The backend is fake and the
sleeps are artificial, so the times in the record are a
simulation of the service flow, not a performance result.

Usage, from the repository root:

    python benchmarks/service_demo.py
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class SlowFakeExecutor:
    """Wrap the fake executor with a small decode delay."""

    def __init__(self, delay_seconds: float = 0.05) -> None:
        from miniserve.executor import FakeExecutor

        self._inner = FakeExecutor()
        self._delay = delay_seconds
        self.thread_ids: set[int] = set()

    def prefill(self, requests: list) -> None:  # type: ignore[type-arg]
        self.thread_ids.add(threading.get_ident())
        self._inner.prefill(requests)

    def decode(self, requests: list) -> None:  # type: ignore[type-arg]
        self.thread_ids.add(threading.get_ident())
        if requests:
            time.sleep(self._delay)
        self._inner.decode(requests)

    def release(self, request: object) -> None:
        self.thread_ids.add(threading.get_ident())
        self._inner.release(request)  # type: ignore[arg-type]

    @property
    def released_request_ids(self) -> list[str]:
        return self._inner.released_request_ids


def main() -> int:
    from miniserve.engine import Engine
    from miniserve.request import Request
    from miniserve.scheduler import Scheduler
    from miniserve.service import EngineOwner

    executor = SlowFakeExecutor()
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=100)
    engine = Engine(scheduler, executor)  # type: ignore[arg-type]
    owner = EngineOwner(engine)
    order: list[dict[str, Any]] = []
    start = time.monotonic()

    def stamp(event: str) -> None:
        order.append(
            {"event": event, "at_seconds": round(time.monotonic() - start, 4)}
        )

    owner.start()
    try:
        first = owner.submit(
            Request(
                request_id="demo-a",
                prompt_token_ids=[1] * 5,
                max_new_tokens=6,
            )
        )
        stamp("first_submitted")
        deadline = time.monotonic() + 5
        while first.request.output_length == 0:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        stamp("first_token_observed")
        second = owner.submit(
            Request(
                request_id="demo-b",
                prompt_token_ids=[1] * 3,
                max_new_tokens=4,
            )
        )
        stamp("second_submitted")
        second_submitted_before_first_finished = not first.done()
        first_result = first.result(timeout=10)
        stamp("first_finished")
        second_result = second.result(timeout=10)
        stamp("second_finished")
        health = owner.health()
    finally:
        owner.stop(timeout=10)

    result: dict[str, Any] = {
        "backend": "fake executor with an artificial decode delay",
        "timing_kind": "simulation of the service flow, not a performance result",
        "order": order,
        "second_submitted_before_first_finished": (
            second_submitted_before_first_finished
        ),
        "first": {
            "request_id": "demo-a",
            "state": first_result.state.name,
            "generated_tokens": first_result.output_length,
        },
        "second": {
            "request_id": "demo-b",
            "state": second_result.state.name,
            "generated_tokens": second_result.output_length,
        },
        "engine_steps": engine.step_count,
        "executor_thread_ids": sorted(executor.thread_ids),
        "worker_thread_id": owner.worker_thread_id,
        "single_worker_thread": executor.thread_ids
        == {owner.worker_thread_id},
        "released_request_ids": list(executor.released_request_ids),
        "health_while_running": health,
    }
    result["all_checks_passed"] = bool(
        result["second_submitted_before_first_finished"]
        and result["single_worker_thread"]
        and first_result.output_length == 6
        and second_result.output_length == 4
        and sorted(executor.released_request_ids) == ["demo-a", "demo-b"]
    )
    out_path = Path(__file__).resolve().parent / "service_demo_output.json"
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
