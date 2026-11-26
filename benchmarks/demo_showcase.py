"""Run the three act showcase used by the demo script.

Act 1, concurrent streaming: two requests are submitted to one
engine owner; the second submission returns while the first is
still generating, and both finish with their full token counts.

Act 2, dynamic join: a third request is submitted while a long
request is mid generation; it is admitted, joins the running
work at a step boundary, and finishes without waiting for the
long request to end first.

Act 3, cancel and overload: with a waiting limit of one, a third
submission is refused with the queue full error while one request
runs and one waits; the waiting request is then cancelled, and
the running request still completes normally.

The backend is the fake executor with an artificial decode delay,
so the ordering is visible in wall clock time. Every number in
the saved record is a simulation of the service flow, not a
model performance result. The real model path is demonstrated
separately by benchmarks/service_streaming_demo.py and
benchmarks/batched_join_generate.py.

Usage, from the repository root:

    python benchmarks/demo_showcase.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class SlowFakeExecutor:
    """Wrap the fake executor with a small decode delay."""

    def __init__(self, delay_seconds: float = 0.05) -> None:
        from miniserve.executor import FakeExecutor

        self._inner = FakeExecutor()
        self._delay = delay_seconds

    def prefill(self, requests: list) -> None:  # type: ignore[type-arg]
        self._inner.prefill(requests)

    def decode(self, requests: list) -> None:  # type: ignore[type-arg]
        if requests:
            time.sleep(self._delay)
        self._inner.decode(requests)

    def release(self, request: object) -> None:
        self._inner.release(request)  # type: ignore[arg-type]

    @property
    def released_request_ids(self) -> list[str]:
        return self._inner.released_request_ids


def _request(request_id: str, prompt_len: int, output_len: int):  # type: ignore[no-untyped-def]
    from miniserve.request import Request

    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt_len,
        max_new_tokens=output_len,
    )


def main() -> int:
    from miniserve.engine import Engine
    from miniserve.scheduler import Scheduler
    from miniserve.service import EngineOwner, QueueFullError

    record: dict[str, Any] = {"kind": "demo_showcase", "backend": "fake_simulation"}
    checks: dict[str, bool] = {}

    # Act 1 and Act 2 share one owner.
    executor = SlowFakeExecutor()
    engine = Engine(Scheduler(max_batch_size=4, max_batch_tokens=100), executor)  # type: ignore[arg-type]
    owner = EngineOwner(engine)
    owner.start()
    try:
        first = owner.submit(_request("show-a", 5, 12))
        second = owner.submit(_request("show-b", 5, 6))
        checks["second_submit_returned_while_first_running"] = not first.done()
        first_result = first.result(timeout=30)
        second_result = second.result(timeout=30)
        checks["act1_both_finished_full_length"] = (
            first_result.output_length == 12 and second_result.output_length == 6
        )
        record["act1"] = {
            "show_a_output": first_result.output_length,
            "show_b_output": second_result.output_length,
        }

        long_handle = owner.submit(_request("show-long", 5, 16))
        time.sleep(0.2)
        joined = owner.submit(_request("show-join", 5, 4))
        checks["join_submit_returned_while_long_running"] = not long_handle.done()
        joined_result = joined.result(timeout=30)
        long_result = long_handle.result(timeout=30)
        checks["act2_join_finished_full_length"] = (
            joined_result.output_length == 4 and long_result.output_length == 16
        )
        record["act2"] = {
            "show_join_output": joined_result.output_length,
            "show_long_output": long_result.output_length,
        }
    finally:
        owner.stop()

    # Act 3 uses its own owner with a waiting limit of one.
    executor3 = SlowFakeExecutor()
    engine3 = Engine(Scheduler(max_batch_size=1, max_batch_tokens=100), executor3)  # type: ignore[arg-type]
    owner3 = EngineOwner(engine3, max_waiting_requests=1)
    owner3.start()
    try:
        running = owner3.submit(_request("show-run", 5, 16))
        time.sleep(0.2)
        waiting = owner3.submit(_request("show-wait", 5, 4))
        refused: str | None = None
        try:
            owner3.submit(_request("show-refused", 5, 4))
        except QueueFullError as exc:
            refused = exc.error_code
        checks["act3_overload_refused_queue_full"] = refused == "queue_full"
        cancelled = owner3.cancel("show-wait")
        waiting_result = waiting.result(timeout=30)
        running_result = running.result(timeout=30)
        checks["act3_waiting_cancelled"] = (
            cancelled and waiting_result.finish_reason == "cancelled"
        )
        checks["act3_running_still_finished"] = (
            running_result.output_length == 16
            and running_result.finish_reason == "finished"
        )
        record["act3"] = {
            "refused_error_code": refused,
            "waiting_finish_reason": waiting_result.finish_reason,
            "running_output": running_result.output_length,
            "admission_stats": owner3.admission_stats(),
        }
    finally:
        owner3.stop()

    record["checks"] = checks
    record["all_checks_passed"] = all(checks.values())
    out = Path(__file__).with_name("demo_showcase_output.json")
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"all_checks_passed={record['all_checks_passed']}")
    return 0 if record["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
