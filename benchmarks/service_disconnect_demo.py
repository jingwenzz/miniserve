"""Demonstrate client disconnect mapped to cancellation.

Three requests are served by one engine owner with a batch
size of one and a fake executor whose decode sleeps briefly,
which stretches generation over wall clock time so the order
is visible. A waiting request is disconnected through its
handle before it ever starts, and a running request is
disconnected by closing its event stream after the first
token event, which is the signal a stream consumer gives
when it goes away. A third request submitted afterwards
completes normally. The saved record shows that the waiting
request produced no tokens and was never released (it held
no backend resources), that the running request froze at the
token count of its last completed step and was released,
that no request kept generating in the background, and that
the survivor was unaffected. The backend is fake and the
sleeps are artificial, so the record is a simulation of the
service flow, not a performance result.

Usage, from the repository root:

    python benchmarks/service_disconnect_demo.py
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

    executor = SlowFakeExecutor(delay_seconds=0.05)
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    engine = Engine(scheduler, executor)  # type: ignore[arg-type]
    owner = EngineOwner(engine)
    owner.start()
    try:
        running = owner.submit(
            Request(
                request_id="disc-running",
                prompt_token_ids=[1] * 5,
                max_new_tokens=20,
            )
        )
        waiting = owner.submit(
            Request(
                request_id="disc-waiting",
                prompt_token_ids=[1] * 3,
                max_new_tokens=4,
            )
        )
        # The waiting request's client goes away while the
        # running request is still generating.
        waiting_disconnect = waiting.disconnect()
        waiting_result = waiting.result(timeout=10)
        # The running request's client reads one token event
        # and then closes the stream, which disconnects it.
        stream = running.events()
        first_event = next(stream)
        stream.close()
        running_result = running.result(timeout=10)
        frozen_count = running_result.output_length
        time.sleep(0.3)
        still_frozen = running_result.output_length == frozen_count
        no_pending = not scheduler.has_pending_requests()
        survivor = owner.submit(
            Request(
                request_id="disc-survivor",
                prompt_token_ids=[1] * 3,
                max_new_tokens=3,
            )
        )
        survivor_result = survivor.result(timeout=10)
        stats = owner.admission_stats()
    finally:
        owner.stop(timeout=10)

    result: dict[str, Any] = {
        "backend": "fake executor with an artificial decode delay",
        "timing_kind": "simulation of the service flow, not a performance result",
        "waiting": {
            "request_id": "disc-waiting",
            "disconnect_returned": waiting_disconnect,
            "state": waiting_result.state.name,
            "generated_tokens": waiting_result.output_length,
            "released": "disc-waiting" in executor.released_request_ids,
        },
        "running": {
            "request_id": "disc-running",
            "first_event_type": first_event.type,
            "state": running_result.state.name,
            "generated_tokens": frozen_count,
            "output_limit": 20,
            "still_frozen_after_0.3s": still_frozen,
            "released": "disc-running" in executor.released_request_ids,
        },
        "survivor": {
            "request_id": "disc-survivor",
            "state": survivor_result.state.name,
            "generated_tokens": survivor_result.output_length,
        },
        "no_pending_requests_after_disconnects": no_pending,
        "released_request_ids": list(executor.released_request_ids),
        "disconnect_cancellations": stats["disconnect_cancellations"],
    }
    result["all_checks_passed"] = bool(
        waiting_disconnect
        and waiting_result.state.name == "CANCELLED"
        and waiting_result.output_length == 0
        and "disc-waiting" not in executor.released_request_ids
        and first_event.type == "token"
        and running_result.state.name == "CANCELLED"
        and 1 <= frozen_count < 20
        and still_frozen
        and no_pending
        and survivor_result.state.name == "FINISHED"
        and survivor_result.output_length == 3
        and stats["disconnect_cancellations"] == 2
    )
    out_path = (
        Path(__file__).resolve().parent / "service_disconnect_output.json"
    )
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
