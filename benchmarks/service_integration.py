"""Run the minimal service integration pass over HTTP.

Five scenarios run against real HTTP servers on ephemeral
ports, mirroring the integration checklist: two normal
clients at once, one streaming client cancelled mid
generation, one invalid request and one admission rejection,
one overload refusal, and one executor failure reported as a
visible error while the worker survives. A sixth scenario is
a small real model smoke test through the same HTTP layer.
Fake backend scenarios are labelled simulation of the service
flow; the smoke test times are wall clock observations on a
shared CPU. Neither is a performance result.

Usage, from the repository root:

    python benchmarks/service_integration.py
    python benchmarks/service_integration.py --skip-real

The full record is saved to
benchmarks/service_integration_output.json.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Self


def post_json(url: str, body: Any, timeout: float = 60) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def get_json(url: str, timeout: float = 10) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class RunningService:
    def __init__(self, owner: Any, front_end: Any) -> None:
        from miniserve.server import make_server

        self.owner = owner
        self.server = make_server(front_end, port=0)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        host, port = self.server.server_address[:2]
        self.base = f"http://{host}:{port}"

    def __enter__(self) -> Self:
        self.owner.start()
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.owner.stop(timeout=30)


def fake_service(
    executor: Any = None,
    *,
    max_batch_size: int = 4,
    max_batch_tokens: int = 100,
    max_waiting_requests: int | None = 32,
) -> RunningService:
    import time as time_module

    from miniserve.engine import Engine
    from miniserve.executor import FakeExecutor
    from miniserve.scheduler import Scheduler
    from miniserve.server import ServiceFrontEnd, simple_tokenize
    from miniserve.service import EngineOwner

    if executor is None:

        class SlowFakeExecutor(FakeExecutor):
            def decode(self, requests: list) -> None:  # type: ignore[type-arg]
                if requests:
                    time_module.sleep(0.05)
                super().decode(requests)

        executor = SlowFakeExecutor()
    scheduler = Scheduler(
        max_batch_size=max_batch_size, max_batch_tokens=max_batch_tokens
    )
    owner = EngineOwner(
        Engine(scheduler, executor), max_waiting_requests=max_waiting_requests
    )
    front_end = ServiceFrontEnd(
        owner, simple_tokenize, model_id="fake-model", revision="test"
    )
    return RunningService(owner, front_end)


def scenario_two_clients() -> dict[str, Any]:
    with fake_service() as service:
        health_status, health = get_json(f"{service.base}/health")
        results: dict[str, tuple[int, Any]] = {}

        def call(name: str, prompt: str) -> None:
            results[name] = post_json(
                f"{service.base}/generate",
                {"prompt": prompt, "max_new_tokens": 4},
            )

        threads = [
            threading.Thread(target=call, args=("a", "one two three")),
            threading.Thread(target=call, args=("b", "four five")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        worker_alive_after = service.owner.health()["worker_alive"]
    passed = bool(
        health_status == 200
        and health["status"] == "ok"
        and all(
            status == 200 and body["generated_tokens"] == 4
            for status, body in results.values()
        )
        and results["a"][1]["request_id"] != results["b"][1]["request_id"]
    )
    return {
        "health": health,
        "results": {name: body for name, (_s, body) in results.items()},
        "worker_alive_after": worker_alive_after,
        "passed": passed,
    }


def scenario_cancel() -> dict[str, Any]:
    with fake_service() as service:
        payload = json.dumps(
            {"prompt": "long running prompt", "max_new_tokens": 30}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{service.base}/generate/stream",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        events: list[dict[str, Any]] = []
        cancel_result: dict[str, Any] = {}

        def read_stream() -> None:
            with urllib.request.urlopen(request, timeout=60) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        data = json.loads(line.split(":", 1)[1].strip())
                        events.append(data)
                        if len(events) == 2:
                            _status, body = post_json(
                                f"{service.base}/requests/"
                                f"{data['request_id']}/cancel",
                                {},
                            )
                            cancel_result.update(body)

        reader = threading.Thread(target=read_stream)
        reader.start()
        reader.join(timeout=60)
        follow_status, follow = post_json(
            f"{service.base}/generate",
            {"prompt": "next one", "max_new_tokens": 2},
        )
    terminals = [e for e in events if e["type"] in {"finished", "error"}]
    passed = bool(
        len(terminals) == 1
        and terminals[0]["finish_reason"] == "cancelled"
        and cancel_result.get("cancelled") is True
        and follow_status == 200
        and follow["generated_tokens"] == 2
    )
    return {
        "events": events,
        "cancel_response": cancel_result,
        "follow_up": follow,
        "passed": passed,
    }


def scenario_invalid_and_rejected() -> dict[str, Any]:
    with fake_service(max_batch_tokens=10) as service:
        invalid_status, invalid = post_json(
            f"{service.base}/generate", {"prompt": ""}
        )
        rejected_status, rejected = post_json(
            f"{service.base}/generate",
            {
                "prompt": "one two three four five six seven eight nine ten x",
                "max_new_tokens": 2,
            },
        )
    passed = bool(
        invalid_status == 400
        and invalid["error"]["code"] == "invalid_request"
        and rejected_status == 400
        and rejected["error"]["code"] == "request_rejected"
    )
    return {
        "invalid": {"status": invalid_status, "body": invalid},
        "rejected": {"status": rejected_status, "body": rejected},
        "passed": passed,
    }


def scenario_overload() -> dict[str, Any]:
    with fake_service(max_batch_size=1, max_waiting_requests=1) as service:
        outcomes: dict[str, tuple[int, Any]] = {}

        def call(name: str, tokens: int) -> None:
            outcomes[name] = post_json(
                f"{service.base}/generate",
                {"prompt": "some prompt words", "max_new_tokens": tokens},
                timeout=90,
            )

        running = threading.Thread(target=call, args=("running", 30))
        running.start()
        deadline = time.monotonic() + 15
        while service.owner.health()["active_requests"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        waiting = threading.Thread(target=call, args=("waiting", 2))
        waiting.start()
        deadline = time.monotonic() + 15
        while service.owner.health()["waiting_requests"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        overflow_status, overflow = post_json(
            f"{service.base}/generate",
            {"prompt": "one more", "max_new_tokens": 1},
            timeout=15,
        )
        running.join(timeout=90)
        waiting.join(timeout=90)
        stats = service.owner.admission_stats()
    passed = bool(
        overflow_status == 429
        and overflow["error"]["code"] == "queue_full"
        and outcomes["running"][0] == 200
        and outcomes["waiting"][0] == 200
        and stats["rejected_queue_full"] == 1
    )
    return {
        "overflow": {"status": overflow_status, "body": overflow},
        "admission_stats": stats,
        "passed": passed,
    }


def scenario_failure() -> dict[str, Any]:
    from miniserve.executor import FakeExecutor
    from miniserve.request import Request

    class PromptFailingExecutor(FakeExecutor):
        def decode(self, requests: list[Request]) -> None:
            for request in requests:
                if request.prompt_length == 13:
                    request.fail(
                        "decode forward failed: RuntimeError: injected"
                    )
                    raise RuntimeError("injected")
            super().decode(requests)

    with fake_service(PromptFailingExecutor()) as service:
        fail_status, failure = post_json(
            f"{service.base}/generate",
            {"prompt": " ".join(["word"] * 13), "max_new_tokens": 3},
        )
        health_status, health = get_json(f"{service.base}/health")
        ok_status, ok = post_json(
            f"{service.base}/generate",
            {"prompt": "short one", "max_new_tokens": 2},
        )
    passed = bool(
        fail_status == 500
        and failure["error"]["code"] == "generation_failed"
        and health_status == 200
        and health["worker_alive"] is True
        and ok_status == 200
    )
    return {
        "failure": {"status": fail_status, "body": failure},
        "health_after": health,
        "follow_up": ok,
        "passed": passed,
    }


def scenario_real_smoke() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniserve.batched_executor import BatchedRealExecutor
    from miniserve.engine import Engine
    from miniserve.scheduler import Scheduler
    from miniserve.server import ServiceFrontEnd
    from miniserve.service import EngineOwner

    root = Path(__file__).resolve().parent.parent
    recorded = json.loads(
        (root / "configs" / "model.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        recorded["tokenizer"], revision=recorded["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        recorded["model_id"],
        revision=recorded["revision"],
        dtype=torch.float32,
        attn_implementation=recorded["attention_backend"],
    )
    model.eval()
    executor = BatchedRealExecutor(
        model, eos_token_id=int(model.config.eos_token_id)
    )
    scheduler = Scheduler(
        max_batch_size=2,
        max_batch_tokens=int(model.config.max_position_embeddings),
        max_model_len=int(model.config.max_position_embeddings),
    )
    owner = EngineOwner(Engine(scheduler, executor), tokenizer=tokenizer)

    def tokenize(prompt: str) -> list[int]:
        result: list[int] = tokenizer(prompt, add_special_tokens=False)[
            "input_ids"
        ]
        return result

    front_end = ServiceFrontEnd(
        owner,
        tokenize,
        model_id=recorded["model_id"],
        revision=recorded["revision"],
    )
    service = RunningService(owner, front_end)
    with service:
        started = time.monotonic()
        status, body = post_json(
            f"{service.base}/generate",
            {"prompt": "The capital of France is", "max_new_tokens": 4},
            timeout=120,
        )
        elapsed = round(time.monotonic() - started, 4)
        health_status, health = get_json(f"{service.base}/health")
    passed = bool(
        status == 200
        and body.get("generated_tokens") == 4
        and body.get("prompt_tokens") == 5
        and bool(body.get("text"))
        and health_status == 200
        and executor.active_cache_count == 0
    )
    return {
        "model_id": recorded["model_id"],
        "revision": recorded["revision"],
        "timing_kind": (
            "wall clock observation on a shared CPU, "
            "not a performance result"
        ),
        "response": body,
        "elapsed_seconds": elapsed,
        "health": health,
        "active_caches_after_run": executor.active_cache_count,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    scenarios: dict[str, Any] = {
        "two_normal_clients": scenario_two_clients(),
        "cancel_client": scenario_cancel(),
        "invalid_and_rejected": scenario_invalid_and_rejected(),
        "overload": scenario_overload(),
        "generation_failure": scenario_failure(),
    }
    if not args.skip_real:
        scenarios["real_model_smoke"] = scenario_real_smoke()
    result: dict[str, Any] = {
        "backend": (
            "fake executors for the five flow scenarios, "
            "real locked model for the smoke test"
        ),
        "timing_kind": (
            "simulation of the service flow plus one wall clock "
            "smoke observation, not a performance result"
        ),
        "scenarios": scenarios,
    }
    result["all_checks_passed"] = bool(
        all(scenario["passed"] for scenario in scenarios.values())
    )
    out_path = (
        Path(__file__).resolve().parent / "service_integration_output.json"
    )
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
