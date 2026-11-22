"""Integration tests for the minimal HTTP service.

Every test talks to a real HTTP server on an ephemeral port
through urllib, against fake executors, so the whole service
path (handler threads, front end translation, engine owner,
scheduler) is exercised exactly as a client sees it. One
executor sleeps briefly per decode so concurrent clients
genuinely overlap; one fails a named prompt so the failure
response is checked end to end.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import Request
from miniserve.scheduler import Scheduler
from miniserve.server import ServiceFrontEnd, make_server, simple_tokenize
from miniserve.service import EngineOwner


class SlowFakeExecutor(FakeExecutor):
    def decode(self, requests: list[Request]) -> None:
        if requests:
            time.sleep(0.05)
        super().decode(requests)


class PromptFailingExecutor(FakeExecutor):
    """Fail the request whose prompt tokenizes to length 13."""

    def decode(self, requests: list[Request]) -> None:
        for request in requests:
            if request.prompt_length == 13:
                request.fail("decode forward failed: RuntimeError: injected")
                raise RuntimeError("injected")
        super().decode(requests)


@contextmanager
def running_service(
    executor: FakeExecutor | None = None,
    *,
    max_batch_size: int = 4,
    max_batch_tokens: int = 100,
    max_waiting_requests: int | None = 32,
) -> Iterator[tuple[str, EngineOwner]]:
    scheduler = Scheduler(
        max_batch_size=max_batch_size, max_batch_tokens=max_batch_tokens
    )
    engine = Engine(scheduler, executor or SlowFakeExecutor())
    owner = EngineOwner(engine, max_waiting_requests=max_waiting_requests)
    front_end = ServiceFrontEnd(
        owner, simple_tokenize, model_id="fake-model", revision="test"
    )
    server = make_server(front_end, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    owner.start()
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", owner
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        owner.stop(timeout=10)


def post_json(
    url: str, body: Any, timeout: float = 30
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def get_json(url: str, timeout: float = 10) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_two_normal_clients_complete_concurrently() -> None:
    with running_service() as (base, owner):
        status, health = get_json(f"{base}/health")
        assert status == 200
        assert health["status"] == "ok"
        assert health["model_id"] == "fake-model"
        results: dict[str, tuple[int, dict[str, Any]]] = {}

        def call(name: str, prompt: str) -> None:
            results[name] = post_json(
                f"{base}/generate", {"prompt": prompt, "max_new_tokens": 4}
            )

        threads = [
            threading.Thread(target=call, args=("a", "one two three")),
            threading.Thread(target=call, args=("b", "four five")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        for name in ("a", "b"):
            status, body = results[name]
            assert status == 200
            assert body["finish_reason"] == "finished"
            assert body["generated_tokens"] == 4
            assert body["request_id"].startswith("req-")
        assert results["a"][1]["request_id"] != results["b"][1]["request_id"]
        assert owner.health()["worker_alive"] is True


def test_streaming_client_and_cancellation() -> None:
    with running_service() as (base, owner):
        payload = json.dumps(
            {"prompt": "long running prompt", "max_new_tokens": 30}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/generate/stream",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        events: list[dict[str, Any]] = []
        request_ids: list[str] = []

        def read_stream() -> None:
            with urllib.request.urlopen(request, timeout=30) as response:
                current_event = ""
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line.split(":", 1)[1].strip())
                        data["_sse_event"] = current_event
                        events.append(data)
                        if data.get("request_id"):
                            request_ids.append(data["request_id"])
                        if len(events) == 2:
                            status, body = post_json(
                                f"{base}/requests/{request_ids[0]}/cancel", {}
                            )
                            assert status == 200
                            assert body["cancelled"] is True

        reader = threading.Thread(target=read_stream)
        reader.start()
        reader.join(timeout=30)
        assert not reader.is_alive()
        terminals = [e for e in events if e["type"] in {"finished", "error"}]
        assert len(terminals) == 1
        assert terminals[0]["type"] == "finished"
        assert terminals[0]["finish_reason"] == "cancelled"
        assert terminals[0]["generated_tokens"] < 30
        # A second cancel of the ended request is a safe no-op,
        # and an unknown id is the contract's 404.
        status, body = post_json(
            f"{base}/requests/{request_ids[0]}/cancel", {}
        )
        assert status == 200
        assert body["cancelled"] is False
        assert body["finish_reason"] == "cancelled"
        status, body = post_json(f"{base}/requests/req-never/cancel", {})
        assert status == 404
        assert body["error"]["code"] == "unknown_request"
        # The service is unaffected and serves the next client.
        status, body = post_json(
            f"{base}/generate", {"prompt": "next one", "max_new_tokens": 2}
        )
        assert status == 200
        assert body["generated_tokens"] == 2
        assert owner.admission_stats()["accepted"] >= 2


def test_invalid_and_rejected_requests_get_error_responses() -> None:
    with running_service(max_batch_tokens=10) as (base, _owner):
        for bad_body in (
            {},
            {"prompt": ""},
            {"prompt": 42},
            {"prompt": "ok", "max_new_tokens": 0},
            {"prompt": "ok", "max_new_tokens": "four"},
            ["not", "an", "object"],
        ):
            status, body = post_json(f"{base}/generate", bad_body)
            assert status == 400, bad_body
            assert body["error"]["code"] == "invalid_request"
            assert body["error"]["request_id"] is None
        # Eleven words tokenize to eleven prompt tokens, over
        # the step budget of ten: admission refuses it at once.
        status, body = post_json(
            f"{base}/generate",
            {
                "prompt": "one two three four five six seven eight nine ten x",
                "max_new_tokens": 2,
            },
        )
        assert status == 400
        assert body["error"]["code"] == "request_rejected"
        # Malformed JSON is an invalid request, not a hang.
        raw = urllib.request.Request(
            f"{base}/generate",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(raw, timeout=10)
            raise AssertionError("malformed JSON was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read().decode("utf-8"))
            assert body["error"]["code"] == "invalid_request"


def test_overload_over_http_returns_queue_full() -> None:
    with running_service(max_batch_size=1, max_waiting_requests=1) as (
        base,
        owner,
    ):
        outcomes: dict[str, tuple[int, dict[str, Any]]] = {}

        def call(name: str, tokens: int) -> None:
            outcomes[name] = post_json(
                f"{base}/generate",
                {"prompt": "some prompt words", "max_new_tokens": tokens},
                timeout=60,
            )

        running = threading.Thread(target=call, args=("running", 30))
        running.start()
        deadline = time.monotonic() + 10
        while owner.health()["active_requests"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        waiting = threading.Thread(target=call, args=("waiting", 2))
        waiting.start()
        deadline = time.monotonic() + 10
        while owner.health()["waiting_requests"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        status, body = post_json(
            f"{base}/generate",
            {"prompt": "one more", "max_new_tokens": 1},
            timeout=10,
        )
        assert status == 429
        assert body["error"]["code"] == "queue_full"
        running.join(timeout=60)
        waiting.join(timeout=60)
        assert outcomes["running"][0] == 200
        assert outcomes["waiting"][0] == 200
        stats = owner.admission_stats()
        assert stats["rejected_queue_full"] == 1
        assert stats["queue_full_rejection_rate"] > 0


def test_generation_failure_is_a_visible_error_and_worker_survives() -> None:
    executor = PromptFailingExecutor()
    with running_service(executor) as (base, owner):
        # Thirteen words tokenize to thirteen prompt tokens,
        # which is the prompt length this executor fails on.
        status, body = post_json(
            f"{base}/generate",
            {
                "prompt": " ".join(["word"] * 13),
                "max_new_tokens": 3,
            },
        )
        assert status == 500
        assert body["error"]["code"] == "generation_failed"
        assert "injected" in body["error"]["message"]
        assert body["error"]["request_id"] is not None
        status, health = get_json(f"{base}/health")
        assert status == 200
        assert health["worker_alive"] is True
        status, body = post_json(
            f"{base}/generate", {"prompt": "short one", "max_new_tokens": 2}
        )
        assert status == 200
        assert body["finish_reason"] == "finished"
        assert owner.last_step_error is not None


def test_shutdown_leaves_no_worker_behind() -> None:
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=100)
    engine = Engine(scheduler, FakeExecutor())
    owner = EngineOwner(engine)
    front_end = ServiceFrontEnd(owner, simple_tokenize)
    server = make_server(front_end, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    owner.start()
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    status, _body = post_json(
        f"{base}/generate", {"prompt": "quick one", "max_new_tokens": 1}
    )
    assert status == 200
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    owner.stop(timeout=10)
    assert not thread.is_alive()
    assert owner.health()["worker_alive"] is False
    assert owner.health()["status"] == "unavailable"
