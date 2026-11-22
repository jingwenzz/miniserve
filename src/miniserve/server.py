"""Minimal HTTP front end for the engine owner.

This module is the fifth staged step of the service contract
in ``docs/service.md``: a thin HTTP layer over
:class:`miniserve.service.EngineOwner`. It implements exactly
the four endpoints of the contract (``GET /health``,
``POST /generate``, ``POST /generate/stream``, and
``POST /requests/{request_id}/cancel``) with the standard
library HTTP server, so the project gains no new runtime
dependency. The layer only parses and validates request
bodies, assigns request ids, translates runtime outcomes
into the contract's result and error shapes, and carries
stream events to the client. It never touches the scheduler,
the engine, or an executor directly: every generation goes
through the owner, which keeps model execution on its single
worker thread.

Error mapping follows the contract table: a malformed body
is ``invalid_request`` (400), a scheduler admission refusal
is ``request_rejected`` (400), a full waiting queue is
``queue_full`` (429), an executor failure after admission is
``generation_failed`` (500), and cancelling an id the service
never admitted is ``unknown_request`` (404). No error is
reported as a hanging connection: every failure path writes
a response, and a streaming client that goes away is mapped
to the owner's disconnect command, so its request is
cancelled instead of generating into a dead socket.
"""

import json
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from miniserve.request import Request
from miniserve.scheduler import RequestRejectedError
from miniserve.service import (
    EngineOwner,
    GenerationFailedError,
    QueueFullError,
    RequestHandle,
)

TokenizeFn = Callable[[str], list[int]]

DEFAULT_MAX_NEW_TOKENS = 16
RESULT_TIMEOUT_SECONDS = 300.0


def simple_tokenize(prompt: str) -> list[int]:
    """Tokenize for the fake backend: one token per word.

    The fake executor needs prompt token ids only as a length
    and a budget cost, so a word count is an honest stand in.
    Real serving passes the model tokenizer instead and never
    uses this function.
    """
    return [1] * max(1, len(prompt.split()))


class InvalidRequestError(ValueError):
    """The request body failed API level validation."""


def validate_generate_body(body: Any) -> tuple[str, int]:
    """Validate one generation body; return prompt and limit."""
    if not isinstance(body, dict):
        raise InvalidRequestError("Request body must be a JSON object.")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise InvalidRequestError(
            "Field 'prompt' is required and must be a non-empty string."
        )
    max_new_tokens = body.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise InvalidRequestError(
            "Field 'max_new_tokens' must be a positive integer."
        )
    return prompt, max_new_tokens


def error_body(code: str, message: str, request_id: str | None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


class ServiceFrontEnd:
    """Translate between HTTP payloads and the engine owner.

    One instance is shared by every HTTP handler thread. It
    holds no per request generation state of its own beyond
    the handles the owner returned, which it keeps only so a
    cancel call can report the finished reason of a request
    that already ended.
    """

    def __init__(
        self,
        owner: EngineOwner,
        tokenize: TokenizeFn,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.owner = owner
        self.tokenize = tokenize
        self.model_id = model_id
        self.revision = revision
        self.handles: dict[str, RequestHandle] = {}

    def health(self) -> tuple[int, dict[str, Any]]:
        report = self.owner.health()
        body: dict[str, Any] = {
            "status": report["status"],
            "model_id": self.model_id,
            "revision": self.revision,
            "accepting_requests": report["accepting_requests"],
            "worker_alive": report["worker_alive"],
            "active_requests": report["active_requests"],
            "waiting_requests": report["waiting_requests"],
        }
        return (200 if report["status"] == "ok" else 503), body

    def submit_validated(self, body: Any) -> RequestHandle:
        """Validate, tokenize, and admit one generation request."""
        prompt, max_new_tokens = validate_generate_body(body)
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        request = Request(
            request_id=request_id,
            prompt_token_ids=self.tokenize(prompt),
            max_new_tokens=max_new_tokens,
        )
        handle = self.owner.submit(request)
        self.handles[request_id] = handle
        return handle

    def generate(self, body: Any) -> tuple[int, dict[str, Any]]:
        """Run one request to completion and build its result."""
        try:
            handle = self.submit_validated(body)
        except InvalidRequestError as exc:
            return 400, error_body("invalid_request", str(exc), None)
        except QueueFullError as exc:
            return 429, error_body("queue_full", str(exc), None)
        except RequestRejectedError as exc:
            return 400, error_body("request_rejected", str(exc), None)
        try:
            request = handle.result(timeout=RESULT_TIMEOUT_SECONDS)
        except GenerationFailedError as exc:
            return 500, error_body(
                "generation_failed", exc.reason, handle.request_id
            )
        except TimeoutError:
            return 500, error_body(
                "generation_failed",
                "Generation did not finish within the service timeout.",
                handle.request_id,
            )
        return 200, {
            "request_id": request.request_id,
            "text": handle.stream_text,
            "finish_reason": request.finish_reason,
            "prompt_tokens": request.prompt_length,
            "generated_tokens": request.output_length,
        }

    def cancel(self, request_id: str) -> tuple[int, dict[str, Any]]:
        try:
            cancelled = self.owner.cancel(request_id)
        except KeyError:
            return 404, error_body(
                "unknown_request",
                f"No request with id {request_id} was ever admitted.",
                None,
            )
        body: dict[str, Any] = {"request_id": request_id, "cancelled": cancelled}
        if not cancelled:
            handle = self.handles.get(request_id)
            if handle is not None:
                body["finish_reason"] = handle.request.finish_reason
        return 200, body


def make_server(
    front_end: ServiceFrontEnd,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Build the HTTP server for one front end (not yet serving)."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_json_body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/health":
                status, body = front_end.health()
                self._send_json(status, body)
            else:
                self._send_json(
                    404,
                    error_body("unknown_request", "Unknown path.", None),
                )

        def do_POST(self) -> None:
            if self.path == "/generate":
                try:
                    body = self._read_json_body()
                except (ValueError, UnicodeDecodeError) as exc:
                    self._send_json(
                        400,
                        error_body(
                            "invalid_request",
                            f"Request body is not valid JSON: {exc}",
                            None,
                        ),
                    )
                    return
                status, response = front_end.generate(body)
                self._send_json(status, response)
                return
            if self.path == "/generate/stream":
                self._handle_stream()
                return
            prefix = "/requests/"
            suffix = "/cancel"
            if self.path.startswith(prefix) and self.path.endswith(suffix):
                request_id = self.path[len(prefix) : -len(suffix)]
                status, response = front_end.cancel(request_id)
                self._send_json(status, response)
                return
            self._send_json(
                404, error_body("unknown_request", "Unknown path.", None)
            )

        def _handle_stream(self) -> None:
            try:
                body = self._read_json_body()
            except (ValueError, UnicodeDecodeError) as exc:
                self._send_json(
                    400,
                    error_body(
                        "invalid_request",
                        f"Request body is not valid JSON: {exc}",
                        None,
                    ),
                )
                return
            try:
                handle = front_end.submit_validated(body)
            except InvalidRequestError as exc:
                self._send_json(
                    400, error_body("invalid_request", str(exc), None)
                )
                return
            except QueueFullError as exc:
                self._send_json(
                    429, error_body("queue_full", str(exc), None)
                )
                return
            except RequestRejectedError as exc:
                self._send_json(
                    400, error_body("request_rejected", str(exc), None)
                )
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for event in handle.events():
                    self.wfile.write(event.format_sse().encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The client went away mid stream: that is the
                # disconnect signal at this transport, and the
                # request is cancelled through the owner rather
                # than left generating into a dead socket.
                handle.disconnect(reason="stream_closed")
            except Exception:  # noqa: BLE001
                handle.disconnect(reason="stream_error")

    return ThreadingHTTPServer((host, port), Handler)
