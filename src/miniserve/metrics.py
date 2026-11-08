"""Per request metrics export.

This module turns the raw timestamps stored on a Request into the
derived values defined in docs/metrics.md. It reads only those raw
fields, so any exported value can be traced back to its inputs and
recomputed by hand. A value that has not happened yet is exported
as None, in Python and as null in JSON, never as zero.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from miniserve.request import Request


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    state: str
    finish_reason: str | None
    prompt_length: int
    output_length: int
    max_new_tokens: int
    arrival_time: float | None
    admission_time: float | None
    prefill_start_time: float | None
    first_token_time: float | None
    last_token_time: float | None
    finished_time: float | None
    queue_time: float | None
    ttft: float | None
    latency: float | None
    tpot: float | None


def _difference(later: float | None, earlier: float | None) -> float | None:
    if later is None or earlier is None:
        return None
    return later - earlier


def summarize_request(request: Request) -> RequestMetrics:
    """Compute the metrics record for one request from its raw fields."""
    queue_time = _difference(request.prefill_start_time, request.arrival_time)
    ttft = _difference(request.first_token_time, request.arrival_time)
    latency = _difference(request.finished_time, request.arrival_time)

    tpot: float | None = None
    if (
        request.output_length >= 2
        and request.first_token_time is not None
        and request.last_token_time is not None
    ):
        tpot = (request.last_token_time - request.first_token_time) / (
            request.output_length - 1
        )

    return RequestMetrics(
        request_id=request.request_id,
        state=request.state.name,
        finish_reason=request.finish_reason,
        prompt_length=request.prompt_length,
        output_length=request.output_length,
        max_new_tokens=request.max_new_tokens,
        arrival_time=request.arrival_time,
        admission_time=request.admission_time,
        prefill_start_time=request.prefill_start_time,
        first_token_time=request.first_token_time,
        last_token_time=request.last_token_time,
        finished_time=request.finished_time,
        queue_time=queue_time,
        ttft=ttft,
        latency=latency,
        tpot=tpot,
    )


def summarize_requests(requests: Iterable[Request]) -> list[RequestMetrics]:
    return [summarize_request(request) for request in requests]


def metrics_to_dict(metrics: RequestMetrics) -> dict[str, object]:
    return asdict(metrics)


def to_jsonl(requests: Iterable[Request]) -> str:
    """Render one JSON object per request, one per line."""
    lines = [
        json.dumps(metrics_to_dict(summarize_request(request)), sort_keys=True)
        for request in requests
    ]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def write_jsonl(requests: Iterable[Request], path: Path) -> None:
    path.write_text(to_jsonl(requests), encoding="utf-8")


def count_by_finish_reason(requests: Iterable[Request]) -> dict[str, int]:
    """Count terminal outcomes separately, with unfinished apart.

    Finished and cancelled are never merged. A request that is not
    terminal yet counts under the key unfinished.
    """
    counts = {"finished": 0, "cancelled": 0, "unfinished": 0}
    for request in requests:
        reason = request.finish_reason
        if reason is None:
            counts["unfinished"] += 1
        else:
            counts[reason] += 1
    return counts
