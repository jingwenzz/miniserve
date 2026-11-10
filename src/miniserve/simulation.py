"""Deterministic simulation time for scheduling experiments.

This module separates simulation time from Python wall clock time.
A simulation run never reads the wall clock: every request in the
run shares one SimulationClock, and that clock only moves when the
simulated executor charges an explicit cost for prefill or decode
work, or when the runner waits for the next scheduled arrival.

The cost model is deliberately simple and is stated in full:

- Prefill of one request costs ``prefill_seconds_per_token`` times
  its prompt length, in logical seconds.
- Decode of one token for one request costs
  ``decode_seconds_per_token`` logical seconds.
- Costs inside one step accumulate one request after another. This
  is a serial work model chosen so a step can be costed by hand. It
  is not a model of parallel GPU execution.

Every number produced here is simulation data. A logical second is
not a measured second, the fake token stream is not a model, and no
value from this module may be reported as model throughput, model
latency, or evidence of GPU speedup. Wall clock timing around a
simulation run would only measure the test harness.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from miniserve.engine import Engine
from miniserve.request import Request, RequestState
from miniserve.scheduler import create_scheduler
from miniserve.workload import WorkloadRequestSpec

MAX_SIMULATION_STEPS = 100_000


class SimulationClock:
    """A logical clock that only moves when it is advanced.

    The clock is callable so it can be injected as the clock of
    every Request in a run. Two runs started from the same value
    and advanced by the same costs read exactly the same times,
    whatever the wall clock does in between.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def __call__(self) -> float:
        return self._now

    @property
    def now(self) -> float:
        return self._now

    def advance(self, amount: float) -> float:
        """Move the clock forward and return the new reading."""
        if amount < 0:
            raise ValueError("A simulation clock cannot move backwards")
        self._now += amount
        return self._now


@dataclass(frozen=True)
class SimulationCosts:
    """Explicit per token costs, in logical seconds."""

    prefill_seconds_per_token: float = 0.1
    decode_seconds_per_token: float = 0.5

    def __post_init__(self) -> None:
        if self.prefill_seconds_per_token <= 0:
            raise ValueError("prefill_seconds_per_token must be positive")
        if self.decode_seconds_per_token <= 0:
            raise ValueError("decode_seconds_per_token must be positive")

    def prefill_cost(self, prompt_length: int) -> float:
        return self.prefill_seconds_per_token * prompt_length

    def decode_cost(self) -> float:
        return self.decode_seconds_per_token


class SimulatedExecutor:
    """Executor that charges simulation costs on a shared clock.

    Prefill and decode follow the same state transitions as the
    fake executor, with one difference: before a request finishes
    prefill, the clock advances by that request's prefill cost, and
    before a token is appended, the clock advances by the decode
    cost. Token timestamps therefore land after the work that
    produced them, in logical time. This executor holds no per
    request resources, so release only records the request id,
    exactly like the fake executor.
    """

    def __init__(
        self,
        clock: SimulationClock,
        costs: SimulationCosts,
        generated_token_id: int = 1,
    ) -> None:
        self.clock = clock
        self.costs = costs
        self.generated_token_id = generated_token_id
        self.released_request_ids: list[str] = []

    def prefill(self, requests: list[Request]) -> None:
        for request in requests:
            request.start_prefill()
            self.clock.advance(self.costs.prefill_cost(request.prompt_length))
            request.finish_prefill()

    def decode(self, requests: list[Request]) -> None:
        for request in requests:
            if request.state is not RequestState.DECODING:
                raise RuntimeError(
                    f"Request {request.request_id} is not ready for decode"
                )
            self.clock.advance(self.costs.decode_cost())
            request.append_token(self.generated_token_id)

    def release(self, request: Request) -> None:
        if request.request_id not in self.released_request_ids:
            self.released_request_ids.append(request.request_id)


@dataclass(frozen=True)
class StepRecord:
    """One row of the per step simulation timeline.

    The waiting and running id lists are snapshots taken at the
    start of the step, before the scheduler selects its work, so a
    reader can see who was queued, who was already running, and who
    the step actually executed. Times are logical seconds read from
    the simulation clock at the start and end of the step.
    """

    step: int
    start_time: float
    end_time: float
    prefill_request_ids: tuple[str, ...]
    decode_request_ids: tuple[str, ...]
    waiting_request_ids: tuple[str, ...]
    running_request_ids: tuple[str, ...]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def active_count(self) -> int:
        return len(self.prefill_request_ids) + len(self.decode_request_ids)


@dataclass
class SimulationResult:
    """The outcome of one simulation run."""

    policy: str
    costs: SimulationCosts
    requests: list[Request]
    timeline: list[StepRecord]
    released_request_ids: list[str]

    @property
    def total_steps(self) -> int:
        return len(self.timeline)

    @property
    def simulated_duration(self) -> float:
        if not self.timeline:
            return 0.0
        return self.timeline[-1].end_time - self.timeline[0].start_time

    @property
    def total_output_tokens(self) -> int:
        return sum(request.output_length for request in self.requests)

    def waiting_steps(self, request_id: str) -> int:
        """Count the steps that began with this request still waiting."""
        return sum(
            1 for record in self.timeline if request_id in record.waiting_request_ids
        )


def run_simulation(
    specs: Sequence[WorkloadRequestSpec],
    *,
    policy: str,
    max_batch_size: int,
    max_batch_tokens: int,
    max_model_len: int | None = None,
    costs: SimulationCosts | None = None,
    generated_token_id: int = 1,
) -> SimulationResult:
    """Replay one workload under one policy in simulation time.

    Requests are admitted at step boundaries: a request is offered
    to the scheduler at the first step boundary whose clock reading
    is at or after its arrival time. If nothing is active and the
    next arrival is still in the future, the clock jumps straight
    to that arrival instead of stepping through empty rounds, and
    no timeline row is written for the idle gap. The run is fully
    determined by the specifications and the costs.
    """
    effective_costs = costs if costs is not None else SimulationCosts()
    clock = SimulationClock(0.0)
    scheduler = create_scheduler(
        policy,
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        max_model_len=max_model_len,
    )
    executor = SimulatedExecutor(
        clock, effective_costs, generated_token_id=generated_token_id
    )
    engine = Engine(scheduler, executor)

    pending = sorted(specs, key=lambda spec: spec.arrival_time)
    next_arrival = 0
    requests: list[Request] = []
    timeline: list[StepRecord] = []

    while next_arrival < len(pending) or scheduler.has_pending_requests():
        if len(timeline) >= MAX_SIMULATION_STEPS:
            raise RuntimeError(
                f"Simulation exceeded {MAX_SIMULATION_STEPS} steps"
            )

        while (
            next_arrival < len(pending)
            and pending[next_arrival].arrival_time <= clock.now
        ):
            request = pending[next_arrival].to_request(clock=clock)
            scheduler.add_request(request)
            requests.append(request)
            next_arrival += 1

        if not scheduler.has_pending_requests():
            clock.advance(pending[next_arrival].arrival_time - clock.now)
            continue

        waiting_before = tuple(
            request.request_id for request in scheduler.waiting
        )
        running_before = tuple(
            request.request_id for request in scheduler.running
        )
        start_time = clock.now
        output = engine.step()
        timeline.append(
            StepRecord(
                step=len(timeline) + 1,
                start_time=start_time,
                end_time=clock.now,
                prefill_request_ids=tuple(
                    request.request_id for request in output.prefill_requests
                ),
                decode_request_ids=tuple(
                    request.request_id for request in output.decode_requests
                ),
                waiting_request_ids=waiting_before,
                running_request_ids=running_before,
            )
        )

    return SimulationResult(
        policy=policy,
        costs=effective_costs,
        requests=requests,
        timeline=timeline,
        released_request_ids=list(executor.released_request_ids),
    )


def step_record_to_dict(record: StepRecord) -> dict[str, object]:
    return asdict(record)


def timeline_to_jsonl(timeline: Sequence[StepRecord]) -> str:
    """Render the timeline as one JSON object per step."""
    lines = [
        json.dumps(step_record_to_dict(record), sort_keys=True)
        for record in timeline
    ]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def write_timeline_jsonl(timeline: Sequence[StepRecord], path: Path) -> None:
    path.write_text(timeline_to_jsonl(timeline), encoding="utf-8")
