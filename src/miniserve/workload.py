"""Deterministic workload generation for scheduling experiments.

A workload is a list of request specifications. Each specification
carries everything needed to rebuild the same request later: the
request id, the prompt length and the exact prompt token ids, the
output limit, and the arrival time on the simulation timetable.

Both scheduling policies read the same specifications, so a serial
run and a continuous run can be built from one generated workload
and compared fairly. Generating a workload twice with the same
profile, seed, and request count produces exactly the same list.
Generating it with a different seed produces a different list, for
any profile and count where the ranges allow variation.

Arrival times are timetable data. They record when each request is
meant to arrive in simulation time, in non decreasing order,
starting at 0.0. The current engine admits every request it is
given at once, so the runner that replays arrival times belongs to
the experiment pipeline work that follows. The specifications keep
the arrival time now so that no later stage has to invent it.

All prompt token ids and times in this module are simulation data
produced by a seeded generator. They are not text from a real
tokenizer and they say nothing about real model performance.
"""

import random
from dataclasses import dataclass

from miniserve.request import Clock, Request

PROFILE_SHORT = "short"
PROFILE_MIXED = "mixed"
PROFILE_LONG_OUTPUT = "long_output"

PROFILE_NAMES: tuple[str, ...] = (
    PROFILE_SHORT,
    PROFILE_MIXED,
    PROFILE_LONG_OUTPUT,
)

# Prompt and output ranges for each profile, as inclusive bounds.
# The mixed profile alternates short and long prompts by position,
# so a mixed workload always contains both kinds, whatever the seed.
_SHORT_PROMPT = (4, 16)
_SHORT_OUTPUT = (2, 8)
_MIXED_SHORT_PROMPT = (4, 12)
_MIXED_LONG_PROMPT = (32, 64)
_MIXED_OUTPUT = (2, 16)
_LONG_PROMPT = (8, 32)
_LONG_OUTPUT_RANGE = (24, 48)


@dataclass(frozen=True)
class WorkloadRequestSpec:
    """One deterministic request specification in a workload."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_time: float

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    def to_request(self, clock: Clock | None = None) -> Request:
        """Build a fresh Request from this specification.

        A new Request is built on every call, so the same workload
        can be replayed through more than one scheduler without
        sharing request state. The arrival time stored in the
        specification is passed to the request, so its timestamps
        start from the timetable value instead of the wall clock.
        When a clock is given, every timestamp the request records
        comes from that one clock; simulation runs pass their shared
        logical clock here so no wall clock value enters the run.
        """
        if clock is None:
            return Request(
                request_id=self.request_id,
                prompt_token_ids=list(self.prompt_token_ids),
                max_new_tokens=self.max_new_tokens,
                arrival_time=self.arrival_time,
            )
        return Request(
            request_id=self.request_id,
            prompt_token_ids=list(self.prompt_token_ids),
            max_new_tokens=self.max_new_tokens,
            arrival_time=self.arrival_time,
            clock=clock,
        )


def _draw_lengths(
    profile: str,
    index: int,
    rng: random.Random,
) -> tuple[int, int]:
    if profile == PROFILE_SHORT:
        prompt_length = rng.randint(*_SHORT_PROMPT)
        max_new_tokens = rng.randint(*_SHORT_OUTPUT)
        return prompt_length, max_new_tokens
    if profile == PROFILE_MIXED:
        if index % 2 == 0:
            prompt_length = rng.randint(*_MIXED_SHORT_PROMPT)
        else:
            prompt_length = rng.randint(*_MIXED_LONG_PROMPT)
        max_new_tokens = rng.randint(*_MIXED_OUTPUT)
        return prompt_length, max_new_tokens
    prompt_length = rng.randint(*_LONG_PROMPT)
    max_new_tokens = rng.randint(*_LONG_OUTPUT_RANGE)
    return prompt_length, max_new_tokens


def generate_workload(
    profile: str,
    seed: int,
    num_requests: int,
) -> list[WorkloadRequestSpec]:
    """Generate one deterministic workload.

    The profile must be one of PROFILE_NAMES: short for short input
    and short output, mixed for a long and short prompt mix, and
    long_output for requests with a long output limit. The same
    profile, seed, and count always produce the same list, in the
    same order, with the same ids, token ids, output limits, and
    arrival times.
    """
    if profile not in PROFILE_NAMES:
        raise ValueError(
            f"Unknown workload profile {profile!r}. "
            f"Expected one of {', '.join(PROFILE_NAMES)}."
        )
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")

    rng = random.Random(seed)
    specs: list[WorkloadRequestSpec] = []
    arrival_time = 0.0
    for index in range(num_requests):
        prompt_length, max_new_tokens = _draw_lengths(profile, index, rng)
        if index > 0:
            arrival_time += float(rng.randint(0, 2))
        token_id = 100 + index
        specs.append(
            WorkloadRequestSpec(
                request_id=f"{profile}-{index + 1:03d}",
                prompt_token_ids=tuple([token_id] * prompt_length),
                max_new_tokens=max_new_tokens,
                arrival_time=arrival_time,
            )
        )
    return specs


def generate_default_workloads(
    seed: int = 1234,
    num_requests: int = 6,
) -> dict[str, list[WorkloadRequestSpec]]:
    """Generate all three profiles with one shared seed and count."""
    return {
        profile: generate_workload(profile, seed=seed, num_requests=num_requests)
        for profile in PROFILE_NAMES
    }


def build_requests(
    specs: list[WorkloadRequestSpec],
) -> list[Request]:
    """Build fresh Request objects for every specification."""
    return [spec.to_request() for spec in specs]
