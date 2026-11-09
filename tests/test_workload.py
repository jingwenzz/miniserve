import pytest

from miniserve.engine import Engine
from miniserve.executor import FakeExecutor
from miniserve.request import RequestState
from miniserve.scheduler import create_scheduler
from miniserve.workload import (
    PROFILE_NAMES,
    build_requests,
    generate_default_workloads,
    generate_workload,
)


def test_same_seed_generates_identical_workload() -> None:
    first = generate_workload("mixed", seed=1234, num_requests=8)
    second = generate_workload("mixed", seed=1234, num_requests=8)

    assert first == second
    assert [spec.request_id for spec in first] == [
        spec.request_id for spec in second
    ]
    assert [spec.arrival_time for spec in first] == [
        spec.arrival_time for spec in second
    ]


def test_different_seed_generates_different_workload() -> None:
    first = generate_workload("short", seed=1, num_requests=12)
    second = generate_workload("short", seed=2, num_requests=12)

    assert first != second


def test_workload_rejects_unknown_profile_and_bad_count() -> None:
    with pytest.raises(ValueError, match="Unknown workload profile"):
        generate_workload("bogus", seed=1, num_requests=4)
    with pytest.raises(ValueError, match="num_requests must be positive"):
        generate_workload("short", seed=1, num_requests=0)


def test_specs_carry_id_lengths_output_and_arrival() -> None:
    specs = generate_workload("short", seed=7, num_requests=6)

    assert len(specs) == 6
    assert len({spec.request_id for spec in specs}) == 6
    for spec in specs:
        assert spec.prompt_length == len(spec.prompt_token_ids)
        assert spec.prompt_length >= 1
        assert spec.max_new_tokens >= 1
    arrivals = [spec.arrival_time for spec in specs]
    assert arrivals[0] == 0.0
    assert arrivals == sorted(arrivals)


def test_three_profiles_have_their_documented_shapes() -> None:
    short = generate_workload("short", seed=1234, num_requests=6)
    mixed = generate_workload("mixed", seed=1234, num_requests=6)
    long_output = generate_workload("long_output", seed=1234, num_requests=6)

    assert all(4 <= spec.prompt_length <= 16 for spec in short)
    assert all(2 <= spec.max_new_tokens <= 8 for spec in short)

    mixed_lengths = [spec.prompt_length for spec in mixed]
    assert any(length <= 12 for length in mixed_lengths)
    assert any(length >= 32 for length in mixed_lengths)

    assert all(spec.max_new_tokens >= 24 for spec in long_output)
    assert all(
        spec.max_new_tokens > short_spec.max_new_tokens
        for spec, short_spec in zip(long_output, short, strict=True)
    )


def test_default_workloads_cover_all_profiles_deterministically() -> None:
    first = generate_default_workloads(seed=99, num_requests=4)
    second = generate_default_workloads(seed=99, num_requests=4)

    assert list(first) == list(PROFILE_NAMES)
    assert first == second


def test_both_schedulers_read_the_same_workload() -> None:
    specs = generate_workload("short", seed=42, num_requests=4)
    results: dict[str, list[int]] = {}

    for policy in ("serial", "continuous"):
        scheduler = create_scheduler(
            policy, max_batch_size=4, max_batch_tokens=64
        )
        engine = Engine(scheduler, FakeExecutor(generated_token_id=7))
        requests = build_requests(specs)
        for request in requests:
            scheduler.add_request(request)
        engine.run_until_complete()
        assert all(
            request.state is RequestState.FINISHED for request in requests
        )
        results[policy] = [request.output_length for request in requests]

    assert results["serial"] == results["continuous"]
    assert results["serial"] == [spec.max_new_tokens for spec in specs]


def test_build_requests_are_fresh_copies() -> None:
    specs = generate_workload("short", seed=3, num_requests=3)

    first = build_requests(specs)
    second = build_requests(specs)

    assert first[0] is not second[0]
    assert first[0].arrival_time == specs[0].arrival_time
    assert first[0].prompt_token_ids == list(specs[0].prompt_token_ids)
