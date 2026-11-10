import pytest

from miniserve.metrics import summarize_request
from miniserve.simulation import (
    SimulationClock,
    SimulationCosts,
    SimulationResult,
    run_simulation,
)
from miniserve.workload import WorkloadRequestSpec


def hand_specs() -> list[WorkloadRequestSpec]:
    """The hand worked example from docs/simulation.md section 3."""
    return [
        WorkloadRequestSpec(
            request_id="sim-a",
            prompt_token_ids=tuple([1] * 10),
            max_new_tokens=2,
            arrival_time=0.0,
        ),
        WorkloadRequestSpec(
            request_id="sim-b",
            prompt_token_ids=tuple([1] * 5),
            max_new_tokens=1,
            arrival_time=0.0,
        ),
        WorkloadRequestSpec(
            request_id="sim-c",
            prompt_token_ids=tuple([1] * 5),
            max_new_tokens=2,
            arrival_time=1.0,
        ),
    ]


def run_hand_example(
    costs: SimulationCosts | None = None,
) -> SimulationResult:
    return run_simulation(
        hand_specs(),
        policy="continuous",
        max_batch_size=3,
        max_batch_tokens=50,
        costs=costs,
    )


def test_clock_only_moves_when_advanced() -> None:
    clock = SimulationClock(10.0)

    assert clock() == 10.0
    assert clock() == 10.0
    assert clock.advance(2.5) == 12.5
    assert clock.now == 12.5
    with pytest.raises(ValueError, match="cannot move backwards"):
        clock.advance(-1.0)


def test_costs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="prefill_seconds_per_token"):
        SimulationCosts(prefill_seconds_per_token=0.0)
    with pytest.raises(ValueError, match="decode_seconds_per_token"):
        SimulationCosts(decode_seconds_per_token=-1.0)

    costs = SimulationCosts()
    assert costs.prefill_cost(10) == pytest.approx(1.0)
    assert costs.decode_cost() == pytest.approx(0.5)


def test_hand_worked_timeline_matches_hand_calculation() -> None:
    result = run_hand_example()

    assert result.total_steps == 4
    assert result.simulated_duration == pytest.approx(4.5)
    assert [record.duration for record in result.timeline] == [
        pytest.approx(1.5),
        pytest.approx(1.5),
        pytest.approx(1.0),
        pytest.approx(0.5),
    ]
    assert [
        (record.prefill_request_ids, record.decode_request_ids)
        for record in result.timeline
    ] == [
        (("sim-a", "sim-b"), ()),
        (("sim-c",), ("sim-a", "sim-b")),
        ((), ("sim-a", "sim-c")),
        ((), ("sim-c",)),
    ]
    assert result.timeline[0].waiting_request_ids == ("sim-a", "sim-b")
    assert result.timeline[1].waiting_request_ids == ("sim-c",)
    assert result.timeline[1].running_request_ids == ("sim-a", "sim-b")
    assert result.waiting_steps("sim-c") == 1
    assert result.total_output_tokens == 5
    assert result.released_request_ids == ["sim-b", "sim-a", "sim-c"]


def test_hand_worked_metrics_use_simulation_time() -> None:
    result = run_hand_example()
    by_id = {request.request_id: request for request in result.requests}

    metrics_a = summarize_request(by_id["sim-a"])
    assert metrics_a.prefill_start_time == pytest.approx(0.0)
    assert metrics_a.first_token_time == pytest.approx(2.5)
    assert metrics_a.ttft == pytest.approx(2.5)
    assert metrics_a.latency == pytest.approx(3.5)

    metrics_b = summarize_request(by_id["sim-b"])
    assert metrics_b.prefill_start_time == pytest.approx(1.0)
    assert metrics_b.ttft == pytest.approx(3.0)
    assert metrics_b.latency == pytest.approx(3.0)

    metrics_c = summarize_request(by_id["sim-c"])
    assert metrics_c.arrival_time == pytest.approx(1.0)
    assert metrics_c.admission_time == pytest.approx(1.5)
    assert metrics_c.queue_time == pytest.approx(0.5)
    assert metrics_c.ttft == pytest.approx(3.0)
    assert metrics_c.latency == pytest.approx(3.5)


def test_doubling_decode_cost_changes_duration_not_schedule() -> None:
    base = run_hand_example()
    doubled = run_hand_example(SimulationCosts(decode_seconds_per_token=1.0))

    assert doubled.total_steps == base.total_steps == 4
    assert [
        (record.prefill_request_ids, record.decode_request_ids)
        for record in doubled.timeline
    ] == [
        (record.prefill_request_ids, record.decode_request_ids)
        for record in base.timeline
    ]
    assert base.simulated_duration == pytest.approx(4.5)
    assert doubled.simulated_duration == pytest.approx(7.0)


def test_simulation_is_deterministic() -> None:
    first = run_hand_example()
    second = run_hand_example()

    assert first.timeline == second.timeline
    assert [
        summarize_request(request).ttft for request in first.requests
    ] == [summarize_request(request).ttft for request in second.requests]


def test_serial_policy_keeps_one_active_request_per_step() -> None:
    result = run_simulation(
        hand_specs(),
        policy="serial",
        max_batch_size=3,
        max_batch_tokens=50,
    )

    assert result.total_output_tokens == 5
    assert all(record.active_count <= 1 for record in result.timeline)
    continuous = run_hand_example()
    assert max(record.active_count for record in continuous.timeline) == 3
