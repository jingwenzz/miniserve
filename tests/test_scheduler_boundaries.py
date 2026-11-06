from miniserve.executor import FakeExecutor
from miniserve.request import Request, RequestState
from miniserve.scheduler import Scheduler


def step_prefill(scheduler: Scheduler, executor: FakeExecutor) -> None:
    output = scheduler.schedule()
    executor.prefill(output.prefill_requests)
    scheduler.update(output)


def selected_cost(prefill: list[Request], decode: list[Request]) -> int:
    """Hand cost model: prefill costs prompt length, decode costs 1."""
    return sum(r.prompt_length for r in prefill) + len(decode)


def test_prefill_exactly_fills_remaining_budget() -> None:
    scheduler = Scheduler(max_batch_size=3, max_batch_tokens=6)
    executor = FakeExecutor()

    running = Request("running", [1], max_new_tokens=3)
    scheduler.add_request(running)
    step_prefill(scheduler, executor)

    waiting = Request("waiting", [2] * 5, max_new_tokens=1)
    scheduler.add_request(waiting)

    output = scheduler.schedule()

    # Decode costs 1 and prefill costs 5, for a total of exactly 6.
    assert output.decode_requests == [running]
    assert output.prefill_requests == [waiting]
    assert selected_cost(output.prefill_requests, output.decode_requests) == 6


def test_prefill_one_token_over_remaining_budget_waits() -> None:
    scheduler = Scheduler(max_batch_size=3, max_batch_tokens=6)
    executor = FakeExecutor()

    running = Request("running", [1], max_new_tokens=3)
    scheduler.add_request(running)
    step_prefill(scheduler, executor)

    waiting = Request("waiting", [2] * 6, max_new_tokens=1)
    scheduler.add_request(waiting)

    output = scheduler.schedule()

    # The prompt fits the full budget of 6 on its own, so admission
    # accepted it, but 1 decode token plus 6 prefill tokens is 7.
    assert output.decode_requests == [running]
    assert output.prefill_requests == []
    assert list(scheduler.waiting) == [waiting]


def test_temporarily_too_large_head_runs_after_running_request_finishes() -> None:
    scheduler = Scheduler(max_batch_size=3, max_batch_tokens=6)
    executor = FakeExecutor()

    running = Request("running", [1], max_new_tokens=1)
    scheduler.add_request(running)
    step_prefill(scheduler, executor)

    waiting = Request("waiting", [2] * 6, max_new_tokens=1)
    scheduler.add_request(waiting)

    blocked = scheduler.schedule()
    assert blocked.prefill_requests == []
    executor.decode(blocked.decode_requests)
    scheduler.update(blocked)
    assert running.state is RequestState.FINISHED

    freed = scheduler.schedule()
    assert freed.decode_requests == []
    assert freed.prefill_requests == [waiting]
    assert selected_cost(freed.prefill_requests, freed.decode_requests) == 6


def test_batch_size_one_serialises_prefill() -> None:
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)

    requests = [
        Request(f"request-{index}", [index], max_new_tokens=1)
        for index in range(1, 4)
    ]
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.prefill_requests == [requests[0]]
    assert list(scheduler.waiting) == requests[1:]


def test_batch_size_one_gives_the_only_slot_to_decode() -> None:
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=100)
    executor = FakeExecutor()

    running = Request("running", [1], max_new_tokens=3)
    scheduler.add_request(running)
    step_prefill(scheduler, executor)

    waiting = Request("waiting", [2], max_new_tokens=1)
    scheduler.add_request(waiting)

    output = scheduler.schedule()

    assert output.decode_requests == [running]
    assert output.prefill_requests == []
    assert list(scheduler.waiting) == [waiting]


def test_two_requests_finishing_together_free_both_slots_next_step() -> None:
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=100)
    executor = FakeExecutor()

    first = Request("first", [1], max_new_tokens=1)
    second = Request("second", [2], max_new_tokens=1)
    scheduler.add_request(first)
    scheduler.add_request(second)
    step_prefill(scheduler, executor)

    third = Request("third", [3], max_new_tokens=1)
    fourth = Request("fourth", [4], max_new_tokens=1)
    scheduler.add_request(third)
    scheduler.add_request(fourth)

    finishing = scheduler.schedule()
    assert finishing.decode_requests == [first, second]
    assert finishing.prefill_requests == []
    executor.decode(finishing.decode_requests)
    scheduler.update(finishing)
    assert scheduler.running == []

    refilled = scheduler.schedule()
    assert refilled.prefill_requests == [third, fourth]
    assert list(scheduler.waiting) == []


def test_fifo_head_blocks_smaller_request_behind_it() -> None:
    scheduler = Scheduler(max_batch_size=3, max_batch_tokens=5)
    executor = FakeExecutor()

    running = Request("running", [1], max_new_tokens=3)
    scheduler.add_request(running)
    step_prefill(scheduler, executor)

    head = Request("head", [2] * 5, max_new_tokens=1)
    small = Request("small", [3], max_new_tokens=1)
    scheduler.add_request(head)
    scheduler.add_request(small)

    output = scheduler.schedule()

    # Decode costs 1, so the head needs 5 of the remaining 4 and does
    # not fit this step. The small request behind it would fit on its
    # own (1 plus 1), but FIFO order means it does not jump ahead.
    assert output.decode_requests == [running]
    assert output.prefill_requests == []
    assert list(scheduler.waiting) == [head, small]


def test_decode_is_selected_before_prefill_when_budget_is_tight() -> None:
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=3)
    executor = FakeExecutor()

    running = Request("running", [1], max_new_tokens=3)
    scheduler.add_request(running)
    step_prefill(scheduler, executor)

    waiting = Request("waiting", [2] * 3, max_new_tokens=1)
    scheduler.add_request(waiting)

    output = scheduler.schedule()

    # If prefill went first it would spend all 3 tokens and starve
    # the in flight decode. Decode goes first and costs 1, which
    # leaves only 2, so the prefill waits for a later step.
    assert output.decode_requests == [running]
    assert output.prefill_requests == []


def test_no_step_exceeds_limits_and_no_request_is_lost_or_duplicated() -> None:
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=8)
    executor = FakeExecutor()

    requests = [
        Request("a", [1] * 3, max_new_tokens=2),
        Request("b", [2] * 4, max_new_tokens=3),
        Request("c", [3] * 2, max_new_tokens=1),
        Request("d", [4] * 5, max_new_tokens=2),
    ]
    for request in requests:
        scheduler.add_request(request)

    steps = 0
    while scheduler.has_pending_requests():
        output = scheduler.schedule()
        selected = output.prefill_requests + output.decode_requests
        assert len(selected) <= scheduler.max_batch_size
        assert (
            selected_cost(output.prefill_requests, output.decode_requests)
            <= scheduler.max_batch_tokens
        )
        assert len({id(r) for r in selected}) == len(selected)
        executor.prefill(output.prefill_requests)
        executor.decode(output.decode_requests)
        scheduler.update(output)
        steps += 1
        assert steps <= 20

    assert all(r.state is RequestState.FINISHED for r in requests)
    assert [r.output_length for r in requests] == [2, 3, 1, 2]
