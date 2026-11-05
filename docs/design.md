# MiniServe Design: Request and Scheduling Contracts

This document records the contracts that the scheduler, the executor,
and the engine rely on. It describes the code as it exists at this
checkpoint and the decisions that later work must not silently change.

## 1. Request lifecycle

A request moves through these states:

```text
WAITING -> PREFILLING -> DECODING -> FINISHED
   |           |            |
   +-----------+------------+------> CANCELLED
```

- `WAITING`: the request has been admitted and is in the waiting queue.
  It has not run any model work yet.
- `PREFILLING`: the executor is processing the prompt. This state is
  transient inside one engine step for the current fake executor, but
  the contract does not assume it is instant.
- `DECODING`: prefill has finished and the request receives one new
  token per decode step.
- `FINISHED`: the request produced `max_new_tokens` tokens. This is a
  terminal state.
- `CANCELLED`: the request was cancelled before finishing. This is a
  terminal state.

Rules for transitions:

- Only a `WAITING` request may start prefill. Calling `start_prefill`
  from any other state is an error.
- Only a `PREFILLING` request may finish prefill and enter `DECODING`.
- Only a `DECODING` request may accept a new token or finish through
  the normal output path.
- A request in `WAITING`, `PREFILLING`, or `DECODING` may be cancelled.
  A terminal request cannot be cancelled again.
- `FINISHED` and `CANCELLED` requests never execute again and never
  produce new tokens.

### Decision on a FAILED state and a failure reason

No `FAILED` state is added at this checkpoint.

Reason: there is currently no runtime failure path in the fake
executor, so a `FAILED` state added now would be unreachable and
untested. Admission problems are different from runtime failures.
A request that can never run is rejected synchronously when it is
offered to the scheduler (see section 4), before the scheduler takes
ownership of it, so it does not need a terminal state to describe it.

This decision must be revisited when a real executor is introduced.
At that point an execution error after admission needs a terminal
outcome that is distinct from both normal completion and caller
cancellation, together with a recorded failure reason. Until then,
callers must not treat `FINISHED` or `CANCELLED` as a place to hide
execution errors.

## 2. Queue ownership and invariants

The scheduler owns two collections:

- `waiting`: a first in, first out queue of admitted requests that
  have not started prefill.
- `running`: the list of requests that have finished prefill and are
  eligible for decode in later steps.

Invariants:

1. One request object is in at most one of `waiting` and `running`
   at any moment. It is never in both, and it never appears twice.
2. Every request in `waiting` is in state `WAITING`. Only a `WAITING`
   request may be newly admitted to `waiting`.
3. Every request in `running` has completed prefill and is in state
   `DECODING`, unless it reached a terminal state during the most
   recent execution and has not been pruned yet.
4. Terminal requests (`FINISHED` or `CANCELLED`) are removed from
   `running` by the scheduler during `schedule` and `update`. They
   are never selected for prefill or decode again.
5. Output never exceeds the request limit: a request finishes as soon
   as `output_length` reaches `max_new_tokens`, so
   `output_length <= max_new_tokens` always holds.
6. Queue order in `waiting` is arrival order. The scheduler admits
   from the head of the queue and does not skip a head request that
   does not fit in the current step. This head of line blocking is a
   known, deliberate trade off for this stage; it keeps admission
   order fair and easy to reason about. A request that can never fit
   at all is not left in the queue to block others forever, because
   admission rejects it up front (see section 4).

Responsibilities:

| Concern | Owner |
| --- | --- |
| Deciding which requests run in a step | Scheduler |
| Moving a request from `waiting` to `running` | Scheduler, in `update` |
| Changing request state during prefill and decode | Executor, through the `Request` transition methods |
| Pruning terminal requests from `running` | Scheduler |
| Driving one full step (schedule, execute, update) | Engine |
| Cleaning up a request that finishes or is cancelled | Scheduler prunes the queue entry; executor side resource cleanup becomes explicit when a real executor owns per request resources |

## 3. Two different limits: step budget and resident capacity

The scheduler configuration currently exposes `max_batch_size` and
`max_batch_tokens`. These must not be confused with cache capacity.

- Per step token budget (`max_batch_tokens`): the maximum amount of
  scheduling work selected in a single engine step. In the current
  cost model, a prefill costs `prompt_length` tokens and each decode
  costs 1 token. The sum of the costs of all requests selected in one
  step never exceeds `max_batch_tokens`, and the number of selected
  requests never exceeds `max_batch_size`. Decode requests are
  selected before prefill requests, so in flight work is not starved
  by new arrivals.
- Resident capacity: the total context held for all running requests
  across steps. Each running request holds `prompt_length` plus the
  tokens it has generated so far, and in a real backend that context
  occupies KV cache memory for as long as the request is alive.

These are different limits over different time horizons. The step
budget bounds work per step; resident capacity bounds memory held
between steps. One variable cannot stand in for the other: a step can
be well within its token budget while the combined context of many
long running requests exceeds the cache, and a single very long
prompt can fit in the cache while exceeding one step of work.
Resident capacity enforcement is separate future work; this document
only fixes the distinction so the two are never merged into a single
number.

## 4. Admission contract

Admission happens in `Scheduler.add_request`.

- A terminal request cannot be admitted.
- A request that can never execute is rejected at admission with an
  error, instead of entering `waiting` and blocking the queue
  forever. The concrete rejection rules and error type are defined
  with the input validation work that follows this document.
- A rejected request is not added to any queue, and rejecting it must
  not change the state of requests that were already admitted.

## 5. Boundary test checklist

Tests that must exist or be added as each area lands:

Admission and validation:

- [x] Prompt length exactly equal to the per step token budget is admitted.
- [x] Prompt length one above the per step token budget is rejected and leaves the queues empty.
- [x] A valid request offered after a rejection is admitted and completes normally.
- [x] Empty prompt is rejected.
- [x] `max_new_tokens` of zero is rejected.
- [x] Negative `max_new_tokens` is rejected.
- [x] Prompt plus reserved output exactly equal to the model context limit is admitted, when a context limit is configured.
- [x] Prompt plus reserved output one above the model context limit is rejected, when a context limit is configured.

Cancellation and duplicate handling (next scheduled work):

- [ ] Cancelling a waiting request means it never starts prefill and produces no tokens.
- [ ] Cancelling a running request stops further tokens for it and does not affect other requests.
- [ ] Cancelling an already finished or cancelled request does not corrupt the queues.
- [ ] Offering the same active `request_id` twice is rejected.
- [ ] Offering the same request object twice is rejected.

Budget boundaries (following scheduled work):

- [ ] Prefill cost exactly filling the remaining step budget is admitted in that step.
- [ ] Prefill cost one token above the remaining step budget waits for a later step.
- [ ] `max_batch_size` of 1 serialises selection.
- [ ] Several requests finishing in the same step free their slots for the next step.
- [ ] A temporarily too large head request blocks later waiting requests (documented trade off above), while a permanently too large request is rejected at admission instead.

