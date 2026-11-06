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
- Request ids are unique for the lifetime of one scheduler. An id is
  recorded only when a request is actually admitted, so an offer
  that is rejected for a validation reason does not burn its id, and
  a corrected request may reuse it. Once a request has been
  admitted, offering the same id again, or offering the same request
  object again, is rejected with `RequestRejectedError` and never
  occupies a second queue slot. The duplicate rule also holds after
  the first request finishes or is cancelled: within one scheduler,
  one id names exactly one request, ever. A new scheduler starts
  with an empty set of ids.

## 5. Cancellation contract

`Scheduler.cancel_request(request_id)` is the single cancellation
entry point for callers that hold only an id.

- Cancelling a `WAITING` request removes it from `waiting`. It never
  starts prefill and never produces a token.
- Cancelling a `DECODING` request removes it from `running`. It
  keeps the tokens it already produced and receives no new ones.
  Other requests are unaffected and continue to completion.
- The call returns `True` when it cancelled an active request.
  Cancelling a request that already reached a terminal state is a
  harmless no-op: it returns `False`, the terminal state is left as
  it was (a finished request does not become cancelled), and the
  queues are untouched.
- Cancelling an id the scheduler has never admitted raises
  `KeyError`, so a mistyped id is not silently ignored.
- A caller may also cancel a `Request` object directly. The
  scheduler defends against that path by pruning terminal requests
  from both queues at the start of `schedule` and inside
  `has_pending_requests`, so a directly cancelled waiting request is
  dropped before it can be selected, and the executor is never asked
  to start prefill on a `CANCELLED` request.
- Cancellation takes effect at a scheduling boundary. Work already
  handed to the executor inside the current step is not interrupted
  mid step; the cancelled request is excluded from the next step on.

## 6. Hand traced scheduler budget examples

Cost model used for every trace below: a prefill costs its prompt
length in tokens, each decode costs 1 token, the sum of the costs
selected in one step must stay at or below `max_batch_tokens`, and
the number of selected requests must stay at or below
`max_batch_size`. Decode is selected before prefill.

| Case | Setup | Hand trace | Result |
| --- | --- | --- | --- |
| Exact fill | Budget 6, one running request, one waiting prompt of 5 | Decode 1 plus prefill 5 equals 6 | Both are selected in the same step |
| One token over | Budget 6, one running request, one waiting prompt of 6 | Decode 1 plus prefill 6 equals 7, above 6 | Prefill waits; the prompt fits the budget alone, so this is temporary, not a rejection |
| Temporary block clears | Same as above, running request finishes after its decode | Next step has no decode cost, prefill 6 equals 6 | The waiting request is selected in the next step |
| Batch size 1 | Batch size 1, budget 100, three waiting requests | Only one slot exists | Only the head request is selected for prefill |
| Batch size 1 with running work | Batch size 1, one running request, one waiting request | The single slot goes to decode first | The waiting request stays queued |
| Two finish together | Batch size 2, two running requests each need 1 more token, two waiting | Both decode and finish in one step; both slots free up | Both waiting requests are selected for prefill in the next step |
| FIFO head of line blocking | Budget 5, one running request, waiting head prompt 5, small prompt 1 behind it | Decode 1 plus head 5 equals 6, above 5; the small request would fit (1 plus 1) but sits behind the head | Neither waiting request is selected; queue order is preserved |

Two kinds of "does not fit" must stay distinct. A request that can
never fit, such as a prompt longer than the whole step budget, is
rejected at admission and never queues. A request that only does
not fit this step, because decode work is using part of the budget,
waits and runs in a later step once the budget frees up.

## 7. Boundary test checklist

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

Cancellation and duplicate handling:

- [x] Cancelling a waiting request means it never starts prefill and produces no tokens.
- [x] Cancelling a running request stops further tokens for it and does not affect other requests.
- [x] Cancelling an already finished or cancelled request does not corrupt the queues.
- [x] Offering the same active `request_id` twice is rejected.
- [x] Offering the same request object twice is rejected.

Budget boundaries:

- [x] Prefill cost exactly filling the remaining step budget is admitted in that step.
- [x] Prefill cost one token above the remaining step budget waits for a later step.
- [x] `max_batch_size` of 1 serialises selection.
- [x] Several requests finishing in the same step free their slots for the next step.
- [x] A temporarily too large head request blocks later waiting requests (documented trade off above), while a permanently too large request is rejected at admission instead.

