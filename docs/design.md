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
   |           |            |
   +-----------+------------+------> FAILED
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
- `FAILED`: an executor forward for this request raised after the
  request was admitted. This is a terminal state, distinct from both
  finishing and cancelling, and the request carries the recorded
  `failure_reason` text the executor supplied.

Rules for transitions:

- Only a `WAITING` request may start prefill. Calling `start_prefill`
  from any other state is an error.
- Only a `PREFILLING` request may finish prefill and enter `DECODING`.
- Only a `DECODING` request may accept a new token or finish through
  the normal output path.
- A request in `WAITING`, `PREFILLING`, or `DECODING` may be cancelled.
  A terminal request cannot be cancelled again.
- `FINISHED`, `CANCELLED`, and `FAILED` requests never execute again
  and never produce new tokens.

### Decision on a FAILED state and a failure reason

A `FAILED` state with a recorded failure reason exists now. It was
deliberately deferred while only the fake executor existed, because
a state with no reachable failure path would have been unreachable
and untested, and it was added when the real executor gained a
deliberately injected failure path that could exercise it (see
section 16). Admission problems stay different from runtime
failures: a request that can never run is rejected synchronously
when it is offered to the scheduler (see section 4), before the
scheduler takes ownership of it, so it does not need a terminal
state to describe it. An execution error after admission does:
the executor that ran the failing forward moves that one request
to `FAILED` through `Request.fail`, recording the exception as its
`failure_reason`, and the original exception still propagates to
the caller. Callers must not treat `FINISHED` or `CANCELLED` as a
place to hide execution errors.

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
| Cleaning up a request that finishes or is cancelled | Scheduler prunes the queue entry; Engine calls executor release for backend resources (see section 8) |

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

Executor boundary:

- [x] The engine runs to completion with a custom executor that shares no code with the fake backend.
- [x] A finished request is released exactly once.
- [x] No release happens while a request is still active.
- [x] A running request cancelled between steps is released, and a waiting request cancelled before prefill needs no release.
- [x] Releasing the same request twice on the fake backend is a safe no-op after the first call.

## 8. Executor contract

The engine depends on the `Executor` protocol in
`src/miniserve/executor.py`, not on `FakeExecutor`. The scheduler
never talks to an executor, so replacing the backend does not change
scheduling, admission, or the budget rules. The protocol is three
methods:

| Method | Input | Effect on success | Resources |
| --- | --- | --- | --- |
| `prefill(requests)` | The WAITING requests the scheduler selected for this step | Each request moves WAITING to PREFILLING to DECODING, through the `Request` transition methods | The executor creates any per request backend state here, such as a KV cache in a real backend |
| `decode(requests)` | The DECODING requests the scheduler selected for this step | Exactly one token is appended to each request; a request that reaches its output limit finishes | The executor updates the per request backend state it created in prefill |
| `release(request)` | One request that reached a terminal state | No change to the request state or its tokens | The executor frees the per request backend state for that request |

Responsibility is split once, not duplicated:

- The executor owns the request state transitions that happen
  during execution and the tokens it appends. The engine does not
  also update `Request` state for prefill or decode, and the
  scheduler never calls executor methods.
- The scheduler owns the queues. It decides which requests are
  selected, moves prefill requests into `running` in `update`, and
  prunes terminal requests.
- The engine owns the order of one step and the timing of release.
  It tracks every request the executor started work on, and after
  `update` it calls `release` for each tracked request that reached
  a terminal state, at most once per request. Tracking matters for
  cancellation: a running request cancelled between steps leaves the
  running list without appearing in a later step output, so the
  engine sweeps its tracked set at the end of each step and once
  more at the end of `run_until_complete`. A request cancelled
  while waiting was never tracked, never ran prefill, and holds no
  backend resources, so it needs no release.
- `release` must be idempotent on the executor side as a second
  safety net. The fake backend holds no per request resources, so
  its release only records the request id, in order, in
  `released_request_ids`, which tests use to verify the engine
  drives release correctly.

Failure responsibility: if `prefill` or `decode` raises, the
exception propagates to the caller of the engine. An executor that
knows which request its forward failed on moves that request to
`FAILED` with a recorded reason before the exception propagates
(see section 1). The engine then still runs the scheduler
`update` for that step and releases every tracked request that
reached a terminal state, so a failed request leaves its queue
and its backend resources are freed, before the original exception
is re raised. No release is invented for a request that is not
terminal, and the engine does not invent a terminal state for a
request the executor did not mark.

What a real backend swap changes and does not change: a real
executor must implement the same three methods, produce the first
token semantics agreed at that stage, and free its cache in
`release`. It does not require changes to the scheduler contract,
the admission rules, the cancellation entry point, or the per step
budget accounting.

## 9. First week checkpoint

Checked on 2026-11-06 by rerunning the full suite and the demo:

- Normal requests complete through the engine with both the fake
  backend and a custom protocol executor.
- Overlong and otherwise invalid requests are rejected at admission
  and never enter a queue.
- Cancellation of waiting and running requests and duplicate
  request ids have explicit, tested behaviour.
- The scheduler budget boundaries are covered by hand traced tests.
- Open items carried forward, in priority order: there is no FAILED
  state or failure reason yet, because there is still no real
  execution failure path; resident cache capacity is distinguished
  from the per step budget in this document but not yet enforced;
  cancellation still takes effect at a scheduling boundary, not in
  the middle of a step. None of these block the metrics work that
  comes next, and the FAILED decision must be revisited when a real
  executor is introduced.

## 10. Serial baseline and the policy switch

`SerialScheduler` in `src/miniserve/scheduler.py` is the one at a
time baseline. It subclasses `Scheduler`, so admission, validation,
cancellation, and duplicate id handling are inherited unchanged and
are not reimplemented. The `Request` type, the executor contract,
and the stopping rule (`max_new_tokens`, enforced inside
`Request.append_token`) are also shared. The whole engine is not
copied: one `Engine` drives either scheduler.

The only difference is selection in `schedule`:

- Continuous: decode work is selected first, then waiting requests
  may join for prefill in the same step while batch size and token
  budget allow.
- Serial: at most one request is active. If a request is running,
  only that request is selected for decode and no prefill is
  selected, even when slots and budget are free. If nothing is
  running, only the head waiting request is selected for prefill.

`create_scheduler(policy, ...)` is the single configuration switch.
The value `continuous` returns a `Scheduler`, `serial` returns a
`SerialScheduler`, and any other value raises `ValueError`.

Fairness rule: a comparison is only meaningful when both policies
receive the same request specifications, the same executor, and
the same limits. With the fake executor and the three request demo
set, both policies produce output lengths 4, 2, and 3; the serial
run takes 12 steps and the continuous run takes 5 steps. Those step
counts are simulation scheduling counts, not model throughput.

## 11. Deterministic workloads

`src/miniserve/workload.py` generates fixed workloads. Each
`WorkloadRequestSpec` carries the request id, the exact prompt
token ids, the output limit, and the arrival time, so a workload
can be rebuilt exactly and shared by both schedulers.

- Profiles: `short` (prompt 4 to 16, output 2 to 8), `mixed`
  (prompts alternating short 4 to 12 and long 32 to 64 by position,
  output 2 to 16), and `long_output` (prompt 8 to 32, output 24 to
  48).
- Determinism: generation uses a dedicated seeded random generator.
  The same profile, seed, and count always produce the same ids,
  token ids, output limits, and arrival times. A different seed
  produces a different workload where the ranges allow variation.
- Arrival times start at 0.0 and never decrease. They are
  simulation timetable values. The current engine admits every
  request it is given at once, so replaying requests at their
  arrival times belongs to the experiment pipeline that follows;
  the specification stores the time now so no later stage has to
  invent it.
- `build_requests` builds a fresh `Request` per specification on
  every call, passing the stored arrival time, so replaying one
  workload through two schedulers never shares request state.
- A worked example with seed 1234 and six requests per profile is
  saved in `benchmarks/workload_example.json`. Its token ids and
  times are simulation data, not tokenizer output or measured
  model performance.

## 12. Simulation time and the step timeline

Simulation runs keep their own time, separate from the wall
clock. Every request in a run shares one `SimulationClock` from
`src/miniserve/simulation.py`, and that clock only moves when the
simulated executor charges an explicit cost: prefill costs a
per token parameter times the prompt length, and decode costs a
per token parameter per token, with both parameters recorded in
the run configuration. Costs inside a step accumulate one request
after another, so durations are hand costable and the total
logical duration of a run is its total work under the model,
whichever policy scheduled it.

Requests are replayed at their workload arrival times: a request
is admitted at the first step boundary at or after its arrival,
and an idle clock jumps straight to the next arrival instead of
stepping through empty rounds. Each step writes one timeline row
with its logical start and end times, the prefill and decode ids
it executed, and the waiting and running ids at its start, so
step counts, queue rounds, and per step active counts can be
checked without trusting a summary.

The full contract, the hand worked example, and the limits of
the model are in `docs/simulation.md`. The standing rule from
that document: step counts are scheduling counts and durations
are cost model artefacts; neither is model throughput, model
latency, or evidence about a GPU.

## 13. Experiment pipeline

`benchmarks/run.py` runs one experiment: it generates the
workload for a profile and seed, replays it under each chosen
policy in simulation time, and writes a fresh run directory
holding `config.json` (including the package version and the
simulation costs), the raw per request metrics and per step
timeline for each policy, and a summary. A run id names the
directory, and an existing run id is refused rather than
overwritten, so reruns always leave both traces.

`benchmarks/summarize.py` is a separate entry point that
recomputes the summary from the raw files in a run directory
only. It never touches the objects that produced the run, so
recomputing an unchanged directory reproduces the original
summary exactly. The layout, the commands, and one recorded
simulation comparison are in `docs/experiments.md`.

## 14. Second week checkpoint

Checked at the second week review by rerunning the full suite
(90 passed), the demo (3 requests in 5 steps, unchanged), the
type check, and the experiment pipeline end to end: recomputing
the recorded run `simulation-short-seed1234` with
`benchmarks/summarize.py` left `summary.json` and `summary.md`
byte identical, verified by checksum.

Fairness of the baseline, verified against the code rather than
assumed:

- Both policies replay the same workload specifications, rebuilt
  into fresh request objects per policy, so no execution state
  leaks from one run into the other.
- Both use the same simulated executor, the same cost parameters,
  the same scheduler limits, and the same stopping rule. The only
  difference is the selection policy inside `schedule`.
- Both produced the same work in the recorded run: 6 finished
  requests and 30 output tokens each. Any later real comparison
  must pass the same check, that the generated work is equal or
  that any difference, such as an early EOS, is recorded, before
  its timing numbers are compared.

Why throughput and TTFT can move in opposite directions, in the
terms of this project: the continuous policy admits a new request
for prefill while earlier requests are still decoding, so a
waiting request tends to see its first token earlier and mean
TTFT falls. Under the current serial cost model, however, the
work inside a shared step is charged one request after another,
so a request sharing steps with others can still finish later
than it would have alone, and mean latency rises. Both effects
appeared in the recorded run (mean TTFT 9.2167 against 10.3167,
mean latency 15.8167 against 12.3167, in logical seconds). A real
batched backend executes a shared step differently, so the
direction and size of this trade off must be measured again with
the real executor; the simulation establishes the mechanism and
the vocabulary, not the real numbers.

Open items carried forward, in priority order: the FAILED state
decision must be revisited when the real executor introduces a
real failure path; resident cache capacity is still distinguished
but not enforced, and becomes a concrete question once a real
cache exists; the first token source changes with a real backend,
because prefill logits can produce the first token, and
`docs/metrics.md` already records that this change must be
documented when it happens. None of these block the real model
selection that comes next.

Model selection conditions, listed before any model was picked:
decoder only and small (roughly 0.1B to 0.5B parameters); a
permissive licence that allows this project to be published; a
cache structure that is simple to explain, with no sliding window
or hybrid attention in the first model; official support in the
locked transformers version; a tokenizer shipped in the same
repository; a revision that can be pinned; and runnable on the
machine actually available here, which has no GPU, so CPU
execution within the memory budget is a hard condition, not a
preference.

## 15. Real executor integration checkpoint

The hand written generation loop is now verified against the
reference path and wired into the engine as a real executor.

Correctness, from `benchmarks/correctness.py` and its saved
record `benchmarks/correctness_output.json`: 12 fixed cases
(single token limit, short and multi token prompts, a 121 token
long prompt, EOS cases) matched the reference token for token
with no first divergence, and stop reasons agreed everywhere.
The real EOS token never appeared in any probe or case, so EOS
stopping on the real model was checked with a shared custom EOS
id on both paths, and a literal near limit (8,192 position) run
was not attempted on this CPU machine. Both limits are recorded
in `docs/generation.md` section 5 rather than smoothed over.

Integration, in `src/miniserve/real_executor.py`: the executor
implements the section 8 protocol unchanged, one request per
forward. Prefill appends the first token from its logits, so
first token time is recorded at prefill completion, the real
source `docs/metrics.md` anticipated. Decode appends one token
per call after that, so N tokens take 1 prefill and N - 1 decode
forwards, and a limit of 1 takes no decode forward, verified both
by scripted fake model tests and by real runs through
`benchmarks/real_generate.py` (the default run finished in 16
steps with ids identical to the reference). The executor owns
finishing a request on EOS, stores exactly one cache per active
request, and drops it in `release`; after every recorded run the
active cache count was zero. A model error still propagates with
no invented terminal state, so the FAILED state question stays
open until a failure path is deliberately exercised, which is
the next scheduled item together with cache release checks.

## 16. Stop, failure, and cache release checkpoint

That failure path has now been deliberately exercised, and the
FAILED question from sections 1, 8, and 15 is settled.

What exists now:

- `RequestState.FAILED` is a third terminal state. `Request.fail`
  moves any non terminal request there, records the supplied
  reason in `failure_reason`, and stamps `finished_time` from the
  request clock. Failing a terminal request is an error, so a late
  failure cannot overwrite a finished, cancelled, or earlier
  failed outcome, and a failed request keeps the tokens it already
  produced.
- `RealExecutor` wraps each model forward. A forward that raises
  marks exactly the request it was running as FAILED, with the
  stage and the exception in the reason (for example, prefill
  forward failed or decode forward failed), and then re raises.
  A decode call rejected before any forward ran, such as a missing
  cache, still raises without marking, because that is a caller
  bug, not a model failure.
- `Engine.step` cleans up a failed step before re raising: the
  scheduler update runs and terminal requests are released, so a
  failed request leaves its queue and the executor drops its
  cache. A waiting request behind the failing one is untouched and
  stays WAITING.
- Metrics count failed separately. `summarize_request` exports
  `finish_reason` failed plus the `failure_reason` text, and
  `count_by_finish_reason` and the experiment summary both carry a
  failed count alongside finished, cancelled, and unfinished.

Verification, all with the scripted model in
`tests/test_cache_cleanup.py` and the diagnostic in
`benchmarks/cache_cleanup_check.py` (saved record
`benchmarks/cache_cleanup_output.json`, all scenarios clean):
five sequential requests on one shared executor each returned the
active cache count to zero; a prefill failure produced a FAILED
request with no tokens and no cache left; a decode failure kept
its one produced token, ended FAILED, and released its cache; EOS
from prefill and a running cancellation released theirs as before.
Release here means the executor drops its per request cache
reference. It does not mean a framework allocator returns reserved
memory, and no per request allocator emptying call is made; on
this CPU only machine there is no GPU memory reading to report.

## 17. Third week checkpoint and scope decision

Checked at the third week review by rerunning the reference
comparison, the real engine path, the full test suite, and a
clean environment install:

- Correctness: `benchmarks/correctness.py` again matched 12 of
  12 cases token for token against the reference path, with stop
  reasons agreeing everywhere and no first divergence. The real
  EOS token again appeared in no case, so the EOS evidence stays
  exactly what section 15 and `docs/generation.md` section 5 say
  it is: the shared custom EOS comparison plus the scripted loop
  tests. Nothing about that limit changed and nothing new is
  claimed for it.
- Real path: `benchmarks/real_generate.py` with its default
  prompt finished in 16 steps (1 prefill forward, 15 decode
  forwards) with token ids identical to
  `benchmarks/reference_output.json`, an active cache count of
  zero afterwards, and the request released.
- Tests and types: the full suite passes (113 tests), mypy is
  clean over the 10 source files, and the demo still finishes 3
  requests in 5 steps.
- Reproducible install: a fresh virtual environment created for
  this check installed the project with the dev extra and passed
  94 tests, with the 3 torch dependent test modules skipped
  because the model extra was not installed there. That is the
  expected split: the core and simulation suites need no model
  download, and the model tests run in the main environment where
  the model extra is installed.
- Experiment pipeline after the failed count change: recomputing
  the recorded run `simulation-short-seed1234` changed only the
  summary schema (a failed column, zero everywhere, since that
  run has no failed requests); every recorded number is unchanged
  and a second recomputation is byte identical to the first.

The single request relationship, stated once more in one place:
prefill runs the whole prompt once and its last position logits
produce the first token, so first token time is prefill
completion. Decode then feeds one token at a time, the most
recent generated token, which is the only token the cache has not
seen, and each decode forward produces the next token. The cache
therefore always trails the produced tokens by one position, and
a run of N tokens takes 1 prefill and N minus 1 decode forwards.
No unexplained single request output or cache behaviour remains.

Scope decision: the single request checkpoint passes, so the
batching work comes next as planned. If that work runs long, the
cut order is fixed in advance: HTTP and extra charts are cut
first, CLI and correctness checks are not. The cache structure
recorded for the batching design is in `docs/generation.md`
section 8.

## 18. Resource limits checkpoint

The scheduler now carries two resident resource limits next to
the per step limits, with the full semantics in
`docs/resources.md`:

- A request reserves prompt plus full output limit from the
  step it is prefilled until it reaches a terminal state. A
  waiting request reserves nothing, because it holds no cache.
- A reservation that exceeds the configured cache capacity on
  its own is rejected at admission, like the context limit; a
  reservation that only exceeds the currently free capacity
  waits in the FIFO queue and starts when a finish or a
  cancellation returns capacity. The reserved total is computed
  from the running list, so the return cannot drift.
- The per step token budget stays a logical work limit. It is
  not a memory figure: the padded allocation of a batched step
  follows the longest row, and pack and split temporaries are
  not counted in any token number. No claim is made that token
  counting alone can guarantee a run never runs out of memory.

## 19. Fourth week checkpoint: batched runtime acceptance

The batched runtime was re-verified end to end in one pass, on
top of the per scenario records from the individual steps.
`benchmarks/fourth_week_acceptance.py` runs six scenarios on
the real locked model, compares every generated sequence token
for token against individual single request runs with the same
limits, and saves the full record, including one complete per
step scheduling trace, in
`benchmarks/fourth_week_acceptance_output.json`. The scripted
counterpart suite is `tests/test_acceptance.py`, so the same
six claims also run in the ordinary test command. Every check
passed:

| Scenario | Evidence from the saved record |
| --- | --- |
| Equal lengths (5 and 5, limit 4) | 4 steps, decode batch sizes [2, 2, 2], tokens matched |
| Different lengths (7 and 2, limit 4) | 4 steps, decode batch sizes [2, 2, 2], tokens matched |
| Exit (limits 2 and 5) | 5 steps, decode batch sizes [2, 1, 1, 1], survivor tokens matched |
| Join (third request admitted after 2 steps) | 5 steps, decode batch sizes [2, 2, 3, 3], the step 3 trace shows the joined request prefilled while the original pair decoded, all three token sequences matched |
| Cancel (running request cancelled after 2 steps) | Cancelled request frozen at its 2 produced tokens, survivor finished with its individual sequence, decode batch sizes [2, 1, 1, 1, 1] |
| Failure (batched decode failure injected at the model boundary) | Both rows FAILED with the injected reason recorded, the exception propagated to the caller, both requests kept their 1 produced token |

In every scenario the active cache count after the run was 0
and every request was released. The acceptance conditions for
this stage are therefore met on evidence, not on the earlier
per scenario records alone: at least two requests share single
decode forwards, a new request joins a run in progress, a
request leaving early or cancelled does not disturb the
survivors, and cleanup is complete on the finish, cancel, and
failure paths alike. No throughput claim is made anywhere in
this checkpoint: the machine is CPU only, the runs are small,
and the numbers recorded are step and forward counts, which
are scheduling facts, not performance results.

Remaining risks, carried forward in priority order:

| Risk | Why it matters | Standing response |
| --- | --- | --- |
| Pack and split copy every layer on every decode step | The copy cost grows with batch size and cached length and could erase the batching benefit, especially for a small model | Named suspect for the benchmark stage; measured before any performance wording is used |
| Prefill is still one serial forward per request | A long prompt occupies its step and delays decode rows sharing that step | Accepted by the plan for this design; chunked prefill stays out of scope |
| Reservations charge the full output limit | Requests that stop early waste resident capacity until they end | Accepted conservative bound, recorded in `docs/resources.md` |
| Natural EOS still never observed on this base model | Real model EOS evidence rests on the shared custom EOS comparison and the scripted tests | Stated wherever EOS is claimed; not upgraded silently |
| The layout is specific to this model family and cache type | Other architectures or cache implementations may not pack the same way | No portability claim; a second model would need its own layout check |
| Failure injection covers batched decode, and prefill failure is covered by the single request cleanup suite | A failure mode outside those two paths is unproven | New executor paths must bring their own failure test, per the executor contract |

Scope decision: the core runtime checkpoint passes, so the
service work comes next as planned. If that work runs long,
the cut order from the third week checkpoint still holds: the
CLI and the verified runtime are the deliverable, and no
unverified service behaviour is claimed in their place.

## 20. Service boundary checkpoint

The minimal service boundary is designed and frozen in
`docs/service.md`; no server code exists yet, and none is
claimed. The decisions that the implementation must not
silently change:

- One generation entry point (`POST /generate`), one streaming
  variant with a fixed event vocabulary (`token`, then exactly
  one terminal `finished` or `error`), one health check, and
  one cancel command that mirrors the scheduler cancellation
  contract, including its unknown id error.
- The model is loaded once at startup from the locked
  configuration. The API layer validates shape, assigns the
  request id, and translates outcomes; admission, scheduling,
  execution, and metrics stay in the runtime.
- Error semantics are mapped to runtime sources, not invented
  at the boundary: `invalid_request` from API validation,
  `request_rejected` from admission refusal, `queue_full` from
  the waiting queue bound, `generation_failed` from the
  executor failure path, each in one shared error shape.
- The contract is explicitly not OpenAI compatible, carries no
  sampling controls, and states that cancellation takes effect
  at a scheduling boundary.

The staged implementation order (single engine owner, then
streaming, then disconnect cancellation, then the queue bound,
then an integration pass, then a freeze review) is recorded at
the end of `docs/service.md`.

## 21. Background execution checkpoint

The first staged step of `docs/service.md` is implemented in
`src/miniserve/service.py`. `EngineOwner` runs one engine on
one worker thread. That thread is the only one that calls
scheduler or engine methods, so scheduler state has a single
owner and the executor only ever runs on that thread. Callers
on other threads hand the worker commands (submit, cancel,
shutdown) through one thread safe queue and are answered on a
per command reply queue. Submission returns as soon as
admission is decided, not when generation finishes.

Semantics worth keeping:

- When nothing is pending the worker blocks on the command
  queue; it does not poll or spin. While requests are pending
  it drains queued commands between steps, then runs one step,
  so a command waits at most for the step already in progress,
  which is the scheduling boundary the cancellation contract
  already stated.
- A step that raises does not kill the worker. The engine has
  already cleaned up the failed step; the worker finalises the
  requests that reached a terminal state and keeps serving the
  rest, so a waiting request behind a failing one still runs.
  A FAILED request's handle raises `GenerationFailedError`
  from `result`, carrying the recorded failure reason.
- Shutdown is a command. Once queued, new submissions are
  refused, requests already pending are finished, and then
  the worker exits. Submitting after shutdown raises
  `RuntimeError` instead of queueing work nobody will run.
- Health reports liveness and the active count only; it runs
  no forward and reserves no capacity, as the contract says.

Evidence: `tests/test_service.py` (9 tests, fake executor, a
step gated executor makes the concurrency deterministic), and
`benchmarks/service_demo.py` with its saved record
`benchmarks/service_demo_output.json`, where the second
request was submitted at 0.101 seconds while the first was
still generating (it finished at 0.302), every executor call
ran on the single worker thread, and both requests were
released. The demo's decode delay is artificial and its times
are labelled simulation, not a performance result. No HTTP
layer exists yet; the owner is the boundary a later front end
must use, and the queue bound behind `queue_full` is still
the later overload step, not part of this checkpoint.

## 22. Streaming events checkpoint

The second staged step is implemented in the same owner plus
`src/miniserve/streaming.py`. After every engine step the
worker diffs each active request's generated token list
against what it already emitted and queues one `token` event
per new token on that request's handle, before the next step
runs. A reader therefore receives the first token event while
generation is still in progress. Each handle then receives
exactly one terminal event, `finished` (including a
`cancelled` finish reason) or `error` for a FAILED request,
and nothing after it. `StreamEvent.format_sse` renders the
server sent events frame of the frozen contract; no HTTP
transport is claimed yet.

Incremental text comes from `IncrementalDecoder`, which
re-decodes the whole token sequence produced so far and emits
only the new suffix. Decoding each token in isolation and
concatenating would be wrong whenever a piece's text depends
on its neighbours or a token carries only part of a multi
byte character; a trailing replacement character is held back
as unstable until a later token completes it or the request
ends. With no tokenizer, token events carry ids and empty
text, and no text is invented.

Evidence: `tests/test_streaming.py` (9 tests, scripted
executors and sequence dependent tokenizers, including the
split character case and a first token event observed while
the worker was gated inside the following decode), and
`benchmarks/service_streaming_demo.py` with its saved record
`benchmarks/service_streaming_output.json` on the real locked
model: two prompts streamed concurrently, token events for
the two requests interleaved in arrival time, each stream
ended by exactly one `finished` event, each streamed text
equal to the decoding of its full token sequence, decode
batch sizes [1, 2, 2, 2, 2, 1], active caches back to zero,
and both requests released. Its times are wall clock
observations on a shared CPU, not a performance result.
Client disconnect cancellation and the slow consumer policy
remain later staged steps; a slow reader currently only
backs up its own handle's unbounded event queue, which the
overload step must bound.

## 23. Client disconnect checkpoint

The third staged step of `docs/service.md` is implemented in
`src/miniserve/service.py`. A disconnect reaches the worker
as its own command kind and is then indistinguishable from
an explicit cancel: the worker calls the scheduler's single
cancellation entry point, the engine releases a tracked
request that reached a terminal state, and the handle is
finalised with the single terminal event. Three caller side
signals forward to that command: the owner's `disconnect`
method, the handle's `disconnect` method, and closing a
handle's event stream early, which the events iterator
treats as the client going away. The handle records that it
was disconnected and why, and the owner counts disconnect
cancellations separately from explicit cancels.

Semantics worth keeping:

- Waiting and running requests take the same path and end
  in different resource states, both correct: a waiting
  request was never prefilled, is never tracked by the
  engine, and needs no release; a running request is
  released exactly once on the ordinary cancelled path.
- Cancellation still takes effect only at a scheduling
  boundary. A disconnect that arrives while a step is in
  flight waits for that step; the step's token stays
  produced and counted, and no token is produced after the
  boundary. A forward already running on a real backend is
  not preempted, which the contract already stated.
- After the disconnect the request is gone from both
  queues, so nothing keeps generating in the background,
  and the worker keeps serving every other request.

Evidence: `tests/test_disconnect.py` (3 tests: a waiting
disconnect through the handle with a step gated executor,
asserting zero tokens and no release; a running disconnect
through the owner, frozen at exactly one token with its
release recorded and the service still usable afterwards,
including the repeated disconnect and unknown id cases; and
a stream close disconnect against a slow fake executor,
frozen well below its limit), and
`benchmarks/service_disconnect_demo.py` with its saved
record `benchmarks/service_disconnect_output.json`, where a
waiting request disconnected with zero tokens and no
release, a running request froze at 3 of 20 tokens and was
still frozen 0.3 seconds later, and a survivor submitted
afterwards finished normally. The demo's decode delay is
artificial and its record is labelled simulation, not a
performance result.

## 24. Overload and slow consumer checkpoint

The fourth staged step is implemented in the same owner.
Admission now runs an overload check on the worker thread
before scheduler admission: if the scheduler's waiting
queue is already at the configured bound (default 32
waiting requests; running requests never count), the
submission is refused with `QueueFullError`, the
`queue_full` error of the contract. The refusal reaches no
scheduler state: nothing is queued and the request id is
not consumed, so the same id succeeds once the queue
drains. The owner counts every decision it makes (offered,
accepted, refused for a full queue, refused by scheduler
admission) and reports the queue full rejection rate from
those counts, because an overload report that only covers
accepted requests hides exactly the requests that mattered.

Each handle's event queue is now bounded (default 128
events) and the worker enqueues without ever blocking, so
a slow reader cannot stall the engine for other clients.
The slow consumer rule is cancellation, not silent loss: a
token event that does not fit is dropped and counted on
the handle, the handle is marked as a slow consumer, and
the worker cancels the request at that same boundary
through the scheduler, with the ordinary release and
terminal event. The terminal event itself is guaranteed:
if the buffer is full when it is due, the oldest buffered
token events are dropped and counted to make room, so a
reader that resumes always learns the outcome and the true
generated token count instead of waiting on a stream that
never ends. Disconnect and slow consumer cancellations are
counted in the same admission statistics.

Evidence: `tests/test_overload.py` (4 tests: the waiting
bound under a step gated executor, asserting the error's
code, limit, and observed count, the statistics and
rejection rate, and the refused id being accepted on retry;
a slow consumer with a two event buffer cancelled at
exactly three produced tokens with dropped events counted,
its release recorded, and its terminal event delivered on
a late read; a well behaved request served normally after
a slow consumer; and invalid bound configuration rejected),
and `benchmarks/service_overload_demo.py` with its saved
record `benchmarks/service_overload_output.json`: five
offered, three accepted, two refused with `queue_full`, a
rejection rate of 0.4, the refused id accepted after the
queue drained, and the slow consumer cancelled at three
tokens with two dropped events and its terminal event
intact. Both scenarios use fake executors and are labelled
simulation. What remains for the service week is the
integration pass over mixed clients and the freeze review;
no HTTP transport exists yet, and none is claimed.

## 25. HTTP integration checkpoint

The fifth staged step is implemented in
`src/miniserve/server.py`, using only the standard library
HTTP server, so the project gains no runtime dependency. A
thin front end object holds the owner, the tokenize
function, and the model identity; the handler threads only
parse bodies, call the front end, and write responses or
stream frames. Every generation still goes through the
engine owner, so model execution stays on the single worker
thread and the HTTP layer never touches the scheduler, the
engine, or an executor.

The four contract endpoints answer as specified: health
reports the owner status plus the loaded model id and
revision (200 while accepting, 503 once unavailable); the
non streaming generate path validates the body, assigns the
request id, waits on the handle, and returns the result
object with the runtime finish reason and token counts; the
streaming path writes the frozen server sent events frames
as they arrive and maps a broken socket to the owner's
disconnect command; cancel mirrors the scheduler contract,
including `unknown_request` as a 404. The contract error
table is implemented one to one: `invalid_request` 400 for
malformed JSON and shape failures, `request_rejected` 400
for admission refusals, `queue_full` 429, and
`generation_failed` 500 carrying the recorded failure
reason, so a worker failure is a visible response, never a
hanging connection.

Evidence: `tests/test_http_integration.py` (6 tests over
real HTTP on ephemeral ports: two concurrent clients, a
streaming client cancelled through the cancel endpoint with
exactly one terminal event and a working follow up request,
invalid and rejected bodies, an overload refusal with its
admission statistics, an injected executor failure answered
500 while health stays ok and the next request succeeds, and
a shutdown check that no worker thread survives), and
`benchmarks/service_integration.py` with its saved record
`benchmarks/service_integration_output.json`, which reruns
the same five flow scenarios plus a real model smoke test
through HTTP (the locked model answered one generate call
with 5 prompt tokens, 4 generated tokens, and real text, and
its active cache count returned to zero). The flow scenarios
use fake executors and are labelled simulation; the smoke
timing is a wall clock observation on a shared CPU. The
startup command and curl examples are in the README and in
`benchmarks/run_server.py`, which serves a fake backend by
default and the locked model with `--real`.

## 26. Fifth week checkpoint and feature freeze

Checked at the fifth week review by rerunning the full
suite (169 passed), the HTTP integration script (all
scenarios passed, including the real model smoke test), the
scheduler demo, the type check, and a manual curl pass
against `benchmarks/run_server.py` (health, one generate
call, and one streamed generate call, all answering as the
contract specifies).

The service is frozen here. The supported and unsupported
lists are recorded at the end of `docs/service.md` and are
the claims ceiling from now on: single owner execution,
streaming with the frozen vocabulary, disconnect and
explicit cancellation, the bounded waiting queue and event
buffer, and the five contract errors are supported and
evidenced; authentication, sampling, multi model serving,
OpenAI compatibility, persistence, and exposure beyond
localhost are not supported and are not claimed. The CLI
core and the verified runtime underneath are unchanged by
the freeze.

What remains is the experiment stage: a fixed protocol, a
pilot, the main serial against continuous comparison, and
the write up. From here on only correctness fixes and
experiment work are in scope; a service change that is not
needed for correctness does not land, however small.

## 27. Benchmark protocol checkpoint

The real model experiment protocol is frozen in
`docs/benchmark_protocol.md`, with its machine readable copy in
`configs/benchmark.json`, before any real comparison was run.
The decisions the experiment stage must not silently change:

- The comparison is serial (SerialScheduler plus the single
  request executor) against continuous (the default scheduler
  plus the batched executor), sharing the loaded model, the
  tokenizer, the prompt token ids, the output limit, and the
  scheduler limits. A result is a strategy result, never a pure
  scheduler result, because the executors differ by design.
- Model loading and per strategy warmup (2 requests, discarded)
  are excluded from every timed window. The core runtime window
  is the wall clock around `run_until_complete` with all
  requests already admitted (an all at once closed batch); the
  HTTP benchmark is separate and client timed, and simulation
  numbers never enter a real table.
- Fixed length runs and natural EOS runs are separate modes.
  Fixed length mode sets the executor EOS id to the sentinel
  -1, stated openly in the protocol, so every request generates
  exactly its limit and the workload is identical by
  construction. Natural EOS mode uses the model EOS id 0 and
  must record actual output lengths.
- Timing on this CPU only machine is wall clock over
  synchronous execution: no CUDA events exist to use, and no
  per token synchronisation is inserted. A GPU repeat would
  require rewriting that section first.
- A run may be quoted only after its workload check passes:
  equal finished counts, prompt tokens, and output tokens, and
  token for token identical sequences per request id in fixed
  length mode. Each run writes a fresh run directory and
  refuses to overwrite an existing one.

## 28. Pilot checkpoint

The pilot defined in the protocol ran as run id
`real-pilot-seed1234` (`benchmarks/pilot_benchmark.py`): 20
requests per strategy, the 10 protocol prompts cycled twice
(prompt lengths 5, 5, 6, 6, 9, 8, 5, 8, 9, 6, totalling 134
prompt tokens per strategy), output limit 8, fixed length mode,
continuous batch size 4, serial first. Model load took 5.3644
seconds and warmup took 3.0889 (serial) and 0.9924 (continuous)
seconds; all three figures are excluded observations.

Results, all wall clock on the shared CPU and exploratory at
this single repeat scale: both strategies finished 20 of 20
with 0 failed, 0 cancelled, and 160 output tokens, and the
generated sequences matched token for token per request id, so
the workload check passed. Serial took 160 steps (20 prefill
and 140 decode forwards) and 27.4936 seconds, with mean TTFT
13.0717, mean latency 14.2577, and mean TPOT 0.1694.
Continuous took 40 steps (20 prefill and 35 decode forwards,
every decode batch size 4) and 13.9019 seconds, with mean TTFT
6.1622, mean latency 8.2292, and mean TPOT 0.2953. Active
caches returned to zero and all 20 requests were released
under both strategies. The manual timing check recomputed the
first serial request's queue time, TTFT, latency, and TPOT
from its raw timestamps and matched the stored values exactly.

Problems found by the pilot: none in timing, EOS, queue, or
configuration behaviour. No request stopped early (every output
length is exactly 8 and no generated token is the natural EOS
id 0), no queue or admission error occurred, and the two
strategies used the same backend model and output lengths, so
nothing had to be fixed before the main experiment. Two
observations are recorded instead of hidden: continuous TPOT
is higher even though its TTFT and latency are lower, because
a batched decode step costs more wall time per step while the
wait before the first token shrinks; and with batch size 4 the
continuous run is five sequential groups of four, not one
batch of twenty, which is exactly the concurrency variable the
main experiment varies next. These pilot numbers are a
measurement path check, not a headline comparison: one repeat,
a shared CPU, and no rotation of strategy order.

## 29. Main experiment checkpoint

The main experiment ran as run id `real-main-seed1234`
(`benchmarks/main_benchmark.py`) under the frozen protocol:
closed batches of 20 requests, fixed length mode, output limit
8, three repeats per configuration, strategy order rotated
between repeats, and the continuous concurrency (its scheduler
max batch size) varied over 1, 2, 4, and 8 on two workloads.
Serial has no concurrency variable, so it ran once per workload
and repeat and is paired with each continuous concurrency of
the same repeat. The short workload cycles the 10 protocol
prompts (lengths 5 to 9); the mixed workload alternates those
short prompts with long prompts of 20 to 24 tokens built by
appending fixed suffix texts, so prefill costs differ by about
a factor of three inside one batch. The sample is 60 finished
requests per continuous point, below the plan target of about
100, so every figure is exploratory and ranges across repeats
are reported next to means. Every run finished all 20 requests
with 0 failed and 0 cancelled, and every token comparison
against the paired serial run matched, so the workload check
passed for all 30 runs.

Short workload, means across the three repeats with the wall
range in brackets: serial took 27.39 seconds [25.83, 28.40] at
5.85 output tokens per second, mean TTFT 13.2452, mean latency
14.3988, mean TPOT 0.1648. Continuous at concurrency 1 was
statistically indistinguishable from serial (25.70 seconds,
6.23 tokens per second), which is the expected fairness
sanity check because a batch size of 1 reduces it to one
request per step. At concurrency 2, 4, and 8 the wall time
fell to 15.69, 10.56, and 9.30 seconds and output throughput
rose to 10.21, 15.19, and 17.48 tokens per second, about three
times the serial figure at concurrency 8. Mean TTFT fell from
13.2452 (serial) to 3.5028 and mean latency from 14.3988 to
5.8303 at concurrency 8, while mean TPOT rose from 0.1648 to
0.3325: sharing steps cuts the wait before the first token,
but each batched step is more expensive, so the gap between
two tokens of one request grows. The gain from 4 to 8 is much
smaller than from 1 to 2 or 2 to 4 on every metric.

Mixed workload: serial took 27.46 seconds [26.80, 27.88] at
5.83 tokens per second. Continuous rose from 5.67 tokens per
second at concurrency 1 to 8.78, 13.95, and 15.56 at 2, 4, and
8, with wall times 28.24, 18.23, 11.49, and 10.30 seconds.
Mean latency fell from 14.5063 (serial) to 6.7148 at
concurrency 4 and then stayed at 6.7141 at concurrency 8,
while mean TPOT kept rising to 0.3753: on this workload the
latency benefit of wider batches saturates at 4, and going to
8 buys throughput only, paid for with slower per token
progress for each individual request. These are shared CPU
wall clock observations for this model and workload only; no
slower run was hidden and no GPU conclusion is drawn.

## 30. Token budget mechanism checkpoint

The mechanism experiment ran as run id `real-budget-seed1234`
(`benchmarks/budget_benchmark.py`): the mixed workload,
continuous strategy, concurrency 4, 20 requests, fixed length
mode, three repeats, tier order rotated, with the per step
token budget set to 32, 64, and the protocol default 8192.
The smallest budget still admits the longest prompt (24
tokens) on its own, and admission was verified identical
across tiers, so the accepted request set is the same at
every budget and differences are budget effects. Every run
finished 20 of 20 and all generated sequences matched the
reference run token for token.

Results, means across repeats: budget 32 took 42.0 engine
steps and 13.02 wall seconds [11.90, 14.70], with mean TTFT
6.1510, mean latency 8.1303, and mean TPOT 0.2828. Budget 64
took 40.0 steps and 11.44 seconds [10.81, 12.49], mean TTFT
5.2530, mean latency 6.9560, mean TPOT 0.2433. Budget 8192
took 40.0 steps and 12.50 seconds [11.82, 13.47], mean TTFT
5.4543, mean latency 7.3240, mean TPOT 0.2671. The wall ranges
of the three tiers overlap, so no tier is claimed faster
overall on this machine.

What the mechanism actually shows is narrower and more
useful. The budget binds only when the work a step wants to
pick exceeds it: with concurrency 4 the batch size cap leaves
at most 4 slots, decode costs 1 token each, and only a long
prefill of 20 to 24 tokens can push a step over a budget of
32, which is why budget 32 costs exactly 2 extra steps (one
long prefill deferred in each of two places) and why budgets
64 and 8192 produce identical step counts of 40: above the
largest step the scheduler would ever pick, a bigger budget
changes nothing. The single request timeline makes the same
point per request: the first long request (main-mixed-01,
prompt 20 tokens) waited 1.4155 seconds in the queue in one
budget 32 repeat, against 0.12 to 0.19 seconds in the other
eight runs, because its prefill did not fit the step in which
it first became eligible and it waited for a later step under
FIFO order; in the other budget 32 repeats its prefill
happened to fit and its timeline is unremarkable. So the
budget is a real scheduling lever, but at this scale it is
the second constraint to bind, after batch size, and its
effect appears as occasional deferred prefills and higher
TTFT for the deferred requests rather than as a uniform
slowdown. A workload with longer prompts or a larger batch
size would make the budget bind far more often; that is a
prediction from the step accounting, not a measured result.

## 31. Bottleneck analysis checkpoint

The experiment stage closed with a segmented analysis rather
than a presumed speedup. `benchmarks/bottleneck_analysis.py`
times the three parts of one decode step separately on the
locked model (medians of 7 trials) and recomputes the derived
ratios from the stored main, budget, and pilot summaries; the
record is `benchmarks/bottleneck_output.json`.

Findings, in evidence order:

- The batched model forward dominates one decode step at every
  measured batch size (medians about 175 to 210 ms across
  repeated runs). Pack and split together are about 1 to 3
  percent of a step at batch sizes 1 to 4 and about 6 to 8
  percent at batch size 8 (pack medians about 0.8 to 6.4 ms,
  split medians about 1.2 to 12.0 ms). They copy every
  layer on every step, so the share grows with batch size and
  cached length, but at the cache lengths measured here (at
  most 24 tokens) they are a secondary cost, not the
  bottleneck.
- Prefill is one serial forward per request in both strategies
  (segmented medians typically about 0.18 to 0.36 s per
  request, occasionally higher under shared CPU contention),
  so a 20 request run pays 20 prefill
  forwards that decode batching cannot help, and a long
  prompt's prefill occupies its step while decode rows wait.
  That is the largest structural cost the runtime itself
  controls, and the recorded candidate for any future work;
  batched or chunked prefill stays out of scope by design.
- Padding waste is workload shaped: near zero for equal short
  prompts at small batch sizes, 27.8 percent of packed slots
  for the short workload at batch size 8, and about 37 to 39
  percent for the mixed workload at every batch size.

One small change was tried against the pack cost and rejected
on its own measurements: pack was rewritten as one zero filled
allocation per layer with one copy per row, the packed values
verified identical, and the segmented medians after the change
were within run to run noise of the before medians and
slightly higher in most configurations. The change was
reverted, so the shipped pack code is exactly the code that
produced the main experiment, and the before and after medians
are kept in `benchmarks/bottleneck_pack_attempt.json` as a
tried and rejected change, not a speedup. No baseline, workload,
or measurement boundary was changed at any point in this
analysis.

## 32. Results and architecture checkpoint

The results are organised in `docs/results.md`, which is the
single place the experiment numbers are interpreted:

- The main experiment table (both workloads, every concurrency,
  means across 3 repeats with the repeat range next to each
  mean) and the budget mechanism table are recomputed from the
  saved run summaries, and the three main charts in
  `docs/charts/` (throughput, mean TTFT, mean latency) are
  generated from `benchmarks/results/real-main-seed1234` by
  `benchmarks/make_results_charts.py`, so every figure traces
  back to that run id and one generation command. Whiskers are
  the repeat range, not a confidence interval.
- The text states the benefit and the cost together: throughput
  rises to about 3.0 times serial on short and 2.7 times on
  mixed at concurrency 8, TTFT and latency fall, and TPOT
  roughly doubles over the same range; mixed latency saturates
  at concurrency 4. Sample sizes (60 finished requests per
  continuous point, below the plan target of about 100) and the
  exploratory status of every figure are stated in the
  document, and no percentile beyond the recorded means is
  reported.
- `docs/architecture.md` records the component map, the
  ownership split (HTTP translation, single engine owner,
  scheduler, executors, engine release timing), one request's
  path end to end, and where each contract lives.

With this checkpoint the code is frozen for the remaining
stages: only correctness fixes and the final delivery work
(README, resume notes, final acceptance) follow, per the fifth
week freeze decision.

## 33. Clean environment reproduction checkpoint

The frozen code was rebuilt from a fresh copy of the project
files (no `.venv`, no cache directories, no egg-info) in a new
virtual environment on a large disk, following only the README
and `docs/reproduction.md`:

- With the dev extra alone, installation succeeded and the
  suite reported 152 passed and 8 skipped, the skips being
  exactly the model dependent tests; the fake demo finished
  3 requests in 5 steps with outputs 4, 2, and 3.
- After adding the model extra (torch 2.9.1, transformers
  4.57.6, inside the locked ranges), the same clean environment
  reported 188 passed with no skips, and the real smoke run
  finished with the 16 generated token ids matching
  `benchmarks/reference_output.json` exactly, the request
  released and no active cache left.
- A search of source, tests, benchmarks, docs, and configs for
  home and temporary directory paths returned no hits, and no
  file outside the gitignored caches exceeds 500 KiB, so the
  repository carries no personal path, no credential, and no
  model weight.
- Two environment pitfalls were re-encountered and are recorded
  in `docs/reproduction.md`: a small default temporary
  filesystem (512 MiB) makes the torch install fail with
  `No space left on device` unless the environment and `TMPDIR`
  live on the large disk, and the first model download needs
  `HF_HUB_DISABLE_XET=1` on this network.
- The final issue list (natural EOS never observed, pack and
  split cost beyond 32 cached tokens unmeasured, HTTP exercised
  on loopback only, one pre existing ruff note in
  `tests/test_request.py`, exploratory sample sizes) is kept in
  `docs/reproduction.md`. The code stays frozen.

## 34. README and demo checkpoint

The README was rewritten to match the shipped runtime instead
of the early simulator description, in the order: what it does,
how to run it, architecture, results, limitations. Every claim
in it traces to a record: the result figures are the main
experiment means from `docs/results.md`, the service behaviour
is the frozen contract in `docs/service.md`, and the hygiene
statement is the reproduction check above. The old statements
that there is no real model and no HTTP API were removed
because both shipped.

The demo is reproducible without live parameter changes:
`benchmarks/demo_showcase.py` runs three acts on the fake
backend (concurrent submissions returning while a first request
generates, a request joining mid generation, and an overload
refusal next to a cancellation) and saves
`benchmarks/demo_showcase_output.json` with all seven checks
passing; the real model streaming act and the closing chart are
scripted in `docs/demo.md`, with the saved records named as the
fallback if anything fails live.
