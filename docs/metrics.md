# MiniServe Metrics: Definitions and Timing Contract

This document freezes the timing definitions that every later
experiment must use. It describes engine side time only. A future
HTTP layer will have its own client side times, and the two must
never be mixed in one number without saying which side produced it.

All times in this document and in `src/miniserve/metrics.py` are
seconds, read from one clock per request. By default that clock is
`time.monotonic`. Tests inject a fake clock, so no metric test needs
to sleep or depend on real elapsed time.

## 1. Events and timestamps

Each request carries these raw timestamps. Every derived metric can
be recomputed from them, so a summary never has to guess.

| Field | Meaning | Set by |
| --- | --- | --- |
| `arrival_time` | The engine receives the request. In the current in process code this is the moment the `Request` object is created, using its injected clock. | `Request` construction |
| `admission_time` | The scheduler accepts the request into the waiting queue. A request rejected at admission never gets this time. | `Scheduler.add_request`, through `Request.mark_admitted` |
| `prefill_start_time` | The request starts its first model execution, meaning prefill begins. This is the end of queue waiting as defined below. | `Request.start_prefill` |
| `first_token_time` | The first generated token becomes available for this request. With the fake executor this is the first decode append, because its prefill produces no token. A real backend that produces the first token from prefill logits must record that moment here instead, and that change of source must be documented when it happens. | `Request.append_token`, first call |
| `last_token_time` | The most recent generated token becomes available. For a one token request it equals `first_token_time`. | `Request.append_token`, every call |
| `finished_time` | The request reaches a terminal state, `FINISHED` or `CANCELLED`. For a request that finishes by reaching its output limit, this is recorded in the same append step as the last token. | `Request.finish` or `Request.cancel` |

Rules:

- A timestamp that has not happened yet stays `None`. It is never
  silently filled with zero and never copied from another event.
- One request uses one clock for all of its timestamps. Mixing
  clocks would make differences meaningless.
- Timestamps are raw evidence. Derived values are computed in
  `metrics.py` from these fields and from nothing else.

## 2. Derived metrics

| Metric | Formula | When it is `None` |
| --- | --- | --- |
| Queue time | `prefill_start_time - arrival_time` | Prefill has not started yet |
| Time to first token, TTFT | `first_token_time - arrival_time` | No token has been produced yet |
| End to end latency | `finished_time - arrival_time` | The request is not in a terminal state yet |
| Time per output token, TPOT | `(last_token_time - first_token_time) / (output_length - 1)` | Fewer than two tokens were produced, so there is no interval to average. A one token request reports `None`, never zero and never a division by zero. |
| Output length | `len(generated_token_ids)` at export time | Never `None`. It is a count, not a time, and zero is a real count for a request that has produced nothing. |
| Prompt length | `len(prompt_token_ids)` | Never `None` |

Throughput is a property of a run, not of one request, so it is not
part of the per request record:

- Output throughput for a run is the total output tokens counted in
  a stated measurement window divided by that window in seconds.
  The window, and whether tokens from cancelled requests count,
  must be stated with any reported number.
- Request throughput for a run is the number of requests that
  reached a terminal state in the window divided by the window.
  Rejected, failed, and cancelled counts are reported alongside it,
  never hidden inside it.

Percentiles such as p50 and p95 are computed later, over a stated
sample of per request values, by the experiment pipeline. This
document only fixes the per request values they are computed from.

## 3. How terminal and unfinished requests are counted

- `FINISHED` requests carry `finish_reason` equal to `finished`.
  They contribute TTFT, latency, and, when they have at least two
  tokens, TPOT.
- `CANCELLED` requests carry `finish_reason` equal to `cancelled`.
  They are counted separately from finished requests in every
  summary. A cancelled request keeps the tokens and timestamps it
  already earned, so its TTFT is reported if it produced a token,
  and its latency is the time until cancellation. A request
  cancelled while waiting produced no token, so its TTFT and TPOT
  are `None`.
- Requests that are still waiting or decoding at export time carry
  `finish_reason` equal to `None` and latency equal to `None`.
  They are not counted as finished or cancelled, and their missing
  values are exported as `null` in JSON, not as zero.
- There is still no `FAILED` state at this checkpoint, for the same
  reason recorded in `docs/design.md`. When a real executor adds a
  failure path, failed requests must get their own reason and their
  own count, distinct from both finished and cancelled.

## 4. Engine time and client time

- Engine TTFT and engine latency use the timestamps above, all read
  inside the engine process from the request clock.
- Client first content latency, for a future HTTP layer, is the
  time from the client sending a request to the client receiving
  the first non empty text chunk. A text chunk is not guaranteed to
  be exactly one token, so this number is not engine TTFT and must
  be reported under its own name.
- No result may average engine times and client times together.

## 5. Hand worked example with three tokens

A fake clock drives one request through these events:

| Event | Clock reading |
| --- | --- |
| Arrival, request created | 100.0 |
| Admission into the waiting queue | 101.0 |
| Prefill starts | 103.0 |
| First token | 105.0 |
| Second token | 107.0 |
| Third and last token, request finishes | 109.0 |

Hand calculation:

- Queue time is 103.0 minus 100.0, which is 3.0 seconds.
- TTFT is 105.0 minus 100.0, which is 5.0 seconds.
- End to end latency is 109.0 minus 100.0, which is 9.0 seconds.
- TPOT is (109.0 minus 105.0) divided by (3 minus 1), which is
  2.0 seconds per token.

The test `test_hand_worked_three_token_example` in
`tests/test_metrics.py` builds exactly this timeline with a fake
clock and asserts that the program produces exactly these values.

Single token check: if the same request produced only its first
token at 105.0 and finished there, its TTFT would be 5.0 seconds,
its latency would be 5.0 seconds, and its TPOT would be `None`,
because there is no interval between two tokens to average.

## 6. Simulation warning

The fake executor appends a fixed token per decode step and has no
real model cost. Times measured with a real wall clock around the
fake executor describe the test harness, not a model. Only the
logical relationships in this document, queue time, ordering of
events, and which values are missing, carry over to a real backend.
No number produced with the fake executor may be reported as model
throughput or model latency.

## 7. Frozen review: units, denominators, and counting rules

This section was added at the second week review checkpoint and
freezes the rules below for every later experiment, simulation or
real. Changing any of them requires updating this section and
saying so in the devlog.

Units:

- Every per request time value is in seconds: queue time, TTFT,
  latency, and TPOT (seconds per token). In simulation runs the
  seconds are logical seconds under the recorded cost model; in
  later real runs they are seconds from the request clock. The
  unit is the same, the clock is not, and the clock must be named
  with any quoted number.
- Throughput values are tokens per second or requests per second,
  over a stated window, never a bare count.

Denominators for the means in the experiment summary:

| Mean | Numerator | Denominator |
| --- | --- | --- |
| Mean TTFT | Sum of TTFT over requests that produced at least one token | The number of those requests only |
| Mean latency | Sum of latency over terminal requests (finished plus cancelled) | The number of terminal requests only |
| Mean TPOT | Sum of TPOT over requests that produced at least two tokens | The number of those requests only |

A request with a missing value never enters either side of a
mean. It is not counted as zero and it does not enlarge the
denominator, so a mean can always be read as the average over
exactly the requests that earned that value.

Counting rules:

- Finished, cancelled, and unfinished are three separate counts
  in every summary, and they add up to the number of exported
  request records.
- A request rejected at admission produces no metrics record at
  all, because it was never admitted. Rejection counts, when they
  matter, must be counted at the admission call site and reported
  alongside the summary, never inferred from the records.
- The current simulated output throughput numerator is the total
  output tokens of all exported records, including tokens a
  cancelled request produced before it was cancelled. In the
  recorded runs every request finished, so this nuance changes
  nothing there; any future run that contains cancelled requests
  must state whether its throughput numerator includes their
  tokens.
- There is still no failed count, because there is still no
  FAILED state. When a real executor adds one, failed requests
  get their own count and their own denominator treatment here
  before any real experiment is reported.
- Percentiles are not computed anywhere yet. When they are added,
  the sample, the percentile method, and the sample size must be
  stated, and small samples must be labelled exploratory.
