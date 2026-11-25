# MiniServe Architecture

This document describes the system as it stands at the results
checkpoint: what each part owns, how a request flows through it,
and where each contract is written down. It describes only what
is implemented and verified; the frozen service contract in
`docs/service.md` lists what is deliberately not supported.

## Component map

```text
HTTP clients
    |
    v
server.py  ServiceFrontEnd + HTTP handlers (stdlib only)
    |  validates bodies, assigns request ids, translates errors,
    |  writes SSE frames, maps a broken socket to disconnect
    v
service.py  EngineOwner (single worker thread)
    |  command queue: submit / cancel / disconnect / shutdown
    |  waiting bound, per handle event buffer, admission statistics
    |  streaming: one token event per new token, one terminal event
    v
engine.py  Engine
    |  one step: schedule, then prefill, then decode, then update,
    |  then release every tracked request that reached a terminal
    |  state; a failed step is cleaned up the same way before the
    |  exception propagates
    +----------------------+-----------------------+
    v                      v                       v
scheduler.py           executor.py             metrics.py
Scheduler /            Executor protocol       per request records
SerialScheduler        FakeExecutor            and summaries
waiting / running      RealExecutor            (docs/metrics.md)
queues, admission,     (one request per
token budget,          forward)
resident limits        BatchedRealExecutor
(docs/resources.md)    (batched decode,
                       docs/batching.md)
```

## Who owns what

- The HTTP layer owns nothing but translation. Handler threads
  never touch the scheduler, the engine, or an executor; every
  generation goes through the engine owner, so model execution
  happens on exactly one thread.
- The engine owner owns the engine. Callers hand it commands
  through one thread safe queue and are answered per command;
  submission returns once admission is decided, not when
  generation finishes. While requests are pending, the worker
  drains queued commands between steps, which is why cancellation
  and disconnect take effect at a scheduling boundary and never
  interrupt a forward already running.
- The scheduler owns the queues and the selection policy:
  decode before prefill, FIFO waiting order, the per step token
  budget, the batch size cap, and the resident reservation limits.
  The serial baseline is the same scheduler family with a one at
  a time selection policy, which is what makes the comparison in
  `docs/results.md` a strategy comparison rather than two
  different systems.
- An executor owns request state transitions during execution
  and its per request resources. The real executors store one
  independent KV cache per active request. The batched executor
  keeps those per request caches as the source of truth and, for
  each decode step, packs copies into one left padded batched
  cache, runs a single forward, and splits the result back, so a
  step's batch rows are rebuilt from the surviving requests every
  time and a finished, cancelled, or failed request simply stops
  appearing.
- The engine owns release timing: after each step's queue update
  it releases every tracked request that reached a terminal
  state, at most once per request, with one final sweep at the
  end of a run. Release drops the executor's cache reference for
  that request; it is not a framework allocator call.

## One request, end to end

1. A client posts a prompt. The front end validates the body,
   tokenizes the prompt, assigns a request id, and submits a
   command to the owner.
2. On the worker thread, admission runs: shape and context
   checks first, then the waiting queue bound. A refusal is a
   typed error (`request_rejected` or `queue_full`) and consumes
   neither a queue slot nor the request id.
3. The scheduler selects work per step under the token budget:
   running requests decode first, then waiting requests prefill
   in FIFO order while slots, budget, and resident reservations
   allow.
4. Prefill runs one serial forward per request and produces the
   first token from its logits, so time to first token is
   prefill completion. Decode then feeds one token per request
   per step; in the batched executor those rows share one
   forward per step.
5. After every step the owner diffs each active request's token
   list and queues one streaming event per new token, so a
   streaming client sees its first text while generation is
   still running. Text is decoded incrementally from the whole
   token sequence so far, never by concatenating per token
   decodings.
6. The request ends in exactly one terminal state: FINISHED at
   its output limit or on EOS, CANCELLED through cancel or
   disconnect or the slow consumer rule, or FAILED when its
   forward raised, with the reason recorded. Exactly one
   terminal streaming event is sent, the cache is released, and
   the queues and reservations no longer count the request.

## Where the contracts live

| Question | Document |
| --- | --- |
| Request lifecycle, queue invariants, cancellation, executor contract | `docs/design.md` |
| Cache packing layout, mask and position rules | `docs/batching.md` |
| Metric definitions and frozen denominators | `docs/metrics.md` |
| Resident limits and the token versus memory distinction | `docs/resources.md` |
| HTTP and streaming contract, error table, supported list | `docs/service.md` |
| Experiment protocol (fixed model, workloads, windows) | `docs/benchmark_protocol.md` |
| Measured results and their limits | `docs/results.md` |
