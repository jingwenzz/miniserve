# MiniServe Resume and Interview Notes

Every bullet below points at the code or the saved record that
supports it. Numbers carry their measurement conditions; nothing
here claims a result that was not measured on this machine, with
this model, under the frozen protocol in
`docs/benchmark_protocol.md`.

## Resume bullets

### Function bullet (core runtime)

Implemented a single device LLM inference runtime with
continuous batching, token budget scheduling, and per request
KV cache lifecycle management in Python: the batched executor
packs per request caches into one left padded decode forward
with per row masks and logical positions, then splits the
caches back, and new requests join and finished ones leave the
batch at step boundaries.

Evidence: `src/miniserve/batched_executor.py`,
`src/miniserve/scheduler.py`, `docs/batching.md`, and the six
scenario acceptance record in
`benchmarks/fourth_week_acceptance_output.json` (equal length,
different lengths, exit, join, cancel, and injected failure all
pass, with every token sequence matching the same request run
on its own).

### Service bullet (streaming API)

Built a streaming inference API on top of the runtime with a
single engine owner thread, bounded admission, and request
cancellation: clients receive token events over HTTP with
server sent events, a full waiting queue is refused with an
explicit queue full error instead of waiting silently, client
disconnect maps to cancellation at the next step boundary, and
a slow consumer is cancelled instead of blocking the engine.

Evidence: `src/miniserve/service.py`,
`src/miniserve/server.py`, `src/miniserve/streaming.py`, the
contract in `docs/service.md`, and the integration record in
`benchmarks/service_integration_output.json`.

### Experiment bullet (measured, with conditions)

Benchmarked serial and continuous batching execution on
SmolLM2-135M (CPU only, AMD EPYC 9D25, float32, greedy,
fixed length runs of 20 requests, 3 repeats, 60 finished
requests per continuous point): continuous batching at
concurrency 8 reached about 3.0 times the serial output
throughput on a short workload (17.48 against 5.85 output
tokens per second) and about 2.7 times on a mixed length
workload, with mean time to first token and mean latency down
at the same time, at the cost of roughly double the time per
output token.

Evidence: `benchmarks/main_benchmark.py`, the raw records and
summary under `benchmarks/results/real-main-seed1234/`, the
tables and limits in `docs/results.md`, and the bottleneck
analysis in `benchmarks/bottleneck_output.json`, which found
that cache pack and split cost about 1 to 8 percent of a decode
step at these sizes and that serial per request prefill is the
largest structural cost the runtime controls.

Wording limits kept on purpose: no production grade claim, no
comparison with a production serving framework, and no
percentage without its hardware, model, workload, and sample
size next to it. The figures are exploratory at this sample
size and are stated that way wherever they are quoted.

## 30 second overview

I built a small LLM inference runtime that focuses on how
requests are scheduled while they generate token by token. It
started as a scheduling simulator, then gained a real model
backend, and now batches the decode step across requests with
different KV cache lengths, lets requests join and leave mid
generation, and serves streaming results over HTTP with bounded
admission and cancellation. I compared one at a time execution
with continuous batching on the same model and workload: on
this CPU machine throughput rose to about three times the
serial baseline at concurrency 8, while the time per output
token roughly doubled, because one batched forward costs more
than one single forward. The main trade off is that batching
buys waiting time, not cheaper tokens, and prefill is still one
serial forward per request.

## 3 minute technical walkthrough

Start with the request lifecycle. A request is admitted only
if it can ever run: a prompt longer than the per step token
budget is rejected at admission, and so is a prompt whose
length plus its reserved output exceeds the model context.
Admitted requests wait in a FIFO queue. Each engine step, the
scheduler picks decode work for running requests first, then
new prefill while the batch size, the token budget, and the
resident reservations allow. Cancellation, whether explicit,
from a client disconnect, or from the slow consumer rule, takes
effect at a step boundary and releases the request cache.

Prefill and decode are different operations. Prefill runs the
whole prompt once, and its last position logits produce the
first token, so time to first token is prefill completion.
Decode then feeds only the most recent generated token, because
that is the only token the cache has not seen, so the cache
trails the produced tokens by one position and a run of N
tokens costs one prefill forward and N minus 1 decode forwards
per request. The hand written loop was verified against the
reference generate path on 12 cases, token for token, including
a 121 token prompt and a shared custom EOS case, because the
base model never produced its natural EOS under greedy decoding
in any test.

Batching keeps one independent KV cache per request as the
source of truth. For one decode step, the executor left pads
the shorter caches to the longest valid length in the batch,
builds a mask that zeroes the padded positions, passes one
shared physical cache position for the append slot and a per
row logical position for the rotary embeddings, runs a single
forward whose batch dimension is the number of live requests,
and splits each row back to its own cache from its trailing
slice. Rows are rebuilt from the surviving requests every step,
so a finished request simply stops appearing, and a new request
is prefilled at a step boundary and joins the decode batch on
the next step. The acceptance suite proves the tokens of a
batched run match the same requests run one at a time for equal
lengths, different lengths, exit, join, and cancel, and that a
batched forward failure marks the whole batch failed and still
cleans every cache.

The service layer never touches the scheduler. One owner thread
holds the engine and the model, callers send submit, cancel,
disconnect, and shutdown commands through a queue, and the HTTP
front end only validates, translates errors, and writes server
sent events. Streaming text is decoded incrementally from the
whole token sequence so far, never by concatenating per token
decodings, and a trailing replacement character is held back
until later tokens complete it or the request ends. The waiting
queue is bounded and refuses with a typed error; the per
request event buffer is bounded and a reader that falls behind
loses token events, is marked a slow consumer, and is cancelled
at the same boundary, while its terminal event is still
delivered.

Measurement was set up before the results: one frozen protocol,
the same model, tokenizer, greedy rule, and output length for
both strategies, model loading and warmup outside the timed
window, and raw records that a separate script can recompute.
The serial baseline is the same engine and scheduler family
with a one at a time selection policy, which is what makes the
comparison fair and also what limits it: it shows what this
runtime own serial execution costs, not how it compares with a
production serving system. The budget experiment showed the
token budget mostly did not bind at concurrency 4, because the
batch size bound first; the one visible effect was a long
prefill deferred by one step at the smallest budget.

## Interview questions, with the short answer for each

1. What do prefill and decode each compute, and where does the
   first token come from? Prefill scores the whole prompt in
   one forward and its last position logits are the first
   token. Decode then runs one new token at a time against the
   cached context. Evidence: `docs/generation.md`,
   `benchmarks/prefill_output.json`.

2. How is continuous batching different from a fixed batch? A
   fixed batch waits for its slowest member and admits new work
   only between batches. Here the row set is rebuilt every
   step: finished rows leave, a newly prefilled request joins
   the next decode step, and the batch size changes while other
   requests are mid generation. Evidence: the join and exit
   scenarios in `benchmarks/fourth_week_acceptance_output.json`.

3. Why is a token budget not a memory budget? The budget caps
   the work picked in one step, in logical tokens. Real memory
   also pays for padding to the longest row, for the pack and
   split copies made every step, and for activations and logits
   that no token count covers. The runtime therefore treats its
   token reservation as a conservative bound and says so in
   `docs/resources.md` instead of promising it can never run
   out of memory.

4. How do caches of different lengths form one batch, and why
   must padding not affect real positions? Shorter caches are
   left padded, the mask zeroes padded past positions, the
   append slot is one shared physical position, and each row
   carries its own logical position in position_ids for the
   rotary embeddings. Omitting position_ids would make every
   row use the longest length as its position. Layout and the
   hand worked 7 and 3 example: `docs/batching.md`.

5. What happens to batch rows when a request exits? The next
   step rebuilds the row mapping from the surviving requests in
   decode list order, the exited request cache is released by
   the engine after the queue update, and the remaining rows
   keep producing the tokens they produce on their own, which
   the exit scenario checks token for token.

6. Why does cancellation take effect at a step boundary? The
   scheduler owns the queues and the executor owns a forward
   already running; interrupting mid forward would leave cache
   and state half updated. The cancel is recorded immediately
   and applied at the next selection point, the request produces
   no further tokens after that point, and its cache is
   released on the terminal path. A forward already running is
   not preempted, and the docs say so.

7. On what workload could this implementation be slower? Small
   batches on a CPU, where one batched forward costs more than
   a single forward and the waiting time saved is small, show up
   as higher time per output token; at concurrency 1 continuous
   and serial are within noise of each other. Long prompts also
   dilute the gain because prefill is serial in both strategies
   and padding waste on the mixed workload reaches about 37 to
   39 percent of packed slots.

8. Why is the baseline comparison fair, and why does it not
   prove anything about production serving systems? Both
   strategies share the model, tokenizer, greedy rule, output
   length, engine, admission, and stopping rules, and differ
   only in selection policy and the executor that policy needs;
   the workload check requires identical finished counts and
   token sequences before a run may be quoted. The baseline is
   this runtime own serial path, so the result is a statement
   about scheduling inside this codebase, not about any external
   system, and no external comparison is claimed.

9. With one more week, what would the evidence say to do next?
   Attack serial prefill, the largest structural cost the
   runtime controls: 20 requests pay 20 prefill forwards before
   decode batching helps at all. The next step would be a
   segmented timing pass over prefill under the same protocol,
   then a batched or chunked prefill prototype measured against
   the current records, keeping the workload check as the gate.

## One hard case, one wrong assumption, one evidenced trade off

Hard case: building a batched cache. Hand assigning per layer
tensors to a cache object leaves its layers uninitialized and
the first probe failed that way; the working path builds the
batched cache through the cache constructor with per layer key
and value pairs. Recorded in `docs/batching.md`.

Wrong assumption: that rewriting cache pack to one zero filled
allocation per layer would speed it up. The packed values were
verified identical, the segmented medians after the change were
within noise and slightly higher in most configurations, and
the change was reverted. The shipped pack code is the code that
produced the main experiment. Record:
`benchmarks/bottleneck_pack_attempt.json`.

Evidenced trade off: batching buys waiting time, not cheaper
tokens. Throughput, time to first token, and latency improve
together up to concurrency 8 on the short workload, while time
per output token roughly doubles, and on the mixed workload
latency stops improving after concurrency 4 while time per
output token keeps rising. All three effects are in the same
tables in `docs/results.md`.
