# MiniServe

A small LLM inference runtime with continuous batching, built
step by step from a scheduling simulator. MiniServe runs one
locked model on one device, batches the decode step across
requests with different KV cache lengths, lets new requests join
and finished ones leave mid generation, and serves the result
over a small streaming HTTP API with bounded admission and
cancellation.

## What it does

- Schedules requests first come first served under a per step
  token budget and a batch size limit, with decode work chosen
  before new prefill.
- Rejects requests that can never run (a prompt longer than the
  step budget, or prompt plus reserved output beyond the model
  context) at admission instead of letting them wait forever.
- Runs a real model two ways against the same engine: a serial
  baseline that executes one request at a time, and a batched
  executor that packs per request KV caches into a single decode
  forward with left padding, per row masks and logical
  positions, then splits the caches back.
- Admits new requests while others are decoding, removes
  finished, cancelled, and failed requests at step boundaries,
  and releases their caches on every terminal path.
- Serves generation over HTTP through a single engine owner
  thread: streaming token events, client disconnect mapped to
  cancellation, a bounded waiting queue with an explicit queue
  full error, and a per request event buffer that cancels slow
  consumers instead of blocking the engine.
- Measures everything against a frozen protocol and keeps the
  raw records, so every reported number can be recomputed from
  the run that produced it. Simulation results are labelled as
  simulation and never mixed into real model tables.

## How to run it

Requires Python 3.11 or newer. The core and the fake backend
need no model download; the real model is an optional extra.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python benchmarks/demo_scheduler.py
```

The three act showcase (concurrent submissions, a request
joining mid generation, and a cancellation next to an overload
refusal, all on the fake backend) is:

```bash
python benchmarks/demo_showcase.py
```

For the real model, install the model extra and run one of the
real entry points. The model is HuggingFaceTB/SmolLM2-135M at
the revision pinned in `configs/model.json`; the first run
downloads about 260 MiB into the Hugging Face cache.

```bash
python -m pip install -e '.[model]'
python benchmarks/real_generate.py
python benchmarks/service_streaming_demo.py
```

The HTTP service, fake backend by default and real model with
`--real`:

```bash
python benchmarks/run_server.py --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"prompt": "The capital of France is", "max_new_tokens": 8}'
curl -N -X POST http://127.0.0.1:8000/generate/stream -H 'Content-Type: application/json' -d '{"prompt": "The capital of France is", "max_new_tokens": 8}'
curl -X POST http://127.0.0.1:8000/requests/<request_id>/cancel
```

Stop the server with Ctrl+C: pending requests finish, then the
worker exits. The full reproduction checklist, including the
clean environment record and the known environment pitfalls, is
in `docs/reproduction.md`. A timed demo script is in
`docs/demo.md`.

## Architecture

One engine loop drives everything: the scheduler picks work,
the executor runs prefill and decode, and the engine updates
queues and releases terminal requests. The HTTP layer never
touches the scheduler; it translates requests and events for
the single engine owner thread that holds the model.

| Component | Responsibility |
| --- | --- |
| `src/miniserve/scheduler.py` | Waiting and running queues, admission, token budget, cancellation, resource limits, serial baseline policy |
| `src/miniserve/engine.py` | The step loop and terminal cleanup |
| `src/miniserve/executor.py` | The executor protocol and the fake backend |
| `src/miniserve/real_executor.py` | Single request real model execution over the KV cache |
| `src/miniserve/batched_executor.py` | Batched decode: pack, one forward, split |
| `src/miniserve/service.py` | The engine owner, request handles, streaming events, overload and disconnect handling |
| `src/miniserve/server.py` | The minimal HTTP front end and its error mapping |
| `src/miniserve/metrics.py`, `workload.py`, `simulation.py`, `experiment.py` | Per request metrics, deterministic workloads, the labelled simulation pipeline |

The fuller map, with the end to end path of one request, is in
`docs/architecture.md`. Design decisions live in
`docs/design.md`, the cache layout in `docs/batching.md`, the
service contract in `docs/service.md`, and the measurement
rules in `docs/benchmark_protocol.md` and `docs/metrics.md`.

## Results

Measured on a CPU only machine (AMD EPYC 9D25, no GPU) with the
locked model in float32, greedy decoding, fixed length runs of
20 requests, 3 repeats per configuration, 60 finished requests
per continuous point. Full tables, ranges, and limits are in
`docs/results.md`; charts are in `docs/charts/`.

- Short workload: continuous batching at concurrency 8 reached
  17.48 output tokens per second against 5.85 for the serial
  baseline, about 3.0 times, with mean time to first token down
  from 13.25 to 3.50 seconds and mean latency down from 14.40
  to 5.83 seconds.
- Mixed workload: 15.56 against 5.83 tokens per second, about
  2.7 times; latency gains saturated at concurrency 4.
- The trade off is real: time per output token roughly doubled
  at concurrency 8, because one batched forward costs more than
  one single forward. Continuous batching here buys waiting
  time, not cheaper tokens.
- Token budget was mostly not the binding limit at concurrency
  4; the batch size bound first. Packing and splitting caches
  cost about 1 to 8 percent of a decode step at these sizes and
  were not the bottleneck; serial per request prefill is the
  largest structural cost the runtime controls.

These are exploratory figures for this hardware, model, and
workload, not a general claim about serving systems, and no
comparison with a production serving framework is made.

## Limitations

- One model, one device, one process; greedy decoding only; no
  sampling controls, no authentication, no persistence, and no
  OpenAI API compatibility. The HTTP server is the Python
  standard library server and was exercised on loopback only.
- Prefill runs one serial forward per request; there is no
  batched or chunked prefill, no paged cache, and no prefix
  caching.
- Resource limits are logical token counts. They are a
  conservative bound, not a guarantee against running out of
  memory, and padding plus pack and split copies add real
  allocation on top of the logical count.
- Cancellation takes effect at the next step boundary; a
  forward already running is not interrupted.
- Natural end of sequence from this base model under greedy
  decoding was never observed in testing; the EOS path is
  verified with a shared custom EOS id and scripted models.
- Benchmark samples are small (60 finished requests per point)
  on a shared CPU, so wall clock figures carry visible noise
  and are reported with their repeat ranges.

## Repository hygiene

The repository contains no credentials, no model weights, and
no virtual environment; `.gitignore` covers the environment,
caches, and build artifacts, and the model is downloaded on
demand at the pinned revision. Development history, including
every verification result quoted above, is in
`docs/devlog.md`.
