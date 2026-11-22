# MiniServe

A minimal LLM inference serving simulator with **continuous batching**, written in Python.

MiniServe models the core scheduling loop of an LLM serving engine (in the style of vLLM) so that scheduling behavior can be studied in isolation: requests arrive, are batched continuously under batch-size and token-budget limits, and are executed against a deterministic timing model instead of a real model. It is a from-scratch learning project built to understand how modern inference servers work under the hood.

## How it works

Each step of the engine:

1. The **scheduler** picks waiting requests in first-come-first-served order, respecting `max_batch_size` and `max_tokens_per_step`, and moves them through prefill and decode. Admission is validated up front: oversized prompts and duplicate request IDs are rejected before they can stall a queue, and any admitted request can be cancelled while waiting or running.
2. The **executor** simulates execution time (prefill at 0.1 ms/token, decode at 0.5 ms/token) and reports batch timing and throughput.
3. The **engine** drives the loop, updates request state after every step, and collects finished requests with their metrics (TTFT, end-to-end latency, throughput).

On top of the engine sits a **simulation layer**: a deterministic simulated clock and cost model produce a full request timeline for every run, and a **workload** module generates reproducible request streams from a seed. An **experiment pipeline** runs configurations end to end and writes summaries that are independently recomputed from the raw timelines, so a reported number can always be traced back to the run that produced it. All results are simulation results and are labeled as such.

## Project layout

| File | Purpose |
| --- | --- |
| `src/miniserve/request.py` | Request lifecycle (`WAITING → PREFILLING → DECODING → FINISHED / CANCELLED`), token tracking, latency and TTFT metrics |
| `src/miniserve/scheduler.py` | FCFS continuous-batching scheduler with waiting/running queues, per-step token budgets, admission validation, and cancellation |
| `src/miniserve/executor.py` | Simulated executor: prefill/decode timing model and batch throughput helpers |
| `src/miniserve/engine.py` | `MiniServeEngine`: the step loop tying scheduler and executor together |
| `src/miniserve/simulation.py` | Deterministic simulated clock, cost model, and per-request timelines |
| `src/miniserve/workload.py` | Seeded workload generation for reproducible request streams |
| `src/miniserve/experiment.py` | Experiment pipeline: run configurations, collect results, summarize |
| `src/miniserve/metrics.py` | Shared metric definitions and aggregation helpers |
| `benchmarks/demo_scheduler.py` | Demo: runs 3 sample requests to completion and prints TTFT, total time, and throughput |
| `benchmarks/run.py`, `benchmarks/summarize.py` | Experiment runner and independent summary recomputation |
| `tests/` | 90 unit tests covering engine, executor, request, scheduler, validation, cancellation, simulation, workload, and experiment behavior |

## Quickstart

Requires Python 3.11+.

Create a clean virtual environment and install the project with development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the demo and the test suite from the repository root:

```bash
python benchmarks/demo_scheduler.py
python -m pytest -q
```

Verified on 2026-11-09: Python 3.12.3, pytest 9.2.0, 90 passed, demo completed 3 requests in 5 steps. See `docs/devlog.md` for the full record.

## Running the HTTP service

The runtime also serves over HTTP through a single engine owner (see `docs/service.md` for the contract and the frozen list of what is and is not supported). Start it from the repository root:

```bash
python benchmarks/run_server.py --port 8000          # fake backend, no model download
python benchmarks/run_server.py --port 8000 --real   # locked real model, needs the model extra
```

The fake backend tokenizes by word count and returns empty generated text, because it has no tokenizer; it exists so the service shape can be tried in seconds. The `--real` mode loads the locked model once and returns real text.

From another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"prompt": "The capital of France is", "max_new_tokens": 8}'
curl -N -X POST http://127.0.0.1:8000/generate/stream -H 'Content-Type: application/json' -d '{"prompt": "The capital of France is", "max_new_tokens": 8}'
curl -X POST http://127.0.0.1:8000/requests/<request_id>/cancel
```

Stop the server with Ctrl+C: pending requests finish, then the worker exits and no thread is left behind. The integration pass over two normal clients, a cancellation, an invalid request, an overload refusal, a failure, and a real model smoke test is `python benchmarks/service_integration.py`, with its record in `benchmarks/service_integration_output.json`.

## Scope and design choices

- **Simulator, not a server.** There is deliberately no real model, tokenizer, or HTTP API. The executor is a timing model, which makes scheduling behavior deterministic and easy to test.
- **FCFS scheduling** with continuous batching: new requests join the running batch as capacity frees up, instead of waiting for a whole batch to finish.
- **Deterministic by design.** Seeded workloads and a simulated clock make every run exactly reproducible; summaries are recomputed independently from raw output before they are trusted.
- **Metrics first.** Every request tracks time-to-first-token and total latency so scheduling changes can be compared quantitatively.

## Possible future work

- Priority-aware scheduling and preemption
- Chunked prefill
- KV-cache paging simulation
- An adapter for a real backend (e.g. vLLM) behind the same scheduler interface
