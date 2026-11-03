# MiniServe

A minimal LLM inference serving simulator with **continuous batching**, written in Python.

MiniServe models the core scheduling loop of an LLM serving engine (in the style of vLLM) so that scheduling behavior can be studied in isolation: requests arrive, are batched continuously under batch-size and token-budget limits, and are executed against a deterministic timing model instead of a real model. It is a from-scratch learning project built to understand how modern inference servers work under the hood.

## How it works

Each step of the engine:

1. The **scheduler** picks waiting requests in first-come-first-served order, respecting `max_batch_size` and `max_tokens_per_step`, and moves them through prefill and decode.
2. The **executor** simulates execution time (prefill at 0.1 ms/token, decode at 0.5 ms/token) and reports batch timing and throughput.
3. The **engine** drives the loop, updates request state after every step, and collects finished requests with their metrics (TTFT, end-to-end latency, throughput).

## Project layout

| File | Purpose |
| --- | --- |
| `src/miniserve/request.py` | Request lifecycle (`WAITING → PREFILLING → DECODING → FINISHED / CANCELLED`), token tracking, latency and TTFT metrics |
| `src/miniserve/scheduler.py` | FCFS continuous-batching scheduler with waiting/running queues and per-step token budgets |
| `src/miniserve/executor.py` | Simulated executor: prefill/decode timing model and batch throughput helpers |
| `src/miniserve/engine.py` | `MiniServeEngine`: the step loop tying scheduler and executor together |
| `benchmarks/demo_scheduler.py` | Demo: runs 3 sample requests to completion and prints TTFT, total time, and throughput |
| `tests/` | 17 unit tests covering engine, executor, request, and scheduler behavior |

## Quickstart

Requires Python 3.11+.

```bash
pip install -e .
python benchmarks/demo_scheduler.py
```

Run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

## Scope and design choices

- **Simulator, not a server.** There is deliberately no real model, tokenizer, or HTTP API. The executor is a timing model, which makes scheduling behavior deterministic and easy to test.
- **FCFS scheduling** with continuous batching: new requests join the running batch as capacity frees up, instead of waiting for a whole batch to finish.
- **Metrics first.** Every request tracks time-to-first-token and total latency so scheduling changes can be compared quantitatively.

## Possible future work

- Priority-aware scheduling and preemption
- Chunked prefill
- KV-cache paging simulation
- An adapter for a real backend (e.g. vLLM) behind the same scheduler interface
