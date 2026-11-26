# MiniServe Demo Script

A two to three minute demo that can be repeated from a clean
checkout without changing any parameter live. Each act names the
command, what to point at while it runs, and the one sentence
that explains it.

## Before starting

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Say: MiniServe is a small LLM inference runtime. It started as a
scheduling simulator and now runs a real model with continuous
batching, a streaming HTTP service, and a reproducible benchmark
suite. Everything in this demo is labelled: the service acts use
the fake backend so the ordering is visible, and the chart at
the end is from the real model runs.

## Act 1 and 2: concurrency and dynamic join (about 60 seconds)

```bash
python benchmarks/demo_showcase.py
```

Point at the PASS lines as they appear. The second request is
submitted, and its submission returns, while the first request
is still generating, because one engine owner thread holds the
model and every caller talks to it through a command queue. A
third request then joins while a long request is mid generation:
it is admitted at a step boundary, prefilled, and added to the
decode batch on the next step, so it finishes without waiting
for the long request to end.

## Act 3: cancel and overload (about 40 seconds)

The same script continues with a waiting limit of one. A third
submission is refused immediately with the queue full error
instead of waiting silently, the waiting request is cancelled
and produces no further tokens, and the running request still
completes all of its tokens.

Say: overload and cancellation are part of the contract, not
accidents. A refused request consumes no id and can be retried,
and a cancelled request releases its cache at the next step
boundary.

## Real model streaming (about 40 seconds)

```bash
python -m pip install -e '.[model]'
python benchmarks/service_streaming_demo.py
python benchmarks/run_server.py --port 8000 --real
```

In a second terminal:

```bash
curl -N -X POST http://127.0.0.1:8000/generate/stream -H 'Content-Type: application/json' -d '{"prompt": "The capital of France is", "max_new_tokens": 8}'
```

Point at the token events arriving before the single finished
event. The same model is loaded once by the server; the batched
executor packs per request KV caches into one decode forward
and splits them back afterwards.

## Closing chart (about 20 seconds)

Open `docs/charts/throughput.svg` (source data and method in
`docs/results.md`). On this CPU only machine, continuous
batching at concurrency 8 reached about 3.0 times the serial
output throughput on the short workload and about 2.7 times on
the mixed workload, with lower time to first token and latency,
at the cost of roughly double the time per output token. The
sample is 60 finished requests per point over 3 repeats, so the
figures are exploratory and tied to this hardware and workload.

## If something fails live

Fall back to the saved records, which were produced by the same
commands: `benchmarks/demo_showcase_output.json`,
`benchmarks/service_streaming_output.json`, and
`benchmarks/results/real-main-seed1234/summary.json`. Do not
change parameters to make a run pass; the records state what
the shipped code actually did.
