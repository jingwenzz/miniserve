# MiniServe Experiments: Run Pipeline and Summaries

This document describes how a simulation experiment is run, what
it writes, and how its summary can be checked independently. All
experiments at this stage are simulations in logical time (see
`docs/simulation.md`); nothing here is a measured model or GPU
result.

## 1. One command runs an experiment

From the repository root:

```bash
python benchmarks/run.py --profile short --seed 1234 \
    --num-requests 6 --run-id simulation-short-seed1234
```

The command generates the deterministic workload for the profile
and seed, replays it once per policy (serial and continuous by
default), and writes one run directory under `benchmarks/results`.
The configuration carries the profile, seed, request count,
policy list, scheduler limits, the simulation cost parameters,
and the package version, so the run can be repeated exactly.

## 2. What a run directory contains

```text
benchmarks/results/<run_id>/
    config.json                    full configuration and workload shape
    serial_requests.jsonl          raw per request metrics, serial policy
    serial_timeline.jsonl          raw per step timeline, serial policy
    continuous_requests.jsonl      raw per request metrics, continuous policy
    continuous_timeline.jsonl      raw per step timeline, continuous policy
    summary.json                   summary recomputed from the raw files
    summary.md                     the same summary as a table
```

Rules:

- Every run has a run id. A generated id carries the start time,
  the profile, and the seed; an explicit id may be passed with
  `--run-id` for a reproducible example.
- A run directory is never reused. Asking for a run id that
  already exists raises FileExistsError and leaves the earlier
  run untouched, so a rerun always leaves traces of both runs
  instead of silently overwriting the first one.

## 3. The summary is an independent recomputation

```bash
python benchmarks/summarize.py benchmarks/results/<run_id>
```

The summarize entry point reads only `config.json` and the raw
request and timeline files in the run directory. It never sees
the request objects or the engine that produced them. Recomputing
the summary of an unchanged run directory produces exactly the
summary the original run wrote, which the tests in
`tests/test_experiment.py` assert. If a raw file is edited, the
recomputed summary changes to match the edited data; that is the
point of keeping the raw files as the source of truth.

Per policy, the summary reports the request count split into
finished, cancelled, and unfinished, the total prompt and output
tokens, the step count, the total simulated duration, the mean
time to first token, latency, and time per output token over the
requests that have those values, and a simulated output tokens
per second figure that is only the total output tokens divided
by the simulated duration under the recorded cost model.

## 4. One recorded simulation comparison

Run id `simulation-short-seed1234`, profile short, seed 1234,
six requests, batch size 4, token budget 64, costs 0.1 logical
seconds per prefill token and 0.5 per decode token. The full raw
files are in `benchmarks/results/simulation-short-seed1234`.

| Policy | Finished | Output tokens | Steps | Simulated duration | Mean TTFT | Mean latency | Mean TPOT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| serial | 6 | 30 | 36 | 21.3000 | 10.3167 | 12.3167 | 0.5000 |
| continuous | 6 | 30 | 12 | 21.3000 | 9.2167 | 15.8167 | 1.6491 |

Both policies generated exactly the same work from the same
workload, which is the fairness check that matters at this stage.
The continuous policy needed a third of the steps, showed the
lower mean time to first token, and still showed the higher mean
latency, because under the serial cost model the requests share
each step's work instead of overlapping it. The equal durations
are the total work under the cost model, not a coincidence and
not a performance claim. Reading the numbers any further than
that would require a real backend, which this pipeline is built
to accept later without changing its file layout.
