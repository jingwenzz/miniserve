# MiniServe Benchmark Protocol: Real Model Experiments

This document freezes how the real model experiments are run, so
the numbers produced in the experiment stage can be checked
against a protocol that was written before the results. It covers
the core runtime benchmark only. The end to end HTTP benchmark is
a separate measurement with client side timing; its numbers must
never be mixed into a core runtime table. The simulation pipeline
in `docs/experiments.md` is also separate: its logical seconds are
not real times and never appear in a real result table.

## 1. What is fixed

| Item | Frozen value | Source |
| --- | --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | `configs/model.json` |
| Tokenizer | Same repository and revision, no special token added in front | `docs/model.md` |
| Decoding | Greedy, `do_sample=False` everywhere | `configs/model.json` |
| dtype / attention | float32 / eager | `configs/model.json` |
| Device | CPU only, AMD EPYC 9D25, no GPU present, PyTorch reports CUDA unavailable | `docs/model.md` section 1 |
| Software | Python 3.12.3, torch 2.9.1+cpu, transformers 4.57.6 | `docs/model.md` section 3 |
| Seed | 1234, recorded in every run config | `configs/benchmark.json` |
| Scheduler limits | `max_batch_tokens` 8192, `max_model_len` 8192, both strategies | `configs/benchmark.json` |
| Prompt set | The 10 fixed text prompts in `configs/benchmark.json`, cycled in order | `configs/benchmark.json` |
| Arrival trace | All at once: every request is constructed and admitted just before the timed window, so the run is a closed batch, not an arrival rate test | This document |
| Output limit | `max_new_tokens` 8 for the pilot and for the main short workload unless a run config says otherwise | `configs/benchmark.json` |

The machine readable copy of this table is
`configs/benchmark.json`. A run that changes any frozen value must
record the new value in its own run config and say so in its
summary; it must not edit this protocol silently to match.

## 2. The two strategies and what differs

The main comparison is serial against continuous, and it is the
only comparison the headline results may use.

- Serial: `SerialScheduler` (one request active at a time) driving
  `RealExecutor`, which runs one request per forward.
- Continuous: the default `Scheduler` driving
  `BatchedRealExecutor`, which packs per request caches into one
  batched decode forward per step. Prefill stays one serial
  forward per request in both strategies, as the plan allows; no
  fused mixed batch is claimed.

Everything else is shared: the same loaded model object class and
weights, the same tokenizer, the same prompt token ids rebuilt
into fresh `Request` objects per strategy, the same output limit,
the same scheduler limits above, and the same stopping rule
(section 3). Because the executors differ, a result is described
as a strategy result (scheduler plus its executor), never as a
pure scheduler result.

## 3. Fixed length runs and natural EOS runs are separate

Two stopping modes exist and are never mixed in one table.

- Fixed length mode (the pilot and the main performance runs):
  the executor EOS id is set to -1, a value no generated token
  can equal, so no request can stop early on EOS and every
  request generates exactly `max_new_tokens` tokens. This is the
  documented way this project closes off EOS for fixed length
  measurement, as the plan requires the method to be stated, not
  hidden. Work is then identical across strategies by
  construction, and any remaining token mismatch is a correctness
  failure, not a stopping difference.
- Natural EOS mode (correctness and behaviour runs only): the
  executor uses the model EOS id 0. A request that stops early
  records its actual output length, and a comparison must report
  the differing lengths instead of comparing timing as if the
  work were equal. Natural EOS has in fact never been observed
  from this base model under greedy decoding (see
  `docs/generation.md`), which is one reason the performance runs
  do not rely on it.

## 4. Warmup and model loading are excluded

The model is loaded once per benchmark process, before any timed
window. Before each strategy's timed run, that strategy runs 2
warmup requests (the first two protocol prompts, output limit 4,
fixed length mode) whose results are discarded and never counted.
A timed window therefore contains no model loading, no first ever
forward, and no warmup request. Warmup counts and the load time
are recorded in the run summary as observations, not as results.

## 5. Measurement boundaries

Core runtime benchmark, per strategy:

- Window: `time.perf_counter()` taken just before
  `Engine.run_until_complete()` and just after it returns, with
  every request already admitted. The window is wall clock time
  for the whole closed batch.
- Per request times: the `Request` timestamps, all on
  `time.monotonic()`. Arrival is the construction time, which the
  runner sets just before admission, so queue time, TTFT,
  latency, and TPOT follow `docs/metrics.md` unchanged. TTFT ends
  at the first token, which for this backend is prefill
  completion.
- Timing method: this machine is CPU only and PyTorch CPU
  execution is synchronous, so a wall clock window measures
  completed work and needs no synchronisation call. There are no
  CUDA events to use and no GPU asynchronous timing error to
  correct. The runner must not insert per token synchronisation
  or timing calls inside the window, because on this backend
  there is nothing to synchronise and extra calls would only
  perturb the workload. If these experiments are ever repeated
  on a GPU, this section must be rewritten first: end to end
  numbers would then be taken from observable outputs, and any
  microbenchmark would need CUDA events or an explicit
  synchronisation at the window edge only.
- Derived figures: output throughput is total generated tokens
  divided by the window; request throughput is finished requests
  divided by the window; means follow the frozen denominators in
  `docs/metrics.md` (TTFT over requests that produced a token,
  latency over terminal requests, TPOT over requests with at
  least two tokens). Percentiles are not reported at pilot scale.

## 6. Workload verification, run layout, and repeats

Before any timing number from a run may be quoted, the run must
pass the workload check: both strategies report the same number
of finished requests, the same total prompt tokens, the same
total output tokens, and, in fixed length mode, token for token
identical generated sequences per request id. A run that fails
the check is a correctness record, not a performance result.

Each run writes a fresh directory under `benchmarks/results`
named by its run id and refuses to overwrite an existing one:

```text
benchmarks/results/<run_id>/
    config.json              the frozen protocol values as applied
    serial_requests.jsonl    raw per request records, serial
    continuous_requests.jsonl
    summary.json             counts, forwards, windows, means
    summary.md               the same summary as text
```

Raw per request records carry the metrics fields plus the
generated token ids, so the workload check and every mean can be
recomputed from the raw files alone.

Repeats: the pilot is a single run per strategy and its numbers
are exploratory. The main experiment runs each configuration at
least 3 times, rotates the strategy order between repeats, and
reports the range across repeats next to any mean. Sample sizes
stay in every table; a small sample is labelled exploratory and
p95 or p99 are not headline figures at this scale.

## 7. Pilot definition

The pilot is the first run under this protocol: 20 requests per
strategy (the 10 protocol prompts cycled twice), output limit 8,
fixed length mode, continuous batch size 4, serial first. Its job
is to expose timing, EOS, queue, and configuration errors before
the main experiment, not to produce a headline comparison. Its
raw data, its summary, and the list of problems it found are
recorded with the run; the pilot record is
`benchmarks/results/real-pilot-seed1234`, produced by
`benchmarks/pilot_benchmark.py`.
