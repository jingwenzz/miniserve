# MiniServe Final Acceptance

The final check of the shipped version against the plan
acceptance list, run on the frozen code. Each item states what
was verified, and where the record lives. Items that are only
partly true say so.

## Completion tier: C

- Tier A (sound simulator, real single request inference,
  defined metrics, reproducible experiments, README): met.
- Tier B (tier A plus real batched decode, dynamic join and
  exit, cache correctness, and a real performance comparison):
  met. The batched executor shares one decode forward across
  rows, requests join and leave mid generation, and the six
  scenario acceptance record passes on the real locked model.
- Tier C (tier B plus a streaming HTTP and SSE interface, a
  bounded queue, client cancellation, and demo and resume
  material): met. The HTTP layer, the streaming events, the
  bounded waiting queue and event buffer, disconnect mapped to
  cancellation, the demo script in `docs/demo.md`, and the
  resume and interview notes in `docs/resume_notes.md` are all
  part of the shipped version.

## Final verification run

Run on the frozen code in the development environment, and
matching the clean environment record in
`docs/reproduction.md`:

- Full test suite: 188 passed, exit code 0.
- Type check: mypy reports no issues in 14 source files.
- Scheduling demo: 5 steps, 3 requests finished with outputs
  4, 2, and 3.
- Showcase demo: all 7 checks pass, `all_checks_passed` true.
- Real model acceptance: the six scenarios (equal length,
  different lengths, exit, join, cancel, injected failure) all
  pass on the locked model, every token sequence matching the
  same requests run one at a time, every cache cleaned and
  every request released.
- Correctness: the hand written generation loop matches the
  reference generate path on 12 of 12 cases with no first
  divergence.
- Real single request run: finished, 1 prefill forward, the
  request released and no active cache left.

## Correctness and resources

- [x] Overlong, invalid, and duplicate requests have defined
  behaviour and never wait forever: admission rejects a prompt
  longer than the step budget and a prompt plus reserved output
  beyond the context limit, construction rejects an empty prompt
  and a non positive output limit, and duplicate ids are
  rejected for the lifetime of one scheduler.
- [x] Waiting and running cancellation produces no new tokens
  after the next step boundary and does not disturb other
  requests, in the scheduler tests, the disconnect tests, and
  the real model cancel scenario.
- [x] First token timing, the output limit, and EOS semantics
  agree: the first token comes from prefill logits and its time
  is prefill completion, a run of N tokens costs 1 prefill and
  N minus 1 decode forwards, and EOS stops generation and counts
  as a generated token. One limit is stated plainly: natural EOS
  from this base model under greedy decoding was never observed,
  so the real model EOS evidence is the shared custom EOS
  comparison plus the scripted loop tests.
- [x] Real single request output was compared with the
  reference implementation: 12 of 12 cases token for token.
- [x] A real decode forward has batch dimension greater than 1:
  the acceptance record shows shared decode forwards with batch
  sizes above 1, and the batched executor counts forwards, not
  requests.
- [x] Different lengths, dynamic join, and dynamic exit pass
  output and cache checks on the real model.
- [x] Cache and queue entries are cleaned after finish, failure,
  and cancellation: every acceptance scenario ends with zero
  active caches and every request released, and the engine
  releases a terminal request at most once.
- [x] The logical token budget, the padding cost, and the real
  memory limit are distinguished in `docs/resources.md`: the
  token reservation is a conservative bound, padding waste is
  measured per workload, and no claim is made that token
  counting guarantees memory safety.

## Service and reproduction

- [x] The model is loaded once and engine state has a single
  owner: one worker thread holds the engine, the scheduler, and
  the executor, and the HTTP layer only translates.
- [x] Streaming output, disconnect cancellation, a full queue,
  and slow consumers have defined behaviour, verified in the
  service, streaming, disconnect, overload, and HTTP integration
  tests and demos. Nothing in this list is claimed beyond what
  those records show: the HTTP server ran on loopback only.
- [x] A clean environment runs the CPU tests and the demo from
  the README alone: 152 passed and 8 model tests skipped with
  the dev extra, 188 passed after adding the model extra.
- [x] Model, dependencies, device, and run commands are
  recorded in `configs/model.json`, `configs/benchmark.json`,
  `docs/model.md`, and `docs/reproduction.md`.
- [x] The repository depends on no personal path and contains
  no virtual environment, credential, or model weight: the
  final hygiene search found no personal absolute path in
  source, tests, benchmarks, docs, or configs, and no file
  outside gitignored caches above 500 KiB.

## Experiments and presentation

- [x] The baseline and the continuous strategy use the same
  model, backend family, and output rules, and every quoted run
  first passed the workload check (same finished count, prompt
  tokens, output tokens, and token sequences).
- [x] Raw results, the summary script, and the charts trace to
  each other: charts are generated from
  `benchmarks/results/real-main-seed1234/summary.json` by one
  command, and summaries recompute from the raw records.
- [x] Failed, cancelled, and rejected counts are recorded:
  they are zero in the main experiment runs and are reported
  where they occur in the service and overload records, together
  with the sample size of 60 finished requests per continuous
  point.
- [x] Simulation data, core runtime data, and HTTP data are
  kept apart: simulation records live in the simulation
  pipeline and are labelled simulation, the real tables come
  only from the frozen real protocol, and HTTP behaviour is
  verified functionally and never mixed into the runtime tables.
- [x] No conclusion exceeds its hardware, workload, repeat
  count, or implementation scope: every figure is stated as
  exploratory for one small model on one shared CPU machine,
  with repeat ranges, and no comparison with a production
  serving system is made.
- [x] Every resume bullet points at code or a saved record:
  `docs/resume_notes.md` names the supporting file under each
  bullet.
- [x] One hard case, one wrong assumption, and one evidenced
  trade off are written down at the end of
  `docs/resume_notes.md`.

## Future work, deliberately not done

- Batched or chunked prefill, the largest structural cost the
  runtime controls, left as the evidenced next step.
- Pack and split cost at cached lengths beyond the measured
  range (at most 24 tokens in the bottleneck runs).
- HTTP behaviour beyond loopback, and any exposure beyond a
  local demo.
- Sampling controls, multiple models, and OpenAI compatibility,
  all outside the frozen service contract.
- A natural EOS observation for this base model under greedy
  decoding, if a prompt set that produces one is ever found.
