# Reproduction Guide

How to rebuild MiniServe from the project files alone, what was
verified in a clean environment, and the final list of known
issues. Nothing in this project depends on a previous virtual
environment, on a file that exists only in a local cache, or on
a personal absolute path: a search of the source, tests,
benchmarks, docs, and configs for home directory and temporary
directory paths returns no hits, and the only large files in a
working copy are the gitignored caches (`.venv`, `.mypy_cache`,
`.pytest_cache`, `.ruff_cache`, `__pycache__`, egg-info).

## Clean environment procedure

Use a fresh copy of the repository (a new clone or a copy
without `.venv` and the cache directories) on a disk with
several GiB free, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python benchmarks/demo_scheduler.py
python -m pip install -e '.[model]'
python -m pytest -q
python benchmarks/real_generate.py
```

The dev extra alone is enough for the scheduler, simulation,
service, and HTTP layers. Tests that need the real model skip
themselves when the model extra is absent instead of failing.
The model extra installs torch and transformers inside the
ranges pinned in `pyproject.toml`; the first real run downloads
the model recorded in `configs/model.json` (about 260 MiB) into
the Hugging Face cache, which is a download cache, not a
project file, and can be deleted and rebuilt at any time.

## Clean environment record

Verified on 2026-11-25 on this machine (Linux x86_64,
Python 3.12.3, CPU only, AMD EPYC 9D25, no GPU):

- Fresh copy plus fresh virtual environment, dev extra only:
  install succeeded; `pytest -q` reported 152 passed and
  8 skipped, the skips being exactly the model dependent tests;
  the fake demo finished 3 requests in 5 steps with outputs
  4, 2, and 3.
- After adding the model extra in the same clean environment
  (torch 2.9.1, transformers 4.57.6, matching the locked
  ranges): `pytest -q` reported 188 passed with no skips, the
  fake demo again finished 3 requests in 5 steps, and
  `benchmarks/real_generate.py` finished with the 16 generated
  token ids matching `benchmarks/reference_output.json` exactly,
  the request released and no active cache left.
- The long lived development environment reports the same
  188 passed, so the clean copy and the development copy agree
  once the model extra is present. The three charts in
  `docs/charts/` also regenerate byte identically from the
  saved summary in the development environment.

## Environment pitfalls, all previously hit and recorded

- pip needs a temporary directory with room for the torch
  wheel. On a machine where the default temporary filesystem
  is small (512 MiB here), the install fails with
  `No space left on device` even though the target disk has
  space. Point `TMPDIR` at a directory on the large disk
  before installing the model extra, and create the virtual
  environment itself on the large disk, not in `/tmp`.
- The model download can fail on the Xet transfer path with a
  proxy error. Set `HF_HUB_DISABLE_XET=1` for the first real
  run; the plain HTTPS download then succeeds.
- Shared CPU wall clock times vary run to run. Reproduction
  checks token identity and counts, not seconds; the benchmark
  protocol in `docs/benchmark_protocol.md` states which numbers
  may be compared and how.

## What counts as reproduced

- CPU route: the test suite passes and the fake demo completes
  with the outputs above, with no model download at all.
- Real route: `benchmarks/real_generate.py` finishes with the
  same 16 token ids as `benchmarks/reference_output.json`, and
  `benchmarks/fourth_week_acceptance.py` reports
  `all_checks_passed` true across its six scenarios.
- Results route: `benchmarks/make_results_charts.py` regenerates
  the three charts in `docs/charts/` from
  `benchmarks/results/real-main-seed1234/summary.json`, and the
  experiment summaries under `benchmarks/results/` recompute
  byte identically from their raw files with
  `benchmarks/summarize.py` where that pipeline applies.

## Final issue list

Open items at freeze, none blocking the routes above:

- Natural end of sequence from the locked base model under
  greedy decoding has never been observed; EOS behaviour is
  verified with a shared custom EOS id and scripted models.
- Pack and split cost at cache lengths far beyond the tested
  32 tokens is an accounting prediction, not a measurement.
- The HTTP layer was exercised on loopback only; real network
  disconnect timing is untested.
- One pre existing ruff import sorting note in
  `tests/test_request.py` predates the current code and is left
  untouched.
- Benchmark samples are 60 finished requests per continuous
  point, below the plan target of about 100, so all headline
  figures stay exploratory.

Code is frozen at this point: only correctness fixes and
delivery work follow, no new product features.
