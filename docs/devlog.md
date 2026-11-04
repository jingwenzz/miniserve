# MiniServe Devlog

## 2026-11-04

Goal for the day: make the project reproducible from a clean environment, run the existing tests and demo, and leave a version record that can be checked later.

What actually changed: created a local virtual environment .venv (already in .gitignore, not committed); installed the dev dependencies from pyproject.toml in editable mode; created this file as the first devlog entry; updated the Quickstart section of the README to match the virtual environment flow verified below. The core code (request.py, scheduler.py, executor.py, engine.py) was only read today, not modified. The local copy is not a git repository, so there is no local branch; changes go to the remote repository through the GitHub web editor.

Commands and observed results: `python3 -m venv .venv` succeeded, with Python 3.12.3 inside the environment. `.venv/bin/python -m pip install -e '.[dev]'` succeeded; key versions are pytest 9.2.0, ruff 0.17.3, mypy 2.4.1, pip 24.0, on Linux x86_64 glibc 2.39. `.venv/bin/python -m pytest -q` reported 17 passed in 0.06s, exit code 0. `.venv/bin/python benchmarks/demo_scheduler.py` exited 0 and finished 3 requests in 5 steps: request-1 prompt 10 output 4, request-2 prompt 40 output 2, request-3 prompt 5 output 3, each matching its max_new_tokens.

One thing I can now explain: in Engine.step, the scheduler first picks the requests, then prefill and decode run, then the queues are updated.

Still unsure: a prompt longer than the whole token budget currently waits in the queue forever and can never run. This is a known issue and is already scheduled to be fixed in a later session, so it was left alone today.

Next single step: define the request and scheduling contracts and write them up in docs/design.md.
