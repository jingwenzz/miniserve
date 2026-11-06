# MiniServe Devlog

## 2026-11-04

Goal for the day: make the project reproducible from a clean environment, run the existing tests and demo, and leave a version record that can be checked later.

What actually changed: created a local virtual environment .venv (already in .gitignore, not committed); installed the dev dependencies from pyproject.toml in editable mode; created this file as the first devlog entry; updated the Quickstart section of the README to match the virtual environment flow verified below. The core code (request.py, scheduler.py, executor.py, engine.py) was only read today, not modified. The local copy is not a git repository, so there is no local branch; changes go to the remote repository through the GitHub web editor.

Commands and observed results: `python3 -m venv .venv` succeeded, with Python 3.12.3 inside the environment. `.venv/bin/python -m pip install -e '.[dev]'` succeeded; key versions are pytest 9.2.0, ruff 0.17.3, mypy 2.4.1, pip 24.0, on Linux x86_64 glibc 2.39. `.venv/bin/python -m pytest -q` reported 17 passed in 0.06s, exit code 0. `.venv/bin/python benchmarks/demo_scheduler.py` exited 0 and finished 3 requests in 5 steps: request-1 prompt 10 output 4, request-2 prompt 40 output 2, request-3 prompt 5 output 3, each matching its max_new_tokens.

One thing I can now explain: in Engine.step, the scheduler first picks the requests, then prefill and decode run, then the queues are updated.

Still unsure: a prompt longer than the whole token budget currently waits in the queue forever and can never run. This is a known issue and is already scheduled to be fixed in a later session, so it was left alone today.

Next single step: define the request and scheduling contracts and write them up in docs/design.md.

## 2026-11-04

Goal for the day: write down the request and scheduling contracts, then stop requests that can never run from waiting in the queue forever.

What actually changed: added docs/design.md with the request lifecycle, the queue invariants, the ownership table, the distinction between the per step token budget and resident cache capacity, the admission contract, a decision to defer a FAILED state until a real executor creates a runtime failure path, and a boundary test checklist. In request.py, constructing a request now rejects an empty prompt and a non positive max_new_tokens with a ValueError. In scheduler.py, the scheduler gained an optional max_model_len limit and a RequestRejectedError type, which is a ValueError subclass. Admission now rejects a prompt longer than the per step token budget, because prefill is not chunked and such a request could never be scheduled, and it rejects a request whose prompt length plus reserved max_new_tokens exceeds max_model_len when that limit is configured. A rejected request is never queued. Added tests/test_validation.py with eight regression tests covering the empty prompt, zero and negative output limits, the budget boundary at exactly the limit and one above it, queue cleanliness after a rejection, a valid request completing after a rejection, and the context limit boundary at exactly the limit and one above it.

Commands and observed results: `.venv/bin/python -m pytest -q` reported 25 passed in 0.06s, exit code 0, which is the previous 17 tests plus the 8 new ones. `.venv/bin/python benchmarks/demo_scheduler.py` exited 0 and again finished 3 requests in 5 steps with the same outputs as before. `.venv/bin/python -m mypy src/miniserve` reported success with no issues in 5 source files. `.venv/bin/python -m ruff check src/miniserve/request.py src/miniserve/scheduler.py tests/test_validation.py` reported all checks passed. A full ruff run over src, tests, and benchmarks still reports one pre existing import sorting warning in tests/test_request.py, which was left untouched because it is outside the scope of this change.

One thing I can now explain: the per step token budget limits the work picked in a single step, while resident capacity limits the total context held between steps, so one number cannot represent both. I can also explain that a prompt longer than the whole step budget used to wait forever because prefill is not chunked, and that admission now rejects it up front instead.

Still unsure: cancellation of waiting and running requests and duplicate request ids are not handled yet. They are the next scheduled items and are listed in the checklist in docs/design.md.

Next single step: handle cancellation and duplicate request ids.

## 2026-11-05

Goal for the day: make cancellation and duplicate requests safe, then check the scheduler budget boundaries with hand computed examples.

What actually changed: the scheduler gained a single cancellation entry point, cancel_request, which removes a waiting or running request from its queue and returns True, treats a repeated cancel or a cancel after finish as a no-op that returns False, and raises KeyError for an id the scheduler never admitted. The scheduler now prunes terminal requests from both queues at the start of schedule and inside has_pending_requests, so a request cancelled directly on the object is dropped before the executor can be asked to start prefill on it. Request ids are now unique for the lifetime of one scheduler: an id is recorded only on successful admission, so a validation rejection does not burn its id, while a duplicate active id, the same object offered twice, and an id reused after finish are all rejected with RequestRejectedError and never take a second slot. Added tests/test_cancellation.py with twelve tests covering waiting and running cancellation through both entry paths, repeated cancellation, unknown ids, and the duplicate rules. Added tests/test_scheduler_boundaries.py with nine tests built from hand traced examples: a prefill that exactly fills the remaining budget, one token over, a temporarily blocked head that runs after the running request finishes, batch size 1 serialising prefill and giving its only slot to decode, two requests finishing together and freeing both slots, FIFO head of line blocking, decode priority under a tight budget, and a full run asserting that no step exceeds the limits and no request is lost or selected twice. The hand trace table and the cancellation and duplicate contracts were added to docs/design.md, and the matching checklist items are now ticked.

Commands and observed results: `.venv/bin/python -m pytest -q` reported 46 passed in 0.07s, exit code 0, which is the previous 25 tests plus 12 cancellation tests and 9 boundary tests. One cancellation test failed on its first run because a directly cancelled running request stayed in the running list when no further step ran; has_pending_requests now prunes terminal requests first, and the full suite passed after that fix. `.venv/bin/python benchmarks/demo_scheduler.py` exited 0 and again finished 3 requests in 5 steps with the same outputs as before. `.venv/bin/python -m mypy src/miniserve` reported success with no issues in 5 source files. `.venv/bin/python -m ruff check src/miniserve/scheduler.py tests/test_cancellation.py tests/test_scheduler_boundaries.py` reported all checks passed.

One thing I can now explain: cancelling a request is a scheduler queue operation, not just a state flag, because the request has to leave waiting or running before the next selection, or it would be handed to the executor again. I can also explain the difference between a request that does not fit this step, which waits and runs later, and a request that can never fit, which admission rejects up front.

Still unsure: cancellation only takes effect at a scheduling boundary, so a token already being produced inside the current step is not interrupted. That is acceptable for the fake executor and is recorded in the design document; it will need revisiting when a real executor owns per request cache.

Next single step: define the executor boundary so the engine no longer depends on the fake executor type.
