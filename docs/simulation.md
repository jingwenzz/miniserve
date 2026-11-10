# MiniServe Simulation Time: Logical Clock and Cost Model

This document defines how simulation runs keep time. It exists so
that no simulation number is ever mistaken for a measured model
number, and so that every simulation result can be recomputed by
hand from the assumptions written here.

## 1. Two clocks, kept apart

- Wall clock time is what `time.monotonic` reports while Python
  code happens to run. Around the fake executor it measures the
  test harness on this machine, nothing else.
- Simulation time is a logical clock, `SimulationClock` in
  `src/miniserve/simulation.py`. It starts at a stated value and
  only moves when simulated work is charged against it, or when
  the runner waits for the next scheduled arrival. It never reads
  the wall clock.

In a simulation run, every request shares the one simulation
clock, so all of its timestamps (arrival, admission, prefill
start, token times, finish) live on the same logical timeline.
Requests built outside a simulation run still use the monotonic
clock by default; the two kinds of timestamps must never be mixed
in one comparison.

The fake executor appends a fixed token per decode step. Its
token production rate is a property of the test double, so it is
never reported as model throughput. Only the ordering of events,
the step counts, and the values derived from the cost model below
are simulation results.

## 2. The cost model, stated in full

Costs are explicit parameters of `SimulationCosts`, in logical
seconds:

| Work | Cost | Default |
| --- | --- | --- |
| Prefill of one request | `prefill_seconds_per_token` times its prompt length | 0.1 per token |
| Decode of one token for one request | `decode_seconds_per_token` | 0.5 per token |

Assumptions a reader must know before quoting any number:

- Costs inside one step accumulate one request after another.
  A step that prefills two requests and decodes two more is
  charged for all four pieces of work in sequence. This serial
  work model is chosen so a step can be costed by hand.
- Because work is charged per request, the total logical duration
  of a run is the total work in the workload, whichever policy
  scheduled it. Batching under this model changes step counts
  and when individual requests finish; it does not reduce total
  logical time. A cost model that charged once per batched step
  instead of once per request would produce different duration
  conclusions, and that difference is a property of the model,
  not of a GPU.
- When nothing is active and the next arrival is in the future,
  the clock jumps straight to that arrival. Idle gaps produce no
  timeline rows and cost nothing.

## 3. The per step timeline

Each simulation run records one `StepRecord` per engine step:
the step number, the logical start and end times, the prefill and
decode request ids the step executed, and the waiting and running
id lists as they stood at the start of the step. From these rows
a reader can count steps, queue rounds (the steps that began with
a given request still waiting), and active requests per step,
without trusting any summary.

Hand worked example, with the default costs, batch size 3, and
token budget 50. Requests: sim-a prompt 10, output 2, arrival 0;
sim-b prompt 5, output 1, arrival 0; sim-c prompt 5, output 2,
arrival 1.0. The timeline is saved in
`benchmarks/simulation_timeline_example.jsonl`.

| Step | Start | End | Prefill | Decode | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0 | 1.5 | sim-a, sim-b | none | Prefill costs 10 times 0.1 plus 5 times 0.1 |
| 2 | 1.5 | 3.0 | sim-c | sim-a, sim-b | sim-c arrived at 1.0 and is admitted at this step boundary; prefill 0.5, then two decodes at 0.5 each; sim-b finishes |
| 3 | 3.0 | 4.0 | none | sim-a, sim-c | Two decodes; sim-a finishes |
| 4 | 4.0 | 4.5 | none | sim-c | Last decode; sim-c finishes |

Metrics read off the same run: sim-a has time to first token 2.5
and latency 3.5; sim-b has time to first token 3.0 and latency
3.0; sim-c waited one step in the queue, has queue time 0.5, time
to first token 3.0 measured from its arrival at 1.0, and latency
3.5. The tests in `tests/test_simulation.py` assert every one of
these values against the program output.

## 4. Changing the cost parameters changes the conclusions

The same hand worked example with the decode cost doubled to 1.0
logical second per token: the schedule is identical (the same
four steps select the same request ids in the same order), but
the total logical duration grows from 4.5 to 7.0. Step counts and
selection order are properties of the scheduler and the token
budget; durations are properties of the cost parameters. Any
report that quotes a duration must quote the costs it used, and
any conclusion drawn from durations must be rechecked when the
costs change.

## 5. What simulation cannot prove

- It cannot prove GPU speedup, or anything about a real model:
  there is no model, no kernel, and no parallel execution in the
  cost model.
- A step count is a scheduling count, not a throughput figure.
- Under the serial cost model, sharing a step can stretch an
  individual request: in the example experiment run recorded in
  `docs/experiments.md`, the continuous policy used 12 steps
  against the serial policy's 36 for the same 30 output tokens,
  both runs totalled the same 21.3 logical seconds of work, and
  the continuous run still showed the higher mean latency
  (15.8167 against 12.3167 logical seconds) alongside the lower
  mean time to first token (9.2167 against 10.3167). That is a
  real trade off of this model, reported as observed, not tuned
  away.
