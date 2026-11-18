# MiniServe Resource Limits and Budget Semantics

This document fixes what each limit in the scheduler actually
bounds, how cache capacity is reserved and returned, and what
the numbers do not promise. The limits are configuration on the
scheduler (`Scheduler` and `create_scheduler` accept all of
them; both default to no limit where they are optional):

| Limit | Bounds | Unit | Scope |
| --- | --- | --- | --- |
| `max_batch_size` | requests picked in one step | requests | per step |
| `max_batch_tokens` | work picked in one step: prompt tokens per prefill plus one per decode row | logical tokens | per step |
| `max_model_len` | one request's prompt plus reserved output | tokens | per request, at admission |
| `max_active_requests` | requests resident at once (prefilled, holding a cache) | requests | resident |
| `max_reserved_tokens` | total reserved cache capacity of resident requests | tokens | resident |

The first two are per step work limits. The last two are
resident limits. They answer different questions, and no one of
them can stand in for another: a step can be well inside its
token budget while the resident caches are already at capacity,
and a nearly empty system can still be handed a single step it
cannot afford.

## Reservation model

A request reserves `prompt_length + max_new_tokens` tokens of
cache capacity. The reservation is deliberately conservative:
the full output limit is reserved up front, even though the
cache only reaches that size if the request really generates up
to its limit, and an early EOS returns the unused part only by
ending the request.

- A waiting request reserves nothing. It holds no cache yet, so
  admission does not charge it against the resident capacity.
- The reservation starts when the request is selected for
  prefill and becomes active, and it is held for the request's
  whole active life.
- The reservation ends when the request reaches a terminal
  state: finished, failed, or cancelled. The scheduler computes
  `reserved_tokens` and `active_request_count` from the running
  list on every read, so the return needs no separate
  bookkeeping and cannot drift from the queues. A cancelled
  request's reservation is already returned before the next
  step runs.

## Reject versus wait

The same capacity limit produces two different outcomes,
depending on whether the request could ever fit:

- A request whose reservation alone exceeds
  `max_reserved_tokens` is rejected at admission with
  `RequestRejectedError`. It could never become active, even in
  an empty system, so letting it wait would repeat the
  permanent waiting failure the admission contract exists to
  prevent. The rejection does not burn its request id, and a
  corrected request can reuse the id. The per request
  `max_model_len` context check works the same way.
- A request whose reservation fits on its own, but not right
  now because active requests hold the capacity, stays in the
  waiting queue. It is selected for prefill at the first step
  where the active count is below `max_active_requests` and its
  reservation fits the returned capacity. The active count
  limit holds a request back the same way, even when capacity
  is free. Both waits use the existing FIFO rule: the head
  request blocks later ones, which is the recorded trade off of
  this scheduler, not a new behaviour.

With no limits configured, selection behaves exactly as it did
before these limits existed. The serial baseline inherits the
admission rejection, and its one active request policy is
stricter than any active limit of one or more.

## Logical budget versus padded allocation

The token numbers above are logical counts. They are not bytes,
and they are not the shape of any real tensor:

- The per step token budget charges a decode row one token,
  because that is the logical work of producing one token. The
  real batched decode step allocates a packed cache of shape
  [B, 3, L_max + 1, 64] per layer across 30 layers, where L_max
  is the longest row. Shorter rows carry padding up to L_max,
  so the padded allocation of a step is B times L_max, which is
  more than the sum of the rows' logical lengths whenever
  lengths differ. Padding waste for a step is the sum of
  (L_max - L_i) over rows.
- Pack and split copy every layer on every decode step, so a
  step also allocates temporary tensors on the order of the
  whole packed cache, twice over. None of that appears in any
  token count.
- Prefill and decode are separate forwards. A prefill forward
  allocates its own activations and its output cache, and the
  per step budget describes it only as prompt tokens of work,
  not as a memory figure.
- For the locked model in float32, one cached token costs
  3 heads times 64 head size times 4 bytes times 2 (keys and
  values) times 30 layers, which is 46,080 bytes per token.
  That conversion is exact for the stored cache tensors of
  this model and dtype, and it still excludes activations,
  logits (one decode step produces [B, 1, 49152] floats),
  padding, pack and split temporaries, and framework overhead.

## Known limitations

- No token count in this project guarantees that a run cannot
  run out of memory. The limits are a conservative early bound
  that keeps obviously oversized requests out and caps what is
  resident at once; actual memory depends on padding, temporary
  copies, activations, and the framework allocator, none of
  which the scheduler measures.
- Reserving the full output limit wastes capacity for requests
  that stop early. That waste is accepted for the first version
  because it makes the admission decision final: a request that
  was activated can always grow to its limit without evicting
  another request.
- A large head request can hold back smaller later requests
  while it waits for capacity (FIFO head of line blocking).
  Skipping the head would trade fairness for utilization and is
  not done here.
- The limits are enforced by the scheduler, which is the only
  component that selects work. Calling an executor directly, as
  some unit tests do, bypasses them by construction.
