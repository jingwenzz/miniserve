# MiniServe Batched Decode: Cache Layout Design

This document fixes one layout for running several requests in a
single decode forward on the locked model (SmolLM2-135M, 30
layers, 3 KV heads, head size 64, float32, eager attention,
transformers 4.57.6). It started as a design plus a measured
feasibility probe; the equal length case is now implemented as
recorded below, while different length rows still wait for the
padding path. Prefill stays per request for now, as the plan
allows: each request prefills on its own, and only decode
forwards are batched.

## Decision

Per request caches stay the source of truth. The executor keeps
one independent cache per request, exactly as the single request
executor does today. For one batched decode step it packs copies
of the active caches into one left padded batched cache, runs a
single forward, and splits the result back into per request
caches. Packing builds new tensors with concatenation, so the
stored per request caches are not mutated by the batched
forward; if the forward raises, the packed copy is discarded and
the originals are untouched. No fixed slot cache is introduced,
and the two schemes are not mixed.

## Row mapping

Rows follow the order of the request list handed to the decode
step, and the mapping is rebuilt for every step:

| Row | Request | Valid length before the step |
| --- | --- | --- |
| 0 | first request in the decode list | its own cached length |
| 1 | second request in the decode list | its own cached length |

The packed cache reports only the longest row length through
`get_seq_length()`, so per request valid lengths are tracked
outside the cache, next to each stored cache. A length is never
read back from the packed tensor shape for an individual row.

## Layout, per layer (all 30 layers share these shapes)

Let row i have valid length L_i and let L_max be the longest
valid length in the step. Padding count for row i is
P_i = L_max - L_i.

| Tensor | Before the step | After the step |
| --- | --- | --- |
| per request keys and values | [1, 3, L_i, 64] | [1, 3, L_i + 1, 64] |
| packed keys and values | [B, 3, L_max, 64] | [B, 3, L_max + 1, 64] |
| input_ids | [B, 1], each row its last generated token | n/a |
| logits | n/a | [B, 1, 49152], row i picks its next token from logits[i, -1] |

Padding direction is left. A shorter row is padded with zeros
at the front, so its valid positions occupy the trailing slots
and every row ends at the same physical slot. That makes the
append uniform: the new token lands at physical index L_max for
every row, so one shared `cache_position` of [L_max] describes
the step, and the cache layer update, which concatenates along
the sequence axis in this version, needs no per row index.

Two different positions must not be confused:

- Physical position is where a token sits in the packed tensor.
  Left padding shifts a short row by P_i physical slots.
- Logical position is where the token sits in its own request.
  Rotary embeddings must use the logical position, so the step
  passes `position_ids` of shape [B, 1] with value L_i per row,
  not the physical L_max. If `position_ids` were omitted, the
  model would derive it from `cache_position` and every row
  would wrongly share L_max.

The attention mask has shape [B, L_max + 1]: for row i, zeros on
the P_i padded past positions, ones on its L_i valid past
positions, and one on the current token. Padded keys therefore
never contribute to attention, and padded length is never
counted as valid length anywhere.

Splitting back is the mirror of packing. After the forward, row
i holds L_i + 1 valid positions in the trailing slots, starting
at physical index (L_max + 1) - (L_i + 1), which equals P_i.
Slicing from that index to the end and taking batch row i gives
a [1, 3, L_i + 1, 64] tensor pair per layer, which becomes the
request's new stored cache. Nothing valid is dropped and no
padded slot is kept.

## Hand worked example: lengths 7 and 3

Request A has L_A = 7, request B has L_B = 3, so L_max = 7,
P_A = 0, and P_B = 4.

- Before: A keys [1, 3, 7, 64] at physical slots 0 to 6. B keys
  [1, 3, 3, 64] packed into physical slots 4 to 6, with zeros in
  slots 0 to 3. Packed shape [2, 3, 7, 64].
- Step inputs: input_ids [[token A], [token B]] with shape
  [2, 1]; mask A is eight ones; mask B is [0, 0, 0, 0, 1, 1, 1,
  1]; position_ids [[7], [3]]; cache_position [7].
- After: packed shape [2, 3, 8, 64], the new token for both rows
  at physical slot 7. Split A from index 0, giving [1, 3, 8, 64]
  with valid length 8. Split B from index 8 - 4 = 4, giving
  [1, 3, 4, 64] with valid length 4. Both rows grew by exactly
  one valid position.

## API check on the locked version

The design was checked against the installed transformers
4.57.6 instead of assumed:

- `DynamicLayer.update` concatenates new keys and values along
  the sequence axis and ignores per row positions, which is why
  the uniform physical append slot matters.
- A batched cache is built through the `DynamicCache`
  constructor taking per layer key and value pairs
  (`ddp_cache_data`). Assigning tensors to fresh layer objects
  by hand does not work, because their key and value fields stay
  uninitialized until an update runs.
- The cache helpers recorded earlier (`batch_repeat_interleave`,
  `batch_select_indices`, `reorder_cache`) replicate, select, or
  reorder whole rows of a cache that is already batched. None of
  them merges independent caches of different lengths, so they
  do not replace pack and split.
- The model forward accepts per row `position_ids` separately
  from `cache_position`, and rotary embeddings are built from
  `position_ids`, which is what makes the logical versus
  physical split above expressible.

Right padding was rejected: the new token would still land at
one shared physical slot, but valid positions would start at
slot 0 with padding behind them, so after the append the valid
region of a short row would no longer be one trailing slice,
and the split rule would need per row gap handling for no
benefit.

## Measured feasibility probe

`benchmarks/batch_layout_probe.py` ran the layout once on the
real model with prompts of length 7 and 2 (the second row padded
by 5). Recorded in `benchmarks/batch_layout_output.json`:

- Packed layer shape before the step [2, 3, 7, 64], after the
  step [2, 3, 8, 64]; batched logits shape [2, 1, 49152].
- Both rows produced the same next token in the batched forward
  as in their individual decode steps (657 and 198).
- Logits differed from the individual runs by at most about
  1.9e-05, and the split caches differed from the individually
  updated caches by at most about 1.4e-05 across all layers,
  which is float batching noise, not a token divergence.

## Implemented: equal length batched decode

`src/miniserve/batched_executor.py` implements the first slice
of this layout. `BatchedRealExecutor` extends the single request
executor, inherits its per request prefill, and overrides only
decode:

- All requests in one decode call must hold caches of the same
  valid length. Their caches are stacked along the batch axis
  with concatenation, which for equal lengths is exactly the
  pack step above with every padding count equal to zero.
- One forward runs with input shape [B, 1], an all ones mask of
  shape [B, L + 1], a shared `cache_position` of [L], and
  `position_ids` of [B, 1] holding L per row (logical and
  physical positions coincide when nothing is padded).
- The updated batched cache is split back by taking row slices,
  and each request appends the greedy token from its own logits
  row. A failed batched forward marks every row FAILED, because
  the forward was shared.
- `decode_forward_count` counts forwards, not requests, and
  `decode_batch_sizes` records the batch dimension of each
  decode forward in order. Requests with different cache lengths
  are rejected with a ValueError before any forward runs; they
  are not silently run one by one.

Measured on the real model by `benchmarks/batched_generate.py`,
saved in `benchmarks/batched_generate_output.json`: prompts
"The capital of France is" and "The capital of Italy is", both
5 tokens, each limited to 8 new tokens. The batched run finished
both requests in 8 engine steps with 2 prefill forwards and 7
decode forwards, every decode batch size equal to 2, and the
generated token ids matched the individual single request runs
token for token for both prompts. The active cache count after
the run was 0 and both requests were released. Scripted model
tests in `tests/test_batched_executor.py` pin the same claims
without the real model, including the unequal length rejection
and the shared failure marking.

## Known costs and open items

Pack and split copy every layer on every decode step, so the
overhead grows with batch size and total cached length. That
cost is accepted for the first batched implementation and is a
named suspect to measure later, not a hidden one. Joining and
leaving the batch need no layout change: a request joins by
being included in the next pack with its own length, and leaves
by being excluded after its split cache is released. Resource
limits on the packed allocation are a separate later step.
