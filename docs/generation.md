# MiniServe Real Generation: Prefill, First Token, and the Decode Loop

This document records how the locked model actually generates,
measured on the reference prompt with the scripts in this
repository. The machine readable runs are
`benchmarks/prefill_output.json` (one prefill forward, from
`benchmarks/prefill_inspect.py`) and
`benchmarks/manual_generate_output.json` (the hand written loop,
from `benchmarks/manual_generate.py`). Both use the locked
configuration in `configs/model.json`: SmolLM2-135M at the pinned
revision, float32, eager attention, CPU, greedy decoding.

## 1. One prefill, measured

Prompt: `The capital of France is`, tokenized with no added
special token to the five ids 504, 3575, 282, 4649, 314.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| input_ids | [1, 5] | batch 1, five prompt positions |
| attention_mask | [1, 5], all ones | no padding in a single request |
| logits | [1, 5, 49152] | one vocabulary row per prompt position |
| last position logits | [49152] | the only row generation uses |
| layer 0 keys | [1, 3, 5, 64] | batch, KV heads, positions, head size |
| layer 0 values | [1, 3, 5, 64] | same layout as keys |

All 30 layers reported the same key and value shapes. The head
size 64 is the hidden size 576 divided by the 9 attention heads;
the cache stores keys and values for the 3 key value heads of the
grouped query attention layout, not for all 9 query heads. After
prefill the cache sequence length is 5: the cache holds exactly
the prompt, nothing more.

Prefill computes a logits row for every prompt position, but only
the last row is a prediction of something unknown. Each earlier
row predicts the next prompt token, whose true value is already in
the input, so those rows are computed and discarded.

## 2. Where the first token comes from

The first generated token is the argmax of the last prompt
position logits row. For the reference prompt that row scores
token 260 highest (logit 14.0228, ahead of 7042 at 12.9973), and
260 is exactly the first generated token of the reference run in
`benchmarks/reference_output.json`. No decode step is involved in
producing it.

This settles a semantic difference the metrics document flagged
earlier: the fake executor produces its first token inside decode,
while the real model produces it inside prefill. For a real
backend, the first token time is therefore the moment the prefill
forward completes. The fake and real conventions are not silently
mixed: unifying the event and metrics handling around the real
convention is part of wiring the real executor into the engine,
and until then the two backends keep the convention each one
actually implements.

## 3. The hand written decode loop

After prefill, generation continues one token per forward:

1. The input is a single token in a [1, 1] tensor: the most
   recently generated token, which is the only token the cache
   has not seen yet. The prompt is never fed again, and no
   already generated token is fed twice.
2. The attention mask for that step has one entry per cached
   position plus the new token (length L + 1 at cache length L),
   all ones, because a single request has no padding.
3. The cache position passed for the new token is L, the index it
   occupies, so positions stay aligned with the cached context.
4. The forward appends exactly one position to every layer of the
   cache and returns logits whose last (only) row picks the next
   token by argmax.

The bookkeeping consequence is an off by one that is easy to get
wrong: the generated tokens lead the cache by one. A token is
produced by a forward before it is ever an input. When generation
stops after N generated tokens, only the first N - 1 of them were
fed back, so the final cache length is prompt length + N - 1. The
last generated token needs no forward of its own unless generation
continues past it. In the recorded run the cache grows 5, 6, 7,
and so on to 20 for 16 generated tokens: 5 prompt positions plus
15 fed back generated tokens, with the 16th token produced by the
last decode forward and never fed.

Stopping rules, matching the reference generate path:

- A produced EOS token (id 0 for this model) counts as a
  generated token and ends the run; it is not fed back.
- Reaching max_new_tokens ends the run after that token is
  produced, again without feeding it back.
- A limit of max_new_tokens 1 therefore needs no decode forward
  at all: prefill alone produces the single token.

## 4. What was verified, and what was not

The hand written loop reproduced the reference run on the fixed
prompt exactly: the same 16 generated token ids in the same
order, stopping by the token limit, with every decode input shape
[1, 1] and the cache growing by exactly one position per decode
step. The loop bookkeeping (prompt fed once, one token per decode
input, cache growth, the off by one, EOS and limit stopping) is
also covered without the model by tests/test_manual_generate.py,
which drives the loop with a fake scripted model.

This is a single prompt check, recorded as such. A broader
correctness pass across several prompts, including EOS and near
limit cases against the reference path, is a separate scheduled
step and has not been done yet. No timing from these scripts is a
performance result; they run on a shared CPU and exist to make
the mechanics checkable.
