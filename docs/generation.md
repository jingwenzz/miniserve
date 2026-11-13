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
forward completes. The two backends keep the convention each one
actually implements, and neither is silently mixed into the
other; the real executor wiring that records the first token at
prefill completion is described in section 6.

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

At the time that was written this was a single prompt check. The
broader correctness pass has since been run and is recorded in
section 5 below, and the loop has since been wired into the
engine as a real executor, recorded in section 6. No timing from
these scripts is a performance result; they run on a shared CPU
and exist to make the mechanics checkable.

## 5. Correctness pass across prompts

`benchmarks/correctness.py` runs both paths, reference
`generate` and the hand written loop, over 12 fixed cases with
the same model, tokenizer, dtype, and greedy stopping rules, and
saves the full record in `benchmarks/correctness_output.json`.
The cases cover a one token limit, a three token limit, a one
token prompt, several multi token prompts of different kinds
(story opening, question and answer, arithmetic, list
continuation), a long prompt of 121 tokens built by repeating one
sentence, and the two EOS cases described below.

Result: all 12 cases matched exactly, token for token, with no
first divergence in any case, and the stop reasons agreed in
every case (the limit in 11 cases, EOS in the custom EOS case).
No floating point divergence was observed, so there is no
divergence to attribute: under float32, eager attention, and
greedy decoding on this CPU, the two paths are token identical on
this case set. That is a statement about these 12 cases, not a
proof for every prompt.

Two honest limits of the pass:

- The real EOS token (id 0) never appeared. While preparing the
  pass, 16 probe prompts were run with a 32 token limit and none
  produced token 0, and in the pass itself the search case ran 48
  tokens without producing it either; the output record marks
  `real_eos_appeared` false for all 12 cases. EOS stopping on the
  real model is therefore verified with a shared custom EOS id:
  both paths were given EOS id 198, a token the reference prompt
  produces as its seventh token, and both stopped there with the
  identical seven tokens ending in 198. EOS id 0 stopping inside
  the loop itself is covered by the scripted fake model tests.
- The long prompt case is 121 tokens, not a run near the 8,192
  position limit. A literal near limit prefill was not attempted
  on this CPU only machine; the 121 token case exercises the same
  mask and position path at a larger length, and the record does
  not claim more than that.

## 6. The loop as an engine executor

`src/miniserve/real_executor.py` turns the same bookkeeping into
a `RealExecutor` that implements the executor protocol, so the
scheduler and the engine are unchanged. Each forward still
processes one request only. Prefill runs the prompt once, stores
the cache under the request id, and appends the first token from
the prefill logits, which settles the convention this document
flagged in section 2: for the real backend the first token time
is the prefill completion moment, recorded by the same
`append_token` call the fake backend's decode uses, so the
metrics definition in `docs/metrics.md` now has its documented
real source. Decode feeds the most recent generated token and
appends the produced one; a run of N tokens takes one prefill
forward and N - 1 decode forwards, and a limit of one takes no
decode forward. A produced EOS token finishes the request in the
executor, because the generic request type only finishes on the
output limit, and the EOS token is never fed back. `release`
drops the stored cache for the request and is idempotent; the
engine owns when it runs.

`benchmarks/real_generate.py` is the command line entry point:
one prompt goes through a batch size one scheduler and the engine
and prints the text plus the per request metrics. Recorded runs,
saved in `benchmarks/real_generate_output.json` for the default
prompt: the default 16 token run finished in 16 engine steps (1
prefill forward, 15 decode forwards) with token ids identical to
the reference run, no cache left afterwards, and the request
released; a limit of 1 finished in 1 step with 0 decode forwards;
a second prompt (`Hello`, 4 tokens) also finished with its cache
released. The metrics in that file are wall clock engine times
on this shared CPU, recorded as observed values, not as a
performance result.
