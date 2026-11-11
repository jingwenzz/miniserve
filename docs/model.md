# MiniServe Model Selection and Reference Run

This document records which real model MiniServe will use, why it
was chosen, the exact environment it was verified in, and the
reference generation that later custom code must match. The
machine readable version of the locked configuration is
`configs/model.json`, and the raw reference output is
`benchmarks/reference_output.json`, produced by
`benchmarks/reference_generate.py`.

## 1. The machine that is actually available

Verified directly on 2026-11-10:

- No GPU is present. There is no NVIDIA device and no
  `nvidia-smi`, and PyTorch reports CUDA unavailable.
- CPU: AMD EPYC 9D25, Linux x86_64. RAM reported by the system is
  7.7 GiB total, and disk space is ample (about 144 GiB free).
- Python 3.12.3.

Consequence: the first real model must run on CPU within a few
gigabytes of RAM. That rules out starting at the top of the size
range for comfort, and it means every timing number in this phase
is a CPU number on this machine, not a GPU result.

## 2. Selection

Chosen model: `HuggingFaceTB/SmolLM2-135M`, the base model, at
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

How it meets the conditions listed at the second week checkpoint
(see `docs/design.md`, section 14):

| Condition | Evidence from the verified run |
| --- | --- |
| Decoder only, 0.1B to 0.5B | Llama architecture causal LM, 134,515,008 parameters counted from the loaded model |
| Permissive licence | The model repository records licence apache-2.0 |
| Simple cache structure | One standard KV cache per layer: 30 layers, 9 attention heads, 3 key value heads (grouped query attention), hidden size 576, no sliding window or hybrid attention in the configuration |
| Official transformers support | Loaded and generated with transformers 4.57.6 through `AutoModelForCausalLM`, no custom code |
| Tokenizer in the same repository | Loaded with `AutoTokenizer` from the same model id and revision; vocabulary size 49,152 |
| Pinnable revision | The revision above is recorded in `configs/model.json` and passed on every load |
| Runnable on this machine | Reference generation below ran on CPU in float32 and was repeated with identical token ids |

The base model was chosen over the instruct variant on purpose:
the next stages study prefill, decode, and cache mechanics with
greedy decoding on plain prompts, and a base model keeps chat
templates out of that path.

Alternatives considered and set aside:

- `Qwen/Qwen2.5-0.5B`: permissive licence and a capable model,
  but at the top of the size range, roughly four times the
  parameters, so slower and heavier on a CPU only machine.
  SmolLM2-135M was picked first on size alone; Qwen remains a
  reasonable fallback if SmolLM2 hits a blocking problem.
- `facebook/opt-125m`: the right size, but its model card
  records licence `other` with commercial use marked false, which
  fails the publishable licence condition.

## 3. Locked software versions

Installed into the project virtual environment and verified:

| Package | Version |
| --- | --- |
| torch | 2.9.1+cpu (CPU wheel from the PyTorch CPU wheel index) |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| numpy | 2.5.4 |
| Python | 3.12.3 |

The model extra in `pyproject.toml` records the torch and
transformers ranges. Two environment notes from the install,
recorded so the next install does not rediscover them: pip needs
a temporary directory on the large disk here, because the
default temporary filesystem is only 512 MiB and the torch wheel
does not fit in it; and the Hugging Face download in this
environment only succeeded with the Xet transfer path disabled
(`HF_HUB_DISABLE_XET=1`). The downloaded model cache is about
260 MiB and lives outside the repository; the repository records
the model id and revision, never the weights.

Locked generation settings for the reference: dtype float32 (the
checkpoint is stored in bfloat16, but float32 is the safer
default for CPU correctness comparisons), attention backend
eager, greedy decoding (`do_sample=False`), seed 1234 set on the
torch generator state before the run, maximum context from the
model configuration is 8,192 positions, and the EOS token id is
0.

## 4. Reference generation

Command, from the repository root:

```bash
HF_HUB_DISABLE_XET=1 python benchmarks/reference_generate.py
```

Fixed prompt: `The capital of France is`, tokenized by the
locked tokenizer to the five token ids 504, 3575, 282, 4649,
314, with no extra special token added in front. Greedy
generation with a limit of 16 new tokens produced these 16 token
ids, in order:

260, 3575, 282, 260, 1798, 30, 198, 198, 504, 3575, 282, 4649,
314, 260, 3575, 282

Decoded, the generated text is ` the capital of the
country.` followed by a blank line and `The capital of France is
the capital of`. The text is repetitive, which is expected from a
135M base model under greedy decoding; the point of this run is
not text quality. The run was repeated once and produced
identical prompt and generated token ids; only the wall clock
time differed (2.72 and 3.30 seconds, recorded as observed run
times on a shared CPU, not as a performance result).

No performance comparison is made at this stage. The next steps
split prefill out of this reference path, then rebuild decode by
hand, and compare the custom loop against exactly these token
ids under the same model, tokenizer, dtype, and stopping rules.
