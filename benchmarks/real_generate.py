"""Generate one prompt through the engine with the real executor.

This is the command line entry point for the real single request
path: a prompt is tokenized, admitted to a scheduler configured
with batch size one, and driven by the engine with the real model
executor until it finishes. The output records the generated token
ids and text, the per request metrics from
``src/miniserve/metrics.py`` (wall clock engine times on this
machine, recorded as observed values and not as a performance
result), the number of prefill and decode forwards the executor
actually ran, and the executor cache state after the run, which
must show no cache left and the request released.

Usage, from the repository root, with the model extra installed:

    python benchmarks/real_generate.py
    python benchmarks/real_generate.py --prompt "Hello" --max-new-tokens 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def run_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    request_id: str = "real-1",
) -> dict[str, Any]:
    """Run one prompt through scheduler, engine, and real executor."""
    from miniserve.engine import Engine
    from miniserve.metrics import metrics_to_dict, summarize_request
    from miniserve.real_executor import RealExecutor
    from miniserve.request import Request
    from miniserve.scheduler import Scheduler

    prompt_token_ids: list[int] = tokenizer(prompt)["input_ids"]
    executor = RealExecutor(model, eos_token_id=int(model.config.eos_token_id))
    scheduler = Scheduler(
        max_batch_size=1,
        max_batch_tokens=int(model.config.max_position_embeddings),
        max_model_len=int(model.config.max_position_embeddings),
    )
    engine = Engine(scheduler, executor)
    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
    )
    scheduler.add_request(request)
    engine.run_until_complete()

    stop_reason = "unknown"
    if request.generated_token_ids and (
        request.generated_token_ids[-1] == executor.eos_token_id
    ):
        stop_reason = "eos"
    elif request.output_length == max_new_tokens:
        stop_reason = "max_new_tokens"

    return {
        "request_id": request_id,
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "prompt_length": len(prompt_token_ids),
        "max_new_tokens": max_new_tokens,
        "generated_token_ids": request.generated_token_ids,
        "generated_text": tokenizer.decode(request.generated_token_ids),
        "state": request.state.name,
        "stop_reason": stop_reason,
        "engine_steps": engine.step_count,
        "prefill_forwards": executor.prefill_forward_count,
        "decode_forwards": executor.decode_forward_count,
        "active_caches_after_run": executor.active_cache_count,
        "released_request_ids": list(executor.released_request_ids),
        "metrics": metrics_to_dict(summarize_request(request)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--request-id", default="real-1")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parent.parent
    recorded = json.loads((root / "configs" / "model.json").read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(
        recorded["tokenizer"], revision=recorded["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        recorded["model_id"],
        revision=recorded["revision"],
        dtype=torch.float32,
        attn_implementation=recorded["attention_backend"],
    )
    model.eval()

    result = run_one(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        request_id=args.request_id,
    )
    result.update(
        {
            "model_id": recorded["model_id"],
            "revision": recorded["revision"],
            "decoding": recorded["decoding"],
            "dtype": recorded["dtype"],
            "device": recorded["device"],
        }
    )
    out_path = Path(__file__).resolve().parent / "real_generate_output.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
