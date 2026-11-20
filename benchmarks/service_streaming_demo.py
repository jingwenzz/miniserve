"""Stream two real generations through the engine owner at once.

Two prompts run together on the batched real executor behind
one engine owner, and a reader thread per request collects its
stream events as they arrive. The saved record shows, per
request, the token events in order with their arrival offsets,
that the first token event arrived before that request's
terminal event, that exactly one terminal event ended each
stream, and that the concatenation of the streamed text deltas
equals the decoding of the full generated token sequence,
which is the incremental decoding claim. Times are wall clock
observations on a shared CPU and are not a performance result.

Usage, from the repository root, with the model extra installed:

    python benchmarks/service_streaming_demo.py
    python benchmarks/service_streaming_demo.py --max-new-tokens 4
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

PROMPTS = ["The capital of France is", "The capital of Italy is"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniserve.batched_executor import BatchedRealExecutor
    from miniserve.engine import Engine
    from miniserve.request import Request
    from miniserve.scheduler import Scheduler
    from miniserve.service import EngineOwner

    root = Path(__file__).resolve().parent.parent
    recorded = json.loads(
        (root / "configs" / "model.json").read_text(encoding="utf-8")
    )
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

    executor = BatchedRealExecutor(
        model, eos_token_id=int(model.config.eos_token_id)
    )
    scheduler = Scheduler(
        max_batch_size=2,
        max_batch_tokens=int(model.config.max_position_embeddings),
        max_model_len=int(model.config.max_position_embeddings),
    )
    engine = Engine(scheduler, executor)
    owner = EngineOwner(engine, tokenizer=tokenizer)

    start = time.monotonic()
    collected: dict[str, list[dict[str, Any]]] = {}

    def collect(request_id: str, handle: Any) -> None:
        events: list[dict[str, Any]] = []
        for event in handle.events():
            entry = event.to_dict()
            entry["arrived_at_seconds"] = round(time.monotonic() - start, 4)
            events.append(entry)
        collected[request_id] = events

    owner.start()
    handles = []
    readers = []
    try:
        for index, prompt in enumerate(PROMPTS):
            request_id = f"stream-{index}"
            handle = owner.submit(
                Request(
                    request_id=request_id,
                    prompt_token_ids=tokenizer(
                        prompt, add_special_tokens=False
                    )["input_ids"],
                    max_new_tokens=args.max_new_tokens,
                )
            )
            handles.append(handle)
            reader = threading.Thread(
                target=collect, args=(request_id, handle)
            )
            reader.start()
            readers.append(reader)
        for reader in readers:
            reader.join(timeout=60)
        for handle in handles:
            handle.result(timeout=5)
    finally:
        owner.stop(timeout=10)

    per_request: list[dict[str, Any]] = []
    all_ok = True
    for handle in handles:
        events = collected[handle.request_id]
        token_events = [e for e in events if e["type"] == "token"]
        terminals = [e for e in events if e["type"] in {"finished", "error"}]
        streamed_text = "".join(e["text"] for e in token_events)
        full_text = tokenizer.decode(handle.request.generated_token_ids)
        first_before_terminal = bool(
            token_events
            and terminals
            and token_events[0]["arrived_at_seconds"]
            < terminals[0]["arrived_at_seconds"]
        )
        ok = bool(
            len(terminals) == 1
            and terminals[0]["type"] == "finished"
            and streamed_text == full_text
            and first_before_terminal
            and len(token_events) == len(handle.request.generated_token_ids)
        )
        all_ok = all_ok and ok
        per_request.append(
            {
                "request_id": handle.request_id,
                "events": events,
                "token_event_count": len(token_events),
                "terminal_event_count": len(terminals),
                "streamed_text": streamed_text,
                "full_decoded_text": full_text,
                "streamed_text_matches_full_decoding": streamed_text
                == full_text,
                "first_token_event_before_terminal": first_before_terminal,
                "generated_token_ids": list(
                    handle.request.generated_token_ids
                ),
                "state": handle.request.state.name,
            }
        )

    result: dict[str, Any] = {
        "model_id": recorded["model_id"],
        "revision": recorded["revision"],
        "decoding": recorded["decoding"],
        "dtype": recorded["dtype"],
        "device": recorded["device"],
        "timing_kind": (
            "wall clock observations on a shared CPU, "
            "not a performance result"
        ),
        "prompts": PROMPTS,
        "max_new_tokens": args.max_new_tokens,
        "requests": per_request,
        "engine_steps": engine.step_count,
        "decode_batch_sizes": list(executor.decode_batch_sizes),
        "active_caches_after_run": executor.active_cache_count,
        "released_request_ids": list(executor.released_request_ids),
        "all_checks_passed": bool(
            all_ok
            and executor.active_cache_count == 0
            and sorted(executor.released_request_ids)
            == ["stream-0", "stream-1"]
        ),
    }
    out_path = (
        Path(__file__).resolve().parent / "service_streaming_output.json"
    )
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
