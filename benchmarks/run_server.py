"""Start the MiniServe HTTP service.

Two modes:

- Default (fake backend): no model download, prompt tokenized
  by word count, generated text is empty because the fake
  executor has no tokenizer. This mode exists so the service
  shape, streaming, cancellation, and overload behaviour can
  be tried with curl in seconds.
- ``--real``: loads the locked model from configs/model.json
  once, serves it through the batched real executor, and
  returns real generated text. Requires the model extra.

Usage, from the repository root:

    python benchmarks/run_server.py --port 8000
    python benchmarks/run_server.py --port 8000 --real

Then, from another terminal:

    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"prompt": "The capital of France is", "max_new_tokens": 8}'
    curl -N -X POST http://127.0.0.1:8000/generate/stream -H 'Content-Type: application/json' -d '{"prompt": "The capital of France is", "max_new_tokens": 8}'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--max-waiting-requests", type=int, default=32)
    args = parser.parse_args()

    from miniserve.engine import Engine
    from miniserve.scheduler import Scheduler
    from miniserve.server import ServiceFrontEnd, make_server, simple_tokenize
    from miniserve.service import EngineOwner

    if args.real:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from miniserve.batched_executor import BatchedRealExecutor

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
            max_batch_size=4,
            max_batch_tokens=int(model.config.max_position_embeddings),
            max_model_len=int(model.config.max_position_embeddings),
        )
        owner = EngineOwner(
            Engine(scheduler, executor),
            tokenizer=tokenizer,
            max_waiting_requests=args.max_waiting_requests,
        )

        def tokenize(prompt: str) -> list[int]:
            result: list[int] = tokenizer(prompt, add_special_tokens=False)[
                "input_ids"
            ]
            return result

        front_end = ServiceFrontEnd(
            owner,
            tokenize,
            model_id=recorded["model_id"],
            revision=recorded["revision"],
        )
        mode = f"real model {recorded['model_id']}"
    else:
        from miniserve.executor import FakeExecutor

        scheduler = Scheduler(max_batch_size=4, max_batch_tokens=100)
        owner = EngineOwner(
            Engine(scheduler, FakeExecutor()),
            max_waiting_requests=args.max_waiting_requests,
        )
        front_end = ServiceFrontEnd(
            owner, simple_tokenize, model_id="fake", revision="none"
        )
        mode = "fake backend (no model, empty generated text)"

    server = make_server(front_end, host=args.host, port=args.port)
    owner.start()
    print(f"MiniServe listening on http://{args.host}:{args.port} ({mode})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        owner.stop(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
