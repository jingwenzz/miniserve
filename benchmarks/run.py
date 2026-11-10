"""Run one simulation experiment and write its run directory.

Usage, from the repository root:

    python benchmarks/run.py --profile short --seed 1234 \
        --num-requests 6 --run-id simulation-short-seed1234

The run directory lands under benchmarks/results by default and
contains config.json, the raw per request and per step files for
each policy, and a summary that benchmarks/summarize.py can
recompute from those raw files at any time. Every value is a
simulation value in logical seconds, not a measured model time.
"""

import argparse
from pathlib import Path

from miniserve.experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="mixed")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-requests", type=int, default=6)
    parser.add_argument(
        "--policies",
        default="serial,continuous",
        help="Comma separated policy names, for example serial,continuous.",
    )
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--prefill-per-token", type=float, default=0.1)
    parser.add_argument("--decode-per-token", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = tuple(
        policy.strip() for policy in args.policies.split(",") if policy.strip()
    )
    config = ExperimentConfig(
        profile=args.profile,
        seed=args.seed,
        num_requests=args.num_requests,
        policies=policies,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_model_len=args.max_model_len,
        prefill_seconds_per_token=args.prefill_per_token,
        decode_seconds_per_token=args.decode_per_token,
        run_id=args.run_id,
    )
    run_dir = run_experiment(config, args.output_dir)
    print(f"Run directory: {run_dir}")
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
