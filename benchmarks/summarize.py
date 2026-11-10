"""Recompute the summary of an existing experiment run.

Usage, from the repository root:

    python benchmarks/summarize.py benchmarks/results/<run_id>

The summary is recomputed from the raw config, request, and
timeline files in the run directory only, and summary.json and
summary.md in that directory are rewritten with the result. If the
raw files are unchanged, the recomputed summary is identical to
the one the original run wrote.
"""

import argparse
from pathlib import Path

from miniserve.experiment import summarize_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_run(args.run_dir)
    print((args.run_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
