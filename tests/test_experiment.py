import json
from pathlib import Path

import pytest

from miniserve.experiment import ExperimentConfig, run_experiment, summarize_run


def make_config(run_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        profile="short",
        seed=1234,
        num_requests=4,
        policies=("serial", "continuous"),
        run_id=run_id,
    )


def load_summary(run_dir: Path) -> dict[str, object]:
    data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_run_experiment_writes_config_raw_files_and_summary(
    tmp_path: Path,
) -> None:
    run_dir = run_experiment(make_config("run-one"), tmp_path)

    assert run_dir == tmp_path / "run-one"
    for name in (
        "config.json",
        "serial_requests.jsonl",
        "serial_timeline.jsonl",
        "continuous_requests.jsonl",
        "continuous_timeline.jsonl",
        "summary.json",
        "summary.md",
    ):
        assert (run_dir / name).is_file(), name

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["kind"] == "simulation"
    assert config["simulation"] is True
    assert config["run_id"] == "run-one"
    assert config["profile"] == "short"
    assert config["seed"] == 1234
    assert len(config["workload"]) == 4

    summary = load_summary(run_dir)
    policies = summary["policies"]
    assert isinstance(policies, dict)
    assert set(policies) == {"serial", "continuous"}
    output_totals = set()
    for stats in policies.values():
        assert isinstance(stats, dict)
        assert stats["requests"] == 4
        assert stats["finished"] == 4
        assert stats["cancelled"] == 0
        assert stats["unfinished"] == 0
        assert stats["steps"] > 0
        assert stats["simulated_duration"] > 0
        output_totals.add(stats["total_output_tokens"])
    # Both policies generated exactly the same work from one workload.
    assert len(output_totals) == 1

    summary_md = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "simulation" in summary_md
    assert "serial" in summary_md
    assert "continuous" in summary_md


def test_summary_recomputed_from_raw_files_is_identical(tmp_path: Path) -> None:
    run_dir = run_experiment(make_config("run-recompute"), tmp_path)
    first_summary = load_summary(run_dir)
    first_md = (run_dir / "summary.md").read_text(encoding="utf-8")

    recomputed = summarize_run(run_dir)

    assert recomputed == first_summary
    assert load_summary(run_dir) == first_summary
    assert (run_dir / "summary.md").read_text(encoding="utf-8") == first_md


def test_rerun_with_same_run_id_is_refused_and_keeps_first_run(
    tmp_path: Path,
) -> None:
    run_dir = run_experiment(make_config("run-fixed"), tmp_path)
    config_before = (run_dir / "config.json").read_text(encoding="utf-8")
    summary_before = (run_dir / "summary.json").read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_experiment(make_config("run-fixed"), tmp_path)

    assert (run_dir / "config.json").read_text(encoding="utf-8") == config_before
    assert (run_dir / "summary.json").read_text(encoding="utf-8") == summary_before


def test_second_run_id_leaves_first_run_untouched(tmp_path: Path) -> None:
    first_dir = run_experiment(make_config("run-a"), tmp_path)
    summary_before = (first_dir / "summary.json").read_text(encoding="utf-8")

    second_dir = run_experiment(make_config("run-b"), tmp_path)

    assert second_dir != first_dir
    assert (first_dir / "summary.json").read_text(encoding="utf-8") == summary_before
    # Same configuration, so the per policy numbers match exactly;
    # only the run ids differ.
    assert load_summary(first_dir)["policies"] == load_summary(second_dir)[
        "policies"
    ]


def test_unknown_policy_is_rejected_before_any_output(tmp_path: Path) -> None:
    config = ExperimentConfig(policies=("bogus",), run_id="run-bogus")

    with pytest.raises(ValueError, match="Unknown policy"):
        run_experiment(config, tmp_path)

    assert not (tmp_path / "run-bogus").exists()
