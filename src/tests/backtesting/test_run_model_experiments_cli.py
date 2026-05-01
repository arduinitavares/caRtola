from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_model_experiments import main, parse_args  # noqa: E402


def test_parse_args_defaults() -> None:
    args = parse_args(["--group", "production-parity", "--current-year", "2026"])

    assert args.group == "production-parity"
    assert args.seasons == "2023,2024,2025"
    assert args.start_round == 5
    assert args.budget == 100.0
    assert args.jobs == 1


def test_main_calls_runner(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    def fake_run_model_experiment(**kwargs):
        observed.update(kwargs)

        class Result:
            output_path = tmp_path / "out"
            experiment_id = "exp"

        return Result()

    monkeypatch.setattr("scripts.run_model_experiments.run_model_experiment", fake_run_model_experiment)

    exit_code = main(
        [
            "--group",
            "matchup-research",
            "--seasons",
            "2023,2024",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
            "--output-root",
            "data/08_reporting/experiments/model_feature/test",
            "--jobs",
            "12",
        ]
    )

    assert exit_code == 0
    assert observed["group"] == "matchup-research"
    assert observed["seasons"] == (2023, 2024)
    assert observed["current_year"] == 2026
    assert observed["project_root"] == tmp_path
    assert observed["output_root"] == Path("data/08_reporting/experiments/model_feature/test")
    assert observed["jobs"] == 12


def test_main_rejects_empty_seasons_without_traceback(capsys) -> None:
    exit_code = main(["--group", "production-parity", "--seasons", "", "--current-year", "2026"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "At least one season is required" in captured.err
    assert "Traceback" not in captured.err
