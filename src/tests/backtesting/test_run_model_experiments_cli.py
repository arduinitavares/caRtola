from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import pytest

from cartola.backtesting.experiment_tracking import TrackerWarning

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
    assert args.tracker == "none"
    assert args.mlflow_tracking_uri is None


def test_main_calls_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_model_experiment(**kwargs: object) -> object:
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
    assert callable(observed["progress_callback"])
    assert observed["tracker"].__class__.__name__ == "NoOpExperimentTracker"


def test_main_passes_mlflow_tracker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_model_experiment(**kwargs: object) -> object:
        observed.update(kwargs)

        class Result:
            output_path = tmp_path / "out"
            experiment_id = "exp"

        return Result()

    monkeypatch.setattr("scripts.run_model_experiments.run_model_experiment", fake_run_model_experiment)

    exit_code = main(
        [
            "--group",
            "production-parity",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
            "--tracker",
            "mlflow",
            "--mlflow-tracking-uri",
            "file:///tmp/cartola-mlruns",
        ]
    )

    assert exit_code == 0
    assert observed["tracker"].__class__.__name__ == "MLflowExperimentTracker"
    assert getattr(observed["tracker"], "tracking_uri") == "file:///tmp/cartola-mlruns"


def test_main_prints_all_tracker_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_model_experiment(**kwargs: object) -> object:
        tracker = kwargs["tracker"]
        for index in range(6):
            tracker.warnings.append(TrackerWarning(phase=f"phase-{index}", message=f"warning-{index}"))

        class Result:
            output_path = tmp_path / "out"
            experiment_id = "exp"

        return Result()

    monkeypatch.setattr("scripts.run_model_experiments.run_model_experiment", fake_run_model_experiment)

    exit_code = main(["--group", "production-parity", "--current-year", "2026", "--project-root", str(tmp_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "phase-0: warning-0" in captured.err
    assert "phase-5: warning-5" in captured.err


def test_main_rejects_empty_seasons_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--group", "production-parity", "--seasons", "", "--current-year", "2026"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "At least one season is required" in captured.err
    assert "Traceback" not in captured.err


def test_main_reports_child_run_failure_without_traceback(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_run_model_experiment(**_kwargs: object) -> NoReturn:
        raise RuntimeError("child failed")

    monkeypatch.setattr("scripts.run_model_experiments.run_model_experiment", fake_run_model_experiment)

    exit_code = main(["--group", "production-parity", "--current-year", "2026"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "child failed" in captured.err
    assert "Traceback" not in captured.err
