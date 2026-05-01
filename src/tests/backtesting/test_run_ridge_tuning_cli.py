from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ridge_tuning import main, parse_args  # noqa: E402


def test_parse_args_defaults() -> None:
    args = parse_args(["--current-year", "2026"])

    assert args.seasons == "2023,2024,2025"
    assert args.start_round == 5
    assert args.budget == 100.0
    assert args.current_year == 2026
    assert args.jobs == 1
    assert args.output_root == Path("data/08_reporting/experiments/model_tuning")
    assert args.skip_final_rerun is False


def test_main_calls_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_ridge_tuning(**kwargs: object) -> object:
        observed.update(kwargs)

        class Result:
            output_path = tmp_path / "out"
            experiment_id = "exp"

        return Result()

    monkeypatch.setattr("scripts.run_ridge_tuning.run_ridge_tuning", fake_run_ridge_tuning)

    exit_code = main(
        [
            "--seasons",
            "2023,2024",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
            "--output-root",
            "data/08_reporting/experiments/model_tuning/test",
            "--jobs",
            "12",
            "--skip-final-rerun",
        ]
    )

    assert exit_code == 0
    assert observed["seasons"] == (2023, 2024)
    assert observed["current_year"] == 2026
    assert observed["project_root"] == tmp_path
    assert observed["output_root"] == Path("data/08_reporting/experiments/model_tuning/test")
    assert observed["jobs"] == 12
    assert observed["skip_final_rerun"] is True
    assert callable(observed["progress_callback"])


def test_main_rejects_empty_seasons_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--seasons", "", "--current-year", "2026"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "At least one season is required" in captured.err
    assert "Traceback" not in captured.err


def test_main_reports_runner_failure_without_traceback(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_run_ridge_tuning(**_kwargs: object) -> NoReturn:
        raise RuntimeError("ridge tuning failed")

    monkeypatch.setattr("scripts.run_ridge_tuning.run_ridge_tuning", fake_run_ridge_tuning)

    exit_code = main(["--current-year", "2026"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ridge tuning failed" in captured.err
    assert "Traceback" not in captured.err
