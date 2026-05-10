from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_ebm_feature_diagnostic",
        PROJECT_ROOT / "scripts" / "run_ebm_feature_diagnostic.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Expected EBM feature diagnostic CLI module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli_module()


def test_parse_args_defaults() -> None:
    args = cli.parse_args(
        [
            "--experiment-path",
            "data/08_reporting/experiments/model_feature/example",
            "--model-id",
            "xgboost_depth2_l2_heavy",
            "--feature-pack",
            "ppg_xg_matchup",
            "--seasons",
            "2021,2022,2023,2024,2025",
            "--current-year",
            "2026",
        ]
    )

    assert args.experiment_path == Path("data/08_reporting/experiments/model_feature/example")
    assert args.output_root == Path("data/08_reporting/ebm_diagnostics")
    assert args.model_id == "xgboost_depth2_l2_heavy"
    assert args.feature_pack == "ppg_xg_matchup"
    assert args.seasons == (2021, 2022, 2023, 2024, 2025)
    assert args.current_year == 2026
    assert args.fixture_mode == "exploratory"


def test_parse_args_rejects_duplicate_seasons(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(
            [
                "--experiment-path",
                "data/08_reporting/experiments/model_feature/example",
                "--model-id",
                "xgboost_depth2_l2_heavy",
                "--feature-pack",
                "ppg_xg_matchup",
                "--seasons",
                "2021,2022,2022",
                "--current-year",
                "2026",
            ]
        )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Duplicate seasons" in captured.err
    assert "Traceback" not in captured.err


def test_main_success_reports_progress_and_forwards_profile_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        progress_callback = cast("Callable[[str], None]", kwargs["progress_callback"])
        progress_callback("fake validation progress")
        return SimpleNamespace(
            output_path=kwargs["output_path"],
            decision={
                "diagnostic_status": "diagnostic_complete",
                "diagnostic_phase": "metadata_only",
            },
        )

    monkeypatch.setattr(cli, "build_ebm_feature_diagnostic", fake_builder)
    result = cli.main(
        [
            "--experiment-path",
            "data/08_reporting/experiments/model_feature/example",
            "--output-root",
            "data/08_reporting/ebm_diagnostics_test",
            "--model-id",
            "xgboost_depth2_l2_heavy",
            "--feature-pack",
            "ppg_xg_matchup",
            "--seasons",
            "2021,2022,2023",
            "--current-year",
            "2026",
            "--profile-runtime",
        ]
    )

    assert result == 0
    assert observed["profile_runtime"] is True
    assert callable(observed["progress_callback"])
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "EBM diagnostic started" in output
    assert "fake validation progress" in output
    assert "EBM diagnostic complete" in output
    assert "output_path=data/08_reporting/ebm_diagnostics_test/ebm_diagnostic_started_at=" in output
    assert "metadata_only" in output


def test_main_failure_reports_status_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_builder(**_: object) -> SimpleNamespace:
        raise RuntimeError("source validation failed")

    monkeypatch.setattr(cli, "build_ebm_feature_diagnostic", fake_builder)
    result = cli.main(
        [
            "--experiment-path",
            "data/08_reporting/experiments/model_feature/example",
            "--output-root",
            "data/08_reporting/ebm_diagnostics_test",
            "--model-id",
            "xgboost_depth2_l2_heavy",
            "--feature-pack",
            "ppg_xg_matchup",
            "--seasons",
            "2021,2022,2023",
            "--current-year",
            "2026",
        ]
    )

    assert result == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "EBM diagnostic failed" in output or "diagnostic_status=failed" in output
    assert "output_path=data/08_reporting/ebm_diagnostics_test/ebm_diagnostic_started_at=" in output
    assert "source validation failed" in output
    assert "Traceback" not in output
