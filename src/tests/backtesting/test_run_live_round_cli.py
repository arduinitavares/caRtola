from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cartola.backtesting.live_workflow import LiveWorkflowConfig, LiveWorkflowResult

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_live_round.py"
SPEC = importlib.util.spec_from_file_location("run_live_round", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_live_round_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_live_round_cli)
main = run_live_round_cli.main
parse_args = run_live_round_cli.parse_args


def test_parse_args_builds_live_workflow_defaults() -> None:
    args = parse_args(["--season", "2026", "--current-year", "2026"])

    assert args.season == 2026
    assert args.budget == 100.0
    assert args.footystats_mode == "ppg"
    assert args.capture_policy == "fresh"
    assert args.output_root == Path("data/08_reporting/recommendations")


def test_parse_args_rejects_target_round() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--season", "2026", "--target-round", "14", "--current-year", "2026"])


def test_main_builds_workflow_config_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[LiveWorkflowConfig] = []

    def fake_run_live_round(config: LiveWorkflowConfig) -> LiveWorkflowResult:
        observed.append(config)
        return LiveWorkflowResult(
            recommendation=None,
            output_path=tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=x",
            workflow_metadata={
                "status": "ok",
                "capture_policy": "fresh",
                "season": 2026,
                "target_round": 14,
                "capture_captured_at_utc": "2026-04-29T12:00:00Z",
                "capture_age_seconds": 300.0,
                "selected_count": 12,
                "predicted_points": 73.25,
                "budget_used": 99.5,
                "recommendation_output_path": str(
                    tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=x"
                ),
                "capture_metadata_path": str(tmp_path / "data/01_raw/2026/rodada-14.capture.json"),
                "footystats_mode": "ppg",
            },
        )

    monkeypatch.setattr(run_live_round_cli, "run_live_round", fake_run_live_round)

    exit_code = main(["--season", "2026", "--project-root", str(tmp_path), "--current-year", "2026"])

    assert exit_code == 0
    assert observed == [LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026)]
    output = capsys.readouterr().out
    assert "Live round complete" in output
    assert "Capture policy" in output
    assert "fresh" in output
    assert "Target round" in output
    assert "14" in output
    assert "Predicted points" in output
    assert "73.25" in output
    assert "FootyStats mode" in output
    assert "ppg" in output


def test_main_prints_expected_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_live_round(config: LiveWorkflowConfig) -> LiveWorkflowResult:
        raise ValueError("live workflow requires season 2025 to equal current_year 2026")

    monkeypatch.setattr(run_live_round_cli, "run_live_round", fake_run_live_round)

    exit_code = main(["--season", "2025", "--project-root", str(tmp_path), "--current-year", "2026"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Live round failed" in captured.err
    assert "current_year 2026" in captured.err
    assert "Traceback" not in captured.err


def test_main_prints_missing_skip_capture_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_live_round(config: LiveWorkflowConfig) -> LiveWorkflowResult:
        raise FileNotFoundError("live capture files missing for season=2026 target_round=14")

    monkeypatch.setattr(run_live_round_cli, "run_live_round", fake_run_live_round)

    exit_code = main(
        [
            "--season",
            "2026",
            "--project-root",
            str(tmp_path),
            "--current-year",
            "2026",
            "--capture-policy",
            "skip",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Live round failed" in captured.err
    assert "live capture files missing" in captured.err
    assert "Traceback" not in captured.err
