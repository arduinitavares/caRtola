from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cartola.backtesting.recommendation import RecommendationConfig, RecommendationResult

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "recommend_squad.py"
SPEC = importlib.util.spec_from_file_location("recommend_squad", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
recommend_squad = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recommend_squad)
main = recommend_squad.main
parse_args = recommend_squad.parse_args


def test_parse_args_requires_target_round() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--season", "2026", "--mode", "live"])


def test_parse_args_has_no_fixture_mode(capsys) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--season", "2026", "--target-round", "14", "--mode", "live", "--fixture-mode", "none"])

    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err
    assert "--fixture-mode" in captured.err


def test_parse_args_builds_live_defaults() -> None:
    args = parse_args(["--season", "2026", "--target-round", "14", "--mode", "live", "--current-year", "2026"])

    assert args.season == 2026
    assert args.target_round == 14
    assert args.mode == "live"
    assert args.budget == 100.0
    assert args.footystats_mode == "ppg"
    assert args.output_root == Path("data/08_reporting/recommendations")


def test_main_builds_recommendation_config(monkeypatch, tmp_path: Path, capsys) -> None:
    observed: list[RecommendationConfig] = []

    def fake_run_recommendation(config: RecommendationConfig) -> RecommendationResult:
        observed.append(config)
        return RecommendationResult(
            recommended_squad=None,
            candidate_predictions=None,
            summary={
                "actual_points": 38.25,
                "budget": 100.0,
                "budget_used": 97.55,
                "mode": "replay",
                "output_directory": str(config.output_path),
                "oracle_actual_points": 50.0,
                "oracle_capture_rate": 0.765,
                "oracle_gap": 11.75,
                "oracle_optimizer_status": "Optimal",
                "predicted_points": 42.0,
                "season": 2026,
                "selected_count": 12,
                "target_round": 14,
            },
            metadata={},
        )

    monkeypatch.setattr(recommend_squad, "run_recommendation", fake_run_recommendation)

    exit_code = main(
        [
            "--season",
            "2026",
            "--target-round",
            "14",
            "--mode",
            "replay",
            "--project-root",
            str(tmp_path),
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 0
    assert observed == [
        RecommendationConfig(
            season=2026,
            target_round=14,
            mode="replay",
            project_root=tmp_path,
            current_year=2026,
        )
    ]
    output = capsys.readouterr().out
    assert "Recommendation complete" in output
    assert "Predicted points" in output
    assert "42.00" in output
    assert "Actual points" in output
    assert "38.25" in output
    assert "Delta" in output
    assert "-3.75" in output
    assert "Best in candidate pool" in output
    assert "50.00" in output
    assert "Gap to best" in output
    assert "11.75" in output
    assert "Capture rate" in output
    assert "76.50%" in output
    assert "Budget used" in output


def test_main_prints_live_summary_without_actual_points(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_run_recommendation(config: RecommendationConfig) -> RecommendationResult:
        return RecommendationResult(
            recommended_squad=None,
            candidate_predictions=None,
            summary={
                "actual_points": None,
                "budget": 100.0,
                "budget_used": 99.0,
                "mode": "live",
                "output_directory": str(config.output_path),
                "oracle_actual_points": None,
                "oracle_capture_rate": None,
                "oracle_gap": None,
                "oracle_optimizer_status": None,
                "predicted_points": 51.25,
                "season": 2026,
                "selected_count": 12,
                "target_round": 14,
            },
            metadata={},
        )

    monkeypatch.setattr(recommend_squad, "run_recommendation", fake_run_recommendation)

    exit_code = main(
        [
            "--season",
            "2026",
            "--target-round",
            "14",
            "--mode",
            "live",
            "--project-root",
            str(tmp_path),
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Recommendation complete" in output
    assert "Predicted points" in output
    assert "51.25" in output
    assert "Actual points" in output
    assert "n/a (live mode)" in output
    assert "Best in candidate pool" in output
    assert "Capture rate" in output


def test_main_prints_expected_error_without_traceback(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_run_recommendation(config: RecommendationConfig) -> RecommendationResult:
        raise ValueError("Target round 14 not found in season 2026 data.")

    monkeypatch.setattr(recommend_squad, "run_recommendation", fake_run_recommendation)

    exit_code = main(
        [
            "--season",
            "2026",
            "--target-round",
            "14",
            "--mode",
            "live",
            "--project-root",
            str(tmp_path),
            "--current-year",
            "2026",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Recommendation failed" in captured.err
    assert "Target round 14 not found in season 2026 data." in captured.err
    assert "Save data/01_raw/2026/rodada-14.csv before running live mode." in captured.err
    assert "Traceback" not in captured.err
