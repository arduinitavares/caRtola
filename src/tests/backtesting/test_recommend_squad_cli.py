from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from cartola.backtesting.recommendation import RecommendationConfig, RecommendationResult

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
recommend_squad = importlib.import_module("scripts.recommend_squad")
main = recommend_squad.main
parse_args = recommend_squad.parse_args


def test_parse_args_requires_target_round() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--season", "2026", "--mode", "live"])


def test_parse_args_has_no_fixture_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--season", "2026", "--target-round", "14", "--fixture-mode", "none"])


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
            summary={"predicted_points": 42.0, "output_directory": str(config.output_path)},
            metadata={},
        )

    monkeypatch.setattr("scripts.recommend_squad.run_recommendation", fake_run_recommendation)

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
    assert observed == [
        RecommendationConfig(
            season=2026,
            target_round=14,
            mode="live",
            project_root=tmp_path,
            current_year=2026,
        )
    ]
    assert "Recommendation complete" in capsys.readouterr().out
