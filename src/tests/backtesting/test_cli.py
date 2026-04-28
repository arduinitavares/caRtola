from pathlib import Path

import pandas as pd

from cartola.backtesting.cli import main, parse_args
from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.runner import BacktestMetadata, BacktestResult


def test_parse_args_accepts_v1_options():
    args = parse_args(["--season", "2025", "--start-round", "5", "--budget", "100"])

    assert args.season == 2025
    assert args.start_round == 5
    assert args.budget == 100.0


def test_parse_args_uses_v1_defaults():
    args = parse_args([])

    assert args.season == 2025
    assert args.start_round == 5
    assert args.budget == 100.0
    assert args.project_root == Path(".")


def test_cli_parses_fixture_mode_and_alignment_policy() -> None:
    from cartola.backtesting.cli import parse_args

    args = parse_args(
        [
            "--season",
            "2026",
            "--fixture-mode",
            "strict",
            "--strict-alignment-policy",
            "exclude_round",
        ]
    )

    assert args.fixture_mode == "strict"
    assert args.strict_alignment_policy == "exclude_round"


def test_parse_args_accepts_footystats_and_output_options() -> None:
    args = parse_args(
        [
            "--output-root",
            "custom/backtests",
            "--footystats-mode",
            "ppg",
            "--footystats-evaluation-scope",
            "live_current",
            "--footystats-league-slug",
            "england-premier-league",
            "--footystats-dir",
            "custom/footystats",
            "--current-year",
            "2026",
        ]
    )

    assert args.output_root == Path("custom/backtests")
    assert args.footystats_mode == "ppg"
    assert args.footystats_evaluation_scope == "live_current"
    assert args.footystats_league_slug == "england-premier-league"
    assert args.footystats_dir == Path("custom/footystats")
    assert args.current_year == 2026


def test_main_builds_config_and_prints_completion(monkeypatch, capsys, tmp_path):
    observed_configs: list[BacktestConfig] = []

    def fake_run_backtest(config: BacktestConfig) -> BacktestResult:
        observed_configs.append(config)
        return BacktestResult(
            round_results=pd.DataFrame(),
            selected_players=pd.DataFrame(),
            player_predictions=pd.DataFrame(),
            summary=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            metadata=BacktestMetadata(
                season=config.season,
                start_round=config.start_round,
                max_round=0,
                fixture_mode=config.fixture_mode,
                strict_alignment_policy=config.strict_alignment_policy,
                fixture_source_directory=None,
                fixture_manifest_paths=[],
                fixture_manifest_sha256={},
                generator_versions=[],
                excluded_rounds=[],
                warnings=[],
                footystats_mode=config.footystats_mode,
                footystats_evaluation_scope=config.footystats_evaluation_scope,
                footystats_league_slug=config.footystats_league_slug,
                footystats_matches_source_path=None,
                footystats_matches_source_sha256=None,
                footystats_feature_columns=[],
                footystats_missing_join_keys_by_round={},
                footystats_duplicate_join_keys_by_round={},
                footystats_extra_club_rows_by_round={},
            ),
        )

    monkeypatch.setattr("cartola.backtesting.cli.run_backtest", fake_run_backtest)

    exit_code = main(
        [
            "--season",
            "2025",
            "--start-round",
            "5",
            "--budget",
            "100",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert observed_configs == [BacktestConfig(season=2025, start_round=5, budget=100.0, project_root=tmp_path)]
    observed_config = observed_configs[0]
    assert observed_config.output_root == Path("data/08_reporting/backtests")
    assert observed_config.footystats_mode == "none"
    assert observed_config.footystats_evaluation_scope == "historical_candidate"
    assert observed_config.footystats_league_slug == "brazil-serie-a"
    assert observed_config.footystats_dir == Path("data/footystats")
    assert observed_config.current_year is None
    assert "Backtest complete" in capsys.readouterr().out


def test_main_prints_metadata_warnings(monkeypatch, capsys, tmp_path):
    def fake_run_backtest(config: BacktestConfig) -> BacktestResult:
        return BacktestResult(
            round_results=pd.DataFrame(),
            selected_players=pd.DataFrame(),
            player_predictions=pd.DataFrame(),
            summary=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            metadata=BacktestMetadata(
                season=config.season,
                start_round=config.start_round,
                max_round=0,
                fixture_mode=config.fixture_mode,
                strict_alignment_policy=config.strict_alignment_policy,
                fixture_source_directory=None,
                fixture_manifest_paths=[],
                fixture_manifest_sha256={},
                generator_versions=[],
                excluded_rounds=[],
                warnings=["first warning", "second warning"],
                footystats_mode=config.footystats_mode,
                footystats_evaluation_scope=config.footystats_evaluation_scope,
                footystats_league_slug=config.footystats_league_slug,
                footystats_matches_source_path=None,
                footystats_matches_source_sha256=None,
                footystats_feature_columns=[],
                footystats_missing_join_keys_by_round={},
                footystats_duplicate_join_keys_by_round={},
                footystats_extra_club_rows_by_round={},
            ),
        )

    monkeypatch.setattr("cartola.backtesting.cli.run_backtest", fake_run_backtest)

    exit_code = main(["--project-root", str(tmp_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "WARNING: first warning" in output
    assert "WARNING: second warning" in output
    assert "Backtest complete" in output
