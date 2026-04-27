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
            ),
        )

    monkeypatch.setattr("cartola.backtesting.cli.run_backtest", fake_run_backtest)

    exit_code = main(["--project-root", str(tmp_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "WARNING: first warning" in output
    assert "WARNING: second warning" in output
    assert "Backtest complete" in output
