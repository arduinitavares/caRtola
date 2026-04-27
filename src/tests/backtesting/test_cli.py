from pathlib import Path

from cartola.backtesting.cli import main, parse_args
from cartola.backtesting.config import BacktestConfig


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

    def fake_run_backtest(config: BacktestConfig) -> None:
        observed_configs.append(config)

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
