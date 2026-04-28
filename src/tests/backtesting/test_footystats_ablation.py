from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cartola.backtesting import footystats_ablation as ablation


def test_parse_seasons_preserves_order_and_rejects_duplicates() -> None:
    assert ablation.parse_seasons("2025,2023,2024") == (2025, 2023, 2024)

    with pytest.raises(ValueError, match="duplicate"):
        ablation.parse_seasons("2023,2024,2023")


@pytest.mark.parametrize("value", ["", "2023,", "2023,,2024", "0", "-2023"])
def test_parse_seasons_rejects_empty_entries_and_non_positive_values(value: str) -> None:
    with pytest.raises(ValueError):
        ablation.parse_seasons(value)


def test_config_from_default_args() -> None:
    config = ablation.config_from_args(ablation.parse_args([]))

    assert config.seasons == (2023, 2024, 2025)
    assert config.start_round == 5
    assert config.budget == 100.0
    assert config.project_root == Path(".")
    assert config.output_root == Path("data/08_reporting/backtests/footystats_ablation")
    assert config.footystats_league_slug == "brazil-serie-a"
    assert config.force is False


def test_parse_args_preserves_duplicate_season_error_message(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        ablation.parse_args(["--seasons", "2023,2023"])

    captured = capsys.readouterr()
    assert "duplicate season" in captured.err


def test_script_imports_main_from_footystats_ablation() -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_footystats_ppg_ablation.py"
    spec = importlib.util.spec_from_file_location("run_footystats_ppg_ablation", script_path)
    assert spec is not None
    assert spec.loader is not None
    script = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(script)

    assert script.main is ablation.main
