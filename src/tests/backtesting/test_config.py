from dataclasses import fields
from pathlib import Path

from cartola.backtesting.config import (
    DEFAULT_FORMATIONS,
    DEFAULT_SCOUT_COLUMNS,
    POSITION_ID_TO_CODE,
    STATUS_ID_TO_NAME,
    BacktestConfig,
)


def test_default_config_matches_v1_scope():
    config = BacktestConfig()

    assert config.season == 2025
    assert config.start_round == 5
    assert config.budget == 100.0
    assert config.playable_statuses == ("Provavel",)
    assert config.output_path == Path("data/08_reporting/backtests/2025")


def test_default_formations_are_all_official_cartola_formations() -> None:
    assert DEFAULT_FORMATIONS == {
        "3-4-3": {"gol": 1, "lat": 0, "zag": 3, "mei": 4, "ata": 3, "tec": 1},
        "3-5-2": {"gol": 1, "lat": 0, "zag": 3, "mei": 5, "ata": 2, "tec": 1},
        "4-3-3": {"gol": 1, "lat": 2, "zag": 2, "mei": 3, "ata": 3, "tec": 1},
        "4-4-2": {"gol": 1, "lat": 2, "zag": 2, "mei": 4, "ata": 2, "tec": 1},
        "4-5-1": {"gol": 1, "lat": 2, "zag": 2, "mei": 5, "ata": 1, "tec": 1},
        "5-3-2": {"gol": 1, "lat": 2, "zag": 3, "mei": 3, "ata": 2, "tec": 1},
        "5-4-1": {"gol": 1, "lat": 2, "zag": 3, "mei": 4, "ata": 1, "tec": 1},
    }


def test_backtest_config_has_no_public_fixed_formation_fields() -> None:
    field_names = {field.name for field in fields(BacktestConfig)}
    assert "formation_name" not in field_names
    assert "formations" not in field_names
    assert not hasattr(BacktestConfig(), "selected_formation")


def test_default_mappings_cover_cartola_values():
    assert STATUS_ID_TO_NAME[7] == "Provavel"
    assert STATUS_ID_TO_NAME[2] == "Duvida"
    assert POSITION_ID_TO_CODE[1] == "gol"
    assert POSITION_ID_TO_CODE[2] == "lat"
    assert POSITION_ID_TO_CODE[3] == "zag"
    assert POSITION_ID_TO_CODE[4] == "mei"
    assert POSITION_ID_TO_CODE[5] == "ata"
    assert POSITION_ID_TO_CODE[6] == "tec"


def test_default_scout_columns_include_v():
    assert "V" in DEFAULT_SCOUT_COLUMNS
    assert {"G", "A", "DS", "SG", "CA", "FC"}.issubset(DEFAULT_SCOUT_COLUMNS)


def test_backtest_config_defaults_to_no_fixture_mode() -> None:
    from cartola.backtesting.config import BacktestConfig

    config = BacktestConfig()

    assert config.fixture_mode == "none"
    assert config.strict_alignment_policy == "fail"


def test_backtest_config_accepts_fixture_modes() -> None:
    from cartola.backtesting.config import BacktestConfig

    assert BacktestConfig(fixture_mode="exploratory").fixture_mode == "exploratory"
    assert BacktestConfig(fixture_mode="strict").fixture_mode == "strict"
    assert BacktestConfig(strict_alignment_policy="exclude_round").strict_alignment_policy == "exclude_round"


def test_backtest_config_accepts_ppg_xg_footystats_mode() -> None:
    config = BacktestConfig(footystats_mode="ppg_xg")

    assert config.footystats_mode == "ppg_xg"
