from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cartola.backtesting import compatibility_audit as audit


def _touch_round(root: Path, season: int, round_name: str) -> Path:
    path = root / "data" / "01_raw" / str(season) / round_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")
    return path


def _season_frame(rounds: range) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rodada": list(rounds),
            "id_atleta": list(range(1, len(list(rounds)) + 1)),
            "id_clube": [10] * len(list(rounds)),
        }
    )


def _fake_backtest_result(summary: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(summary=summary)


def test_discover_seasons_includes_numeric_dirs_with_round_files(tmp_path: Path) -> None:
    _touch_round(tmp_path, 2025, "rodada-1.csv")
    _touch_round(tmp_path, 2025, "rodada-2.csv")
    _touch_round(tmp_path, 2026, "rodada-1.csv")
    (tmp_path / "data" / "01_raw" / "fixtures" / "2025").mkdir(parents=True)
    (tmp_path / "data" / "01_raw" / "notes").mkdir(parents=True)
    (tmp_path / "data" / "01_raw" / "2024").mkdir(parents=True)

    config = audit.AuditConfig(project_root=tmp_path, current_year=2026)

    seasons = audit.discover_seasons(config)

    assert [season.season for season in seasons] == [2025, 2026]
    assert seasons[0].detected_rounds == [1, 2]
    assert seasons[0].round_file_count == 2
    assert seasons[0].min_round == 1
    assert seasons[0].max_round == 2


def test_discover_seasons_records_malformed_round_filename(tmp_path: Path) -> None:
    _touch_round(tmp_path, 2025, "rodada-final.csv")

    config = audit.AuditConfig(project_root=tmp_path, current_year=2026)

    seasons = audit.discover_seasons(config)

    assert len(seasons) == 1
    assert seasons[0].season == 2025
    assert seasons[0].discovery_error is not None
    assert seasons[0].discovery_error.stage == "discovery"
    assert "Invalid round CSV filename" in seasons[0].discovery_error.message


def test_parse_round_number_rejects_zero_and_non_matching_names() -> None:
    with pytest.raises(ValueError, match="positive"):
        audit.parse_round_number(Path("rodada-0.csv"))

    with pytest.raises(ValueError, match="Invalid round CSV filename"):
        audit.parse_round_number(Path("round-1.csv"))


def test_classify_season_requires_contiguous_complete_rounds() -> None:
    config = audit.AuditConfig(current_year=2026, expected_complete_rounds=38)

    complete = audit.classify_season(2025, list(range(1, 39)), config)
    gapped = audit.classify_season(2025, [*range(1, 7), *range(8, 40)], config)
    irregular_extra = audit.classify_season(2022, list(range(1, 40)), config)
    partial_current = audit.classify_season(2026, list(range(1, 14)), config)

    assert complete == ("complete_historical", True, [])
    assert gapped[0] == "irregular_historical"
    assert gapped[1] is False
    assert irregular_extra[0] == "irregular_historical"
    assert partial_current == (
        "partial_current",
        False,
        ["partial current season; metrics are smoke-test only"],
    )
