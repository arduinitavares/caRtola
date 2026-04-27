from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cartola.backtesting import compatibility_audit as audit
from cartola.backtesting.config import BacktestConfig


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


def test_short_error_message_caps_to_configured_limit() -> None:
    message = audit._short_error_message("x" * 350)

    assert len(message) == audit.CSV_ERROR_MESSAGE_LIMIT


def test_load_failure_records_row_and_skips_later_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch_round(tmp_path, 2025, "rodada-1.csv")

    def fail_load(season: int, project_root: Path) -> pd.DataFrame:
        raise ValueError(f"bad load {season} {project_root}")

    monkeypatch.setattr(audit, "load_season_data", fail_load)

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    record = result.seasons[0]
    assert record.load_status == "failed"
    assert record.feature_status == "skipped"
    assert record.backtest_status == "skipped"
    assert record.error_stage == "load"
    assert record.error_type == "ValueError"
    assert record.error_detail is not None
    assert record.error_detail.stage == "load"


def test_feature_check_covers_every_eligible_target_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 8):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    training_calls: list[int] = []
    prediction_calls: list[int] = []

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 8)))
    monkeypatch.setattr(
        audit,
        "build_training_frame",
        lambda season_df, target_round, playable_statuses, fixtures: training_calls.append(target_round)
        or pd.DataFrame(),
    )
    monkeypatch.setattr(
        audit,
        "build_prediction_frame",
        lambda season_df, target_round, fixtures: prediction_calls.append(target_round) or pd.DataFrame(),
    )
    monkeypatch.setattr(
        audit,
        "run_backtest",
        lambda config: _fake_backtest_result(
            pd.DataFrame(
                [
                    {"strategy": "baseline", "average_actual_points": 1.0},
                    {"strategy": "random_forest", "average_actual_points": 2.0},
                    {"strategy": "price", "average_actual_points": 3.0},
                ]
            )
        ),
    )

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026, start_round=5))

    assert training_calls == [5, 6, 7]
    assert prediction_calls == [5, 6, 7]
    assert result.seasons[0].feature_status == "ok"


def test_feature_failure_records_target_round_and_skips_backtest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for round_number in range(1, 8):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    def fail_training(
        season_df: pd.DataFrame, target_round: int, playable_statuses: tuple[str, ...], fixtures: None
    ) -> pd.DataFrame:
        if target_round == 6:
            raise RuntimeError("feature broke")
        return pd.DataFrame()

    def fail_if_called(config: BacktestConfig) -> SimpleNamespace:
        raise AssertionError("backtest should be skipped")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 8)))
    monkeypatch.setattr(audit, "build_training_frame", fail_training)
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "run_backtest", fail_if_called)

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026, start_round=5))

    record = result.seasons[0]
    assert record.feature_status == "failed"
    assert record.backtest_status == "skipped"
    assert record.error_stage == "feature"
    assert record.error_detail is not None
    assert record.error_detail.target_round == 6


def test_max_round_before_start_round_marks_feature_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for round_number in range(1, 4):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 4)))

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026, start_round=5))

    record = result.seasons[0]
    assert record.load_status == "ok"
    assert record.feature_status == "not_applicable"
    assert record.backtest_status == "skipped"
    assert record.evaluated_rounds == 0
