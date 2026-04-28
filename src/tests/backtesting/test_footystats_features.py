from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.footystats_features import (
    FOOTYSTATS_XG_SOURCE_COLUMNS,
    REQUIRED_MATCH_COLUMNS,
    build_footystats_join_diagnostics,
    load_footystats_feature_rows,
    load_footystats_ppg_rows,
    merge_footystats_ppg,
)

SEASON = 2025
LEAGUE_SLUG = "brazil-serie-a"


def test_load_footystats_ppg_rows_builds_home_away_rows_without_outcomes(tmp_path: Path) -> None:
    source_path = _write_matches_csv(
        tmp_path,
        [
            _match_row(
                week=week,
                home="Flamengo" if week % 2 else "Palmeiras",
                away="Palmeiras" if week % 2 else "Flamengo",
                home_ppg=1.25 if week == 1 else 2.0,
                away_ppg=0.75 if week == 1 else 1.0,
            )
            for week in range(1, 39)
        ],
    )
    _write_cartola_round(tmp_path)

    result = load_footystats_ppg_rows(
        season=SEASON,
        project_root=tmp_path,
        footystats_dir=Path("data/footystats"),
        league_slug=LEAGUE_SLUG,
        evaluation_scope="historical_candidate",
        current_year=None,
    )

    assert result.source_path == source_path
    assert result.source_sha256
    assert len(result.rows) == 76
    assert "home_team_goal_count" not in result.rows.columns
    assert pd.api.types.is_integer_dtype(result.rows["is_home_footystats"])

    home_row = result.rows[(result.rows["rodada"] == 1) & (result.rows["id_clube"] == 262)].iloc[0]
    away_row = result.rows[(result.rows["rodada"] == 1) & (result.rows["id_clube"] == 275)].iloc[0]
    assert home_row.to_dict() == {
        "rodada": 1,
        "id_clube": 262,
        "opponent_id_clube": 275,
        "is_home_footystats": 1,
        "footystats_team_pre_match_ppg": 1.25,
        "footystats_opponent_pre_match_ppg": 0.75,
        "footystats_ppg_diff": 0.5,
    }
    assert away_row["is_home_footystats"] == 0
    assert away_row["footystats_team_pre_match_ppg"] == 0.75
    assert away_row["footystats_opponent_pre_match_ppg"] == 1.25
    assert away_row["footystats_ppg_diff"] == -0.5


def test_load_footystats_ppg_rows_reads_only_required_safe_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = _write_matches_csv(tmp_path, [_match_row(week=week) for week in range(1, 39)])
    _write_cartola_round(tmp_path)
    original_read_csv = pd.read_csv
    footystats_usecols: list[object] = []

    def capture_read_csv(*args, **kwargs):
        if args and Path(args[0]) == source_path:
            footystats_usecols.append(kwargs.get("usecols"))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", capture_read_csv)

    _load_historical(tmp_path)

    assert footystats_usecols == [None, list(REQUIRED_MATCH_COLUMNS)]


def test_load_footystats_feature_rows_ppg_xg_builds_xg_features(tmp_path: Path) -> None:
    _write_matches_csv(
        tmp_path,
        [
            _match_row(
                week=week,
                home_xg=1.8 if week == 1 else 2.1,
                away_xg=0.7 if week == 1 else 1.0,
            )
            for week in range(1, 39)
        ],
    )
    _write_cartola_round(tmp_path)

    result = _load_historical(tmp_path, footystats_mode="ppg_xg")

    assert result.footystats_mode == "ppg_xg"
    assert "team_a_xg" not in result.rows.columns
    assert "team_b_xg" not in result.rows.columns
    home_row = result.rows[(result.rows["rodada"] == 1) & (result.rows["id_clube"] == 262)].iloc[0]
    away_row = result.rows[(result.rows["rodada"] == 1) & (result.rows["id_clube"] == 275)].iloc[0]
    assert home_row["footystats_team_pre_match_xg"] == 1.8
    assert home_row["footystats_opponent_pre_match_xg"] == 0.7
    assert home_row["footystats_xg_diff"] == pytest.approx(1.1)
    assert away_row["footystats_team_pre_match_xg"] == 0.7
    assert away_row["footystats_opponent_pre_match_xg"] == 1.8
    assert away_row["footystats_xg_diff"] == pytest.approx(-1.1)


def test_load_footystats_feature_rows_ppg_xg_reads_only_required_safe_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _write_matches_csv(tmp_path, [_match_row(week=week) for week in range(1, 39)])
    _write_cartola_round(tmp_path)
    original_read_csv = pd.read_csv
    footystats_usecols: list[object] = []

    def capture_read_csv(*args, **kwargs):
        if args and Path(args[0]) == source_path:
            footystats_usecols.append(kwargs.get("usecols"))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", capture_read_csv)

    _load_historical(tmp_path, footystats_mode="ppg_xg")

    assert footystats_usecols == [None, [*REQUIRED_MATCH_COLUMNS, *FOOTYSTATS_XG_SOURCE_COLUMNS]]


def test_load_footystats_feature_rows_ppg_does_not_require_xg_columns(tmp_path: Path) -> None:
    _write_matches_csv(
        tmp_path,
        [_match_row(week=week) for week in range(1, 39)],
        drop_columns=list(FOOTYSTATS_XG_SOURCE_COLUMNS),
    )
    _write_cartola_round(tmp_path)

    result = _load_historical(tmp_path, footystats_mode="ppg")

    assert result.footystats_mode == "ppg"
    assert "footystats_team_pre_match_xg" not in result.rows.columns


def test_load_footystats_feature_rows_ppg_xg_rejects_missing_xg_column(tmp_path: Path) -> None:
    _write_matches_csv(
        tmp_path,
        [_match_row(week=week) for week in range(1, 39)],
        drop_columns=["Away Team Pre-Match xG"],
    )
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="missing required columns.*Away Team Pre-Match xG"):
        _load_historical(tmp_path, footystats_mode="ppg_xg")


def test_load_footystats_feature_rows_ppg_xg_rejects_missing_xg_value(tmp_path: Path) -> None:
    rows = [_match_row(week=week) for week in range(1, 39)]
    rows[0]["Home Team Pre-Match xG"] = None
    _write_matches_csv(tmp_path, rows)
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="missing or non-numeric xG values.*Home Team Pre-Match xG"):
        _load_historical(tmp_path, footystats_mode="ppg_xg")


def test_load_footystats_ppg_rows_rejects_missing_required_ppg_column(tmp_path: Path) -> None:
    _write_matches_csv(tmp_path, [_match_row(week=week) for week in range(1, 39)], drop_columns=["Pre-Match PPG (Away)"])
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="missing required columns.*Pre-Match PPG \\(Away\\)"):
        _load_historical(tmp_path)


def test_load_footystats_ppg_rows_rejects_missing_ppg_value(tmp_path: Path) -> None:
    rows = [_match_row(week=week) for week in range(1, 39)]
    rows[0]["Pre-Match PPG (Home)"] = None
    _write_matches_csv(tmp_path, rows)
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="missing or non-numeric PPG values.*Pre-Match PPG \\(Home\\)"):
        _load_historical(tmp_path)


def test_load_footystats_ppg_rows_rejects_historical_non_complete_status(tmp_path: Path) -> None:
    rows = [_match_row(week=week) for week in range(1, 39)]
    rows[0]["status"] = "incomplete"
    _write_matches_csv(tmp_path, rows)
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="historical_candidate requires all statuses to be complete"):
        _load_historical(tmp_path)


def test_load_footystats_ppg_rows_rejects_historical_incomplete_game_week_coverage(tmp_path: Path) -> None:
    _write_matches_csv(tmp_path, [_match_row(week=week) for week in range(1, 38)])
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="game-week coverage 1\\.\\.38"):
        _load_historical(tmp_path)


def test_load_footystats_ppg_rows_rejects_live_current_for_non_current_year(tmp_path: Path) -> None:
    _write_matches_csv(tmp_path, [_match_row(week=1, status="incomplete")])
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="live_current requires season 2025 to equal current_year 2026"):
        load_footystats_ppg_rows(
            season=SEASON,
            project_root=tmp_path,
            footystats_dir=Path("data/footystats"),
            league_slug=LEAGUE_SLUG,
            evaluation_scope="live_current",
            current_year=2026,
        )


def test_load_footystats_ppg_rows_rejects_duplicate_normalized_round_club_rows(tmp_path: Path) -> None:
    rows = [_match_row(week=week) for week in range(1, 39)]
    rows.append(_match_row(week=1, home="Flamengo", away="Palmeiras"))
    _write_matches_csv(tmp_path, rows)
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="duplicate normalized \\(rodada, id_clube\\)"):
        _load_historical(tmp_path)


def test_load_footystats_ppg_rows_rejects_invalid_game_week_values(tmp_path: Path) -> None:
    rows = [_match_row(week=week) for week in range(1, 39)]
    rows[0]["Game Week"] = 1.5
    _write_matches_csv(tmp_path, rows)
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="positive integer Game Week"):
        _load_historical(tmp_path)


def test_load_footystats_ppg_rows_rejects_missing_team_names(tmp_path: Path) -> None:
    rows = [_match_row(week=week) for week in range(1, 39)]
    rows[0]["home_team_name"] = None
    _write_matches_csv(tmp_path, rows)
    _write_cartola_round(tmp_path)

    with pytest.raises(ValueError, match="missing team names.*home_team_name"):
        _load_historical(tmp_path)


def test_build_footystats_join_diagnostics_reports_missing_keys() -> None:
    season_df = pd.DataFrame(
        [
            {"rodada": 1, "id_clube": 10},
            {"rodada": 1, "id_clube": 20},
        ]
    )
    footystats_rows = pd.DataFrame([{"rodada": 1, "id_clube": 10}])

    diagnostics = build_footystats_join_diagnostics(season_df, footystats_rows)

    assert diagnostics.missing_join_keys_by_round == {"1": [{"rodada": 1, "id_clube": 20}]}
    assert diagnostics.duplicate_join_keys_by_round == {}
    assert diagnostics.extra_club_rows_by_round == {}


def test_build_footystats_join_diagnostics_ignores_rows_without_club_identity() -> None:
    season_df = pd.DataFrame(
        [
            {"rodada": 18, "id_clube": 10, "nome_clube": "Club 10"},
            {"rodada": 18, "id_clube": 1, "nome_clube": None},
        ]
    )
    footystats_rows = pd.DataFrame([{"rodada": 18, "id_clube": 10}])

    diagnostics = build_footystats_join_diagnostics(season_df, footystats_rows)

    assert diagnostics.missing_join_keys_by_round == {}
    assert diagnostics.duplicate_join_keys_by_round == {}
    assert diagnostics.extra_club_rows_by_round == {}


def test_merge_footystats_ppg_ignores_rows_without_club_identity() -> None:
    frame = pd.DataFrame(
        [
            {"rodada": 18, "id_clube": 10, "nome_clube": "Club 10"},
            {"rodada": 18, "id_clube": 1, "nome_clube": None},
        ]
    )
    footystats_rows = pd.DataFrame(
        [
            {
                "rodada": 18,
                "id_clube": 10,
                "footystats_team_pre_match_ppg": 1.5,
                "footystats_opponent_pre_match_ppg": 1.0,
                "footystats_ppg_diff": 0.5,
            }
        ]
    )

    result = merge_footystats_ppg(frame, footystats_rows, target_round=18)

    real_club = result.loc[result["id_clube"].eq(10)].iloc[0]
    missing_club = result.loc[result["id_clube"].eq(1)].iloc[0]
    assert real_club["footystats_team_pre_match_ppg"] == 1.5
    assert pd.isna(missing_club["footystats_team_pre_match_ppg"])


def test_build_footystats_join_diagnostics_reports_duplicate_keys() -> None:
    season_df = pd.DataFrame([{"rodada": 2, "id_clube": 10}])
    footystats_rows = pd.DataFrame(
        [
            {"rodada": 2, "id_clube": 10},
            {"rodada": 2, "id_clube": 10},
        ]
    )

    diagnostics = build_footystats_join_diagnostics(season_df, footystats_rows)

    assert diagnostics.missing_join_keys_by_round == {}
    assert diagnostics.duplicate_join_keys_by_round == {"2": [{"rodada": 2, "id_clube": 10, "count": 2}]}
    assert diagnostics.extra_club_rows_by_round == {}


def test_build_footystats_join_diagnostics_reports_extra_club_rows() -> None:
    season_df = pd.DataFrame([{"rodada": 3, "id_clube": 10}])
    footystats_rows = pd.DataFrame(
        [
            {"rodada": 3, "id_clube": 10},
            {"rodada": 3, "id_clube": 20},
        ]
    )

    diagnostics = build_footystats_join_diagnostics(season_df, footystats_rows)

    assert diagnostics.missing_join_keys_by_round == {}
    assert diagnostics.duplicate_join_keys_by_round == {}
    assert diagnostics.extra_club_rows_by_round == {"3": [{"rodada": 3, "id_clube": 20}]}


def _load_historical(project_root: Path, *, footystats_mode: str = "ppg"):
    if footystats_mode == "ppg":
        return load_footystats_ppg_rows(
            season=SEASON,
            project_root=project_root,
            footystats_dir=Path("data/footystats"),
            league_slug=LEAGUE_SLUG,
            evaluation_scope="historical_candidate",
            current_year=None,
        )
    return load_footystats_feature_rows(
        season=SEASON,
        project_root=project_root,
        footystats_dir=Path("data/footystats"),
        league_slug=LEAGUE_SLUG,
        evaluation_scope="historical_candidate",
        current_year=None,
        footystats_mode=footystats_mode,
    )


def _match_row(
    *,
    week: int,
    home: str = "Flamengo",
    away: str = "Palmeiras",
    home_ppg: float = 1.0,
    away_ppg: float = 2.0,
    home_xg: float = 1.4,
    away_xg: float = 0.8,
    status: str = "complete",
) -> dict[str, object]:
    return {
        "Game Week": week,
        "home_team_name": home,
        "away_team_name": away,
        "Pre-Match PPG (Home)": home_ppg,
        "Pre-Match PPG (Away)": away_ppg,
        "Home Team Pre-Match xG": home_xg,
        "Away Team Pre-Match xG": away_xg,
        "status": status,
        "home_team_goal_count": 3,
        "team_a_xg": 2.4,
        "team_b_xg": 0.6,
    }


def _write_matches_csv(
    project_root: Path,
    rows: list[dict[str, object]],
    *,
    drop_columns: list[str] | None = None,
) -> Path:
    footystats_dir = project_root / "data" / "footystats"
    footystats_dir.mkdir(parents=True)
    source_path = footystats_dir / f"{LEAGUE_SLUG}-matches-{SEASON}-to-{SEASON}-stats.csv"
    df = pd.DataFrame(rows)
    if drop_columns:
        df = df.drop(columns=drop_columns)
    df.to_csv(source_path, index=False)
    return source_path


def _write_cartola_round(project_root: Path) -> None:
    season_dir = project_root / "data" / "01_raw" / str(SEASON)
    season_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "atletas.clube_id": [262, 275],
            "atletas.clube.id.full.name": ["FLA", "PAL"],
        }
    ).to_csv(season_dir / "rodada-1.csv", index=False)
