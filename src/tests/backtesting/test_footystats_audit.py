from __future__ import annotations

from pathlib import Path

import pandas as pd

from cartola.backtesting import footystats_audit as audit


def test_parse_footystats_filename_extracts_table_and_year() -> None:
    parsed = audit.parse_footystats_filename(Path("brazil-serie-a-matches-2025-to-2025-stats.csv"))

    assert parsed.league_slug == "brazil-serie-a"
    assert parsed.table_type == "matches"
    assert parsed.start_year == 2025
    assert parsed.end_year == 2025
    assert parsed.season == 2025


def test_parse_footystats_filename_rejects_cross_year_season() -> None:
    try:
        audit.parse_footystats_filename(Path("brazil-serie-a-matches-2024-to-2025-stats.csv"))
    except ValueError as exc:
        assert "single-year seasons" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_discover_footystats_files_ignores_non_csv_and_groups_by_season(tmp_path: Path) -> None:
    footystats_dir = tmp_path / "data" / "footystats"
    footystats_dir.mkdir(parents=True)
    (footystats_dir / ".DS_Store").write_text("ignored", encoding="utf-8")
    (footystats_dir / "brazil-serie-a-matches-2025-to-2025-stats.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (footystats_dir / "brazil-serie-a-teams-2025-to-2025-stats.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    discoveries = audit.discover_footystats_files(audit.FootyStatsAuditConfig(project_root=tmp_path))

    assert len(discoveries) == 1
    assert discoveries[0].season == 2025
    assert sorted(discoveries[0].files) == ["matches", "teams"]


def test_profile_match_file_classifies_pre_match_and_outcome_columns(tmp_path: Path) -> None:
    path = tmp_path / "brazil-serie-a-matches-2025-to-2025-stats.csv"
    pd.DataFrame(
        [
            {
                "status": "complete",
                "Game Week": 1,
                "home_team_name": "Flamengo",
                "away_team_name": "Palmeiras",
                "Pre-Match PPG (Home)": 0.0,
                "Pre-Match PPG (Away)": 0.0,
                "Home Team Pre-Match xG": 0.0,
                "Away Team Pre-Match xG": 0.0,
                "odds_ft_home_team_win": 1.8,
                "home_team_goal_count": 2,
                "away_team_goal_count": 1,
                "team_a_xg": 1.4,
            },
            {
                "status": "complete",
                "Game Week": 2,
                "home_team_name": "Palmeiras",
                "away_team_name": "Flamengo",
                "Pre-Match PPG (Home)": 3.0,
                "Pre-Match PPG (Away)": 0.0,
                "Home Team Pre-Match xG": 1.2,
                "Away Team Pre-Match xG": 0.8,
                "odds_ft_home_team_win": 2.2,
                "home_team_goal_count": 0,
                "away_team_goal_count": 0,
                "team_a_xg": 0.7,
            },
        ]
    ).to_csv(path, index=False)

    profile = audit.profile_match_file(path)

    assert profile.row_count == 2
    assert profile.min_game_week == 1
    assert profile.max_game_week == 2
    assert profile.status_counts == {"complete": 2}
    assert "Pre-Match PPG (Home)" in profile.pre_match_safe_columns
    assert "Home Team Pre-Match xG" in profile.pre_match_safe_columns
    assert "odds_ft_home_team_win" in profile.pre_match_safe_columns
    assert "home_team_goal_count" in profile.post_match_outcome_columns
    assert "team_a_xg" in profile.post_match_outcome_columns
    assert profile.team_names == ["Flamengo", "Palmeiras"]


def test_profile_match_file_only_marks_explicit_plan_columns_as_pre_match_safe(tmp_path: Path) -> None:
    path = tmp_path / "brazil-serie-a-matches-2025-to-2025-stats.csv"
    expected_plan_columns = [
        "over_15_HT_FHG_percentage_pre_match",
        "over_05_HT_FHG_percentage_pre_match",
        "over_15_2HG_percentage_pre_match",
        "over_05_2HG_percentage_pre_match",
        "average_corners_per_match_pre_match",
        "average_cards_per_match_pre_match",
        "odds_ft_over15",
        "odds_ft_over25",
        "odds_ft_over35",
        "odds_ft_over45",
    ]
    pd.DataFrame([{**dict.fromkeys(expected_plan_columns, 1), "odds_ft_unplanned_live": 2}]).to_csv(path, index=False)

    profile = audit.profile_match_file(path)

    assert set(expected_plan_columns).issubset(profile.pre_match_safe_columns)
    assert "odds_ft_unplanned_live" not in profile.pre_match_safe_columns


def test_profile_match_file_excludes_extra_pre_match_fields_and_includes_half_time_goals(tmp_path: Path) -> None:
    path = tmp_path / "brazil-serie-a-matches-2025-to-2025-stats.csv"
    extra_pre_match_columns = [
        "over_05_percentage_pre_match",
        "over_55_percentage_pre_match",
        "over_65_percentage_pre_match",
        "home_team_corner_count_pre_match",
        "away_team_corner_count_pre_match",
        "home_team_cards_pre_match",
        "away_team_cards_pre_match",
    ]
    half_time_goal_columns = [
        "total_goals_at_half_time",
        "home_team_goal_count_half_time",
        "away_team_goal_count_half_time",
    ]
    pd.DataFrame([{**dict.fromkeys(extra_pre_match_columns, 1), **dict.fromkeys(half_time_goal_columns, 1)}]).to_csv(
        path,
        index=False,
    )

    profile = audit.profile_match_file(path)

    assert not set(extra_pre_match_columns).intersection(profile.pre_match_safe_columns)
    assert set(half_time_goal_columns).issubset(profile.post_match_outcome_columns)
