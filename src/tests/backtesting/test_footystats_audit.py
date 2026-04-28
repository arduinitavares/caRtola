from __future__ import annotations

from pathlib import Path

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
