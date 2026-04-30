from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting import matchup_fixture_audit as audit


def test_parse_seasons_accepts_comma_separated_positive_unique_values() -> None:
    assert audit.parse_seasons("2023,2024,2025") == (2023, 2024, 2025)


@pytest.mark.parametrize("value", ["", "2023,", "2023,2023", "0", "-2024", "abc"])
def test_parse_seasons_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        audit.parse_seasons(value)


def test_parse_args_defaults_to_requested_historical_gate() -> None:
    args = audit.parse_args(["--current-year", "2026"])

    assert args.seasons == (2023, 2024, 2025)
    assert args.current_year == 2026
    assert args.project_root == Path(".")
    assert args.output_root == Path("data/08_reporting/fixtures")
    assert args.expected_complete_rounds == 38
    assert args.complete_round_threshold == 38


def test_config_from_args_preserves_values() -> None:
    args = audit.parse_args(
        [
            "--seasons",
            "2024,2025",
            "--current-year",
            "2026",
            "--project-root",
            "/tmp/cartola",
            "--output-root",
            "/tmp/reports",
            "--expected-complete-rounds",
            "30",
            "--complete-round-threshold",
            "20",
        ]
    )

    config = audit.config_from_args(args)

    assert config.seasons == (2024, 2025)
    assert config.current_year == 2026
    assert config.project_root == Path("/tmp/cartola")
    assert config.output_root == Path("/tmp/reports")
    assert config.expected_complete_rounds == 30
    assert config.complete_round_threshold == 20


def test_fixture_context_rows_expand_home_and_away_with_exploratory_source(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "data/01_raw/fixtures/2025"
    fixture_dir.mkdir(parents=True)
    fixture_path = fixture_dir / "partidas-1.csv"
    fixture_path.write_text(
        "rodada,id_clube_home,id_clube_away,data\n"
        "1,10,20,2025-04-01\n",
        encoding="utf-8",
    )

    rows = audit.load_round_fixture_context(
        project_root=tmp_path,
        season=2025,
        round_number=1,
    )

    assert rows[["rodada", "id_clube", "opponent_id_clube", "is_home", "fixture_source"]].to_dict("records") == [
        {"rodada": 1, "id_clube": 10, "opponent_id_clube": 20, "is_home": 1, "fixture_source": "exploratory"},
        {"rodada": 1, "id_clube": 20, "opponent_id_clube": 10, "is_home": 0, "fixture_source": "exploratory"},
    ]
    assert rows["source_file"].tolist() == ["data/01_raw/fixtures/2025/partidas-1.csv"] * 2
    assert rows["source_manifest"].isna().all()
    assert rows["source_sha256"].notna().all()


def test_fixture_context_prefers_strict_when_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strict_dir = tmp_path / "data/01_raw/fixtures_strict/2025"
    exploratory_dir = tmp_path / "data/01_raw/fixtures/2025"
    strict_dir.mkdir(parents=True)
    exploratory_dir.mkdir(parents=True)
    strict_path = strict_dir / "partidas-1.csv"
    manifest_path = strict_dir / "partidas-1.manifest.json"
    strict_path.write_text("rodada,id_clube_home,id_clube_away,data\n1,10,20,2025-04-01\n", encoding="utf-8")
    manifest_path.write_text('{"mode":"strict"}\n', encoding="utf-8")
    (exploratory_dir / "partidas-1.csv").write_text(
        "rodada,id_clube_home,id_clube_away,data\n1,30,40,2025-04-01\n",
        encoding="utf-8",
    )

    calls: list[Path] = []

    def fake_validate_strict_manifest(
        *, project_root: Path, fixture_path: Path, season: int, round_number: int
    ) -> object:
        calls.append(fixture_path)
        return object()

    monkeypatch.setattr(audit, "validate_strict_manifest", fake_validate_strict_manifest)

    rows = audit.load_round_fixture_context(tmp_path, season=2025, round_number=1)

    assert calls == [strict_path]
    assert rows["fixture_source"].unique().tolist() == ["strict"]
    assert rows["id_clube"].tolist() == [10, 20]
    assert rows["source_manifest"].tolist() == ["data/01_raw/fixtures_strict/2025/partidas-1.manifest.json"] * 2


def test_classify_complete_historical_and_partial_current() -> None:
    config = audit.MatchupFixtureAuditConfig(current_year=2026)

    assert audit.classify_season(2025, list(range(1, 39)), config) == ("complete_historical", True, [])
    assert audit.classify_season(2026, list(range(1, 14)), config) == (
        "partial_current",
        False,
        ["partial current season; matchup coverage is not historically comparable"],
    )
    irregular = audit.classify_season(2024, [1, 2, 4], config)
    assert irregular[0] == "irregular_historical"
    assert irregular[1] is False


def test_audit_season_passes_when_every_played_club_has_one_context_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = pd.DataFrame(
        [
            _context_row(2025, 1, 10, 20, 1, "f1"),
            _context_row(2025, 1, 20, 10, 0, "f1"),
            _context_row(2025, 2, 10, 30, 1, "f2"),
            _context_row(2025, 2, 30, 10, 0, "f2"),
        ]
    )

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_df())
    monkeypatch.setattr(
        audit,
        "load_round_fixture_context",
        lambda project_root, *, season, round_number: context[context["rodada"].eq(round_number)].copy(),
    )

    record = audit.audit_one_season(
        2025,
        audit.MatchupFixtureAuditConfig(project_root=tmp_path, current_year=2026, expected_complete_rounds=2),
    )

    assert record.fixture_status == "ok"
    assert record.metrics_comparable is True
    assert record.expected_club_round_count == 4
    assert record.fixture_context_row_count == 4
    assert record.missing_context_count == 0
    assert record.duplicate_context_count == 0
    assert record.extra_context_count == 0


def test_audit_season_populates_opponent_names_from_cartola_clubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = pd.DataFrame(
        [
            _context_row(2025, 1, 10, 20, 1, "f1"),
            _context_row(2025, 1, 20, 10, 0, "f1"),
        ]
    )

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_df().query("rodada == 1"))
    monkeypatch.setattr(
        audit,
        "load_round_fixture_context",
        lambda project_root, *, season, round_number: context[context["rodada"].eq(round_number)].copy(),
    )

    record = audit.audit_one_season(
        2025,
        audit.MatchupFixtureAuditConfig(project_root=tmp_path, current_year=2026, expected_complete_rounds=1),
    )

    assert record.fixture_context_rows == [
        {
            "season": 2025,
            "rodada": 1,
            "id_clube": 10,
            "opponent_id_clube": 20,
            "opponent_nome_clube": "B",
            "is_home": 1,
            "fixture_source": "exploratory",
            "source_file": "f1",
            "source_manifest": None,
            "source_sha256": "a",
            "source_manifest_sha256": None,
        },
        {
            "season": 2025,
            "rodada": 1,
            "id_clube": 20,
            "opponent_id_clube": 10,
            "opponent_nome_clube": "A",
            "is_home": 0,
            "fixture_source": "exploratory",
            "source_file": "f1",
            "source_manifest": None,
            "source_sha256": "a",
            "source_manifest_sha256": None,
        },
    ]
    assert record.to_json_object()["fixture_context_rows"][0]["opponent_nome_clube"] == "B"


def test_audit_season_fails_missing_duplicate_and_extra_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = pd.DataFrame(
        [
            _context_row(2025, 1, 10, 20, 1, "f1"),
            _context_row(2025, 1, 10, 20, 1, "f1"),
            _context_row(2025, 1, 99, 20, 0, "f1"),
        ]
    )

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_df())
    monkeypatch.setattr(
        audit,
        "load_round_fixture_context",
        lambda project_root, *, season, round_number: context[context["rodada"].eq(round_number)].copy(),
    )

    record = audit.audit_one_season(2025, audit.MatchupFixtureAuditConfig(project_root=tmp_path, current_year=2026))

    assert record.fixture_status == "failed"
    assert record.metrics_comparable is False
    assert record.missing_context_count == 3
    assert record.duplicate_context_count == 1
    assert record.extra_context_count == 1
    assert {"rodada": 1, "id_clube": 20} in record.missing_context_keys
    assert {"rodada": 1, "id_clube": 10, "count": 2} in record.duplicate_context_keys
    assert {"rodada": 1, "id_clube": 99} in record.extra_context_keys


def test_decision_ready_when_2023_2024_2025_are_comparable() -> None:
    decision = audit.build_decision(
        [_record(2023, comparable=True), _record(2024, comparable=True), _record(2025, comparable=True)]
    )

    assert decision["status"] == "ready_for_matchup_context"
    assert decision["recommended_next_step"] == "implement matchup_context_mode=cartola_matchup_v1"


def test_decision_exploratory_only_when_only_2025_passes() -> None:
    decision = audit.build_decision(
        [_record(2023, comparable=False), _record(2024, comparable=False), _record(2025, comparable=True)]
    )

    assert decision["status"] == "exploratory_only"
    assert decision["recommended_next_step"] == "keep matchup context exploratory until 2023-2024 fixture coverage is fixed"


def test_decision_blocks_when_2025_fails() -> None:
    decision = audit.build_decision(
        [_record(2023, comparable=True), _record(2024, comparable=True), _record(2025, comparable=False)]
    )

    assert decision["status"] == "coverage_blocked"
    assert decision["recommended_next_step"] == "fix or import fixture coverage before feature work"


def test_write_reports_outputs_csv_and_json(tmp_path: Path) -> None:
    config = audit.MatchupFixtureAuditConfig(project_root=tmp_path, current_year=2026)
    records = [_record(2025, comparable=True)]

    csv_path, json_path = audit.write_audit_reports(
        records,
        config,
        generated_at_utc="2026-04-30T00:00:00+00:00",
    )

    csv_frame = pd.read_csv(csv_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert csv_path == tmp_path / "data/08_reporting/fixtures/matchup_fixture_coverage.csv"
    assert json_path == tmp_path / "data/08_reporting/fixtures/matchup_fixture_coverage.json"
    assert csv_frame.loc[0, "season"] == 2025
    assert csv_frame.loc[0, "fixture_status"] == "ok"
    assert payload["config"]["current_year"] == 2026
    assert payload["decision"]["status"] == "exploratory_only"
    assert payload["seasons"][0]["season"] == 2025


def test_run_matchup_fixture_audit_audits_requested_seasons_and_writes_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audited_seasons: list[int] = []

    def fake_audit_one_season(season: int, config: audit.MatchupFixtureAuditConfig) -> audit.SeasonMatchupFixtureRecord:
        audited_seasons.append(season)
        assert config.current_year == 2026
        return _record(season, comparable=season == 2025)

    monkeypatch.setattr(audit, "audit_one_season", fake_audit_one_season)

    result = audit.run_matchup_fixture_audit(
        audit.MatchupFixtureAuditConfig(seasons=(2024, 2025), project_root=tmp_path),
        clock=lambda: datetime(2026, 4, 30, tzinfo=UTC),
    )

    assert audited_seasons == [2024, 2025]
    assert result.config.current_year == 2026
    assert result.csv_path == tmp_path / "data/08_reporting/fixtures/matchup_fixture_coverage.csv"
    assert result.json_path == tmp_path / "data/08_reporting/fixtures/matchup_fixture_coverage.json"
    assert result.decision["status"] == "exploratory_only"
    assert result.records[1].season == 2025


def test_main_prints_success_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_matchup_fixture_audit(
        config: audit.MatchupFixtureAuditConfig,
    ) -> audit.MatchupFixtureAuditRunResult:
        assert config.seasons == (2025,)
        return audit.MatchupFixtureAuditRunResult(
            config=config,
            records=[_record(2025, comparable=True)],
            csv_path=tmp_path / "matchup_fixture_coverage.csv",
            json_path=tmp_path / "matchup_fixture_coverage.json",
            decision={"status": "exploratory_only", "recommended_next_step": "keep testing"},
        )

    monkeypatch.setattr(audit, "run_matchup_fixture_audit", fake_run_matchup_fixture_audit)

    assert audit.main(["--seasons", "2025", "--current-year", "2026"]) == 0

    captured = capsys.readouterr()
    assert "Matchup fixture coverage audit complete" in captured.out
    assert "CSV: " in captured.out
    assert "matchup_fixture_coverage.csv" in captured.out
    assert "JSON: " in captured.out
    assert "matchup_fixture_coverage.json" in captured.out
    assert "Decision: exploratory_only" in captured.out


def _season_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rodada": 1, "id_clube": 10, "nome_clube": "A", "entrou_em_campo": True},
            {"rodada": 1, "id_clube": 20, "nome_clube": "B", "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 10, "nome_clube": "A", "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 30, "nome_clube": "C", "entrou_em_campo": True},
        ]
    )


def _record(
    season: int,
    *,
    comparable: bool,
    status: str = "complete_historical",
) -> audit.SeasonMatchupFixtureRecord:
    return audit.SeasonMatchupFixtureRecord(
        season=season,
        season_status=status,
        metrics_comparable=comparable,
        fixture_status="ok" if comparable else "failed",
        round_file_count=38,
        min_round=1,
        max_round=38,
        detected_rounds=list(range(1, 39)),
    )


def _context_row(
    season: int,
    rodada: int,
    id_clube: int,
    opponent_id_clube: int,
    is_home: int,
    source_file: str,
) -> dict[str, object]:
    return {
        "season": season,
        "rodada": rodada,
        "id_clube": id_clube,
        "opponent_id_clube": opponent_id_clube,
        "opponent_nome_clube": None,
        "is_home": is_home,
        "fixture_source": "exploratory",
        "source_file": source_file,
        "source_manifest": None,
        "source_sha256": "a",
        "source_manifest_sha256": None,
    }
