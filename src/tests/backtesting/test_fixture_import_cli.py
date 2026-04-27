import importlib.util
from pathlib import Path

import pandas as pd

from cartola.backtesting.fixture_import import FixtureImportResult

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "import_fixture_schedule.py"
SPEC = importlib.util.spec_from_file_location("import_fixture_schedule", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
import_fixture_schedule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(import_fixture_schedule)


def test_import_fixture_schedule_main_writes_report_on_success(tmp_path, monkeypatch):
    season_df = pd.DataFrame([{"rodada": 1, "id_clube": 262, "entrou_em_campo": True}])
    fixtures = pd.DataFrame([{"rodada": 1, "id_clube_home": 262, "id_clube_away": 277, "data": "2025-04-05"}])
    official_fixtures = fixtures.copy()
    report = pd.DataFrame([{"rodada": 1, "is_valid": True}])

    monkeypatch.setattr(import_fixture_schedule, "load_season_data", lambda season, project_root: season_df)
    monkeypatch.setattr(
        import_fixture_schedule,
        "import_thesportsdb_fixtures",
        lambda **kwargs: FixtureImportResult(fixtures=fixtures, official_fixtures=official_fixtures),
    )
    monkeypatch.setattr(import_fixture_schedule, "load_fixtures", lambda season, project_root: fixtures)
    monkeypatch.setattr(
        import_fixture_schedule,
        "build_round_alignment_report",
        lambda fixtures, season_df, official_fixtures: report,
    )

    result = import_fixture_schedule.main(["--season", "2025", "--project-root", str(tmp_path)])

    assert result == 0
    assert (tmp_path / "data" / "08_reporting" / "fixtures" / "2025" / "round_alignment.csv").exists()


def test_import_fixture_schedule_main_returns_one_for_invalid_alignment(tmp_path, monkeypatch):
    season_df = pd.DataFrame([{"rodada": 1, "id_clube": 262, "entrou_em_campo": True}])
    fixtures = pd.DataFrame([{"rodada": 1, "id_clube_home": 262, "id_clube_away": 277, "data": "2025-04-05"}])
    official_fixtures = fixtures.copy()
    report = pd.DataFrame([{"rodada": 1, "is_valid": False}])

    monkeypatch.setattr(import_fixture_schedule, "load_season_data", lambda season, project_root: season_df)
    monkeypatch.setattr(
        import_fixture_schedule,
        "import_thesportsdb_fixtures",
        lambda **kwargs: FixtureImportResult(fixtures=fixtures, official_fixtures=official_fixtures),
    )
    monkeypatch.setattr(import_fixture_schedule, "load_fixtures", lambda season, project_root: fixtures)
    monkeypatch.setattr(
        import_fixture_schedule,
        "build_round_alignment_report",
        lambda fixtures, season_df, official_fixtures: report,
    )

    result = import_fixture_schedule.main(["--season", "2025", "--project-root", str(tmp_path)])

    assert result == 1
