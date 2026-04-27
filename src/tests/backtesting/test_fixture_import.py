from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.fixture_import import (
    FixtureImportResult,
    events_to_fixture_frame,
    fetch_thesportsdb_round,
    import_thesportsdb_fixtures,
    load_club_mapping,
)


def test_load_club_mapping_reads_external_names_to_cartola_ids(tmp_path):
    mapping_file = tmp_path / "club_mapping.csv"
    pd.DataFrame(
        {
            "external_name": ["Flamengo", "Santos"],
            "id_clube": ["262", "277"],
        }
    ).to_csv(mapping_file, index=False)

    mapping = load_club_mapping(mapping_file)

    assert mapping == {"Flamengo": 262, "Santos": 277}


def test_events_to_fixture_frame_maps_names_and_filters_to_played_clubs():
    events = [
        {
            "strHomeTeam": "Flamengo",
            "strAwayTeam": "Santos",
            "dateEvent": "2025-04-05",
        },
        {
            "strHomeTeam": "Palmeiras",
            "strAwayTeam": "Gremio",
            "dateEvent": "2025-04-06",
        },
    ]
    club_mapping = {"Flamengo": 262, "Santos": 277, "Palmeiras": 275, "Gremio": 284}

    fixtures = events_to_fixture_frame(
        events,
        club_mapping,
        round_number=2,
        played_clubs={262, 277},
    )

    assert fixtures.to_dict("records") == [
        {"rodada": 2, "id_clube_home": 262, "id_clube_away": 277, "data": "2025-04-05"}
    ]
    assert fixtures.columns.tolist() == ["rodada", "id_clube_home", "id_clube_away", "data"]


def test_events_to_fixture_frame_rejects_unmapped_teams():
    events = [
        {
            "strHomeTeam": "Flamengo",
            "strAwayTeam": "Unknown FC",
            "dateEvent": "2025-04-05",
        },
        {
            "strHomeTeam": "Mystery Club",
            "strAwayTeam": "Santos",
            "dateEvent": "2025-04-06",
        },
    ]

    with pytest.raises(ValueError, match="Unmapped fixture team names"):
        events_to_fixture_frame(
            events,
            {"Flamengo": 262, "Santos": 277},
            round_number=2,
        )


@pytest.mark.parametrize("date_event", [None, "NaT", "nan"])
def test_events_to_fixture_frame_rejects_invalid_event_dates(date_event):
    events = [
        {
            "strHomeTeam": "Flamengo",
            "strAwayTeam": "Santos",
            "dateEvent": date_event,
        }
    ]

    with pytest.raises(ValueError, match="Invalid dateEvent"):
        events_to_fixture_frame(
            events,
            {"Flamengo": 262, "Santos": 277},
            round_number=2,
        )


def test_fetch_thesportsdb_round_calls_round_endpoint(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            calls.append(("raise_for_status",))

        def json(self):
            return {"events": [{"idEvent": "1"}]}

    def fake_get(url, *, params, timeout):
        calls.append((url, params, timeout))
        return Response()

    monkeypatch.setattr("cartola.backtesting.fixture_import.requests.get", fake_get)

    events = fetch_thesportsdb_round(round_number=4, season=2025, api_key="abc", league_id=123)

    assert events == [{"idEvent": "1"}]
    assert calls == [
        (
            "https://www.thesportsdb.com/api/v1/json/abc/eventsround.php",
            {"id": 123, "r": 4, "s": 2025},
            30,
        ),
        ("raise_for_status",),
    ]


def test_import_thesportsdb_fixtures_writes_canonical_round_files_and_returns_result(tmp_path, monkeypatch):
    _write_mapping(tmp_path)
    season_df = pd.DataFrame(
        [
            {"rodada": 1, "id_clube": 262, "entrou_em_campo": True},
            {"rodada": 1, "id_clube": 277, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 275, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 284, "entrou_em_campo": True},
        ]
    )
    events_by_round = {
        1: [
            {"strHomeTeam": "Flamengo", "strAwayTeam": "Santos", "dateEvent": "2025-04-05"},
            {"strHomeTeam": "Palmeiras", "strAwayTeam": "Gremio", "dateEvent": "2025-04-05"},
        ],
        2: [{"strHomeTeam": "Palmeiras", "strAwayTeam": "Gremio", "dateEvent": "2025-04-12"}],
    }

    def fake_fetch(*, round_number, season, api_key, league_id):
        assert season == 2025
        assert api_key == "abc"
        assert league_id == 4351
        return events_by_round[round_number]

    monkeypatch.setattr("cartola.backtesting.fixture_import.fetch_thesportsdb_round", fake_fetch)

    result = import_thesportsdb_fixtures(
        season=2025,
        season_df=season_df,
        rounds=[1, 2],
        project_root=tmp_path,
        api_key="abc",
    )

    assert isinstance(result, FixtureImportResult)
    assert result.fixtures[["rodada", "id_clube_home", "id_clube_away"]].to_dict("records") == [
        {"rodada": 1, "id_clube_home": 262, "id_clube_away": 277},
        {"rodada": 2, "id_clube_home": 275, "id_clube_away": 284},
    ]
    assert result.official_fixtures[["rodada", "id_clube_home", "id_clube_away"]].to_dict("records") == [
        {"rodada": 1, "id_clube_home": 262, "id_clube_away": 277},
        {"rodada": 1, "id_clube_home": 275, "id_clube_away": 284},
        {"rodada": 2, "id_clube_home": 275, "id_clube_away": 284},
    ]

    round_1_file = tmp_path / "data" / "01_raw" / "fixtures" / "2025" / "partidas-1.csv"
    round_2_file = tmp_path / "data" / "01_raw" / "fixtures" / "2025" / "partidas-2.csv"
    assert round_1_file.exists()
    assert round_2_file.exists()
    assert pd.read_csv(round_1_file).columns.tolist() == ["rodada", "id_clube_home", "id_clube_away", "data"]


def test_import_thesportsdb_fixtures_rejects_missing_played_clubs_before_writing(tmp_path, monkeypatch):
    _write_mapping(tmp_path)
    season_df = pd.DataFrame(
        [
            {"rodada": 2, "id_clube": 262, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 277, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 275, "entrou_em_campo": True},
        ]
    )

    def fake_fetch(*, round_number, season, api_key, league_id):
        assert round_number == 2
        return [{"strHomeTeam": "Flamengo", "strAwayTeam": "Santos", "dateEvent": "2025-04-05"}]

    monkeypatch.setattr("cartola.backtesting.fixture_import.fetch_thesportsdb_round", fake_fetch)

    with pytest.raises(ValueError, match="Missing Cartola played clubs"):
        import_thesportsdb_fixtures(
            season=2025,
            season_df=season_df,
            rounds=[2],
            project_root=tmp_path,
            api_key="abc",
        )

    assert not (tmp_path / "data" / "01_raw" / "fixtures" / "2025" / "partidas-2.csv").exists()


def test_import_thesportsdb_fixtures_does_not_write_partial_files_when_later_round_fails(tmp_path, monkeypatch):
    _write_mapping(tmp_path)
    season_df = pd.DataFrame(
        [
            {"rodada": 1, "id_clube": 262, "entrou_em_campo": True},
            {"rodada": 1, "id_clube": 277, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 275, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 284, "entrou_em_campo": True},
        ]
    )
    events_by_round = {
        1: [{"strHomeTeam": "Flamengo", "strAwayTeam": "Santos", "dateEvent": "2025-04-05"}],
        2: [{"strHomeTeam": "Palmeiras", "strAwayTeam": "Unknown FC", "dateEvent": "2025-04-12"}],
    }

    def fake_fetch(*, round_number, season, api_key, league_id):
        return events_by_round[round_number]

    monkeypatch.setattr("cartola.backtesting.fixture_import.fetch_thesportsdb_round", fake_fetch)

    with pytest.raises(ValueError):
        import_thesportsdb_fixtures(
            season=2025,
            season_df=season_df,
            rounds=[1, 2],
            project_root=tmp_path,
            api_key="abc",
        )

    fixture_dir = tmp_path / "data" / "01_raw" / "fixtures" / "2025"
    assert not (fixture_dir / "partidas-1.csv").exists()
    assert not (fixture_dir / "partidas-2.csv").exists()


def _write_mapping(root: Path) -> None:
    mapping_dir = root / "data" / "01_raw" / "fixtures"
    mapping_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "external_name": ["Flamengo", "Santos", "Palmeiras", "Gremio"],
            "id_clube": [262, 277, 275, 284],
        }
    ).to_csv(mapping_dir / "club_mapping.csv", index=False)
