from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from cartola.backtesting.data import played_club_set

THESPORTSDB_ROUND_URL = "https://www.thesportsdb.com/api/v1/json/{api_key}/eventsround.php"
DEFAULT_THESPORTSDB_LEAGUE_ID = 4351
CANONICAL_FIXTURE_COLUMNS = ["rodada", "id_clube_home", "id_clube_away", "data"]


@dataclass(frozen=True)
class FixtureImportResult:
    fixtures: pd.DataFrame
    official_fixtures: pd.DataFrame


def load_club_mapping(path: str | Path) -> dict[str, int]:
    mapping_path = Path(path)
    frame = pd.read_csv(mapping_path)
    required_columns = ["external_name", "id_clube"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required club mapping columns in {mapping_path}: {missing}")

    ids = pd.to_numeric(frame["id_clube"], errors="raise").astype(int)
    return dict(zip(frame["external_name"].astype(str), ids))


def fetch_thesportsdb_round(
    *,
    round_number: int,
    season: int,
    api_key: str,
    league_id: int = DEFAULT_THESPORTSDB_LEAGUE_ID,
) -> list[dict[str, Any]]:
    response = requests.get(
        THESPORTSDB_ROUND_URL.format(api_key=api_key),
        params={"id": league_id, "r": round_number, "s": season},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    events = payload.get("events") or []
    return list(events)


def events_to_fixture_frame(
    events: Iterable[dict[str, Any]],
    club_mapping: dict[str, int],
    *,
    round_number: int,
    played_clubs: set[int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unmapped: set[str] = set()

    for event in events:
        home_name = event.get("strHomeTeam")
        away_name = event.get("strAwayTeam")
        if home_name not in club_mapping:
            unmapped.add(str(home_name))
        if away_name not in club_mapping:
            unmapped.add(str(away_name))
        if home_name not in club_mapping or away_name not in club_mapping:
            continue

        home_id = int(club_mapping[str(home_name)])
        away_id = int(club_mapping[str(away_name)])
        if played_clubs is not None and (home_id not in played_clubs or away_id not in played_clubs):
            continue

        rows.append(
            {
                "rodada": int(round_number),
                "id_clube_home": home_id,
                "id_clube_away": away_id,
                "data": pd.to_datetime(event.get("dateEvent"), errors="raise").date(),
            }
        )

    if unmapped:
        raise ValueError(f"Unmapped fixture team names: {sorted(unmapped)}")

    return pd.DataFrame(rows, columns=CANONICAL_FIXTURE_COLUMNS)


def import_thesportsdb_fixtures(
    *,
    season: int,
    season_df: pd.DataFrame,
    rounds: Iterable[int],
    project_root: str | Path = ".",
    api_key: str = "3",
    league_id: int = DEFAULT_THESPORTSDB_LEAGUE_ID,
) -> FixtureImportResult:
    root = Path(project_root)
    mapping = load_club_mapping(root / "data" / "01_raw" / "fixtures" / "club_mapping.csv")
    fixture_dir = root / "data" / "01_raw" / "fixtures" / str(season)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    imported_frames: list[pd.DataFrame] = []
    official_frames: list[pd.DataFrame] = []
    for round_number in rounds:
        events = fetch_thesportsdb_round(
            round_number=int(round_number),
            season=season,
            api_key=api_key,
            league_id=league_id,
        )
        official_fixtures = events_to_fixture_frame(
            events,
            mapping,
            round_number=int(round_number),
            played_clubs=None,
        )
        current_played_clubs = played_club_set(season_df, int(round_number))
        fixtures = events_to_fixture_frame(
            events,
            mapping,
            round_number=int(round_number),
            played_clubs=current_played_clubs,
        )

        fixture_clubs = set(fixtures["id_clube_home"]) | set(fixtures["id_clube_away"])
        missing = sorted(current_played_clubs - fixture_clubs)
        if missing:
            raise ValueError(f"Missing Cartola played clubs in round {round_number}: {missing}")

        fixtures.to_csv(fixture_dir / f"partidas-{round_number}.csv", index=False)
        imported_frames.append(fixtures)
        official_frames.append(official_fixtures)

    return FixtureImportResult(
        fixtures=_concat_fixture_frames(imported_frames),
        official_fixtures=_concat_fixture_frames(official_frames),
    )


def _concat_fixture_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=CANONICAL_FIXTURE_COLUMNS)
    return pd.concat(frames, ignore_index=True)
