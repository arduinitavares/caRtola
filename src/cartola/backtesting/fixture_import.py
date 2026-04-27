from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from cartola.backtesting.data import played_club_set

THESPORTSDB_ROUND_URL = "https://www.thesportsdb.com/api/v1/json/{api_key}/eventsround.php"
DEFAULT_THESPORTSDB_LEAGUE_ID = 4351
CANONICAL_FIXTURE_COLUMNS = ("rodada", "id_clube_home", "id_clube_away", "data")


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
                "data": _event_date(event, round_number=round_number),
            }
        )

    if unmapped:
        raise ValueError(f"Unmapped fixture team names: {sorted(unmapped)}")

    return pd.DataFrame(rows, columns=pd.Index(CANONICAL_FIXTURE_COLUMNS))


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

    imported_frames: list[pd.DataFrame] = []
    official_frames: list[pd.DataFrame] = []
    fixture_writes: list[tuple[int, pd.DataFrame]] = []
    for round_number in rounds:
        current_round = int(round_number)
        events = fetch_thesportsdb_round(
            round_number=current_round,
            season=season,
            api_key=api_key,
            league_id=league_id,
        )
        official_fixtures = events_to_fixture_frame(
            events,
            mapping,
            round_number=current_round,
            played_clubs=None,
        )
        current_played_clubs = played_club_set(season_df, current_round)
        fixtures = events_to_fixture_frame(
            events,
            mapping,
            round_number=current_round,
            played_clubs=current_played_clubs,
        )
        _validate_generated_fixture_frame(fixtures)

        fixture_clubs = set(fixtures["id_clube_home"]) | set(fixtures["id_clube_away"])
        missing = sorted(current_played_clubs - fixture_clubs)
        if missing:
            raise ValueError(f"Missing Cartola played clubs in round {current_round}: {missing}")

        imported_frames.append(fixtures)
        official_frames.append(official_fixtures)
        fixture_writes.append((current_round, fixtures))

    fixture_dir = root / "data" / "01_raw" / "fixtures" / str(season)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for round_number, fixtures in fixture_writes:
        fixtures.to_csv(fixture_dir / f"partidas-{round_number}.csv", index=False)

    return FixtureImportResult(
        fixtures=_concat_fixture_frames(imported_frames),
        official_fixtures=_concat_fixture_frames(official_frames),
    )


def _concat_fixture_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=pd.Index(CANONICAL_FIXTURE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _validate_generated_fixture_frame(fixtures: pd.DataFrame) -> None:
    self_matches = fixtures["id_clube_home"] == fixtures["id_clube_away"]
    if self_matches.any():
        raise ValueError("Fixture rows cannot have the same home and away club")

    for round_number, round_fixtures in fixtures.groupby("rodada", sort=True):
        clubs = pd.concat(
            [round_fixtures["id_clube_home"], round_fixtures["id_clube_away"]],
            ignore_index=True,
        )
        duplicated_clubs = sorted(clubs.loc[clubs.duplicated()].astype(int).unique().tolist())
        if duplicated_clubs:
            raise ValueError(f"Duplicate fixture club entries in round {round_number}: {duplicated_clubs}")


def _event_date(event: dict[str, Any], *, round_number: int) -> str:
    value = event.get("dateEvent")
    if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Invalid dateEvent in round {round_number}: {value!r}")

    try:
        parsed = pd.to_datetime(str(value), errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid dateEvent in round {round_number}: {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"Invalid dateEvent in round {round_number}: {value!r}")
    return parsed.date().isoformat()
