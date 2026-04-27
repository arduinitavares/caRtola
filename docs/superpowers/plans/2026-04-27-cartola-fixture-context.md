# Cartola Fixture Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-safe fixture context to the Cartola walk-forward backtest and measure whether Random Forest improves beyond the player-only feature ceiling.

**Architecture:** Commit a canonical fixture layout under `data/01_raw/fixtures/`, load it through the backtesting data layer, and pass it into feature construction as optional context. Generate 2025 fixture files from TheSportsDB official round data, filtered to the clubs that actually entered the field in each Cartola round so postponed matches do not pollute target-round availability. Report official matches discarded by that filter so source-vs-Cartola mismatches remain visible.

**Tech Stack:** Python 3.13.12, uv, pandas, requests, pytest, Ruff, ty, Bandit, TheSportsDB v1 JSON API.

---

## Source Refinement

The approved design named Football-Data as the preferred source, but local verification showed `https://www.football-data.co.uk/new/BRA.csv` has no round column. Using chronological chunks is unsafe because Cartola rounds can exclude postponed matches.

Use TheSportsDB instead:

```text
https://www.thesportsdb.com/api/v1/json/{api_key}/eventsround.php?id=4351&r={round}&s=2025
```

Verified locally with the public test key `3`: the endpoint returns 10 official Brazilian Serie A events for each round and includes `intRound`, `dateEvent`, `strHomeTeam`, and `strAwayTeam`.

The importer may filter each official round to matches where both clubs are present in the Cartola `entrou_em_campo == True` played set for that same round, because canonical `partidas-{round}.csv` files must represent the Cartola round, not necessarily all official-round matches. Discarded official matches and clubs must remain visible in the validation report. Extra official clubs that were not in the Cartola played set should not automatically fail the import; if any Cartola played club is missing from generated canonical fixtures, the import must fail.

## File Structure

- Modify `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/.gitignore`
  - Allow only committed fixture mapping and canonical `partidas-*.csv` files under `data/01_raw/fixtures/`.
- Modify `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/docs/superpowers/specs/2026-04-27-cartola-fixture-context-design.md`
  - Record the source refinement from Football-Data to TheSportsDB and the Cartola-played-club filter.
- Create `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/data/01_raw/fixtures/club_mapping.csv`
  - Curated external team name to Cartola `id_clube` mapping.
- Create `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/cartola/backtesting/fixture_import.py`
  - Fetch TheSportsDB round events, map names to Cartola IDs, filter to Cartola-played clubs, and write canonical round files.
- Create `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/scripts/import_fixture_schedule.py`
  - CLI wrapper for the import module.
- Modify `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/cartola/backtesting/data.py`
  - Add `load_fixtures()` and `build_round_alignment_report()`.
- Modify `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/cartola/backtesting/features.py`
  - Add `is_home` and `opponent_club_points_roll3`.
- Modify `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/cartola/backtesting/runner.py`
  - Load fixture data once and pass it to training/prediction frame builders.
- Modify `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/README.md`
  - Document fixture import and fixture-aware backtest behavior.
- Modify tests:
  - `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/tests/backtesting/test_data.py`
  - `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/tests/backtesting/test_fixture_import.py`
  - `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/tests/backtesting/test_features.py`
  - `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/tests/backtesting/test_runner.py`

---

### Task 1: Commit Fixture Data Contract

**Files:**
- Modify: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/.gitignore`
- Modify: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/docs/superpowers/specs/2026-04-27-cartola-fixture-context-design.md`
- Create: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/data/01_raw/fixtures/club_mapping.csv`

- [ ] **Step 1: Update `.gitignore` so fixture source files can be committed**

Add these lines after the existing example dataset exceptions:

```gitignore
# keep curated Cartola backtesting fixture inputs
!**/data/01_raw/fixtures/
!**/data/01_raw/fixtures/club_mapping.csv
!**/data/01_raw/fixtures/[0-9][0-9][0-9][0-9]/
!**/data/01_raw/fixtures/[0-9][0-9][0-9][0-9]/partidas-*.csv
```

- [ ] **Step 2: Create the curated club mapping**

Create `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/data/01_raw/fixtures/club_mapping.csv`:

```csv
external_name,id_clube
Atlético Mineiro,282
Bahia,265
Botafogo,263
Botafogo (RJ),263
Bragantino,280
RB Bragantino,280
Ceará,354
Corinthians,264
Cruzeiro,283
Flamengo,262
Flamengo RJ,262
Fluminense,266
Fortaleza,356
Grêmio,284
Internacional,285
Juventude,286
Mirassol,2305
Palmeiras,275
Santos,277
São Paulo,276
Sport Club do Recife,292
Sport Recife,292
Vasco da Gama,267
Vitória,287
```

Extra aliases such as `Botafogo (RJ)`, `Flamengo RJ`, `RB Bragantino`, and `Sport Recife` are intentional compatibility aliases for source-name variations.

- [ ] **Step 3: Update the design source section**

In `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/docs/superpowers/specs/2026-04-27-cartola-fixture-context-design.md`, replace the `Source Strategy` section with:

````markdown
## Source Strategy

The verified source for 2025 match data is TheSportsDB's Brazilian Serie A round endpoint:

```text
https://www.thesportsdb.com/api/v1/json/{api_key}/eventsround.php?id=4351&r={round}&s=2025
```

Local verification showed that this endpoint returns one official Brasileirão round at a time with `intRound`, `dateEvent`, `strHomeTeam`, and `strAwayTeam`. The public test key `3` is enough for this import, and the script also supports `THESPORTSDB_API_KEY`.

Football-Data's `https://www.football-data.co.uk/new/BRA.csv` remains useful for match results, but it is not used for this milestone because it lacks an explicit round column.

The import step must:

1. Fetch TheSportsDB events for rounds 1 through 38.
2. Map `strHomeTeam` and `strAwayTeam` through committed `club_mapping.csv`.
3. Load normalized Cartola player data for the season.
4. Filter official-round matches down to matches where both clubs are in the Cartola `entrou_em_campo == True` played set for that same round, because canonical `partidas-{round}.csv` files must represent the Cartola round, not necessarily all official-round matches.
5. Write one canonical `partidas-{round}.csv` per round.
6. Produce a validation report comparing fixture club sets with clubs that actually entered the field in each Cartola round, while also reporting discarded official matches and clubs.

If any club that entered the field in Cartola is missing from the generated fixture file for that round, the import must fail rather than silently creating unreliable fixture files.
Extra official clubs that were not in the Cartola played set must not automatically fail the import; they must be reported as discarded official clubs so postponed or otherwise non-Cartola matches remain visible.
````

- [ ] **Step 4: Verify fixture files are not ignored**

Run:

```bash
git check-ignore -q data/01_raw/fixtures/club_mapping.csv && exit 1 || exit 0
git check-ignore -q data/01_raw/fixtures/2025/partidas-1.csv && exit 1 || exit 0
git check-ignore -q data/01_raw/fixtures/2025/raw-api-dump.json
```

Expected: all three commands exit `0`; the first two are unignored, and the raw API dump remains ignored.

- [ ] **Step 5: Commit**

Run:

```bash
git add .gitignore docs/superpowers/specs/2026-04-27-cartola-fixture-context-design.md data/01_raw/fixtures/club_mapping.csv
git commit -m "docs: define fixture context data source"
```

Expected: commit succeeds.

---

### Task 2: Add Fixture Loader And Alignment Report

**Files:**
- Modify: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/tests/backtesting/test_data.py`
- Modify: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/cartola/backtesting/data.py`

- [ ] **Step 1: Write failing fixture loader tests**

Append these imports in `test_data.py`:

```python
from cartola.backtesting.data import build_round_alignment_report, load_fixtures
```

Append these tests:

```python
def _write_fixture_round(root: Path, round_number: int, rows: list[dict[str, object]]) -> None:
    fixture_dir = root / "data" / "01_raw" / "fixtures" / "2025"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["rodada", "id_clube_home", "id_clube_away", "data"]).to_csv(
        fixture_dir / f"partidas-{round_number}.csv",
        index=False,
    )


def test_load_fixtures_reads_and_normalizes_round_files(tmp_path):
    _write_fixture_round(
        tmp_path,
        2,
        [
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-05"},
            {"rodada": 2, "id_clube_home": 30, "id_clube_away": 40, "data": "2025-04-06"},
        ],
    )

    loaded = load_fixtures(2025, project_root=tmp_path)

    assert loaded["rodada"].tolist() == [2, 2]
    assert loaded["id_clube_home"].tolist() == [10, 30]
    assert loaded["id_clube_away"].tolist() == [20, 40]
    assert str(loaded.loc[0, "data"]) == "2025-04-05"


def test_load_fixtures_rejects_missing_required_columns(tmp_path):
    fixture_dir = tmp_path / "data" / "01_raw" / "fixtures" / "2025"
    fixture_dir.mkdir(parents=True)
    pd.DataFrame({"rodada": [2], "id_clube_home": [10], "data": ["2025-04-05"]}).to_csv(
        fixture_dir / "partidas-2.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="Missing required fixture columns"):
        load_fixtures(2025, project_root=tmp_path)


def test_load_fixtures_rejects_duplicate_club_appearance_in_round(tmp_path):
    _write_fixture_round(
        tmp_path,
        2,
        [
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-05"},
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 30, "data": "2025-04-06"},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate fixture club entries"):
        load_fixtures(2025, project_root=tmp_path)


def test_load_fixtures_rejects_self_matches(tmp_path):
    _write_fixture_round(
        tmp_path,
        2,
        [{"rodada": 2, "id_clube_home": 10, "id_clube_away": 10, "data": "2025-04-05"}],
    )

    with pytest.raises(ValueError, match="Fixture rows cannot have the same home and away club"):
        load_fixtures(2025, project_root=tmp_path)


def test_build_round_alignment_report_compares_fixture_and_played_clubs():
    fixtures = pd.DataFrame(
        [
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-05"},
            {"rodada": 3, "id_clube_home": 10, "id_clube_away": 30, "data": "2025-04-12"},
        ]
    )
    season_df = pd.DataFrame(
        [
            {"rodada": 2, "id_clube": 10, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 20, "entrou_em_campo": True},
            {"rodada": 3, "id_clube": 10, "entrou_em_campo": True},
            {"rodada": 3, "id_clube": 40, "entrou_em_campo": True},
        ]
    )
    official_fixtures = pd.DataFrame(
        [
            {"rodada": 2, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-05"},
            {"rodada": 3, "id_clube_home": 10, "id_clube_away": 30, "data": "2025-04-12"},
            {"rodada": 3, "id_clube_home": 50, "id_clube_away": 60, "data": "2025-04-12"},
        ]
    )

    report = build_round_alignment_report(fixtures, season_df, official_fixtures=official_fixtures)
    round_2 = report.loc[report["rodada"] == 2].iloc[0]
    round_3 = report.loc[report["rodada"] == 3].iloc[0]

    assert bool(round_2["is_valid"]) is True
    assert bool(round_3["is_valid"]) is False
    assert round_3["missing_from_fixtures"] == "40"
    assert round_3["extra_in_fixtures"] == "30"
    assert round_3["discarded_official_match_count"] == 1
    assert round_3["discarded_official_clubs"] == "50,60"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_data.py::test_load_fixtures_reads_and_normalizes_round_files src/tests/backtesting/test_data.py::test_build_round_alignment_report_compares_fixture_and_played_clubs -q
```

Expected: FAIL because `load_fixtures` and `build_round_alignment_report` do not exist.

- [ ] **Step 3: Add fixture loading implementation**

In `data.py`, add constants near `_ROUND_FILE_RE`:

```python
FIXTURE_REQUIRED_COLUMNS: tuple[str, ...] = ("rodada", "id_clube_home", "id_clube_away", "data")
_FIXTURE_FILE_RE = re.compile(r"partidas-(\d+)\.csv$")
```

Add these functions after `load_season_data`:

```python
def load_fixtures(season: int, project_root: str | Path = ".") -> pd.DataFrame:
    fixture_dir = Path(project_root) / "data" / "01_raw" / "fixtures" / str(season)
    if not fixture_dir.exists():
        raise FileNotFoundError(f"Fixture directory not found: {fixture_dir}")
    if not fixture_dir.is_dir():
        raise NotADirectoryError(f"Fixture path is not a directory: {fixture_dir}")

    fixture_files = sorted(fixture_dir.glob("partidas-*.csv"), key=_fixture_round_number)
    if not fixture_files:
        raise FileNotFoundError(f"No fixture CSV files found in fixture directory: {fixture_dir}")

    frames = [normalize_fixture_frame(pd.read_csv(path), source=path) for path in fixture_files]
    fixtures = pd.concat(frames, ignore_index=True)
    _validate_fixture_club_entries(fixtures)
    return fixtures


def normalize_fixture_frame(frame: pd.DataFrame, source: str | Path) -> pd.DataFrame:
    source_path = Path(source)
    normalized = frame.copy()
    normalized = normalized.drop(columns=[column for column in normalized.columns if _is_saved_index_column(column)])

    missing = [column for column in FIXTURE_REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required fixture columns in {source_path}: {missing}")

    normalized = normalized.loc[:, list(FIXTURE_REQUIRED_COLUMNS)].copy()
    for column in ("rodada", "id_clube_home", "id_clube_away"):
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(int)
    normalized["data"] = pd.to_datetime(normalized["data"], errors="raise").dt.date

    same_club = normalized["id_clube_home"].eq(normalized["id_clube_away"])
    if same_club.any():
        rows = normalized.loc[same_club, ["rodada", "id_clube_home"]].to_dict("records")
        raise ValueError(f"Fixture rows cannot have the same home and away club in {source_path}: {rows}")

    return normalized


def build_round_alignment_report(
    fixtures: pd.DataFrame,
    season_df: pd.DataFrame,
    official_fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    round_sources = [
        set(fixtures["rodada"].dropna().astype(int)),
        set(season_df["rodada"].dropna().astype(int)),
    ]
    if official_fixtures is not None:
        round_sources.append(set(official_fixtures["rodada"].dropna().astype(int)))
    rounds = sorted(set().union(*round_sources))
    rows: list[dict[str, object]] = []
    for round_number in rounds:
        round_fixtures = fixtures[fixtures["rodada"].eq(round_number)]
        fixture_clubs = _fixture_club_set(round_fixtures)
        played_clubs = played_club_set(season_df, round_number)
        missing = sorted(played_clubs - fixture_clubs)
        extra = sorted(fixture_clubs - played_clubs)
        discarded_count = 0
        discarded_clubs: list[int] = []
        if official_fixtures is not None:
            round_official = official_fixtures[official_fixtures["rodada"].eq(round_number)]
            discarded_count, discarded_clubs = _discarded_official_summary(round_official, fixture_clubs)
        rows.append(
            {
                "rodada": round_number,
                "fixture_club_count": len(fixture_clubs),
                "played_club_count": len(played_clubs),
                "missing_from_fixtures": _format_club_set(missing),
                "extra_in_fixtures": _format_club_set(extra),
                "discarded_official_match_count": discarded_count,
                "discarded_official_clubs": _format_club_set(discarded_clubs),
                "is_valid": not missing,
            }
        )
    return pd.DataFrame(rows)
```

Add helper functions near `_round_number`:

```python
def _fixture_round_number(path: Path) -> int:
    match = _FIXTURE_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid fixture CSV filename: {path}")
    return int(match.group(1))


def _validate_fixture_club_entries(fixtures: pd.DataFrame) -> None:
    if fixtures.empty:
        return
    club_entries = pd.concat(
        [
            fixtures[["rodada", "id_clube_home"]].rename(columns={"id_clube_home": "id_clube"}),
            fixtures[["rodada", "id_clube_away"]].rename(columns={"id_clube_away": "id_clube"}),
        ],
        ignore_index=True,
    )
    duplicated = club_entries[club_entries.duplicated(["rodada", "id_clube"], keep=False)]
    if not duplicated.empty:
        entries = sorted(
            {
                (int(row.rodada), int(row.id_clube))
                for row in duplicated.itertuples(index=False)
            }
        )
        raise ValueError(f"Duplicate fixture club entries: {entries}")


def _fixture_club_set(fixtures: pd.DataFrame) -> set[int]:
    if fixtures.empty:
        return set()
    home = fixtures["id_clube_home"].dropna().astype(int)
    away = fixtures["id_clube_away"].dropna().astype(int)
    return set(home) | set(away)


def _discarded_official_summary(official_fixtures: pd.DataFrame, fixture_clubs: set[int]) -> tuple[int, list[int]]:
    discarded_clubs: set[int] = set()
    discarded_count = 0
    for row in official_fixtures.itertuples(index=False):
        match_clubs = {int(row.id_clube_home), int(row.id_clube_away)}
        if not match_clubs.issubset(fixture_clubs):
            discarded_count += 1
            discarded_clubs.update(match_clubs - fixture_clubs)
    return discarded_count, sorted(discarded_clubs)


def played_club_set(season_df: pd.DataFrame, round_number: int) -> set[int]:
    round_players = season_df[season_df["rodada"].eq(round_number)].copy()
    if "entrou_em_campo" in round_players.columns:
        round_players = round_players[round_players["entrou_em_campo"].fillna(False)]
    return set(round_players["id_clube"].dropna().astype(int))


def _format_club_set(values: list[int]) -> str:
    return ",".join(str(value) for value in values)
```

- [ ] **Step 4: Run fixture loader tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_data.py -q
```

Expected: all `test_data.py` tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/data.py src/tests/backtesting/test_data.py
git commit -m "feat: add fixture loader and alignment report"
```

Expected: commit succeeds.

---

### Task 3: Add TheSportsDB Importer

**Files:**
- Create: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/cartola/backtesting/fixture_import.py`
- Create: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/scripts/import_fixture_schedule.py`
- Create: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/tests/backtesting/test_fixture_import.py`

- [ ] **Step 1: Write failing importer tests**

Create `src/tests/backtesting/test_fixture_import.py`:

```python
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


def test_load_club_mapping_reads_external_names(tmp_path):
    mapping_path = tmp_path / "club_mapping.csv"
    mapping_path.write_text("external_name,id_clube\nSão Paulo,276\nSport Club do Recife,292\n")

    mapping = load_club_mapping(mapping_path)

    assert mapping == {"São Paulo": 276, "Sport Club do Recife": 292}


def test_events_to_fixture_frame_maps_and_filters_to_cartola_played_clubs():
    events = [
        {
            "intRound": "2",
            "dateEvent": "2025-04-05",
            "strHomeTeam": "São Paulo",
            "strAwayTeam": "Sport Club do Recife",
        },
        {
            "intRound": "2",
            "dateEvent": "2025-04-05",
            "strHomeTeam": "Palmeiras",
            "strAwayTeam": "Corinthians",
        },
    ]
    mapping = {"São Paulo": 276, "Sport Club do Recife": 292, "Palmeiras": 275, "Corinthians": 264}

    fixtures = events_to_fixture_frame(events, mapping, round_number=2, played_clubs={276, 292})

    assert fixtures.to_dict("records") == [
        {"rodada": 2, "id_clube_home": 276, "id_clube_away": 292, "data": "2025-04-05"}
    ]


def test_events_to_fixture_frame_rejects_unmapped_teams():
    events = [
        {
            "intRound": "2",
            "dateEvent": "2025-04-05",
            "strHomeTeam": "Unknown FC",
            "strAwayTeam": "Sport Club do Recife",
        }
    ]

    with pytest.raises(ValueError, match="Unmapped fixture team names"):
        events_to_fixture_frame(
            events,
            {"Sport Club do Recife": 292},
            round_number=2,
            played_clubs={292},
        )


def test_fetch_thesportsdb_round_uses_round_endpoint(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"events": [{"intRound": "2"}]}

    def fake_get(url: str, *, params: dict[str, object], timeout: int) -> FakeResponse:
        calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("cartola.backtesting.fixture_import.requests.get", fake_get)

    events = fetch_thesportsdb_round(round_number=2, season=2025, api_key="test-key", league_id=4351)

    assert events == [{"intRound": "2"}]
    assert calls == [
        {
            "url": "https://www.thesportsdb.com/api/v1/json/test-key/eventsround.php",
            "params": {"id": 4351, "r": 2, "s": 2025},
            "timeout": 30,
        }
    ]


def test_import_thesportsdb_fixtures_writes_one_file_per_round(tmp_path, monkeypatch):
    mapping_dir = tmp_path / "data" / "01_raw" / "fixtures"
    mapping_dir.mkdir(parents=True)
    (mapping_dir / "club_mapping.csv").write_text(
        "external_name,id_clube\nSão Paulo,276\nSport Club do Recife,292\nPalmeiras,275\nCorinthians,264\n"
    )
    season_df = pd.DataFrame(
        [
            {"rodada": 1, "id_clube": 276, "entrou_em_campo": False},
            {"rodada": 1, "id_clube": 292, "entrou_em_campo": False},
            {"rodada": 2, "id_clube": 276, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 292, "entrou_em_campo": True},
        ]
    )

    def fake_fetch(round_number: int, season: int, api_key: str, league_id: int) -> list[dict[str, object]]:
        return [
            {
                "intRound": str(round_number),
                "dateEvent": "2025-04-05",
                "strHomeTeam": "São Paulo",
                "strAwayTeam": "Sport Club do Recife",
            },
            {
                "intRound": str(round_number),
                "dateEvent": "2025-04-05",
                "strHomeTeam": "Palmeiras",
                "strAwayTeam": "Corinthians",
            },
        ]

    monkeypatch.setattr("cartola.backtesting.fixture_import.fetch_thesportsdb_round", fake_fetch)

    result = import_thesportsdb_fixtures(
        season=2025,
        season_df=season_df,
        project_root=tmp_path,
        api_key="test-key",
        league_id=4351,
        first_round=1,
        last_round=2,
    )

    assert isinstance(result, FixtureImportResult)
    assert len(result.fixtures) == 1
    assert len(result.official_fixtures) == 4
    assert (tmp_path / "data" / "01_raw" / "fixtures" / "2025" / "partidas-1.csv").exists()
    assert (tmp_path / "data" / "01_raw" / "fixtures" / "2025" / "partidas-2.csv").exists()
    round_2 = pd.read_csv(tmp_path / "data" / "01_raw" / "fixtures" / "2025" / "partidas-2.csv")
    assert round_2.to_dict("records") == [
        {"rodada": 2, "id_clube_home": 276, "id_clube_away": 292, "data": "2025-04-05"}
    ]


def test_import_thesportsdb_fixtures_rejects_missing_played_clubs_before_writing(tmp_path, monkeypatch):
    mapping_dir = tmp_path / "data" / "01_raw" / "fixtures"
    mapping_dir.mkdir(parents=True)
    (mapping_dir / "club_mapping.csv").write_text(
        "external_name,id_clube\nSão Paulo,276\nSport Club do Recife,292\nPalmeiras,275\n"
    )
    season_df = pd.DataFrame(
        [
            {"rodada": 2, "id_clube": 276, "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 292, "entrou_em_campo": True},
        ]
    )

    def fake_fetch(round_number: int, season: int, api_key: str, league_id: int) -> list[dict[str, object]]:
        return [
            {
                "intRound": str(round_number),
                "dateEvent": "2025-04-05",
                "strHomeTeam": "São Paulo",
                "strAwayTeam": "Palmeiras",
            }
        ]

    monkeypatch.setattr("cartola.backtesting.fixture_import.fetch_thesportsdb_round", fake_fetch)

    with pytest.raises(ValueError, match="Missing Cartola played clubs"):
        import_thesportsdb_fixtures(
            season=2025,
            season_df=season_df,
            project_root=tmp_path,
            api_key="test-key",
            league_id=4351,
            first_round=2,
            last_round=2,
        )

    assert not (tmp_path / "data" / "01_raw" / "fixtures" / "2025" / "partidas-2.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixture_import.py -q
```

Expected: FAIL because `cartola.backtesting.fixture_import` does not exist.

- [ ] **Step 3: Add importer module**

Create `src/cartola/backtesting/fixture_import.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    mapping = pd.read_csv(path)
    missing = {"external_name", "id_clube"} - set(mapping.columns)
    if missing:
        raise ValueError(f"Missing required mapping columns in {path}: {sorted(missing)}")
    mapping["id_clube"] = pd.to_numeric(mapping["id_clube"], errors="raise").astype(int)
    return dict(zip(mapping["external_name"].astype(str), mapping["id_clube"], strict=True))


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
    return [dict(event) for event in events]


def events_to_fixture_frame(
    events: list[dict[str, Any]],
    club_mapping: dict[str, int],
    *,
    round_number: int,
    played_clubs: set[int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    unmapped: set[str] = set()

    for event in events:
        home_name = str(event.get("strHomeTeam", ""))
        away_name = str(event.get("strAwayTeam", ""))
        if home_name not in club_mapping:
            unmapped.add(home_name)
        if away_name not in club_mapping:
            unmapped.add(away_name)
        if unmapped:
            continue

        home_id = club_mapping[home_name]
        away_id = club_mapping[away_name]
        if played_clubs is not None and (home_id not in played_clubs or away_id not in played_clubs):
            continue

        rows.append(
            {
                "rodada": round_number,
                "id_clube_home": home_id,
                "id_clube_away": away_id,
                "data": str(event["dateEvent"]),
            }
        )

    if unmapped:
        raise ValueError(f"Unmapped fixture team names: {sorted(unmapped)}")

    return pd.DataFrame(rows, columns=CANONICAL_FIXTURE_COLUMNS)


def import_thesportsdb_fixtures(
    *,
    season: int,
    season_df: pd.DataFrame,
    project_root: str | Path = ".",
    api_key: str = "3",
    league_id: int = DEFAULT_THESPORTSDB_LEAGUE_ID,
    first_round: int = 1,
    last_round: int = 38,
) -> FixtureImportResult:
    root = Path(project_root)
    mapping = load_club_mapping(root / "data" / "01_raw" / "fixtures" / "club_mapping.csv")
    output_dir = root / "data" / "01_raw" / "fixtures" / str(season)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    official_frames: list[pd.DataFrame] = []
    for round_number in range(first_round, last_round + 1):
        events = fetch_thesportsdb_round(
            round_number=round_number,
            season=season,
            api_key=api_key,
            league_id=league_id,
        )
        official_fixtures = events_to_fixture_frame(
            events,
            mapping,
            round_number=round_number,
            played_clubs=None,
        )
        played_clubs = played_club_set(season_df, round_number)
        fixtures = events_to_fixture_frame(
            events,
            mapping,
            round_number=round_number,
            played_clubs=played_clubs,
        )
        fixture_clubs = set(fixtures["id_clube_home"]) | set(fixtures["id_clube_away"])
        missing = sorted(played_clubs - fixture_clubs)
        if missing:
            raise ValueError(f"Missing Cartola played clubs in round {round_number}: {missing}")

        fixtures.to_csv(output_dir / f"partidas-{round_number}.csv", index=False)
        official_frames.append(official_fixtures)
        frames.append(fixtures)

    imported = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANONICAL_FIXTURE_COLUMNS)
    official = (
        pd.concat(official_frames, ignore_index=True)
        if official_frames
        else pd.DataFrame(columns=CANONICAL_FIXTURE_COLUMNS)
    )
    return FixtureImportResult(fixtures=imported, official_fixtures=official)
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/import_fixture_schedule.py`:

```python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from cartola.backtesting.data import build_round_alignment_report, load_fixtures, load_season_data
from cartola.backtesting.fixture_import import DEFAULT_THESPORTSDB_LEAGUE_ID, import_thesportsdb_fixtures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Cartola fixture files from TheSportsDB round data.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--api-key", default=os.environ.get("THESPORTSDB_API_KEY", "3"))
    parser.add_argument("--league-id", type=int, default=DEFAULT_THESPORTSDB_LEAGUE_ID)
    parser.add_argument("--first-round", type=int, default=1)
    parser.add_argument("--last-round", type=int, default=38)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    season_df = load_season_data(args.season, project_root=args.project_root)
    result = import_thesportsdb_fixtures(
        season=args.season,
        season_df=season_df,
        project_root=args.project_root,
        api_key=args.api_key,
        league_id=args.league_id,
        first_round=args.first_round,
        last_round=args.last_round,
    )

    fixtures = load_fixtures(args.season, project_root=args.project_root)
    report = build_round_alignment_report(fixtures, season_df, official_fixtures=result.official_fixtures)
    report_path = args.project_root / "data" / "08_reporting" / "fixtures" / str(args.season) / "round_alignment.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    invalid = report[~report["is_valid"]]
    if not invalid.empty:
        print(invalid.to_string(index=False))
        return 1

    print(f"Imported fixtures: season={args.season} rounds={args.first_round}-{args.last_round}")
    print(f"Alignment report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run importer tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixture_import.py -q
```

Expected: all importer tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cartola/backtesting/fixture_import.py scripts/import_fixture_schedule.py src/tests/backtesting/test_fixture_import.py
git commit -m "feat: add fixture schedule importer"
```

Expected: commit succeeds.

---

### Task 4: Add Fixture Features

**Files:**
- Modify: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/tests/backtesting/test_features.py`
- Modify: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/cartola/backtesting/features.py`

- [ ] **Step 1: Write failing fixture feature tests**

Append these tests to `test_features.py`:

```python
def _fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rodada": 3, "id_clube_home": 10, "id_clube_away": 20, "data": "2025-04-12"},
            {"rodada": 4, "id_clube_home": 20, "id_clube_away": 10, "data": "2025-04-19"},
        ]
    )


def test_fixture_features_mark_home_and_away_players() -> None:
    frame = build_prediction_frame(_season_df(), target_round=3, fixtures=_fixture_df())
    home_player = frame.loc[frame["id_atleta"] == 1].iloc[0]
    away_player = frame.loc[frame["id_atleta"] == 2].iloc[0]

    assert home_player["is_home"] == 1
    assert away_player["is_home"] == 0


def test_opponent_club_points_roll3_uses_prior_rounds_only() -> None:
    season_df = _season_df()
    season_df.loc[season_df["id_atleta"].eq(2), "pontuacao"] = [10, 20, 500]

    frame = build_prediction_frame(season_df, target_round=3, fixtures=_fixture_df())
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["opponent_club_points_roll3"] == pytest.approx(15)


def test_fixture_features_fall_back_without_fixtures() -> None:
    frame = build_prediction_frame(_season_df(), target_round=3)
    player = frame.loc[frame["id_atleta"] == 1].iloc[0]

    assert player["is_home"] == 0
    assert player["opponent_club_points_roll3"] == pytest.approx(5)


def test_opponent_id_is_join_only_not_model_feature() -> None:
    assert "opponent_id" not in FEATURE_COLUMNS
    assert "is_home" in FEATURE_COLUMNS
    assert "opponent_club_points_roll3" in FEATURE_COLUMNS
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_features.py::test_fixture_features_mark_home_and_away_players src/tests/backtesting/test_features.py::test_opponent_id_is_join_only_not_model_feature -q
```

Expected: FAIL because `build_prediction_frame` does not accept `fixtures` and the new feature columns do not exist.

- [ ] **Step 3: Add fixture feature columns and signatures**

In `features.py`, add to `FEATURE_COLUMNS` after `club_points_roll3`:

```python
    "is_home",
    "opponent_club_points_roll3",
```

Add to `NUMERIC_PRIOR_COLUMNS` after `club_points_roll3`:

```python
    "is_home",
    "opponent_club_points_roll3",
```

Change signatures:

```python
def build_prediction_frame(
    season_df: pd.DataFrame,
    target_round: int,
    fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidates = season_df[season_df["rodada"] == target_round].copy()
    played_history = _played_history(season_df, target_round)
    all_history = season_df[season_df["rodada"] < target_round].copy()
    return _add_prior_features(candidates, played_history, all_history, fixtures, target_round)
```

```python
def build_training_frame(
    season_df: pd.DataFrame,
    target_round: int,
    playable_statuses: tuple[str, ...] | None = None,
    fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

Inside the training loop, change:

```python
round_frame = build_prediction_frame(season_df, int(round_number), fixtures=fixtures)
```

Change `_add_prior_features` signature:

```python
def _add_prior_features(
    candidates: pd.DataFrame,
    played_history: pd.DataFrame,
    all_history: pd.DataFrame,
    fixtures: pd.DataFrame | None,
    target_round: int,
) -> pd.DataFrame:
```

- [ ] **Step 4: Add fixture context helper**

In `features.py`, add this helper after `_club_history_features`:

```python
def _fixture_context_features(
    fixtures: pd.DataFrame | None,
    *,
    target_round: int,
    played_history: pd.DataFrame,
) -> pd.DataFrame:
    columns = pd.Index(["id_clube", "is_home", "opponent_club_points_roll3"])
    if fixtures is None or fixtures.empty:
        return pd.DataFrame(columns=columns)

    round_fixtures = fixtures[fixtures["rodada"].eq(target_round)]
    if round_fixtures.empty:
        return pd.DataFrame(columns=columns)

    home_context = round_fixtures[["id_clube_home", "id_clube_away"]].rename(
        columns={"id_clube_home": "id_clube", "id_clube_away": "opponent_id"}
    )
    home_context["is_home"] = 1
    away_context = round_fixtures[["id_clube_away", "id_clube_home"]].rename(
        columns={"id_clube_away": "id_clube", "id_clube_home": "opponent_id"}
    )
    away_context["is_home"] = 0
    context = pd.concat([home_context, away_context], ignore_index=True)

    opponent_roll = _club_history_features(played_history).rename(
        columns={"id_clube": "opponent_id", "club_points_roll3": "opponent_club_points_roll3"}
    )
    context = context.merge(opponent_roll, on="opponent_id", how="left")
    return context[["id_clube", "is_home", "opponent_club_points_roll3"]]
```

In `_add_prior_features`, merge fixture context after club history:

```python
    frame = frame.merge(_fixture_context_features(fixtures, target_round=target_round, played_history=played_history), on="id_clube", how="left")
```

Add fills after `club_points_roll3`:

```python
    frame["is_home"] = frame["is_home"].fillna(0).astype(int)
    frame["opponent_club_points_roll3"] = frame["opponent_club_points_roll3"].fillna(global_club_points_prior)
```

- [ ] **Step 5: Run fixture feature tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_features.py::test_fixture_features_mark_home_and_away_players src/tests/backtesting/test_features.py::test_opponent_club_points_roll3_uses_prior_rounds_only src/tests/backtesting/test_features.py::test_fixture_features_fall_back_without_fixtures src/tests/backtesting/test_features.py::test_opponent_id_is_join_only_not_model_feature -q
```

Expected: all four tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cartola/backtesting/features.py src/tests/backtesting/test_features.py
git commit -m "feat: add fixture context features"
```

Expected: commit succeeds.

---

### Task 5: Wire Fixtures Into The Backtest Runner

**Files:**
- Modify: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/tests/backtesting/test_runner.py`
- Modify: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/src/cartola/backtesting/runner.py`
- Modify: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/README.md`

- [ ] **Step 1: Write failing runner test**

Append this test to `test_runner.py`:

```python
def test_run_backtest_uses_fixture_files_when_available(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    fixture_dir = tmp_path / "data" / "01_raw" / "fixtures" / "2025"
    fixture_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"rodada": 5, "id_clube_home": 1, "id_clube_away": 2, "data": "2025-04-26"},
        ],
        columns=["rodada", "id_clube_home", "id_clube_away", "data"],
    ).to_csv(fixture_dir / "partidas-5.csv", index=False)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)
    round_5 = result.player_predictions[result.player_predictions["rodada"] == 5]
    club_1 = round_5[round_5["id_clube"] == 1].iloc[0]
    club_2 = round_5[round_5["id_clube"] == 2].iloc[0]

    assert club_1["is_home"] == 1
    assert club_2["is_home"] == 0
    assert "opponent_club_points_roll3" in result.player_predictions.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_run_backtest_uses_fixture_files_when_available -q
```

Expected: FAIL because the runner does not load fixtures.

- [ ] **Step 3: Update runner**

In `runner.py`, change imports:

```python
from cartola.backtesting.data import load_fixtures, load_season_data
```

Change the `run_backtest` signature:

```python
def run_backtest(
    config: BacktestConfig,
    season_df: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
) -> BacktestResult:
```

After loading `data`, add:

```python
    fixture_data = fixtures.copy() if fixtures is not None else _load_optional_fixtures(config)
```

Change frame construction inside the round loop:

```python
        training = build_training_frame(
            data,
            round_number,
            playable_statuses=config.playable_statuses,
            fixtures=fixture_data,
        )
        candidates = build_prediction_frame(data, round_number, fixtures=fixture_data)
```

Add helper after `_max_round`:

```python
def _load_optional_fixtures(config: BacktestConfig) -> pd.DataFrame | None:
    try:
        return load_fixtures(config.season, project_root=config.project_root)
    except FileNotFoundError:
        return None
```

- [ ] **Step 4: Update README**

In `README.md`, under the backtest section, add:

````markdown
Para importar partidas de 2025 para features de mando/oponente:

```bash
uv run --frozen python scripts/import_fixture_schedule.py --season 2025
```

O backtest carrega automaticamente `data/01_raw/fixtures/{season}/partidas-*.csv` quando esses arquivos existem. Sem esses arquivos, ele roda com os mesmos defaults sem contexto de partidas.
````

- [ ] **Step 5: Run runner test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_run_backtest_uses_fixture_files_when_available -q
```

Expected: test passes.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py README.md
git commit -m "feat: load fixtures in backtest runner"
```

Expected: commit succeeds.

---

### Task 6: Generate 2025 Fixtures, Verify, And Measure

**Files:**
- Generate: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/data/01_raw/fixtures/2025/partidas-*.csv`
- Generate ignored report: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/data/08_reporting/fixtures/2025/round_alignment.csv`
- Generate ignored backtest outputs: `/Users/aaat/projects/caRtola/.worktrees/cartola-fixture-context/data/08_reporting/backtests/2025/*.csv`

- [ ] **Step 1: Run all backtesting tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/ -v
```

Expected: all backtesting tests pass.

- [ ] **Step 2: Import 2025 fixture files**

Run:

```bash
uv run --frozen python scripts/import_fixture_schedule.py --season 2025 --project-root .
```

Expected: command exits `0`, creates `data/01_raw/fixtures/2025/partidas-1.csv` through `partidas-38.csv`, and writes `data/08_reporting/fixtures/2025/round_alignment.csv`.

- [ ] **Step 3: Inspect fixture alignment**

Run:

```bash
uv run --frozen python - <<'PY'
import pandas as pd
report = pd.read_csv("data/08_reporting/fixtures/2025/round_alignment.csv")
print(report.to_string(index=False))
if not report["is_valid"].all():
    raise SystemExit(1)
PY
```

Expected: all rows show `is_valid` as `True`.

- [ ] **Step 4: Run the 2025 backtest with fixtures**

Run:

```bash
uv run --frozen --no-dev python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100
```

Expected: command exits `0` and rewrites `data/08_reporting/backtests/2025/*.csv`.

- [ ] **Step 5: Inspect diagnostics**

Run:

```bash
grep -E 'player_r2|player_correlation' data/08_reporting/backtests/2025/diagnostics.csv | grep ',all,'
uv run --frozen python - <<'PY'
import pandas as pd
summary = pd.read_csv("data/08_reporting/backtests/2025/summary.csv")
print(summary.to_string(index=False))
rf = summary.loc[summary["strategy"].eq("random_forest"), "avg_actual_points"].iloc[0]
baseline = summary.loc[summary["strategy"].eq("baseline"), "avg_actual_points"].iloc[0]
diagnostics = pd.read_csv("data/08_reporting/backtests/2025/diagnostics.csv")
r2 = diagnostics.loc[
    diagnostics["strategy"].eq("random_forest")
    & diagnostics["position"].eq("all")
    & diagnostics["metric"].eq("player_r2"),
    "value",
].iloc[0]
print(f"rf_avg={rf:.4f}")
print(f"baseline_avg={baseline:.4f}")
print(f"rf_minus_baseline={rf - baseline:.4f}")
print(f"rf_player_r2={r2:.6f}")
PY
```

Expected: metrics print clearly. Success criteria are Random Forest `player_r2 > 0.05` and Random Forest average actual points at least `2.0` above baseline.

- [ ] **Step 6: Run full quality checks**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff clean, ty clean, Bandit no issues, pytest all passing.

- [ ] **Step 7: Review changed files**

Run:

```bash
git status --short
git diff --stat
git diff -- data/08_reporting/backtests/2025/summary.csv data/08_reporting/backtests/2025/diagnostics.csv
```

Expected: source/test/docs/data fixture files are tracked changes; reporting outputs remain ignored.

- [ ] **Step 8: Commit**

Run:

```bash
git add data/01_raw/fixtures/2025
git commit -m "data: add validated 2025 fixture files"
```

Expected: commit succeeds.

---

## Self-Review Checklist

- [ ] Spec coverage: committed mapping, canonical fixture files, fixture loader, alignment report, `is_home`, `opponent_club_points_roll3`, runner wiring, README, and verification are covered.
- [ ] Scope boundary: `opponent_id` is join-only, model architecture is unchanged, optimizer is unchanged, opponent defensive weakness is not implemented.
- [ ] Leakage boundary: fixture assignment comes from target-round schedule, while opponent rolling points use only `played_history` where `rodada < target_round`.
- [ ] Source reliability: TheSportsDB provides explicit official round numbers; importer filters official matches to Cartola played club sets before writing canonical fixtures and reports discarded official clubs.
- [ ] Type consistency: `build_prediction_frame`, `build_training_frame`, `load_fixtures`, `build_round_alignment_report(..., official_fixtures=...)`, `import_thesportsdb_fixtures` returning `FixtureImportResult`, and `run_backtest` signatures are consistent across tasks.
- [ ] Verification: targeted tests, full backtesting tests, fixture import, alignment inspection, backtest, diagnostics inspection, and `pyrepo-check --all` are included.
