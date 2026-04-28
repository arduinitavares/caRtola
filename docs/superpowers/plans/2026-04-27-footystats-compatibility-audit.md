# FootyStats Compatibility Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only audit that determines whether `data/footystats/` Brazil Serie A files are complete, leakage-safe, and joinable to Cartola seasons before adding FootyStats-derived model features.

**Architecture:** Add a focused `cartola.backtesting.footystats_audit` module, mirroring the existing compatibility-audit style: discover files, profile each season, compare FootyStats teams to Cartola clubs, classify columns by leakage risk, and write CSV/JSON reports. The audit must not change model code or produce features; it only reports whether a later integration milestone is safe.

**Tech Stack:** Python 3.13, pandas, stdlib dataclasses/argparse/json/pathlib, pytest, `uv run --frozen`.

---

## File Structure

- Create `src/cartola/backtesting/footystats_audit.py`
  - Owns filename parsing, file discovery, schema profiling, team-name comparison, report writing, and CLI entrypoints.
- Create `scripts/audit_footystats_compatibility.py`
  - Thin executable wrapper, matching `scripts/audit_backtest_compatibility.py`.
- Create `src/tests/backtesting/test_footystats_audit.py`
  - Unit tests using temp files; no dependency on local real `data/footystats/` contents.
- Modify `README.md`
  - Add the audit command and report paths.
- Optionally update `roadmap.md`
  - Only if implementation results change the next roadmap decision.

Report output:

```text
data/08_reporting/footystats/footystats_compatibility.csv
data/08_reporting/footystats/footystats_compatibility.json
```

The audit is read-only with respect to `data/footystats/` and `data/01_raw/`.

---

### Task 1: Discover FootyStats Files

**Files:**
- Create: `src/cartola/backtesting/footystats_audit.py`
- Create: `src/tests/backtesting/test_footystats_audit.py`

- [ ] **Step 1: Write failing tests for filename parsing and discovery**

Add this to `src/tests/backtesting/test_footystats_audit.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py -q
```

Expected: FAIL because `cartola.backtesting.footystats_audit` does not exist.

- [ ] **Step 3: Implement minimal discovery code**

Create `src/cartola/backtesting/footystats_audit.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FOOTYSTATS_FILE_RE = re.compile(
    r"^(?P<league_slug>.+)-(?P<table_type>league|matches|players|teams|teams2)-"
    r"(?P<start_year>\d{4})-to-(?P<end_year>\d{4})-stats\.csv$"
)


@dataclass(frozen=True)
class FootyStatsAuditConfig:
    project_root: Path = Path(".")
    footystats_dir: Path = Path("data/footystats")
    output_root: Path = Path("data/08_reporting/footystats")


@dataclass(frozen=True)
class ParsedFootyStatsFilename:
    path: Path
    league_slug: str
    table_type: str
    start_year: int
    end_year: int

    @property
    def season(self) -> int:
        return self.start_year


@dataclass(frozen=True)
class FootyStatsSeasonDiscovery:
    season: int
    league_slug: str
    files: dict[str, Path] = field(default_factory=dict)


def parse_footystats_filename(path: Path) -> ParsedFootyStatsFilename:
    match = FOOTYSTATS_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid FootyStats filename: {path}")
    start_year = int(match.group("start_year"))
    end_year = int(match.group("end_year"))
    if start_year != end_year:
        raise ValueError(f"FootyStats audit only supports single-year seasons: {path}")
    return ParsedFootyStatsFilename(
        path=path,
        league_slug=match.group("league_slug"),
        table_type=match.group("table_type"),
        start_year=start_year,
        end_year=end_year,
    )


def discover_footystats_files(config: FootyStatsAuditConfig) -> list[FootyStatsSeasonDiscovery]:
    root = config.project_root / config.footystats_dir
    if not root.exists():
        return []

    grouped: dict[tuple[int, str], dict[str, Path]] = {}
    for path in sorted(root.glob("*.csv")):
        parsed = parse_footystats_filename(path)
        key = (parsed.season, parsed.league_slug)
        grouped.setdefault(key, {})[parsed.table_type] = path

    return [
        FootyStatsSeasonDiscovery(season=season, league_slug=league_slug, files=files)
        for (season, league_slug), files in sorted(grouped.items())
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_audit.py src/tests/backtesting/test_footystats_audit.py
git commit -m "feat: discover footystats season files"
```

---

### Task 2: Profile Match Files And Leakage-Safe Columns

**Files:**
- Modify: `src/cartola/backtesting/footystats_audit.py`
- Modify: `src/tests/backtesting/test_footystats_audit.py`

- [ ] **Step 1: Write failing tests for match profiling**

Append to `src/tests/backtesting/test_footystats_audit.py`:

```python
import pandas as pd


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py::test_profile_match_file_classifies_pre_match_and_outcome_columns -q
```

Expected: FAIL because `profile_match_file` does not exist.

- [ ] **Step 3: Implement match profiling**

Add to `src/cartola/backtesting/footystats_audit.py`:

```python
import pandas as pd

PRE_MATCH_SAFE_COLUMNS: tuple[str, ...] = (
    "Pre-Match PPG (Home)",
    "Pre-Match PPG (Away)",
    "Home Team Pre-Match xG",
    "Away Team Pre-Match xG",
    "average_goals_per_match_pre_match",
    "btts_percentage_pre_match",
    "over_15_percentage_pre_match",
    "over_25_percentage_pre_match",
    "over_35_percentage_pre_match",
    "over_45_percentage_pre_match",
    "over_15_HT_FHG_percentage_pre_match",
    "over_05_HT_FHG_percentage_pre_match",
    "over_15_2HG_percentage_pre_match",
    "over_05_2HG_percentage_pre_match",
    "average_corners_per_match_pre_match",
    "average_cards_per_match_pre_match",
    "odds_ft_home_team_win",
    "odds_ft_draw",
    "odds_ft_away_team_win",
    "odds_ft_over15",
    "odds_ft_over25",
    "odds_ft_over35",
    "odds_ft_over45",
    "odds_btts_yes",
    "odds_btts_no",
)

POST_MATCH_OUTCOME_COLUMNS: tuple[str, ...] = (
    "home_team_goal_count",
    "away_team_goal_count",
    "total_goal_count",
    "total_goals_at_half_time",
    "home_team_goal_count_half_time",
    "away_team_goal_count_half_time",
    "home_team_corner_count",
    "away_team_corner_count",
    "home_team_yellow_cards",
    "home_team_red_cards",
    "away_team_yellow_cards",
    "away_team_red_cards",
    "home_team_shots",
    "away_team_shots",
    "home_team_shots_on_target",
    "away_team_shots_on_target",
    "home_team_fouls",
    "away_team_fouls",
    "home_team_possession",
    "away_team_possession",
    "team_a_xg",
    "team_b_xg",
)


@dataclass(frozen=True)
class MatchFileProfile:
    path: Path
    row_count: int
    status_counts: dict[str, int]
    min_game_week: int | None
    max_game_week: int | None
    game_week_count: int
    team_names: list[str]
    pre_match_safe_columns: list[str]
    post_match_outcome_columns: list[str]
    pre_match_missing_counts: dict[str, int]
    pre_match_zero_counts: dict[str, int]


def profile_match_file(path: Path) -> MatchFileProfile:
    frame = pd.read_csv(path)
    game_weeks = pd.to_numeric(frame["Game Week"], errors="coerce") if "Game Week" in frame.columns else pd.Series(dtype=float)
    teams: set[str] = set()
    for column in ("home_team_name", "away_team_name"):
        if column in frame.columns:
            teams.update(frame[column].dropna().astype(str).tolist())

    safe_columns = [column for column in PRE_MATCH_SAFE_COLUMNS if column in frame.columns]
    outcome_columns = [column for column in POST_MATCH_OUTCOME_COLUMNS if column in frame.columns]
    missing_counts = {column: int(pd.to_numeric(frame[column], errors="coerce").isna().sum()) for column in safe_columns}
    zero_counts = {
        column: int(pd.to_numeric(frame[column], errors="coerce").fillna(0).eq(0).sum())
        for column in safe_columns
    }

    return MatchFileProfile(
        path=path,
        row_count=len(frame),
        status_counts={str(key): int(value) for key, value in frame.get("status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict().items()},
        min_game_week=None if game_weeks.dropna().empty else int(game_weeks.min()),
        max_game_week=None if game_weeks.dropna().empty else int(game_weeks.max()),
        game_week_count=int(game_weeks.dropna().nunique()),
        team_names=sorted(teams),
        pre_match_safe_columns=safe_columns,
        post_match_outcome_columns=outcome_columns,
        pre_match_missing_counts=missing_counts,
        pre_match_zero_counts=zero_counts,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_audit.py src/tests/backtesting/test_footystats_audit.py
git commit -m "feat: profile footystats match files"
```

---

### Task 3: Compare FootyStats Teams To Cartola Clubs

**Files:**
- Modify: `src/cartola/backtesting/footystats_audit.py`
- Modify: `src/tests/backtesting/test_footystats_audit.py`

- [ ] **Step 1: Write failing tests for team normalization and comparison**

Append to `src/tests/backtesting/test_footystats_audit.py`:

```python
def test_normalize_team_name_handles_accents_suffixes_and_cartola_abbreviations() -> None:
    assert audit.normalize_team_name("São Paulo") == "sao paulo"
    assert audit.normalize_team_name("São Paulo FC") == "sao paulo"
    assert audit.normalize_team_name("Vasco da Gama") == "vasco da gama"
    assert audit.normalize_team_name("FLA") == "flamengo"
    assert audit.normalize_team_name("CAM") == "atletico mineiro"


def test_compare_footystats_teams_to_cartola_clubs_reports_unmapped(tmp_path: Path) -> None:
    season_dir = tmp_path / "data" / "01_raw" / "2025"
    season_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "atletas.clube_id": [262, 275],
            "atletas.clube.id.full.name": ["FLA", "PAL"],
        }
    ).to_csv(season_dir / "rodada-1.csv", index=False)

    comparison = audit.compare_teams_to_cartola(
        season=2025,
        footystats_team_names=["Flamengo", "Palmeiras", "Mirassol"],
        project_root=tmp_path,
    )

    assert comparison.cartola_clubs_by_normalized_name == {"flamengo": 262, "palmeiras": 275}
    assert comparison.mapped_teams == {"Flamengo": 262, "Palmeiras": 275}
    assert comparison.unmapped_footystats_teams == ["Mirassol"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py::test_normalize_team_name_handles_accents_suffixes_and_cartola_abbreviations src/tests/backtesting/test_footystats_audit.py::test_compare_footystats_teams_to_cartola_clubs_reports_unmapped -q
```

Expected: FAIL because the functions do not exist.

- [ ] **Step 3: Implement team comparison**

Add to `src/cartola/backtesting/footystats_audit.py`:

```python
import unicodedata

CARTOLA_ABBREVIATIONS: dict[str, str] = {
    "FLA": "flamengo",
    "BOT": "botafogo",
    "COR": "corinthians",
    "BAH": "bahia",
    "FLU": "fluminense",
    "VAS": "vasco da gama",
    "PAL": "palmeiras",
    "SAO": "sao paulo",
    "SAN": "santos",
    "RBB": "bragantino",
    "CAM": "atletico mineiro",
    "CRU": "cruzeiro",
    "GRE": "gremio",
    "INT": "internacional",
    "JUV": "juventude",
    "VIT": "vitoria",
    "SPT": "sport recife",
    "CEA": "ceara",
    "FOR": "fortaleza",
    "MIR": "mirassol",
    "CAP": "atletico pr",
    "CFC": "coritiba",
    "CHA": "chapecoense",
    "REM": "remo",
}

TEAM_SUFFIXES: tuple[str, ...] = (
    " fc sao paulo",
    " fc",
    " ec",
    " sc",
    " fr",
)


@dataclass(frozen=True)
class TeamComparison:
    cartola_clubs_by_normalized_name: dict[str, int]
    mapped_teams: dict[str, int]
    unmapped_footystats_teams: list[str]


def normalize_team_name(value: str) -> str:
    if value in CARTOLA_ABBREVIATIONS:
        return CARTOLA_ABBREVIATIONS[value]
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    replacements = {
        "athletico pr": "atletico pr",
        "atletico mg": "atletico mineiro",
        "america mg": "america mineiro",
        "vasco": "vasco da gama",
        "sport": "sport recife",
        "cr flamengo": "flamengo",
        "se palmeiras": "palmeiras",
        "sao paulo fc": "sao paulo",
    }
    text = replacements.get(text, text)
    for suffix in TEAM_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return replacements.get(text, text)


def compare_teams_to_cartola(
    *,
    season: int,
    footystats_team_names: list[str],
    project_root: Path,
) -> TeamComparison:
    season_dir = project_root / "data" / "01_raw" / str(season)
    cartola_names: dict[str, int] = {}
    for path in sorted(season_dir.glob("rodada-*.csv")):
        frame = pd.read_csv(path, usecols=lambda column: column in {"atletas.clube_id", "atletas.clube.id.full.name"})
        if {"atletas.clube_id", "atletas.clube.id.full.name"} - set(frame.columns):
            continue
        for _, row in frame.dropna().drop_duplicates().iterrows():
            normalized = normalize_team_name(str(row["atletas.clube.id.full.name"]))
            cartola_names.setdefault(normalized, int(row["atletas.clube_id"]))

    mapped: dict[str, int] = {}
    unmapped: list[str] = []
    for team_name in sorted(set(footystats_team_names)):
        club_id = cartola_names.get(normalize_team_name(team_name))
        if club_id is None:
            unmapped.append(team_name)
        else:
            mapped[team_name] = club_id

    return TeamComparison(
        cartola_clubs_by_normalized_name=cartola_names,
        mapped_teams=mapped,
        unmapped_footystats_teams=unmapped,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_audit.py src/tests/backtesting/test_footystats_audit.py
git commit -m "feat: audit footystats team mapping coverage"
```

---

### Task 4: Build Per-Season Audit Records

**Files:**
- Modify: `src/cartola/backtesting/footystats_audit.py`
- Modify: `src/tests/backtesting/test_footystats_audit.py`

- [ ] **Step 1: Write failing test for a season audit record**

Append to `src/tests/backtesting/test_footystats_audit.py`:

```python
def test_audit_one_footystats_season_marks_joinable_safe_complete_season(tmp_path: Path) -> None:
    footystats_dir = tmp_path / "data" / "footystats"
    footystats_dir.mkdir(parents=True)
    matches_path = footystats_dir / "brazil-serie-a-matches-2025-to-2025-stats.csv"
    pd.DataFrame(
        [
            {
                "status": "complete",
                "Game Week": week,
                "home_team_name": "Flamengo" if week == 1 else "Palmeiras",
                "away_team_name": "Palmeiras" if week == 1 else "Flamengo",
                "Pre-Match PPG (Home)": 0.0,
                "Pre-Match PPG (Away)": 0.0,
                "home_team_goal_count": 1,
                "away_team_goal_count": 0,
            }
            for week in range(1, 39)
        ]
    ).to_csv(matches_path, index=False)
    (footystats_dir / "brazil-serie-a-league-2025-to-2025-stats.csv").write_text(
        "name,season,status,total_matches,matches_completed,game_week,total_game_week,progress\n"
        "Serie A,2025,Completed,380,380,38,38,100\n",
        encoding="utf-8",
    )
    season_dir = tmp_path / "data" / "01_raw" / "2025"
    season_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "atletas.clube_id": [262, 275],
            "atletas.clube.id.full.name": ["FLA", "PAL"],
        }
    ).to_csv(season_dir / "rodada-1.csv", index=False)

    discovery = audit.discover_footystats_files(audit.FootyStatsAuditConfig(project_root=tmp_path))[0]
    record = audit.audit_one_footystats_season(discovery, audit.FootyStatsAuditConfig(project_root=tmp_path))

    assert record.season == 2025
    assert record.league_slug == "brazil-serie-a"
    assert record.match_status == "ok"
    assert record.team_mapping_status == "ok"
    assert record.integration_status == "candidate"
    assert record.match_row_count == 38
    assert record.min_game_week == 1
    assert record.max_game_week == 38
    assert record.unmapped_footystats_teams == []
    assert "Pre-Match PPG (Home)" in record.pre_match_safe_columns
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py::test_audit_one_footystats_season_marks_joinable_safe_complete_season -q
```

Expected: FAIL because `audit_one_footystats_season` does not exist.

- [ ] **Step 3: Implement season audit records**

Add to `src/cartola/backtesting/footystats_audit.py`:

```python
@dataclass
class FootyStatsSeasonAuditRecord:
    season: int
    league_slug: str
    available_files: list[str]
    league_status: str
    match_status: str
    team_mapping_status: str
    integration_status: str
    match_row_count: int | None
    min_game_week: int | None
    max_game_week: int | None
    game_week_count: int | None
    status_counts: dict[str, int]
    footystats_team_count: int
    mapped_team_count: int
    unmapped_footystats_teams: list[str]
    pre_match_safe_columns: list[str]
    post_match_outcome_columns: list[str]
    pre_match_missing_counts: dict[str, int]
    pre_match_zero_counts: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def to_csv_row(self) -> dict[str, object]:
        return {
            "season": self.season,
            "league_slug": self.league_slug,
            "available_files": ",".join(self.available_files),
            "league_status": self.league_status,
            "match_status": self.match_status,
            "team_mapping_status": self.team_mapping_status,
            "integration_status": self.integration_status,
            "match_row_count": self.match_row_count,
            "min_game_week": self.min_game_week,
            "max_game_week": self.max_game_week,
            "game_week_count": self.game_week_count,
            "status_counts": ";".join(f"{key}:{value}" for key, value in sorted(self.status_counts.items())),
            "footystats_team_count": self.footystats_team_count,
            "mapped_team_count": self.mapped_team_count,
            "unmapped_footystats_teams": ",".join(self.unmapped_footystats_teams),
            "pre_match_safe_columns": ",".join(self.pre_match_safe_columns),
            "post_match_outcome_columns": ",".join(self.post_match_outcome_columns),
            "notes": "; ".join(self.notes),
        }

    def to_json_object(self) -> dict[str, object]:
        return {
            "season": self.season,
            "league_slug": self.league_slug,
            "available_files": self.available_files,
            "league_status": self.league_status,
            "match_status": self.match_status,
            "team_mapping_status": self.team_mapping_status,
            "integration_status": self.integration_status,
            "match_row_count": self.match_row_count,
            "min_game_week": self.min_game_week,
            "max_game_week": self.max_game_week,
            "game_week_count": self.game_week_count,
            "status_counts": self.status_counts,
            "footystats_team_count": self.footystats_team_count,
            "mapped_team_count": self.mapped_team_count,
            "unmapped_footystats_teams": self.unmapped_footystats_teams,
            "pre_match_safe_columns": self.pre_match_safe_columns,
            "post_match_outcome_columns": self.post_match_outcome_columns,
            "pre_match_missing_counts": self.pre_match_missing_counts,
            "pre_match_zero_counts": self.pre_match_zero_counts,
            "notes": self.notes,
        }


def audit_one_footystats_season(
    discovery: FootyStatsSeasonDiscovery,
    config: FootyStatsAuditConfig,
) -> FootyStatsSeasonAuditRecord:
    notes: list[str] = []
    league_status = "ok" if "league" in discovery.files else "missing"
    matches_path = discovery.files.get("matches")
    if matches_path is None:
        return FootyStatsSeasonAuditRecord(
            season=discovery.season,
            league_slug=discovery.league_slug,
            available_files=sorted(discovery.files),
            league_status=league_status,
            match_status="missing",
            team_mapping_status="skipped",
            integration_status="not_candidate",
            match_row_count=None,
            min_game_week=None,
            max_game_week=None,
            game_week_count=None,
            status_counts={},
            footystats_team_count=0,
            mapped_team_count=0,
            unmapped_footystats_teams=[],
            pre_match_safe_columns=[],
            post_match_outcome_columns=[],
            pre_match_missing_counts={},
            pre_match_zero_counts={},
            notes=["missing matches file"],
        )

    match_profile = profile_match_file(matches_path)
    team_comparison = compare_teams_to_cartola(
        season=discovery.season,
        footystats_team_names=match_profile.team_names,
        project_root=config.project_root,
    )
    team_mapping_status = "ok" if not team_comparison.unmapped_footystats_teams else "failed"
    if match_profile.max_game_week != 38 or match_profile.min_game_week != 1:
        notes.append("match file does not cover game weeks 1-38")
    if any(status != "complete" for status in match_profile.status_counts):
        notes.append("match file contains non-complete fixtures")

    integration_status = "candidate"
    if team_mapping_status != "ok" or not match_profile.pre_match_safe_columns:
        integration_status = "not_candidate"
    elif any(status != "complete" for status in match_profile.status_counts):
        integration_status = "partial_current"

    return FootyStatsSeasonAuditRecord(
        season=discovery.season,
        league_slug=discovery.league_slug,
        available_files=sorted(discovery.files),
        league_status=league_status,
        match_status="ok",
        team_mapping_status=team_mapping_status,
        integration_status=integration_status,
        match_row_count=match_profile.row_count,
        min_game_week=match_profile.min_game_week,
        max_game_week=match_profile.max_game_week,
        game_week_count=match_profile.game_week_count,
        status_counts=match_profile.status_counts,
        footystats_team_count=len(match_profile.team_names),
        mapped_team_count=len(team_comparison.mapped_teams),
        unmapped_footystats_teams=team_comparison.unmapped_footystats_teams,
        pre_match_safe_columns=match_profile.pre_match_safe_columns,
        post_match_outcome_columns=match_profile.post_match_outcome_columns,
        pre_match_missing_counts=match_profile.pre_match_missing_counts,
        pre_match_zero_counts=match_profile.pre_match_zero_counts,
        notes=notes,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_audit.py src/tests/backtesting/test_footystats_audit.py
git commit -m "feat: audit footystats season integration candidates"
```

---

### Task 5: Write CSV/JSON Reports

**Files:**
- Modify: `src/cartola/backtesting/footystats_audit.py`
- Modify: `src/tests/backtesting/test_footystats_audit.py`

- [ ] **Step 1: Write failing test for report writing**

Append to `src/tests/backtesting/test_footystats_audit.py`:

```python
def test_run_footystats_audit_writes_csv_and_json(tmp_path: Path) -> None:
    footystats_dir = tmp_path / "data" / "footystats"
    footystats_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "complete",
                "Game Week": 1,
                "home_team_name": "Flamengo",
                "away_team_name": "Palmeiras",
                "Pre-Match PPG (Home)": 0.0,
                "Pre-Match PPG (Away)": 0.0,
            }
        ]
    ).to_csv(footystats_dir / "brazil-serie-a-matches-2025-to-2025-stats.csv", index=False)
    season_dir = tmp_path / "data" / "01_raw" / "2025"
    season_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "atletas.clube_id": [262, 275],
            "atletas.clube.id.full.name": ["FLA", "PAL"],
        }
    ).to_csv(season_dir / "rodada-1.csv", index=False)

    result = audit.run_footystats_audit(audit.FootyStatsAuditConfig(project_root=tmp_path))

    assert result.csv_path.exists()
    assert result.json_path.exists()
    csv_text = result.csv_path.read_text(encoding="utf-8")
    assert "season,league_slug" in csv_text
    json_text = result.json_path.read_text(encoding="utf-8")
    assert '"seasons"' in json_text
    assert result.seasons[0].season == 2025
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py::test_run_footystats_audit_writes_csv_and_json -q
```

Expected: FAIL because `run_footystats_audit` does not exist.

- [ ] **Step 3: Implement report writing**

Add to `src/cartola/backtesting/footystats_audit.py`:

```python
import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Sequence

CSV_COLUMNS: tuple[str, ...] = (
    "season",
    "league_slug",
    "available_files",
    "league_status",
    "match_status",
    "team_mapping_status",
    "integration_status",
    "match_row_count",
    "min_game_week",
    "max_game_week",
    "game_week_count",
    "status_counts",
    "footystats_team_count",
    "mapped_team_count",
    "unmapped_footystats_teams",
    "pre_match_safe_columns",
    "post_match_outcome_columns",
    "notes",
)


@dataclass(frozen=True)
class FootyStatsAuditRunResult:
    generated_at_utc: str
    project_root: Path
    config: FootyStatsAuditConfig
    seasons: list[FootyStatsSeasonAuditRecord]
    csv_path: Path
    json_path: Path


def run_footystats_audit(config: FootyStatsAuditConfig) -> FootyStatsAuditRunResult:
    generated_at = datetime.now(UTC).isoformat()
    discoveries = discover_footystats_files(config)
    seasons = [audit_one_footystats_season(discovery, config) for discovery in discoveries]
    csv_path, json_path = write_footystats_audit_reports(seasons, config, generated_at)
    return FootyStatsAuditRunResult(
        generated_at_utc=generated_at,
        project_root=config.project_root,
        config=config,
        seasons=seasons,
        csv_path=csv_path,
        json_path=json_path,
    )


def write_footystats_audit_reports(
    seasons: list[FootyStatsSeasonAuditRecord],
    config: FootyStatsAuditConfig,
    generated_at_utc: str,
) -> tuple[Path, Path]:
    output_dir = config.project_root / config.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "footystats_compatibility.csv"
    json_path = output_dir / "footystats_compatibility.json"
    pd.DataFrame([season.to_csv_row() for season in seasons], columns=pd.Index(CSV_COLUMNS)).to_csv(
        csv_path,
        index=False,
    )
    payload = {
        "generated_at_utc": generated_at_utc,
        "project_root": str(config.project_root),
        "config": {
            "project_root": str(config.project_root),
            "footystats_dir": str(config.footystats_dir),
            "output_root": str(config.output_root),
        },
        "seasons": [season.to_json_object() for season in seasons],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_audit.py src/tests/backtesting/test_footystats_audit.py
git commit -m "feat: write footystats compatibility reports"
```

---

### Task 6: Add CLI Script And README Command

**Files:**
- Create: `scripts/audit_footystats_compatibility.py`
- Modify: `src/cartola/backtesting/footystats_audit.py`
- Modify: `src/tests/backtesting/test_footystats_audit.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI parser test**

Append to `src/tests/backtesting/test_footystats_audit.py`:

```python
def test_config_from_args_uses_project_root_and_output_root(tmp_path: Path) -> None:
    args = audit.parse_args(
        [
            "--project-root",
            str(tmp_path),
            "--footystats-dir",
            "custom/footystats",
            "--output-root",
            "custom/report",
        ]
    )

    config = audit.config_from_args(args)

    assert config.project_root == tmp_path
    assert config.footystats_dir == Path("custom/footystats")
    assert config.output_root == Path("custom/report")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py::test_config_from_args_uses_project_root_and_output_root -q
```

Expected: FAIL because parser helpers do not exist.

- [ ] **Step 3: Implement CLI helpers and wrapper script**

Add to `src/cartola/backtesting/footystats_audit.py`:

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FootyStats compatibility and leakage audit.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--footystats-dir", type=Path, default=Path("data/footystats"))
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/footystats"))
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FootyStatsAuditConfig:
    return FootyStatsAuditConfig(
        project_root=args.project_root,
        footystats_dir=args.footystats_dir,
        output_root=args.output_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    result = run_footystats_audit(config_from_args(parse_args(argv)))
    print("FootyStats compatibility audit complete")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    return 0
```

Create `scripts/audit_footystats_compatibility.py`:

```python
from __future__ import annotations

from cartola.backtesting.footystats_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Add this README section near the existing compatibility audit command:

````markdown
Para auditar a compatibilidade dos arquivos FootyStats em `data/footystats/`:

```bash
uv run --frozen python scripts/audit_footystats_compatibility.py
```

O comando é somente leitura para `data/footystats/` e `data/01_raw/`, e grava:

- `data/08_reporting/footystats/footystats_compatibility.csv`
- `data/08_reporting/footystats/footystats_compatibility.json`
````

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_audit.py src/tests/backtesting/test_footystats_audit.py scripts/audit_footystats_compatibility.py README.md
git commit -m "feat: add footystats compatibility audit cli"
```

---

### Task 7: Run Real Audit And Quality Gate

**Files:**
- No source edits unless this task reveals a defect.

- [ ] **Step 1: Run the FootyStats audit on real local data**

Run:

```bash
uv run --frozen python scripts/audit_footystats_compatibility.py
```

Expected:

```text
FootyStats compatibility audit complete
CSV: data/08_reporting/footystats/footystats_compatibility.csv
JSON: data/08_reporting/footystats/footystats_compatibility.json
```

- [ ] **Step 2: Inspect the report**

Run:

```bash
sed -n '1,20p' data/08_reporting/footystats/footystats_compatibility.csv
```

Expected:

- Seasons `2023`, `2024`, and `2025` should be `integration_status=candidate` if team mapping succeeds.
- Season `2026` should be `integration_status=partial_current` because the match file contains incomplete/suspended fixtures.
- `unmapped_footystats_teams` should be empty for candidate seasons.
- `pre_match_safe_columns` should include `Pre-Match PPG (Home)`, `Pre-Match PPG (Away)`, and odds columns.

- [ ] **Step 3: Run the focused test file**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_audit.py -q
```

Expected: PASS.

- [ ] **Step 4: Run the full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest all pass.

- [ ] **Step 5: Commit any final fixes**

If Step 1-4 required source edits, commit them:

```bash
git add src/cartola/backtesting/footystats_audit.py src/tests/backtesting/test_footystats_audit.py scripts/audit_footystats_compatibility.py README.md
git commit -m "fix: harden footystats compatibility audit"
```

If no source edits were required, do not create an empty commit.

---

## Self-Review

Spec coverage:
- Discovers all files under `data/footystats/`: Task 1.
- Classifies league/season coverage: Tasks 1 and 4.
- Inspects match/team schema and candidate pre-match fields: Task 2.
- Classifies safe and unsafe columns: Task 2.
- Reports joinability to Cartola by round and club: Tasks 3 and 4.
- Writes report artifacts: Task 5.
- Adds command and docs: Task 6.
- Verifies against real data and full checks: Task 7.

Placeholder scan:
- No `TBD`, `TODO`, or open-ended “add tests” placeholders remain.
- Every code task includes concrete test code, implementation code, command, expected outcome, and commit command.

Type consistency:
- `FootyStatsAuditConfig`, `FootyStatsSeasonDiscovery`, `MatchFileProfile`, `TeamComparison`, `FootyStatsSeasonAuditRecord`, and `FootyStatsAuditRunResult` are introduced before later tasks use them.
- Function names are stable across tasks: `parse_footystats_filename`, `discover_footystats_files`, `profile_match_file`, `compare_teams_to_cartola`, `audit_one_footystats_season`, `run_footystats_audit`.
