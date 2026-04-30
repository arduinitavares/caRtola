# Matchup Fixture Coverage Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/audit_matchup_fixture_coverage.py`, a reporting-only audit that proves whether requested seasons have complete fixture context coverage for future opponent/home-away matchup features.

**Architecture:** Add a focused `cartola.backtesting.matchup_fixture_audit` module that discovers requested seasons, loads normalized Cartola player data, resolves per-round fixture context from strict or exploratory local fixture files, validates one context row per played club-round, and writes CSV/JSON reports. Keep the script as a thin CLI wrapper. Do not touch model, optimizer, live recommendation, or feature-generation behavior.

**Tech Stack:** Python 3.13, argparse, dataclasses, pathlib, pandas, pytest, existing `cartola.backtesting.data`, `cartola.backtesting.strict_fixtures`, and scoring/reporting conventions.

---

## File Structure

- Create `src/cartola/backtesting/matchup_fixture_audit.py`
  - Owns CLI parsing helpers, audit dataclasses, fixture context extraction, per-season coverage checks, report writing, and advisory decision logic.

- Create `scripts/audit_matchup_fixture_coverage.py`
  - Thin script wrapper that imports and calls `cartola.backtesting.matchup_fixture_audit.main`.

- Create `src/tests/backtesting/test_matchup_fixture_audit.py`
  - Unit-tests parsing, season classification, fixture context extraction, missing/duplicate/extra context detection, strict/exploratory source metadata, report writing, and decision logic.

- Create `src/tests/backtesting/test_audit_matchup_fixture_coverage_cli.py`
  - Unit-tests script import wiring, parser behavior, and stable CLI output.

- Modify `README.md`
  - Document the audit command and interpretation of its gate decision.

- Modify `roadmap.md`
  - Add the audit as the next prerequisite before `matchup_context_mode=cartola_matchup_v1`.

---

### Task 1: Add CLI Config And Season Parsing

**Files:**
- Create: `src/cartola/backtesting/matchup_fixture_audit.py`
- Test: `src/tests/backtesting/test_matchup_fixture_audit.py`

- [ ] **Step 1: Write failing tests for parsing and config**

Create `src/tests/backtesting/test_matchup_fixture_audit.py` with:

```python
from __future__ import annotations

from pathlib import Path

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
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_matchup_fixture_audit.py -q
```

Expected: import failure because `cartola.backtesting.matchup_fixture_audit` does not exist yet.

- [ ] **Step 3: Implement config and parser**

Create `src/cartola/backtesting/matchup_fixture_audit.py` with:

```python
from __future__ import annotations

import argparse
import json
import traceback as traceback_module
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

DEFAULT_SEASONS = (2023, 2024, 2025)
DEFAULT_OUTPUT_ROOT = Path("data/08_reporting/fixtures")
CSV_ERROR_MESSAGE_LIMIT = 300

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class MatchupFixtureAuditConfig:
    seasons: tuple[int, ...] = DEFAULT_SEASONS
    project_root: Path = Path(".")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    expected_complete_rounds: int = 38
    complete_round_threshold: int = 38
    current_year: int | None = None

    def resolved_current_year(self, clock: Callable[[], datetime] | None = None) -> int:
        if self.current_year is not None:
            return self.current_year
        now = clock() if clock is not None else datetime.now(UTC)
        return now.year


def parse_seasons(value: str) -> tuple[int, ...]:
    raw_parts = value.split(",")
    if any(part.strip() == "" for part in raw_parts):
        raise ValueError("seasons must not contain empty entries")

    seasons: list[int] = []
    seen: set[int] = set()
    for part in raw_parts:
        try:
            season = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid season: {part}") from exc
        if season <= 0:
            raise ValueError("seasons must be positive integers")
        if season in seen:
            raise ValueError(f"duplicate season: {season}")
        seasons.append(season)
        seen.add(season)
    return tuple(seasons)


def parse_seasons_arg(value: str) -> tuple[int, ...]:
    try:
        return parse_seasons(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Cartola matchup fixture coverage.")
    parser.add_argument("--seasons", type=parse_seasons_arg, default=DEFAULT_SEASONS)
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-complete-rounds", type=int, default=38)
    parser.add_argument("--complete-round-threshold", type=int, default=38)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MatchupFixtureAuditConfig:
    return MatchupFixtureAuditConfig(
        seasons=args.seasons,
        project_root=args.project_root,
        output_root=args.output_root,
        expected_complete_rounds=args.expected_complete_rounds,
        complete_round_threshold=args.complete_round_threshold,
        current_year=args.current_year,
    )
```

- [ ] **Step 4: Verify parsing tests pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_matchup_fixture_audit.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/matchup_fixture_audit.py src/tests/backtesting/test_matchup_fixture_audit.py
git commit -m "feat: add matchup fixture audit config"
```

---

### Task 2: Build Fixture Context Rows With Source Metadata

**Files:**
- Modify: `src/cartola/backtesting/matchup_fixture_audit.py`
- Test: `src/tests/backtesting/test_matchup_fixture_audit.py`

- [ ] **Step 1: Write failing tests for exploratory context rows**

Append to `src/tests/backtesting/test_matchup_fixture_audit.py`:

```python
import pandas as pd


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
```

- [ ] **Step 2: Run focused failing tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_fixture_context_rows_expand_home_and_away_with_exploratory_source \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_fixture_context_prefers_strict_when_valid \
  -q
```

Expected: failures because `load_round_fixture_context(...)` is not implemented.

- [ ] **Step 3: Implement context loading**

Add imports near the top of `src/cartola/backtesting/matchup_fixture_audit.py`:

```python
from cartola.backtesting.data import normalize_fixture_frame
from cartola.backtesting.strict_fixtures import sha256_file, validate_strict_manifest
```

Add constants and helpers:

```python
CONTEXT_COLUMNS: tuple[str, ...] = (
    "season",
    "rodada",
    "id_clube",
    "opponent_id_clube",
    "opponent_nome_clube",
    "is_home",
    "fixture_source",
    "source_file",
    "source_manifest",
    "source_sha256",
    "source_manifest_sha256",
)


def _relative_path(path: Path, project_root: Path) -> str:
    resolved_root = project_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _empty_context_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(CONTEXT_COLUMNS))


def _context_rows_from_fixture(
    fixture_frame: pd.DataFrame,
    *,
    season: int,
    source: str,
    source_file: Path,
    source_manifest: Path | None,
    project_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_file_label = _relative_path(source_file, project_root)
    source_manifest_label = None if source_manifest is None else _relative_path(source_manifest, project_root)
    source_hash = sha256_file(source_file)
    manifest_hash = None if source_manifest is None else sha256_file(source_manifest)

    for fixture in fixture_frame.to_dict("records"):
        round_number = int(fixture["rodada"])
        home_id = int(fixture["id_clube_home"])
        away_id = int(fixture["id_clube_away"])
        rows.append(
            {
                "season": season,
                "rodada": round_number,
                "id_clube": home_id,
                "opponent_id_clube": away_id,
                "opponent_nome_clube": None,
                "is_home": 1,
                "fixture_source": source,
                "source_file": source_file_label,
                "source_manifest": source_manifest_label,
                "source_sha256": source_hash,
                "source_manifest_sha256": manifest_hash,
            }
        )
        rows.append(
            {
                "season": season,
                "rodada": round_number,
                "id_clube": away_id,
                "opponent_id_clube": home_id,
                "opponent_nome_clube": None,
                "is_home": 0,
                "fixture_source": source,
                "source_file": source_file_label,
                "source_manifest": source_manifest_label,
                "source_sha256": source_hash,
                "source_manifest_sha256": manifest_hash,
            }
        )

    return pd.DataFrame(rows, columns=pd.Index(CONTEXT_COLUMNS))


def load_round_fixture_context(
    project_root: str | Path,
    *,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    root = Path(project_root)
    strict_path = root / "data" / "01_raw" / "fixtures_strict" / str(season) / f"partidas-{round_number}.csv"
    if strict_path.exists():
        validate_strict_manifest(
            project_root=root,
            fixture_path=strict_path,
            season=season,
            round_number=round_number,
        )
        strict_frame = normalize_fixture_frame(pd.read_csv(strict_path), source=strict_path)
        return _context_rows_from_fixture(
            strict_frame,
            season=season,
            source="strict",
            source_file=strict_path,
            source_manifest=strict_path.with_suffix(".manifest.json"),
            project_root=root,
        )

    exploratory_path = root / "data" / "01_raw" / "fixtures" / str(season) / f"partidas-{round_number}.csv"
    if exploratory_path.exists():
        exploratory_frame = normalize_fixture_frame(pd.read_csv(exploratory_path), source=exploratory_path)
        return _context_rows_from_fixture(
            exploratory_frame,
            season=season,
            source="exploratory",
            source_file=exploratory_path,
            source_manifest=None,
            project_root=root,
        )

    return _empty_context_frame()
```

- [ ] **Step 4: Verify fixture context tests pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_fixture_context_rows_expand_home_and_away_with_exploratory_source \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_fixture_context_prefers_strict_when_valid \
  -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/matchup_fixture_audit.py src/tests/backtesting/test_matchup_fixture_audit.py
git commit -m "feat: build matchup fixture context rows"
```

---

### Task 3: Audit One Season For Missing, Duplicate, And Extra Context

**Files:**
- Modify: `src/cartola/backtesting/matchup_fixture_audit.py`
- Test: `src/tests/backtesting/test_matchup_fixture_audit.py`

- [ ] **Step 1: Write failing tests for season coverage**

Append to `src/tests/backtesting/test_matchup_fixture_audit.py`:

```python
def _season_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rodada": 1, "id_clube": 10, "nome_clube": "A", "entrou_em_campo": True},
            {"rodada": 1, "id_clube": 20, "nome_clube": "B", "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 10, "nome_clube": "A", "entrou_em_campo": True},
            {"rodada": 2, "id_clube": 30, "nome_clube": "C", "entrou_em_campo": True},
        ]
    )


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = pd.DataFrame(
        [
            {"season": 2025, "rodada": 1, "id_clube": 10, "opponent_id_clube": 20, "opponent_nome_clube": None, "is_home": 1, "fixture_source": "exploratory", "source_file": "f1", "source_manifest": None, "source_sha256": "a", "source_manifest_sha256": None},
            {"season": 2025, "rodada": 1, "id_clube": 20, "opponent_id_clube": 10, "opponent_nome_clube": None, "is_home": 0, "fixture_source": "exploratory", "source_file": "f1", "source_manifest": None, "source_sha256": "a", "source_manifest_sha256": None},
            {"season": 2025, "rodada": 2, "id_clube": 10, "opponent_id_clube": 30, "opponent_nome_clube": None, "is_home": 1, "fixture_source": "exploratory", "source_file": "f2", "source_manifest": None, "source_sha256": "b", "source_manifest_sha256": None},
            {"season": 2025, "rodada": 2, "id_clube": 30, "opponent_id_clube": 10, "opponent_nome_clube": None, "is_home": 0, "fixture_source": "exploratory", "source_file": "f2", "source_manifest": None, "source_sha256": "b", "source_manifest_sha256": None},
        ]
    )

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_df())
    monkeypatch.setattr(
        audit,
        "load_round_fixture_context",
        lambda project_root, *, season, round_number: context[context["rodada"].eq(round_number)].copy(),
    )

    record = audit.audit_one_season(2025, audit.MatchupFixtureAuditConfig(project_root=tmp_path, current_year=2026))

    assert record.fixture_status == "ok"
    assert record.metrics_comparable is True
    assert record.expected_club_round_count == 4
    assert record.fixture_context_row_count == 4
    assert record.missing_context_count == 0
    assert record.duplicate_context_count == 0
    assert record.extra_context_count == 0


def test_audit_season_fails_missing_duplicate_and_extra_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = pd.DataFrame(
        [
            {"season": 2025, "rodada": 1, "id_clube": 10, "opponent_id_clube": 20, "opponent_nome_clube": None, "is_home": 1, "fixture_source": "exploratory", "source_file": "f1", "source_manifest": None, "source_sha256": "a", "source_manifest_sha256": None},
            {"season": 2025, "rodada": 1, "id_clube": 10, "opponent_id_clube": 20, "opponent_nome_clube": None, "is_home": 1, "fixture_source": "exploratory", "source_file": "f1", "source_manifest": None, "source_sha256": "a", "source_manifest_sha256": None},
            {"season": 2025, "rodada": 1, "id_clube": 99, "opponent_id_clube": 20, "opponent_nome_clube": None, "is_home": 0, "fixture_source": "exploratory", "source_file": "f1", "source_manifest": None, "source_sha256": "a", "source_manifest_sha256": None},
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
```

- [ ] **Step 2: Run focused failing tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_classify_complete_historical_and_partial_current \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_audit_season_passes_when_every_played_club_has_one_context_row \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_audit_season_fails_missing_duplicate_and_extra_context \
  -q
```

Expected: failures because classification and season audit are not implemented.

- [ ] **Step 3: Implement season record and coverage checks**

Add imports:

```python
from cartola.backtesting.data import load_season_data, played_club_set
```

Add dataclasses and helpers:

```python
@dataclass(frozen=True)
class ErrorDetail:
    stage: str
    exception_type: str
    message: str
    traceback: str | None = None


@dataclass
class SeasonMatchupFixtureRecord:
    season: int
    season_status: str
    metrics_comparable: bool
    fixture_status: str
    round_file_count: int
    min_round: int | None
    max_round: int | None
    detected_rounds: list[int]
    expected_club_round_count: int = 0
    fixture_context_row_count: int = 0
    missing_context_count: int = 0
    duplicate_context_count: int = 0
    extra_context_count: int = 0
    strict_context_count: int = 0
    exploratory_context_count: int = 0
    fixture_sources: list[str] = field(default_factory=list)
    rounds: list[dict[str, object]] = field(default_factory=list)
    missing_context_keys: list[dict[str, int]] = field(default_factory=list)
    duplicate_context_keys: list[dict[str, int]] = field(default_factory=list)
    extra_context_keys: list[dict[str, int]] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    source_manifests: list[str] = field(default_factory=list)
    error_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    notes: list[str] = field(default_factory=list)
    error_detail: ErrorDetail | None = None
```

Add classification:

```python
def classify_season(
    season: int,
    detected_rounds: list[int],
    config: MatchupFixtureAuditConfig,
) -> tuple[str, bool, list[str]]:
    notes: list[str] = []
    max_round = max(detected_rounds) if detected_rounds else 0
    if season == config.resolved_current_year() and max_round < config.complete_round_threshold:
        notes.append("partial current season; matchup coverage is not historically comparable")
        return "partial_current", False, notes

    expected = list(range(1, config.expected_complete_rounds + 1))
    if detected_rounds == expected:
        return "complete_historical", True, notes

    notes.append("historical season has unusual round file count or round sequence")
    return "irregular_historical", False, notes
```

Add season-data discovery and audit helpers:

```python
def _detected_rounds(season_df: pd.DataFrame) -> list[int]:
    rounds = pd.to_numeric(season_df["rodada"], errors="raise").astype(int)
    return sorted(rounds.dropna().unique().tolist())


def _expected_keys(season_df: pd.DataFrame, detected_rounds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for round_number in detected_rounds:
        for club_id in sorted(played_club_set(season_df, round_number)):
            rows.append({"rodada": int(round_number), "id_clube": int(club_id)})
    return pd.DataFrame(rows, columns=pd.Index(["rodada", "id_clube"]))


def _key_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, int]]:
    if frame.empty:
        return []
    return [
        {column: int(row[column]) for column in columns}
        for row in frame[columns].drop_duplicates().sort_values(columns).to_dict("records")
    ]


def _duplicate_key_records(context: pd.DataFrame) -> list[dict[str, int]]:
    if context.empty:
        return []
    counts = context.groupby(["rodada", "id_clube"], as_index=False).size()
    counts = counts[counts["size"].gt(1)].rename(columns={"size": "count"})
    return [
        {"rodada": int(row["rodada"]), "id_clube": int(row["id_clube"]), "count": int(row["count"])}
        for row in counts.sort_values(["rodada", "id_clube"]).to_dict("records")
    ]


def _error_detail(stage: str, exc: BaseException) -> ErrorDetail:
    return ErrorDetail(
        stage=stage,
        exception_type=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback_module.format_exception(type(exc), exc, exc.__traceback__)),
    )


def _short_error_message(message: str) -> str:
    one_line = " ".join(str(message).split())
    if len(one_line) <= CSV_ERROR_MESSAGE_LIMIT:
        return one_line
    return one_line[: CSV_ERROR_MESSAGE_LIMIT - 3] + "..."


def _apply_error(record: SeasonMatchupFixtureRecord, stage: str, error: ErrorDetail) -> None:
    record.fixture_status = STATUS_FAILED
    record.metrics_comparable = False
    record.error_stage = stage
    record.error_type = error.exception_type
    record.error_message = _short_error_message(error.message)
    record.error_detail = error
```

Add `audit_one_season`:

```python
def audit_one_season(season: int, config: MatchupFixtureAuditConfig) -> SeasonMatchupFixtureRecord:
    try:
        season_df = load_season_data(season, project_root=config.project_root)
        detected_rounds = _detected_rounds(season_df)
    except Exception as exc:  # noqa: BLE001 - audit reports per-season failures
        error = _error_detail("load", exc)
        record = SeasonMatchupFixtureRecord(
            season=season,
            season_status="load_failed",
            metrics_comparable=False,
            fixture_status=STATUS_FAILED,
            round_file_count=0,
            min_round=None,
            max_round=None,
            detected_rounds=[],
        )
        _apply_error(record, "load", error)
        return record

    season_status, season_comparable, notes = classify_season(season, detected_rounds, config)
    record = SeasonMatchupFixtureRecord(
        season=season,
        season_status=season_status,
        metrics_comparable=season_comparable,
        fixture_status=STATUS_SKIPPED,
        round_file_count=len(detected_rounds),
        min_round=min(detected_rounds) if detected_rounds else None,
        max_round=max(detected_rounds) if detected_rounds else None,
        detected_rounds=detected_rounds,
        notes=notes,
    )

    expected = _expected_keys(season_df, detected_rounds)
    context_frames: list[pd.DataFrame] = []
    for round_number in detected_rounds:
        try:
            round_context = load_round_fixture_context(
                config.project_root,
                season=season,
                round_number=round_number,
            )
        except Exception as exc:  # noqa: BLE001 - audit reports per-season failures
            _apply_error(record, "fixture", _error_detail("fixture", exc))
            return record
        context_frames.append(round_context)
        record.rounds.append(
            {
                "rodada": int(round_number),
                "expected_club_count": int(len(played_club_set(season_df, round_number))),
                "fixture_context_row_count": int(len(round_context)),
                "fixture_sources": sorted(str(value) for value in round_context["fixture_source"].dropna().unique())
                if not round_context.empty
                else [],
            }
        )

    context = pd.concat(context_frames, ignore_index=True) if context_frames else _empty_context_frame()
    record.expected_club_round_count = int(len(expected))
    record.fixture_context_row_count = int(len(context))
    if not context.empty:
        record.strict_context_count = int(context["fixture_source"].eq("strict").sum())
        record.exploratory_context_count = int(context["fixture_source"].eq("exploratory").sum())
        record.fixture_sources = sorted(str(value) for value in context["fixture_source"].dropna().unique())
        record.source_files = sorted(str(value) for value in context["source_file"].dropna().unique())
        record.source_manifests = sorted(str(value) for value in context["source_manifest"].dropna().unique())

    unique_context = context[["rodada", "id_clube"]].drop_duplicates() if not context.empty else pd.DataFrame(columns=["rodada", "id_clube"])
    missing = expected.merge(unique_context, on=["rodada", "id_clube"], how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")][["rodada", "id_clube"]]
    extra = unique_context.merge(expected, on=["rodada", "id_clube"], how="left", indicator=True)
    extra = extra[extra["_merge"].eq("left_only")][["rodada", "id_clube"]]
    duplicate_keys = _duplicate_key_records(context)

    record.missing_context_keys = _key_records(missing, ["rodada", "id_clube"])
    record.duplicate_context_keys = duplicate_keys
    record.extra_context_keys = _key_records(extra, ["rodada", "id_clube"])
    record.missing_context_count = len(record.missing_context_keys)
    record.duplicate_context_count = len(record.duplicate_context_keys)
    record.extra_context_count = len(record.extra_context_keys)

    if record.missing_context_count or record.duplicate_context_count or record.extra_context_count:
        record.fixture_status = STATUS_FAILED
        record.metrics_comparable = False
        record.notes.append("fixture context coverage has missing, duplicate, or extra club-round keys")
        return record

    record.fixture_status = STATUS_OK
    record.metrics_comparable = season_comparable
    return record
```

- [ ] **Step 4: Verify season audit tests pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_classify_complete_historical_and_partial_current \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_audit_season_passes_when_every_played_club_has_one_context_row \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_audit_season_fails_missing_duplicate_and_extra_context \
  -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/matchup_fixture_audit.py src/tests/backtesting/test_matchup_fixture_audit.py
git commit -m "feat: audit matchup fixture coverage by season"
```

---

### Task 4: Write CSV/JSON Reports And Gate Decision

**Files:**
- Modify: `src/cartola/backtesting/matchup_fixture_audit.py`
- Test: `src/tests/backtesting/test_matchup_fixture_audit.py`

- [ ] **Step 1: Write failing report and decision tests**

Append to `src/tests/backtesting/test_matchup_fixture_audit.py`:

```python
import json


def _record(season: int, *, comparable: bool, status: str = "complete_historical") -> audit.SeasonMatchupFixtureRecord:
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


def test_decision_ready_when_2023_2024_2025_are_comparable() -> None:
    decision = audit.build_decision([_record(2023, comparable=True), _record(2024, comparable=True), _record(2025, comparable=True)])

    assert decision["status"] == "ready_for_matchup_context"
    assert decision["recommended_next_step"] == "implement matchup_context_mode=cartola_matchup_v1"


def test_decision_exploratory_only_when_only_2025_passes() -> None:
    decision = audit.build_decision([_record(2023, comparable=False), _record(2024, comparable=False), _record(2025, comparable=True)])

    assert decision["status"] == "exploratory_only"
    assert decision["recommended_next_step"] == "keep matchup context exploratory until 2023-2024 fixture coverage is fixed"


def test_decision_blocks_when_2025_fails() -> None:
    decision = audit.build_decision([_record(2023, comparable=True), _record(2024, comparable=True), _record(2025, comparable=False)])

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
```

- [ ] **Step 2: Run focused failing tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_decision_ready_when_2023_2024_2025_are_comparable \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_decision_exploratory_only_when_only_2025_passes \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_decision_blocks_when_2025_fails \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_write_reports_outputs_csv_and_json \
  -q
```

Expected: failures because report writing and decisions are not implemented.

- [ ] **Step 3: Implement report serialization**

Add `CSV_COLUMNS`:

```python
CSV_COLUMNS: tuple[str, ...] = (
    "season",
    "season_status",
    "metrics_comparable",
    "fixture_status",
    "round_file_count",
    "min_round",
    "max_round",
    "detected_rounds",
    "expected_club_round_count",
    "fixture_context_row_count",
    "missing_context_count",
    "duplicate_context_count",
    "extra_context_count",
    "strict_context_count",
    "exploratory_context_count",
    "fixture_sources",
    "error_stage",
    "error_type",
    "error_message",
    "notes",
)
```

Add methods to `SeasonMatchupFixtureRecord`:

```python
    def to_csv_row(self) -> dict[str, object]:
        return {
            "season": self.season,
            "season_status": self.season_status,
            "metrics_comparable": self.metrics_comparable,
            "fixture_status": self.fixture_status,
            "round_file_count": self.round_file_count,
            "min_round": self.min_round,
            "max_round": self.max_round,
            "detected_rounds": ",".join(str(round_number) for round_number in self.detected_rounds),
            "expected_club_round_count": self.expected_club_round_count,
            "fixture_context_row_count": self.fixture_context_row_count,
            "missing_context_count": self.missing_context_count,
            "duplicate_context_count": self.duplicate_context_count,
            "extra_context_count": self.extra_context_count,
            "strict_context_count": self.strict_context_count,
            "exploratory_context_count": self.exploratory_context_count,
            "fixture_sources": ",".join(self.fixture_sources),
            "error_stage": self.error_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "notes": "; ".join(self.notes),
        }

    def to_json_object(self) -> dict[str, object]:
        row = self.to_csv_row()
        row["detected_rounds"] = self.detected_rounds
        row["fixture_sources"] = self.fixture_sources
        row["notes"] = self.notes
        row["rounds"] = self.rounds
        row["missing_context_keys"] = self.missing_context_keys
        row["duplicate_context_keys"] = self.duplicate_context_keys
        row["extra_context_keys"] = self.extra_context_keys
        row["source_files"] = self.source_files
        row["source_manifests"] = self.source_manifests
        row["error_detail"] = None if self.error_detail is None else asdict(self.error_detail)
        return row
```

Add decision and writer:

```python
def build_decision(records: list[SeasonMatchupFixtureRecord]) -> dict[str, object]:
    comparable = {record.season for record in records if record.metrics_comparable}
    requested = {record.season for record in records}
    core = {2023, 2024, 2025}
    if core.issubset(comparable):
        status = "ready_for_matchup_context"
        next_step = "implement matchup_context_mode=cartola_matchup_v1"
    elif 2025 in comparable:
        status = "exploratory_only"
        next_step = "keep matchup context exploratory until 2023-2024 fixture coverage is fixed"
    else:
        status = "coverage_blocked"
        next_step = "fix or import fixture coverage before feature work"

    return {
        "status": status,
        "comparable_seasons": sorted(comparable),
        "requested_seasons": sorted(requested),
        "recommended_next_step": next_step,
    }


def _config_json(config: MatchupFixtureAuditConfig) -> dict[str, object]:
    return {
        "seasons": list(config.seasons),
        "project_root": str(config.project_root),
        "output_root": str(config.output_root),
        "expected_complete_rounds": config.expected_complete_rounds,
        "complete_round_threshold": config.complete_round_threshold,
        "current_year": config.current_year,
    }


def write_audit_reports(
    records: list[SeasonMatchupFixtureRecord],
    config: MatchupFixtureAuditConfig,
    *,
    generated_at_utc: str,
) -> tuple[Path, Path]:
    output_dir = config.output_root if config.output_root.is_absolute() else config.project_root / config.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "matchup_fixture_coverage.csv"
    json_path = output_dir / "matchup_fixture_coverage.json"

    pd.DataFrame([record.to_csv_row() for record in records], columns=pd.Index(CSV_COLUMNS)).to_csv(csv_path, index=False)
    payload = {
        "generated_at_utc": generated_at_utc,
        "project_root": str(config.project_root),
        "config": _config_json(config),
        "decision": build_decision(records),
        "seasons": [record.to_json_object() for record in records],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path
```

- [ ] **Step 4: Verify report tests pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_decision_ready_when_2023_2024_2025_are_comparable \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_decision_exploratory_only_when_only_2025_passes \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_decision_blocks_when_2025_fails \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_write_reports_outputs_csv_and_json \
  -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/matchup_fixture_audit.py src/tests/backtesting/test_matchup_fixture_audit.py
git commit -m "feat: report matchup fixture coverage decisions"
```

---

### Task 5: Add Audit Orchestration And CLI Script

**Files:**
- Modify: `src/cartola/backtesting/matchup_fixture_audit.py`
- Create: `scripts/audit_matchup_fixture_coverage.py`
- Create: `src/tests/backtesting/test_audit_matchup_fixture_coverage_cli.py`

- [ ] **Step 1: Write failing orchestration and CLI tests**

Append to `src/tests/backtesting/test_matchup_fixture_audit.py`:

```python
def test_run_matchup_fixture_audit_writes_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_audit_one_season(season: int, config: audit.MatchupFixtureAuditConfig) -> audit.SeasonMatchupFixtureRecord:
        calls.append(season)
        return _record(season, comparable=season == 2025)

    monkeypatch.setattr(audit, "audit_one_season", fake_audit_one_season)

    result = audit.run_matchup_fixture_audit(
        audit.MatchupFixtureAuditConfig(
            seasons=(2024, 2025),
            project_root=tmp_path,
            current_year=2026,
        ),
        clock=lambda: pd.Timestamp("2026-04-30T12:00:00Z").to_pydatetime(),
    )

    assert calls == [2024, 2025]
    assert result.csv_path.exists()
    assert result.json_path.exists()
    assert result.decision["status"] == "exploratory_only"
```

Create `src/tests/backtesting/test_audit_matchup_fixture_coverage_cli.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

from cartola.backtesting import matchup_fixture_audit as audit


def test_script_imports_main_from_matchup_fixture_audit() -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "audit_matchup_fixture_coverage.py"
    spec = importlib.util.spec_from_file_location("audit_matchup_fixture_coverage", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.main is audit.main


def test_main_runs_audit_and_prints_paths(monkeypatch, capsys, tmp_path: Path) -> None:
    observed_configs: list[audit.MatchupFixtureAuditConfig] = []

    def fake_run(config: audit.MatchupFixtureAuditConfig) -> audit.MatchupFixtureAuditRunResult:
        observed_configs.append(config)
        csv_path = tmp_path / "matchup_fixture_coverage.csv"
        json_path = tmp_path / "matchup_fixture_coverage.json"
        return audit.MatchupFixtureAuditRunResult(
            generated_at_utc="2026-04-30T00:00:00+00:00",
            project_root=config.project_root,
            config=config,
            seasons=[],
            csv_path=csv_path,
            json_path=json_path,
            decision={"status": "coverage_blocked"},
        )

    monkeypatch.setattr(audit, "run_matchup_fixture_audit", fake_run)

    exit_code = audit.main(["--project-root", str(tmp_path), "--current-year", "2026"])

    assert exit_code == 0
    assert observed_configs == [audit.MatchupFixtureAuditConfig(project_root=tmp_path, current_year=2026)]
    output = capsys.readouterr().out
    assert "Matchup fixture coverage audit complete" in output
    assert "matchup_fixture_coverage.csv" in output
    assert "matchup_fixture_coverage.json" in output
    assert "Decision: coverage_blocked" in output
```

- [ ] **Step 2: Run focused failing tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_run_matchup_fixture_audit_writes_reports \
  src/tests/backtesting/test_audit_matchup_fixture_coverage_cli.py \
  -q
```

Expected: failures because run result, main, and script are not implemented.

- [ ] **Step 3: Implement orchestration and main**

Add to `src/cartola/backtesting/matchup_fixture_audit.py`:

```python
@dataclass(frozen=True)
class MatchupFixtureAuditRunResult:
    generated_at_utc: str
    project_root: Path
    config: MatchupFixtureAuditConfig
    seasons: list[SeasonMatchupFixtureRecord]
    csv_path: Path
    json_path: Path
    decision: dict[str, object]


def run_matchup_fixture_audit(
    config: MatchupFixtureAuditConfig,
    *,
    clock: Callable[[], datetime] | None = None,
) -> MatchupFixtureAuditRunResult:
    generated_at = (clock() if clock is not None else datetime.now(UTC)).astimezone(UTC)
    resolved_config = replace(config, current_year=config.resolved_current_year(clock))
    seasons = [audit_one_season(season, resolved_config) for season in resolved_config.seasons]
    csv_path, json_path = write_audit_reports(
        seasons,
        resolved_config,
        generated_at_utc=generated_at.isoformat(),
    )
    decision = build_decision(seasons)
    return MatchupFixtureAuditRunResult(
        generated_at_utc=generated_at.isoformat(),
        project_root=resolved_config.project_root,
        config=resolved_config,
        seasons=seasons,
        csv_path=csv_path,
        json_path=json_path,
        decision=decision,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    result = run_matchup_fixture_audit(config)
    print("Matchup fixture coverage audit complete")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Decision: {result.decision['status']}")
    return 0
```

Create `scripts/audit_matchup_fixture_coverage.py`:

```python
from cartola.backtesting.matchup_fixture_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify orchestration and CLI tests pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_matchup_fixture_audit.py::test_run_matchup_fixture_audit_writes_reports \
  src/tests/backtesting/test_audit_matchup_fixture_coverage_cli.py \
  -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add \
  src/cartola/backtesting/matchup_fixture_audit.py \
  scripts/audit_matchup_fixture_coverage.py \
  src/tests/backtesting/test_matchup_fixture_audit.py \
  src/tests/backtesting/test_audit_matchup_fixture_coverage_cli.py
git commit -m "feat: add matchup fixture coverage audit command"
```

---

### Task 6: Document Command And Run Verification

**Files:**
- Modify: `README.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Update README**

Add this section near the fixture/backtesting audit documentation in `README.md`:

```markdown
### Matchup fixture coverage audit

Before adding Cartola opponent/home-away matchup features, audit whether the requested seasons have exactly one fixture context row for each played club-round:

```bash
uv run --frozen python scripts/audit_matchup_fixture_coverage.py \
  --seasons 2023,2024,2025 \
  --current-year 2026
```

Reports are written to:

- `data/08_reporting/fixtures/matchup_fixture_coverage.csv`
- `data/08_reporting/fixtures/matchup_fixture_coverage.json`

Interpretation:

- `ready_for_matchup_context`: 2023, 2024, and 2025 are coverage-clean and comparable for `matchup_context_mode=cartola_matchup_v1`.
- `exploratory_only`: 2025 passes, but 2023 or 2024 needs fixture coverage work before multi-season evaluation.
- `coverage_blocked`: fixture coverage must be fixed before feature work.
```

- [ ] **Step 2: Update roadmap**

Add this item under the next matchup/context work in `roadmap.md`:

```markdown
- [ ] Run `scripts/audit_matchup_fixture_coverage.py` for 2023-2025 and require `ready_for_matchup_context` before implementing `matchup_context_mode=cartola_matchup_v1`.
```

- [ ] **Step 3: Run audit unit tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_matchup_fixture_audit.py \
  src/tests/backtesting/test_audit_matchup_fixture_coverage_cli.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 4: Run existing fixture and compatibility tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_data.py \
  src/tests/backtesting/test_strict_fixtures.py \
  src/tests/backtesting/test_compatibility_audit.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Run the real audit command**

Run:

```bash
uv run --frozen python scripts/audit_matchup_fixture_coverage.py \
  --seasons 2023,2024,2025 \
  --current-year 2026
```

Expected: command exits `0` and writes:

```text
data/08_reporting/fixtures/matchup_fixture_coverage.csv
data/08_reporting/fixtures/matchup_fixture_coverage.json
```

Expected decision with current local data may be `exploratory_only` or `coverage_blocked` until fixture files exist for 2023 and 2024. Do not change feature work based on assumption; use the generated report.

- [ ] **Step 6: Run the full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: all configured checks pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add README.md roadmap.md
git commit -m "docs: document matchup fixture coverage audit"
```

---

## Final Acceptance Checklist

- [ ] `scripts/audit_matchup_fixture_coverage.py` exists and imports `main` from `cartola.backtesting.matchup_fixture_audit`.
- [ ] Audit outputs are written to `data/08_reporting/fixtures/matchup_fixture_coverage.csv` and `.json`.
- [ ] Missing, duplicate, and extra club-round fixture contexts fail the season.
- [ ] Strict fixture rows validate manifests before they count.
- [ ] Exploratory fixture rows record source file and null manifest fields.
- [ ] `partial_current` seasons are never marked comparable.
- [ ] Existing backtest, recommendation, optimizer, and feature code paths are unchanged.
- [ ] The real 2023-2025 audit has been run and the generated decision is recorded in the terminal output.
