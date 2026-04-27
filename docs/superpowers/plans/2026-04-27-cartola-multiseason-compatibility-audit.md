# Cartola Multi-Season Compatibility Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an audit-only command that scans every available raw season, runs loader/feature/no-fixture backtest compatibility checks, and writes isolated CSV/JSON reports.

**Architecture:** Put the testable audit engine in `src/cartola/backtesting/compatibility_audit.py` and keep `scripts/audit_backtest_compatibility.py` as a thin CLI wrapper. The audit always uses `fixture_mode="none"`, passes `project_root` explicitly, and passes `output_root=Path("data/08_reporting/backtests/compatibility/runs")` into `BacktestConfig` so per-season backtest outputs land under `.../runs/{season}/`.

**Tech Stack:** Python 3.13.12, pandas, dataclasses, pathlib, argparse, pytest, uv, Ruff, ty, Bandit.

---

## File Structure

- Create `src/cartola/backtesting/compatibility_audit.py`
  - Audit configuration dataclass.
  - Season discovery and round filename parsing.
  - Season classification.
  - Per-season staged audit execution.
  - Metric extraction.
  - CSV/JSON report writing.
  - CLI argument parsing and `main`.

- Create `scripts/audit_backtest_compatibility.py`
  - Thin wrapper around `cartola.backtesting.compatibility_audit.main`.

- Create `src/tests/backtesting/test_compatibility_audit.py`
  - Unit tests for discovery, classification, staged execution, metrics, reports, and CLI.

- Modify `README.md`
  - Document the local audit command and output files.

---

## Shared Test Helpers

Use these helpers at the top of `src/tests/backtesting/test_compatibility_audit.py` after imports.

```python
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cartola.backtesting import compatibility_audit as audit
from cartola.backtesting.config import BacktestConfig


def _touch_round(root: Path, season: int, round_name: str) -> Path:
    path = root / "data" / "01_raw" / str(season) / round_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")
    return path


def _season_frame(rounds: range) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rodada": list(rounds),
            "id_atleta": list(range(1, len(list(rounds)) + 1)),
            "id_clube": [10] * len(list(rounds)),
        }
    )


def _fake_backtest_result(summary: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(summary=summary)
```

---

### Task 1: Discovery, Round Parsing, And Season Classification

**Files:**
- Create: `src/cartola/backtesting/compatibility_audit.py`
- Create: `src/tests/backtesting/test_compatibility_audit.py`

- [ ] **Step 1: Write failing discovery and classification tests**

Create `src/tests/backtesting/test_compatibility_audit.py` with the shared helpers above and these tests:

```python
def test_discover_seasons_includes_numeric_dirs_with_round_files(tmp_path: Path) -> None:
    _touch_round(tmp_path, 2025, "rodada-1.csv")
    _touch_round(tmp_path, 2025, "rodada-2.csv")
    _touch_round(tmp_path, 2026, "rodada-1.csv")
    (tmp_path / "data" / "01_raw" / "fixtures" / "2025").mkdir(parents=True)
    (tmp_path / "data" / "01_raw" / "notes").mkdir(parents=True)
    (tmp_path / "data" / "01_raw" / "2024").mkdir(parents=True)

    config = audit.AuditConfig(project_root=tmp_path, current_year=2026)

    seasons = audit.discover_seasons(config)

    assert [season.season for season in seasons] == [2025, 2026]
    assert seasons[0].detected_rounds == [1, 2]
    assert seasons[0].round_file_count == 2
    assert seasons[0].min_round == 1
    assert seasons[0].max_round == 2


def test_discover_seasons_records_malformed_round_filename(tmp_path: Path) -> None:
    _touch_round(tmp_path, 2025, "rodada-final.csv")

    config = audit.AuditConfig(project_root=tmp_path, current_year=2026)

    seasons = audit.discover_seasons(config)

    assert len(seasons) == 1
    assert seasons[0].season == 2025
    assert seasons[0].discovery_error is not None
    assert seasons[0].discovery_error.stage == "discovery"
    assert "Invalid round CSV filename" in seasons[0].discovery_error.message


def test_parse_round_number_rejects_zero_and_non_matching_names() -> None:
    with pytest.raises(ValueError, match="positive"):
        audit.parse_round_number(Path("rodada-0.csv"))

    with pytest.raises(ValueError, match="Invalid round CSV filename"):
        audit.parse_round_number(Path("round-1.csv"))


def test_classify_season_requires_contiguous_complete_rounds() -> None:
    config = audit.AuditConfig(current_year=2026, expected_complete_rounds=38)

    complete = audit.classify_season(2025, list(range(1, 39)), config)
    gapped = audit.classify_season(2025, [*range(1, 7), *range(8, 40)], config)
    irregular_extra = audit.classify_season(2022, list(range(1, 40)), config)
    partial_current = audit.classify_season(2026, list(range(1, 14)), config)

    assert complete == ("complete_historical", True, [])
    assert gapped[0] == "irregular_historical"
    assert gapped[1] is False
    assert irregular_extra[0] == "irregular_historical"
    assert partial_current == (
        "partial_current",
        False,
        ["partial current season; metrics are smoke-test only"],
    )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py::test_discover_seasons_includes_numeric_dirs_with_round_files src/tests/backtesting/test_compatibility_audit.py::test_discover_seasons_records_malformed_round_filename src/tests/backtesting/test_compatibility_audit.py::test_parse_round_number_rejects_zero_and_non_matching_names src/tests/backtesting/test_compatibility_audit.py::test_classify_season_requires_contiguous_complete_rounds -q
```

Expected: FAIL because `cartola.backtesting.compatibility_audit` does not exist.

- [ ] **Step 3: Implement discovery and classification**

Create `src/cartola/backtesting/compatibility_audit.py` with this content:

```python
from __future__ import annotations

import argparse
import json
import re
import traceback as traceback_module
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.data import load_season_data
from cartola.backtesting.features import build_prediction_frame, build_training_frame
from cartola.backtesting.runner import run_backtest

ROUND_FILE_RE = re.compile(r"^rodada-(\d+)\.csv$")
EXPECTED_STRATEGIES: tuple[str, ...] = ("baseline", "random_forest", "price")
CSV_ERROR_MESSAGE_LIMIT = 300
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class AuditConfig:
    project_root: Path = Path(".")
    start_round: int = 5
    complete_round_threshold: int = 38
    expected_complete_rounds: int = 38
    current_year: int | None = None
    output_root: Path = Path("data/08_reporting/backtests/compatibility")
    fixture_mode: Literal["none"] = "none"

    def resolved_current_year(self, clock: Callable[[], datetime] | None = None) -> int:
        if self.current_year is not None:
            return self.current_year
        now = clock() if clock is not None else datetime.now(UTC)
        return now.year


@dataclass(frozen=True)
class ErrorDetail:
    stage: str
    exception_type: str
    message: str
    traceback: str | None = None
    target_round: int | None = None


@dataclass(frozen=True)
class SeasonDiscovery:
    season: int
    season_path: Path
    round_files: list[Path]
    round_file_count: int
    min_round: int | None
    max_round: int | None
    detected_rounds: list[int]
    discovery_error: ErrorDetail | None = None


@dataclass
class SeasonAuditRecord:
    season: int
    season_status: str
    metrics_comparable: bool
    round_file_count: int
    min_round: int | None
    max_round: int | None
    detected_rounds: list[int]
    start_round: int
    evaluated_rounds: int
    first_evaluated_round: int | None
    last_evaluated_round: int | None
    fixture_mode: str = "none"
    fixture_status: str = STATUS_NOT_APPLICABLE
    load_status: str = STATUS_SKIPPED
    feature_status: str = STATUS_SKIPPED
    backtest_status: str = STATUS_SKIPPED
    error_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    baseline_avg_points: float | None = None
    random_forest_avg_points: float | None = None
    price_avg_points: float | None = None
    notes: list[str] = field(default_factory=list)
    error_detail: ErrorDetail | None = None

    def to_csv_row(self) -> dict[str, object]:
        return {
            "season": self.season,
            "season_status": self.season_status,
            "metrics_comparable": self.metrics_comparable,
            "round_file_count": self.round_file_count,
            "min_round": self.min_round,
            "max_round": self.max_round,
            "detected_rounds": ",".join(str(round_number) for round_number in self.detected_rounds),
            "start_round": self.start_round,
            "evaluated_rounds": self.evaluated_rounds,
            "first_evaluated_round": self.first_evaluated_round,
            "last_evaluated_round": self.last_evaluated_round,
            "fixture_mode": self.fixture_mode,
            "fixture_status": self.fixture_status,
            "load_status": self.load_status,
            "feature_status": self.feature_status,
            "backtest_status": self.backtest_status,
            "error_stage": self.error_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "baseline_avg_points": self.baseline_avg_points,
            "random_forest_avg_points": self.random_forest_avg_points,
            "price_avg_points": self.price_avg_points,
            "notes": "; ".join(self.notes),
        }

    def to_json_object(self) -> dict[str, object]:
        row = self.to_csv_row()
        row["detected_rounds"] = self.detected_rounds
        row["notes"] = self.notes
        row["error_detail"] = None if self.error_detail is None else self.error_detail.__dict__
        return row


@dataclass(frozen=True)
class AuditRunResult:
    generated_at_utc: str
    project_root: Path
    config: AuditConfig
    seasons: list[SeasonAuditRecord]
    csv_path: Path
    json_path: Path


def parse_round_number(path: Path) -> int:
    match = ROUND_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid round CSV filename: {path}")
    round_number = int(match.group(1))
    if round_number <= 0:
        raise ValueError(f"Round number must be a positive integer: {path}")
    return round_number


def discover_seasons(config: AuditConfig) -> list[SeasonDiscovery]:
    raw_root = config.project_root / "data" / "01_raw"
    if not raw_root.exists():
        return []

    discoveries: list[SeasonDiscovery] = []
    for season_path in sorted((path for path in raw_root.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda p: int(p.name)):
        round_files = sorted(season_path.glob("rodada-*.csv"))
        if not round_files:
            continue

        try:
            detected_rounds = sorted(parse_round_number(path) for path in round_files)
        except Exception as exc:  # noqa: BLE001 - audit reports exceptions per season
            error = _error_detail("discovery", exc)
            discoveries.append(
                SeasonDiscovery(
                    season=int(season_path.name),
                    season_path=season_path,
                    round_files=round_files,
                    round_file_count=len(round_files),
                    min_round=None,
                    max_round=None,
                    detected_rounds=[],
                    discovery_error=error,
                )
            )
            continue

        discoveries.append(
            SeasonDiscovery(
                season=int(season_path.name),
                season_path=season_path,
                round_files=round_files,
                round_file_count=len(round_files),
                min_round=min(detected_rounds),
                max_round=max(detected_rounds),
                detected_rounds=detected_rounds,
            )
        )

    return discoveries


def classify_season(season: int, detected_rounds: list[int], config: AuditConfig) -> tuple[str, bool, list[str]]:
    notes: list[str] = []
    max_round = max(detected_rounds) if detected_rounds else 0
    if season == config.resolved_current_year() and max_round < config.complete_round_threshold:
        notes.append("partial current season; metrics are smoke-test only")
        return "partial_current", False, notes

    expected_rounds = list(range(1, config.expected_complete_rounds + 1))
    if len(detected_rounds) == config.expected_complete_rounds and detected_rounds == expected_rounds:
        return "complete_historical", True, notes

    notes.append("historical season has unusual round file count or round sequence")
    return "irregular_historical", False, notes


def _evaluation_metadata(max_round: int | None, start_round: int) -> tuple[int, int | None, int | None]:
    if max_round is None or max_round < start_round:
        return 0, None, None
    return max_round - start_round + 1, start_round, max_round


def _base_record(discovery: SeasonDiscovery, config: AuditConfig) -> SeasonAuditRecord:
    season_status, metrics_comparable, notes = classify_season(discovery.season, discovery.detected_rounds, config)
    evaluated_rounds, first_round, last_round = _evaluation_metadata(discovery.max_round, config.start_round)
    return SeasonAuditRecord(
        season=discovery.season,
        season_status=season_status,
        metrics_comparable=metrics_comparable,
        round_file_count=discovery.round_file_count,
        min_round=discovery.min_round,
        max_round=discovery.max_round,
        detected_rounds=discovery.detected_rounds,
        start_round=config.start_round,
        evaluated_rounds=evaluated_rounds,
        first_evaluated_round=first_round,
        last_evaluated_round=last_round,
        notes=notes,
    )


def _error_detail(stage: str, exc: BaseException, target_round: int | None = None) -> ErrorDetail:
    return ErrorDetail(
        stage=stage,
        exception_type=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback_module.format_exception(type(exc), exc, exc.__traceback__)),
        target_round=target_round,
    )


def _short_error_message(message: str) -> str:
    one_line = " ".join(str(message).split())
    if len(one_line) <= CSV_ERROR_MESSAGE_LIMIT:
        return one_line
    return one_line[: CSV_ERROR_MESSAGE_LIMIT - 1] + "…"
```

- [ ] **Step 4: Run discovery and classification tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py::test_discover_seasons_includes_numeric_dirs_with_round_files src/tests/backtesting/test_compatibility_audit.py::test_discover_seasons_records_malformed_round_filename src/tests/backtesting/test_compatibility_audit.py::test_parse_round_number_rejects_zero_and_non_matching_names src/tests/backtesting/test_compatibility_audit.py::test_classify_season_requires_contiguous_complete_rounds -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/compatibility_audit.py src/tests/backtesting/test_compatibility_audit.py
git commit -m "feat: add compatibility audit season discovery"
```

---

### Task 2: Load And Feature Compatibility Stages

**Files:**
- Modify: `src/cartola/backtesting/compatibility_audit.py`
- Modify: `src/tests/backtesting/test_compatibility_audit.py`

- [ ] **Step 1: Write failing staged-audit tests**

Append to `src/tests/backtesting/test_compatibility_audit.py`:

```python
def test_load_failure_records_row_and_skips_later_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch_round(tmp_path, 2025, "rodada-1.csv")

    def fail_load(season: int, project_root: Path) -> pd.DataFrame:
        raise ValueError(f"bad load {season} {project_root}")

    monkeypatch.setattr(audit, "load_season_data", fail_load)

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    record = result.seasons[0]
    assert record.load_status == "failed"
    assert record.feature_status == "skipped"
    assert record.backtest_status == "skipped"
    assert record.error_stage == "load"
    assert record.error_type == "ValueError"
    assert record.error_detail is not None
    assert record.error_detail.stage == "load"


def test_feature_check_covers_every_eligible_target_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 8):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    training_calls: list[int] = []
    prediction_calls: list[int] = []

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 8)))
    monkeypatch.setattr(
        audit,
        "build_training_frame",
        lambda season_df, target_round, playable_statuses, fixtures: training_calls.append(target_round) or pd.DataFrame(),
    )
    monkeypatch.setattr(
        audit,
        "build_prediction_frame",
        lambda season_df, target_round, fixtures: prediction_calls.append(target_round) or pd.DataFrame(),
    )
    monkeypatch.setattr(
        audit,
        "run_backtest",
        lambda config: _fake_backtest_result(
            pd.DataFrame(
                [
                    {"strategy": "baseline", "average_actual_points": 1.0},
                    {"strategy": "random_forest", "average_actual_points": 2.0},
                    {"strategy": "price", "average_actual_points": 3.0},
                ]
            )
        ),
    )

    result = audit.run_compatibility_audit(
        audit.AuditConfig(project_root=tmp_path, current_year=2026, start_round=5)
    )

    assert training_calls == [5, 6, 7]
    assert prediction_calls == [5, 6, 7]
    assert result.seasons[0].feature_status == "ok"


def test_feature_failure_records_target_round_and_skips_backtest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 8):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    def fail_training(season_df: pd.DataFrame, target_round: int, playable_statuses: tuple[str, ...], fixtures: None) -> pd.DataFrame:
        if target_round == 6:
            raise RuntimeError("feature broke")
        return pd.DataFrame()

    def fail_if_called(config: BacktestConfig) -> SimpleNamespace:
        raise AssertionError("backtest should be skipped")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 8)))
    monkeypatch.setattr(audit, "build_training_frame", fail_training)
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "run_backtest", fail_if_called)

    result = audit.run_compatibility_audit(
        audit.AuditConfig(project_root=tmp_path, current_year=2026, start_round=5)
    )

    record = result.seasons[0]
    assert record.feature_status == "failed"
    assert record.backtest_status == "skipped"
    assert record.error_stage == "feature"
    assert record.error_detail is not None
    assert record.error_detail.target_round == 6


def test_max_round_before_start_round_marks_feature_not_applicable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 4):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 4)))

    result = audit.run_compatibility_audit(
        audit.AuditConfig(project_root=tmp_path, current_year=2026, start_round=5)
    )

    record = result.seasons[0]
    assert record.load_status == "ok"
    assert record.feature_status == "not_applicable"
    assert record.backtest_status == "skipped"
    assert record.evaluated_rounds == 0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py::test_load_failure_records_row_and_skips_later_stages src/tests/backtesting/test_compatibility_audit.py::test_feature_check_covers_every_eligible_target_round src/tests/backtesting/test_compatibility_audit.py::test_feature_failure_records_target_round_and_skips_backtest src/tests/backtesting/test_compatibility_audit.py::test_max_round_before_start_round_marks_feature_not_applicable -q
```

Expected: FAIL because `run_compatibility_audit` does not exist.

- [ ] **Step 3: Implement load and feature stages**

Append to `src/cartola/backtesting/compatibility_audit.py`:

```python
def run_compatibility_audit(
    config: AuditConfig,
    *,
    clock: Callable[[], datetime] | None = None,
) -> AuditRunResult:
    generated_at = clock() if clock is not None else datetime.now(UTC)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    generated_at_utc = generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z")

    resolved_config = (
        config
        if config.current_year is not None
        else AuditConfig(
            project_root=config.project_root,
            start_round=config.start_round,
            complete_round_threshold=config.complete_round_threshold,
            expected_complete_rounds=config.expected_complete_rounds,
            current_year=config.resolved_current_year(clock),
            output_root=config.output_root,
            fixture_mode=config.fixture_mode,
        )
    )
    seasons = [_audit_one_season(discovery, resolved_config) for discovery in discover_seasons(resolved_config)]
    csv_path, json_path = write_audit_reports(
        generated_at_utc=generated_at_utc,
        config=resolved_config,
        seasons=seasons,
    )
    return AuditRunResult(
        generated_at_utc=generated_at_utc,
        project_root=resolved_config.project_root,
        config=resolved_config,
        seasons=seasons,
        csv_path=csv_path,
        json_path=json_path,
    )


def _audit_one_season(discovery: SeasonDiscovery, config: AuditConfig) -> SeasonAuditRecord:
    record = _base_record(discovery, config)
    if discovery.discovery_error is not None:
        _mark_failure(record, discovery.discovery_error, load_status=STATUS_FAILED)
        record.feature_status = STATUS_SKIPPED
        record.backtest_status = STATUS_SKIPPED
        return record

    try:
        season_df = load_season_data(discovery.season, project_root=config.project_root)
    except Exception as exc:  # noqa: BLE001 - audit reports exceptions per season
        _mark_failure(record, _error_detail("load", exc), load_status=STATUS_FAILED)
        record.feature_status = STATUS_SKIPPED
        record.backtest_status = STATUS_SKIPPED
        return record

    record.load_status = STATUS_OK
    if record.evaluated_rounds == 0 or discovery.max_round is None:
        record.feature_status = STATUS_NOT_APPLICABLE
        record.backtest_status = STATUS_SKIPPED
        return record

    feature_error = _check_feature_compatibility(season_df, config, discovery.max_round)
    if feature_error is not None:
        record.feature_status = STATUS_FAILED
        record.backtest_status = STATUS_SKIPPED
        _apply_error(record, feature_error)
        return record

    record.feature_status = STATUS_OK
    return _run_backtest_stage(record, config)


def _check_feature_compatibility(
    season_df: pd.DataFrame,
    config: AuditConfig,
    max_round: int,
) -> ErrorDetail | None:
    for target_round in range(config.start_round, max_round + 1):
        try:
            build_training_frame(
                season_df,
                target_round,
                playable_statuses=BacktestConfig().playable_statuses,
                fixtures=None,
            )
            build_prediction_frame(season_df, target_round, fixtures=None)
        except Exception as exc:  # noqa: BLE001 - audit reports exceptions per season
            return _error_detail("feature", exc, target_round=target_round)
    return None


def _run_backtest_stage(record: SeasonAuditRecord, config: AuditConfig) -> SeasonAuditRecord:
    record.backtest_status = STATUS_SKIPPED
    return record


def _mark_failure(record: SeasonAuditRecord, error: ErrorDetail, *, load_status: str) -> None:
    record.load_status = load_status
    _apply_error(record, error)


def _apply_error(record: SeasonAuditRecord, error: ErrorDetail) -> None:
    record.error_detail = error
    record.error_stage = error.stage
    record.error_type = error.exception_type
    record.error_message = _short_error_message(error.message)
```

Also append a temporary report writer so the staged tests can run before Task 4:

```python
def write_audit_reports(
    *,
    generated_at_utc: str,
    config: AuditConfig,
    seasons: list[SeasonAuditRecord],
) -> tuple[Path, Path]:
    output_root = config.project_root / config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "season_compatibility.csv"
    json_path = output_root / "season_compatibility.json"
    pd.DataFrame([season.to_csv_row() for season in seasons]).to_csv(csv_path, index=False)
    payload = {
        "generated_at_utc": generated_at_utc,
        "project_root": str(config.project_root),
        "config": _config_json(config),
        "seasons": [season.to_json_object() for season in seasons],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def _config_json(config: AuditConfig) -> dict[str, object]:
    return {
        "start_round": config.start_round,
        "complete_round_threshold": config.complete_round_threshold,
        "expected_complete_rounds": config.expected_complete_rounds,
        "current_year": config.current_year,
        "fixture_mode": config.fixture_mode,
    }
```

- [ ] **Step 4: Run staged-audit tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py::test_load_failure_records_row_and_skips_later_stages src/tests/backtesting/test_compatibility_audit.py::test_feature_check_covers_every_eligible_target_round src/tests/backtesting/test_compatibility_audit.py::test_feature_failure_records_target_round_and_skips_backtest src/tests/backtesting/test_compatibility_audit.py::test_max_round_before_start_round_marks_feature_not_applicable -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/compatibility_audit.py src/tests/backtesting/test_compatibility_audit.py
git commit -m "feat: audit loader and feature compatibility stages"
```

---

### Task 3: Backtest Smoke Stage, Output Isolation, And Metrics

**Files:**
- Modify: `src/cartola/backtesting/compatibility_audit.py`
- Modify: `src/tests/backtesting/test_compatibility_audit.py`

- [ ] **Step 1: Write failing backtest and metric tests**

Append to `src/tests/backtesting/test_compatibility_audit.py`:

```python
def test_backtest_stage_uses_isolated_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 6):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    observed_configs: list[BacktestConfig] = []
    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 6)))
    monkeypatch.setattr(audit, "build_training_frame", lambda season_df, target_round, playable_statuses, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())

    def fake_run_backtest(config: BacktestConfig) -> SimpleNamespace:
        observed_configs.append(config)
        return _fake_backtest_result(
            pd.DataFrame(
                [
                    {"strategy": "baseline", "average_actual_points": 10.0},
                    {"strategy": "random_forest", "average_actual_points": 20.0},
                    {"strategy": "price", "average_actual_points": 5.0},
                ]
            )
        )

    monkeypatch.setattr(audit, "run_backtest", fake_run_backtest)

    result = audit.run_compatibility_audit(
        audit.AuditConfig(project_root=tmp_path, current_year=2026, output_root=Path("data/08_reporting/backtests/compatibility"))
    )

    assert observed_configs == [
        BacktestConfig(
            season=2025,
            start_round=5,
            project_root=tmp_path,
            output_root=Path("data/08_reporting/backtests/compatibility/runs"),
            fixture_mode="none",
        )
    ]
    assert result.seasons[0].backtest_status == "ok"
    assert result.seasons[0].baseline_avg_points == 10.0
    assert result.seasons[0].random_forest_avg_points == 20.0
    assert result.seasons[0].price_avg_points == 5.0


def test_missing_strategy_metric_rows_are_null_without_failing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 6):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 6)))
    monkeypatch.setattr(audit, "build_training_frame", lambda season_df, target_round, playable_statuses, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())
    monkeypatch.setattr(
        audit,
        "run_backtest",
        lambda config: _fake_backtest_result(pd.DataFrame([{"strategy": "baseline", "average_actual_points": 10.0}])),
    )

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    record = result.seasons[0]
    assert record.backtest_status == "ok"
    assert record.baseline_avg_points == 10.0
    assert record.random_forest_avg_points is None
    assert record.price_avg_points is None
    assert "missing expected strategy metrics" in "; ".join(record.notes)


def test_backtest_failure_records_error_and_keeps_metrics_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 6):
        _touch_round(tmp_path, 2025, f"rodada-{round_number}.csv")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 6)))
    monkeypatch.setattr(audit, "build_training_frame", lambda season_df, target_round, playable_statuses, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "run_backtest", lambda config: (_ for _ in ()).throw(RuntimeError("backtest broke")))

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    record = result.seasons[0]
    assert record.backtest_status == "failed"
    assert record.error_stage == "backtest"
    assert record.baseline_avg_points is None
    assert record.random_forest_avg_points is None
    assert record.price_avg_points is None
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py::test_backtest_stage_uses_isolated_output_root src/tests/backtesting/test_compatibility_audit.py::test_missing_strategy_metric_rows_are_null_without_failing src/tests/backtesting/test_compatibility_audit.py::test_backtest_failure_records_error_and_keeps_metrics_null -q
```

Expected: FAIL because `_run_backtest_stage` still skips backtesting.

- [ ] **Step 3: Implement backtest smoke and metrics**

Replace `_run_backtest_stage` in `src/cartola/backtesting/compatibility_audit.py` with:

```python
def _run_backtest_stage(record: SeasonAuditRecord, config: AuditConfig) -> SeasonAuditRecord:
    backtest_config = BacktestConfig(
        season=record.season,
        start_round=config.start_round,
        project_root=config.project_root,
        output_root=config.output_root / "runs",
        fixture_mode="none",
    )
    try:
        result = run_backtest(backtest_config)
    except Exception as exc:  # noqa: BLE001 - audit reports exceptions per season
        record.backtest_status = STATUS_FAILED
        _apply_error(record, _error_detail("backtest", exc))
        return record

    record.backtest_status = STATUS_OK
    _populate_metrics(record, result.summary)
    return record
```

Append metric helpers:

```python
def _populate_metrics(record: SeasonAuditRecord, summary: pd.DataFrame) -> None:
    missing: list[str] = []
    for strategy in EXPECTED_STRATEGIES:
        value = _summary_average_for_strategy(summary, strategy)
        if value is None:
            missing.append(strategy)
        if strategy == "baseline":
            record.baseline_avg_points = value
        elif strategy == "random_forest":
            record.random_forest_avg_points = value
        elif strategy == "price":
            record.price_avg_points = value

    if missing:
        record.notes.append(f"missing expected strategy metrics: {','.join(missing)}")


def _summary_average_for_strategy(summary: pd.DataFrame, strategy: str) -> float | None:
    if summary.empty or "strategy" not in summary.columns or "average_actual_points" not in summary.columns:
        return None
    matches = summary.loc[summary["strategy"].eq(strategy), "average_actual_points"]
    if matches.empty:
        return None
    return float(matches.iloc[0])
```

- [ ] **Step 4: Run backtest and metric tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py::test_backtest_stage_uses_isolated_output_root src/tests/backtesting/test_compatibility_audit.py::test_missing_strategy_metric_rows_are_null_without_failing src/tests/backtesting/test_compatibility_audit.py::test_backtest_failure_records_error_and_keeps_metrics_null -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/compatibility_audit.py src/tests/backtesting/test_compatibility_audit.py
git commit -m "feat: add compatibility audit backtest smoke stage"
```

---

### Task 4: Report Serialization And Error Detail Guarantees

**Files:**
- Modify: `src/cartola/backtesting/compatibility_audit.py`
- Modify: `src/tests/backtesting/test_compatibility_audit.py`

- [ ] **Step 1: Write failing report tests**

Append to `src/tests/backtesting/test_compatibility_audit.py`:

```python
def test_reports_record_current_year_detected_rounds_and_full_error_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _touch_round(tmp_path, 2025, "rodada-1.csv")
    long_message = "x" * 350
    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: (_ for _ in ()).throw(ValueError(long_message)))

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    csv_frame = pd.read_csv(result.csv_path)
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert csv_frame.loc[0, "detected_rounds"] == "1"
    assert len(csv_frame.loc[0, "error_message"]) == 300
    assert payload["config"]["current_year"] == 2026
    assert payload["config"]["fixture_mode"] == "none"
    assert payload["seasons"][0]["detected_rounds"] == [1]
    assert payload["seasons"][0]["error_detail"]["message"] == long_message
    assert "ValueError" in payload["seasons"][0]["error_detail"]["traceback"]


def test_partial_current_metrics_are_recorded_but_not_comparable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for round_number in range(1, 14):
        _touch_round(tmp_path, 2026, f"rodada-{round_number}.csv")

    monkeypatch.setattr(audit, "load_season_data", lambda season, project_root: _season_frame(range(1, 14)))
    monkeypatch.setattr(audit, "build_training_frame", lambda season_df, target_round, playable_statuses, fixtures: pd.DataFrame())
    monkeypatch.setattr(audit, "build_prediction_frame", lambda season_df, target_round, fixtures: pd.DataFrame())
    monkeypatch.setattr(
        audit,
        "run_backtest",
        lambda config: _fake_backtest_result(
            pd.DataFrame(
                [
                    {"strategy": "baseline", "average_actual_points": 1.0},
                    {"strategy": "random_forest", "average_actual_points": 2.0},
                    {"strategy": "price", "average_actual_points": 3.0},
                ]
            )
        ),
    )

    result = audit.run_compatibility_audit(audit.AuditConfig(project_root=tmp_path, current_year=2026))

    record = result.seasons[0]
    assert record.season_status == "partial_current"
    assert record.metrics_comparable is False
    assert record.random_forest_avg_points == 2.0
```

- [ ] **Step 2: Run report tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py::test_reports_record_current_year_detected_rounds_and_full_error_details src/tests/backtesting/test_compatibility_audit.py::test_partial_current_metrics_are_recorded_but_not_comparable -q
```

Expected: PASS if earlier temporary writer already satisfies the report contract. If this fails, the failure should identify a serialization mismatch.

- [ ] **Step 3: Tighten report writer output columns**

In `src/cartola/backtesting/compatibility_audit.py`, add the required CSV column order near constants:

```python
CSV_COLUMNS: tuple[str, ...] = (
    "season",
    "season_status",
    "metrics_comparable",
    "round_file_count",
    "min_round",
    "max_round",
    "detected_rounds",
    "start_round",
    "evaluated_rounds",
    "first_evaluated_round",
    "last_evaluated_round",
    "fixture_mode",
    "fixture_status",
    "load_status",
    "feature_status",
    "backtest_status",
    "error_stage",
    "error_type",
    "error_message",
    "baseline_avg_points",
    "random_forest_avg_points",
    "price_avg_points",
    "notes",
)
```

Replace the `pd.DataFrame(...).to_csv(...)` line in `write_audit_reports` with:

```python
    csv_frame = pd.DataFrame([season.to_csv_row() for season in seasons], columns=pd.Index(CSV_COLUMNS))
    csv_frame.to_csv(csv_path, index=False)
```

- [ ] **Step 4: Run all audit tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py -q
```

Expected: all compatibility audit tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cartola/backtesting/compatibility_audit.py src/tests/backtesting/test_compatibility_audit.py
git commit -m "feat: write compatibility audit reports"
```

---

### Task 5: CLI Script And README

**Files:**
- Modify: `src/cartola/backtesting/compatibility_audit.py`
- Create: `scripts/audit_backtest_compatibility.py`
- Modify: `src/tests/backtesting/test_compatibility_audit.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Append to `src/tests/backtesting/test_compatibility_audit.py`:

```python
def test_parse_args_accepts_current_year_and_output_root() -> None:
    args = audit.parse_args(
        [
            "--project-root",
            "/tmp/cartola",
            "--start-round",
            "6",
            "--complete-round-threshold",
            "20",
            "--expected-complete-rounds",
            "22",
            "--current-year",
            "2026",
            "--output-root",
            "/tmp/audit",
        ]
    )

    assert args.project_root == Path("/tmp/cartola")
    assert args.start_round == 6
    assert args.complete_round_threshold == 20
    assert args.expected_complete_rounds == 22
    assert args.current_year == 2026
    assert args.output_root == Path("/tmp/audit")


def test_main_runs_audit_and_prints_report_paths(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    observed_configs: list[audit.AuditConfig] = []

    def fake_run(config: audit.AuditConfig) -> audit.AuditRunResult:
        observed_configs.append(config)
        csv_path = tmp_path / "season_compatibility.csv"
        json_path = tmp_path / "season_compatibility.json"
        return audit.AuditRunResult(
            generated_at_utc="2026-04-27T00:00:00Z",
            project_root=config.project_root,
            config=config,
            seasons=[],
            csv_path=csv_path,
            json_path=json_path,
        )

    monkeypatch.setattr(audit, "run_compatibility_audit", fake_run)

    exit_code = audit.main(["--project-root", str(tmp_path), "--current-year", "2026"])

    assert exit_code == 0
    assert observed_configs == [audit.AuditConfig(project_root=tmp_path, current_year=2026)]
    output = capsys.readouterr().out
    assert "Compatibility audit complete" in output
    assert "season_compatibility.csv" in output
    assert "season_compatibility.json" in output
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py::test_parse_args_accepts_current_year_and_output_root src/tests/backtesting/test_compatibility_audit.py::test_main_runs_audit_and_prints_report_paths -q
```

Expected: FAIL because `parse_args` and `main` do not exist.

- [ ] **Step 3: Implement CLI parser and main**

Append to `src/cartola/backtesting/compatibility_audit.py`:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Cartola backtest compatibility across raw seasons.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--complete-round-threshold", type=int, default=38)
    parser.add_argument("--expected-complete-rounds", type=int, default=38)
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/backtests/compatibility"))
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> AuditConfig:
    return AuditConfig(
        project_root=args.project_root,
        start_round=args.start_round,
        complete_round_threshold=args.complete_round_threshold,
        expected_complete_rounds=args.expected_complete_rounds,
        current_year=args.current_year,
        output_root=args.output_root,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_compatibility_audit(config_from_args(args))
    print("Compatibility audit complete")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    return 0
```

- [ ] **Step 4: Create script wrapper**

Create `scripts/audit_backtest_compatibility.py`:

```python
from __future__ import annotations

from cartola.backtesting.compatibility_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Add README command**

Add this section to `README.md` after the existing backtest output section:

```markdown
Para auditar quais temporadas brutas são compatíveis com o backtest atual:

```bash
uv run python scripts/audit_backtest_compatibility.py
```

O relatório é gravado em:

- `data/08_reporting/backtests/compatibility/season_compatibility.csv`
- `data/08_reporting/backtests/compatibility/season_compatibility.json`

Os backtests executados pelo audit usam `fixture_mode=none` e gravam saídas isoladas em `data/08_reporting/backtests/compatibility/runs/{season}/`, sem sobrescrever `data/08_reporting/backtests/{season}/`.
```

- [ ] **Step 6: Run CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py::test_parse_args_accepts_current_year_and_output_root src/tests/backtesting/test_compatibility_audit.py::test_main_runs_audit_and_prints_report_paths -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/cartola/backtesting/compatibility_audit.py scripts/audit_backtest_compatibility.py src/tests/backtesting/test_compatibility_audit.py README.md
git commit -m "feat: add compatibility audit CLI"
```

---

### Task 6: Full Verification And Real Audit Run

**Files:**
- Modify only if verification exposes defects:
  - `src/cartola/backtesting/compatibility_audit.py`
  - `scripts/audit_backtest_compatibility.py`
  - `src/tests/backtesting/test_compatibility_audit.py`
  - `README.md`
- Generated ignored outputs:
  - `data/08_reporting/backtests/compatibility/season_compatibility.csv`
  - `data/08_reporting/backtests/compatibility/season_compatibility.json`
  - `data/08_reporting/backtests/compatibility/runs/{season}/`

- [ ] **Step 1: Run all compatibility audit tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_compatibility_audit.py -q
```

Expected: all compatibility audit tests pass.

- [ ] **Step 2: Run all backtesting tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/ -q
```

Expected: all backtesting tests pass.

- [ ] **Step 3: Run the real compatibility audit**

Run:

```bash
uv run --frozen python scripts/audit_backtest_compatibility.py --current-year 2026
```

Expected: command exits 0 and prints paths for `season_compatibility.csv` and `season_compatibility.json`.

- [ ] **Step 4: Inspect key audit outputs**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

csv_path = Path("data/08_reporting/backtests/compatibility/season_compatibility.csv")
json_path = Path("data/08_reporting/backtests/compatibility/season_compatibility.json")
frame = pd.read_csv(csv_path)
payload = json.loads(json_path.read_text())

print(frame[["season", "season_status", "load_status", "feature_status", "backtest_status", "metrics_comparable"]].to_string(index=False))
print(payload["config"])
PY
```

Expected:
- 2026 appears when `data/01_raw/2026` has round files.
- 2026 has `season_status=partial_current` when `max_round < 38`.
- 2022 or any 39-file historical season has `season_status=irregular_historical`.
- `payload["config"]["current_year"] == 2026`.
- Normal outputs under `data/08_reporting/backtests/{season}/` are not rewritten by this command.

- [ ] **Step 5: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest all pass.

- [ ] **Step 6: Commit final verification fixes if any**

If Step 3, Step 4, or Step 5 required code changes, run:

```bash
git add src/cartola/backtesting/compatibility_audit.py scripts/audit_backtest_compatibility.py src/tests/backtesting/test_compatibility_audit.py README.md
git commit -m "fix: harden compatibility audit verification"
```

If no changes were required, do not create an empty commit.

---

## Self-Review Checklist

- [ ] Spec coverage: discovery, classification, staged audit, no fixtures, isolated outputs, CSV/JSON reports, error truncation, full JSON errors, nullable missing-strategy metrics, deterministic current year, and 2026 partial-current handling are covered.
- [ ] Output isolation: `BacktestConfig.output_root` receives `config.output_root / "runs"`, so `BacktestConfig.output_path` writes `.../runs/{season}/`.
- [ ] No fixture coverage is required or loaded; every audit run uses `fixture_mode="none"` and `fixture_status="not_applicable"`.
- [ ] Every loader/backtest call passes `project_root` explicitly.
- [ ] Complete historical seasons require `detected_rounds == [1, 2, ..., expected_complete_rounds]`.
- [ ] CSV uses compact string `detected_rounds`; JSON uses integer arrays.
- [ ] Full verification includes `uv run --frozen scripts/pyrepo-check --all`.
