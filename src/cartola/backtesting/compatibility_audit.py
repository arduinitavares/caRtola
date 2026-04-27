from __future__ import annotations

import re
import traceback as traceback_module
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

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
    season_paths = sorted(
        (path for path in raw_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for season_path in season_paths:
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
    return one_line[: CSV_ERROR_MESSAGE_LIMIT - 1] + "..."
