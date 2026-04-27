from __future__ import annotations

import argparse
import json
import re
import traceback as traceback_module
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal, Sequence

import pandas as pd

from cartola.backtesting.config import BacktestConfig
from cartola.backtesting.data import load_season_data
from cartola.backtesting.features import build_prediction_frame, build_training_frame
from cartola.backtesting.runner import run_backtest

ROUND_FILE_RE = re.compile(r"^rodada-(\d+)\.csv$")
EXPECTED_STRATEGIES: tuple[str, ...] = ("baseline", "random_forest", "price")
CSV_ERROR_MESSAGE_LIMIT = 300
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


def run_compatibility_audit(
    config: AuditConfig,
    *,
    clock: Callable[[], datetime] | None = None,
) -> AuditRunResult:
    generated_at = (clock() if clock is not None else datetime.now(UTC)).astimezone(UTC)
    resolved_config = replace(config, current_year=config.resolved_current_year(clock))
    seasons = [_audit_one_season(discovery, resolved_config) for discovery in discover_seasons(resolved_config)]
    csv_path, json_path = write_audit_reports(
        seasons,
        resolved_config,
        generated_at_utc=generated_at.isoformat(),
    )
    return AuditRunResult(
        generated_at_utc=generated_at.isoformat(),
        project_root=resolved_config.project_root,
        config=resolved_config,
        seasons=seasons,
        csv_path=csv_path,
        json_path=json_path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cartola multi-season backtest compatibility audit.")
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


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    result = run_compatibility_audit(config)
    print("Compatibility audit complete")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    return 0


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
    return one_line[: CSV_ERROR_MESSAGE_LIMIT - 3] + "..."


def _audit_one_season(discovery: SeasonDiscovery, config: AuditConfig) -> SeasonAuditRecord:
    record = _base_record(discovery, config)
    if discovery.discovery_error is not None:
        _mark_failure(record, "discovery", discovery.discovery_error)
        return record

    try:
        season_df = load_season_data(discovery.season, project_root=config.project_root)
    except Exception as exc:  # noqa: BLE001 - audit reports exceptions per season
        _mark_failure(record, "load", _error_detail("load", exc))
        return record

    record.load_status = STATUS_OK
    if discovery.max_round is None or discovery.max_round < config.start_round:
        record.feature_status = STATUS_NOT_APPLICABLE
        record.backtest_status = STATUS_SKIPPED
        return record

    if not _check_feature_compatibility(record, season_df, config):
        return record

    _run_backtest_stage(record, season_df, config)
    return record


def _check_feature_compatibility(
    record: SeasonAuditRecord,
    season_df: pd.DataFrame,
    config: AuditConfig,
) -> bool:
    assert record.last_evaluated_round is not None
    playable_statuses = BacktestConfig().playable_statuses
    for target_round in range(config.start_round, record.last_evaluated_round + 1):
        try:
            build_training_frame(
                season_df,
                target_round,
                playable_statuses=playable_statuses,
                fixtures=None,
            )
            build_prediction_frame(season_df, target_round, fixtures=None)
        except Exception as exc:  # noqa: BLE001 - audit reports exceptions per season
            _mark_failure(record, "feature", _error_detail("feature", exc, target_round=target_round))
            return False

    record.feature_status = STATUS_OK
    return True


def _run_backtest_stage(
    record: SeasonAuditRecord,
    season_df: pd.DataFrame,
    config: AuditConfig,
) -> None:
    _ = season_df
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
        _mark_failure(record, "backtest", _error_detail("backtest", exc))
        return

    record.backtest_status = STATUS_OK
    _populate_metrics(record, result.summary)


def _populate_metrics(record: SeasonAuditRecord, summary: pd.DataFrame) -> None:
    missing_strategies: list[str] = []
    for strategy in EXPECTED_STRATEGIES:
        average = _summary_average_for_strategy(summary, strategy)
        if average is None:
            missing_strategies.append(strategy)
            continue

        if strategy == "baseline":
            record.baseline_avg_points = average
        elif strategy == "random_forest":
            record.random_forest_avg_points = average
        elif strategy == "price":
            record.price_avg_points = average

    if missing_strategies:
        record.notes.append(f"missing expected strategy metrics: {','.join(missing_strategies)}")


def _summary_average_for_strategy(summary: pd.DataFrame, strategy: str) -> float | None:
    if "strategy" not in summary or "average_actual_points" not in summary:
        return None

    strategy_rows = summary.loc[summary["strategy"] == strategy, "average_actual_points"]
    if strategy_rows.empty:
        return None

    average = strategy_rows.iloc[0]
    if pd.isna(average):
        return None
    return float(average)


def _mark_failure(record: SeasonAuditRecord, stage: str, error_detail: ErrorDetail) -> None:
    if stage == "load":
        record.load_status = STATUS_FAILED
        record.feature_status = STATUS_SKIPPED
        record.backtest_status = STATUS_SKIPPED
    elif stage == "feature":
        record.feature_status = STATUS_FAILED
        record.backtest_status = STATUS_SKIPPED
    elif stage == "discovery":
        record.load_status = STATUS_SKIPPED
        record.feature_status = STATUS_SKIPPED
        record.backtest_status = STATUS_SKIPPED
    elif stage == "backtest":
        record.backtest_status = STATUS_FAILED
    _apply_error(record, stage, error_detail)


def _apply_error(record: SeasonAuditRecord, stage: str, error_detail: ErrorDetail) -> None:
    record.error_stage = stage
    record.error_type = error_detail.exception_type
    record.error_message = _short_error_message(error_detail.message)
    record.error_detail = error_detail


def write_audit_reports(
    seasons: list[SeasonAuditRecord],
    config: AuditConfig,
    *,
    generated_at_utc: str,
) -> tuple[Path, Path]:
    output_dir = config.project_root / config.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "compatibility_audit.csv"
    json_path = output_dir / "compatibility_audit.json"

    rows = [season.to_csv_row() for season in seasons]
    pd.DataFrame(rows, columns=pd.Index(CSV_COLUMNS)).to_csv(csv_path, index=False)

    payload = {
        "generated_at_utc": generated_at_utc,
        "project_root": str(config.project_root),
        "config": _config_json(config),
        "seasons": [season.to_json_object() for season in seasons],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def _config_json(config: AuditConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["project_root"] = str(config.project_root)
    payload["output_root"] = str(config.output_root)
    return payload
