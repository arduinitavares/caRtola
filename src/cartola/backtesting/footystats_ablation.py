from __future__ import annotations

import argparse
import json
import math
import shutil
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence, cast

import pandas as pd

from cartola.backtesting.config import BacktestConfig, FootyStatsMode
from cartola.backtesting.footystats_features import FootyStatsPPGLoadResult, load_footystats_ppg_rows
from cartola.backtesting.runner import run_backtest

DEFAULT_SEASONS = (2023, 2024, 2025)
DEFAULT_OUTPUT_ROOT = Path("data/08_reporting/backtests/footystats_ablation")
DEFAULT_LEAGUE_SLUG = "brazil-serie-a"
CSV_COLUMNS = (
    "season",
    "row_type",
    "season_status",
    "metrics_comparable",
    "control_status",
    "treatment_status",
    "metric_status",
    "control_output_path",
    "treatment_output_path",
    "control_baseline_avg_points",
    "treatment_baseline_avg_points",
    "baseline_avg_points",
    "baseline_avg_points_equal",
    "control_rf_avg_points",
    "treatment_rf_avg_points",
    "rf_avg_points_delta",
    "control_player_r2",
    "treatment_player_r2",
    "player_r2_delta",
    "control_player_corr",
    "treatment_player_corr",
    "player_corr_delta",
    "rf_minus_baseline_control",
    "rf_minus_baseline_treatment",
    "error_stage",
    "error_message",
)


@dataclass(frozen=True)
class FootyStatsPPGAblationConfig:
    seasons: tuple[int, ...] = DEFAULT_SEASONS
    start_round: int = 5
    budget: float = 100.0
    project_root: Path = Path(".")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    footystats_league_slug: str = DEFAULT_LEAGUE_SLUG
    current_year: int | None = None
    force: bool = False

    @property
    def resolved_current_year(self) -> int:
        if self.current_year is not None:
            return self.current_year
        return datetime.now(UTC).year


@dataclass(frozen=True)
class AblationError:
    stage: str
    type: str
    message: str
    traceback: str


@dataclass
class SeasonAblationRecord:
    season: int | str
    row_type: str = "season"
    season_status: str = "failed"
    metrics_comparable: bool = False
    control_status: str = "skipped"
    treatment_status: str = "skipped"
    metric_status: str = "skipped"
    control_output_path: str | None = None
    treatment_output_path: str | None = None
    control_baseline_avg_points: float | None = None
    treatment_baseline_avg_points: float | None = None
    baseline_avg_points: float | None = None
    baseline_avg_points_equal: bool | None = None
    control_rf_avg_points: float | None = None
    treatment_rf_avg_points: float | None = None
    rf_avg_points_delta: float | None = None
    control_player_r2: float | None = None
    treatment_player_r2: float | None = None
    player_r2_delta: float | None = None
    control_player_corr: float | None = None
    treatment_player_corr: float | None = None
    player_corr_delta: float | None = None
    rf_minus_baseline_control: float | None = None
    rf_minus_baseline_treatment: float | None = None
    error_stage: str | None = None
    error_message: str | None = None
    treatment_source_path: str | None = None
    treatment_source_sha256: str | None = None
    control_config: dict[str, Any] | None = None
    treatment_config: dict[str, Any] | None = None
    errors: list[AblationError] = field(default_factory=list)


@dataclass(frozen=True)
class FootyStatsPPGAblationResult:
    config: FootyStatsPPGAblationConfig
    resolved_output_root: Path
    seasons: list[SeasonAblationRecord]
    aggregate: SeasonAblationRecord


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
    parser = argparse.ArgumentParser(description="Run the FootyStats PPG ablation report.")
    parser.add_argument("--seasons", type=parse_seasons_arg, default=DEFAULT_SEASONS)
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--footystats-league-slug", default=DEFAULT_LEAGUE_SLUG)
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FootyStatsPPGAblationConfig:
    return FootyStatsPPGAblationConfig(
        seasons=args.seasons,
        start_round=args.start_round,
        budget=args.budget,
        project_root=args.project_root,
        output_root=args.output_root,
        footystats_league_slug=args.footystats_league_slug,
        current_year=args.current_year,
        force=args.force,
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output_root(config: FootyStatsPPGAblationConfig, output_root: Path) -> Path:
    project_root = config.project_root.resolve()
    resolved_output_root = (output_root if output_root.is_absolute() else project_root / output_root).resolve()

    if not _is_relative_to(resolved_output_root, project_root):
        raise ValueError(f"output_root must resolve inside project_root: {project_root}")

    normal_backtests_root = project_root / "data" / "08_reporting" / "backtests"
    protected_paths = {
        project_root,
        project_root / "data",
        project_root / "data" / "08_reporting",
        normal_backtests_root,
        *(normal_backtests_root / str(season) for season in config.seasons),
    }
    if resolved_output_root in protected_paths:
        raise ValueError(f"output_root resolves to a protected path: {resolved_output_root}")

    if resolved_output_root.name != "footystats_ablation":
        raise ValueError("output_root final directory name must be exactly 'footystats_ablation'")

    return resolved_output_root


def resolve_output_root(config: FootyStatsPPGAblationConfig) -> Path:
    return _validate_output_root(config, config.output_root)


def build_backtest_config(
    config: FootyStatsPPGAblationConfig,
    season: int,
    mode: str,
    resolved_output_root: Path,
) -> BacktestConfig:
    resolved_output_root = _validate_output_root(config, resolved_output_root)

    if mode not in ("none", "ppg"):
        raise ValueError(f"Unsupported footystats mode: {mode!r}")
    footystats_mode = cast(FootyStatsMode, mode)

    backtest_config = BacktestConfig(
        season=season,
        start_round=config.start_round,
        budget=config.budget,
        project_root=config.project_root.resolve(),
        output_root=resolved_output_root / "runs" / str(season) / f"footystats_mode={footystats_mode}",
        fixture_mode="none",
        footystats_mode=footystats_mode,
        footystats_evaluation_scope="historical_candidate",
        footystats_league_slug=config.footystats_league_slug,
        current_year=config.resolved_current_year,
    )

    normal_backtest_output_path = backtest_config.project_root / "data" / "08_reporting" / "backtests" / str(season)
    if backtest_config.output_path.resolve() == normal_backtest_output_path.resolve():
        raise ValueError(f"Refusing to use normal backtest output path: {normal_backtest_output_path}")

    return backtest_config


def prepare_output_root(config: FootyStatsPPGAblationConfig, resolved_output_root: Path) -> None:
    resolved_output_root = _validate_output_root(config, resolved_output_root)

    if resolved_output_root.exists():
        if not config.force:
            raise FileExistsError(f"output_root already exists: {resolved_output_root}")
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)


def _error(stage: str, exc: BaseException) -> AblationError:
    return AblationError(
        stage=stage,
        type=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


def _config_dict(config: BacktestConfig) -> dict[str, object]:
    data = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def _load_eligibility(config: FootyStatsPPGAblationConfig, season: int) -> FootyStatsPPGLoadResult:
    return load_footystats_ppg_rows(
        season=season,
        project_root=config.project_root.resolve(),
        footystats_dir=Path("data/footystats"),
        league_slug=config.footystats_league_slug,
        evaluation_scope="historical_candidate",
        current_year=config.resolved_current_year,
    )


def run_footystats_ppg_ablation(config: FootyStatsPPGAblationConfig) -> FootyStatsPPGAblationResult:
    resolved_output_root = resolve_output_root(config)
    prepare_output_root(config, resolved_output_root)
    records: list[SeasonAblationRecord] = []

    for season in config.seasons:
        control_config = build_backtest_config(
            config,
            season=season,
            mode="none",
            resolved_output_root=resolved_output_root,
        )
        treatment_config = build_backtest_config(
            config,
            season=season,
            mode="ppg",
            resolved_output_root=resolved_output_root,
        )
        record = SeasonAblationRecord(
            season=season,
            control_output_path=str(control_config.output_path),
            treatment_output_path=str(treatment_config.output_path),
            control_config=_config_dict(control_config),
            treatment_config=_config_dict(treatment_config),
        )

        try:
            eligibility = _load_eligibility(config, season)
        except Exception as exc:
            record.error_stage = "eligibility"
            record.error_message = str(exc)
            record.errors.append(_error("eligibility", exc))
            records.append(record)
            continue

        record.season_status = "candidate"
        record.treatment_source_path = str(eligibility.source_path)
        record.treatment_source_sha256 = eligibility.source_sha256

        try:
            run_backtest(control_config)
            record.control_status = "ok"
        except Exception as exc:
            record.control_status = "failed"
            record.error_stage = "control_backtest"
            record.error_message = str(exc)
            record.errors.append(_error("control_backtest", exc))
            records.append(record)
            continue

        try:
            run_backtest(treatment_config)
            record.treatment_status = "ok"
        except Exception as exc:
            record.treatment_status = "failed"
            record.error_stage = "treatment_backtest"
            record.error_message = str(exc)
            record.errors.append(_error("treatment_backtest", exc))
            records.append(record)
            continue

        try:
            populate_metrics(record, control_config.output_path, treatment_config.output_path)
        except Exception as exc:
            record.metric_status = "failed"
            record.error_stage = "metric_extraction"
            record.error_message = str(exc)
            record.errors.append(_error("metric_extraction", exc))

        records.append(record)

    aggregate = build_aggregate_record(records)
    return FootyStatsPPGAblationResult(
        config=config,
        resolved_output_root=resolved_output_root,
        seasons=records,
        aggregate=aggregate,
    )


def extract_run_metrics(output_path: Path) -> dict[str, float]:
    summary = pd.read_csv(output_path / "summary.csv")
    diagnostics = pd.read_csv(output_path / "diagnostics.csv")
    return {
        "baseline": _summary_value(summary, "baseline"),
        "rf": _summary_value(summary, "random_forest"),
        "r2": _diagnostic_value(diagnostics, "player_r2"),
        "corr": _diagnostic_value(diagnostics, "player_correlation"),
    }


def _summary_value(summary: pd.DataFrame, strategy: str) -> float:
    rows = summary[summary["strategy"].eq(strategy)]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one summary row for strategy {strategy!r}, found {len(rows)}")
    return _finite_float(
        rows["average_actual_points"].iloc[0],
        context=f"summary strategy {strategy!r} metric 'average_actual_points'",
    )


def _diagnostic_value(diagnostics: pd.DataFrame, metric: str) -> float:
    rows = diagnostics[
        diagnostics["section"].eq("prediction")
        & diagnostics["strategy"].eq("random_forest")
        & diagnostics["position"].eq("all")
        & diagnostics["metric"].eq(metric)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one diagnostics row for metric {metric!r}, found {len(rows)}")
    return _finite_float(
        rows["value"].iloc[0],
        context=f"diagnostics strategy 'random_forest' metric {metric!r} field 'value'",
    )


def _finite_float(value: object, *, context: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"non-finite {context}: {numeric_value!r}")
    return numeric_value


def populate_metrics(record: SeasonAblationRecord, control_path: Path, treatment_path: Path) -> None:
    control = extract_run_metrics(control_path)
    treatment = extract_run_metrics(treatment_path)
    record.control_baseline_avg_points = control["baseline"]
    record.treatment_baseline_avg_points = treatment["baseline"]
    record.control_rf_avg_points = control["rf"]
    record.treatment_rf_avg_points = treatment["rf"]
    record.control_player_r2 = control["r2"]
    record.treatment_player_r2 = treatment["r2"]
    record.control_player_corr = control["corr"]
    record.treatment_player_corr = treatment["corr"]
    record.baseline_avg_points_equal = abs(control["baseline"] - treatment["baseline"]) <= 1e-9
    if not record.baseline_avg_points_equal:
        record.baseline_avg_points = None
        record.metrics_comparable = False
        raise ValueError("baseline average points differ")

    record.baseline_avg_points = control["baseline"]
    record.metrics_comparable = True
    record.rf_avg_points_delta = treatment["rf"] - control["rf"]
    record.player_r2_delta = treatment["r2"] - control["r2"]
    record.player_corr_delta = treatment["corr"] - control["corr"]
    record.rf_minus_baseline_control = control["rf"] - control["baseline"]
    record.rf_minus_baseline_treatment = treatment["rf"] - treatment["baseline"]
    record.metric_status = "ok"


def build_aggregate_record(records: list[SeasonAblationRecord]) -> SeasonAblationRecord:
    included = [record for record in records if _is_included_in_aggregate(record)]
    aggregate = SeasonAblationRecord(
        season="aggregate",
        row_type="aggregate",
        season_status="aggregate",
        control_status="not_applicable",
        treatment_status="not_applicable",
        metric_status="not_applicable",
        metrics_comparable=bool(included),
    )
    if not included:
        return aggregate

    aggregate.control_baseline_avg_points = _mean_metric(included, "control_baseline_avg_points")
    aggregate.treatment_baseline_avg_points = _mean_metric(included, "treatment_baseline_avg_points")
    aggregate.baseline_avg_points = _mean_metric(included, "baseline_avg_points")
    aggregate.baseline_avg_points_equal = all(record.baseline_avg_points_equal is True for record in included)
    aggregate.control_rf_avg_points = _mean_metric(included, "control_rf_avg_points")
    aggregate.treatment_rf_avg_points = _mean_metric(included, "treatment_rf_avg_points")
    aggregate.rf_avg_points_delta = _mean_metric(included, "rf_avg_points_delta")
    aggregate.control_player_r2 = _mean_metric(included, "control_player_r2")
    aggregate.treatment_player_r2 = _mean_metric(included, "treatment_player_r2")
    aggregate.player_r2_delta = _mean_metric(included, "player_r2_delta")
    aggregate.control_player_corr = _mean_metric(included, "control_player_corr")
    aggregate.treatment_player_corr = _mean_metric(included, "treatment_player_corr")
    aggregate.player_corr_delta = _mean_metric(included, "player_corr_delta")
    aggregate.rf_minus_baseline_control = _mean_metric(included, "rf_minus_baseline_control")
    aggregate.rf_minus_baseline_treatment = _mean_metric(included, "rf_minus_baseline_treatment")
    return aggregate


def _mean_metric(records: list[SeasonAblationRecord], metric: str) -> float:
    return sum(_required_metric(record, metric) for record in records) / len(records)


def _required_metric(record: SeasonAblationRecord, metric: str) -> float:
    value = getattr(record, metric)
    if value is None:
        raise ValueError(f"missing aggregate metric {metric!r} for season {record.season!r}")
    if isinstance(value, bool):
        raise ValueError(f"invalid aggregate metric {metric!r} for season {record.season!r}: {value!r}")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"invalid aggregate metric {metric!r} for season {record.season!r}: {numeric_value!r}")
    return numeric_value


def _csv_row(record: SeasonAblationRecord) -> dict[str, object]:
    row = {column: getattr(record, column) for column in CSV_COLUMNS}
    error_message = row["error_message"]
    if isinstance(error_message, str):
        row["error_message"] = error_message[:500]
    return row


def _json_record(record: SeasonAblationRecord) -> dict[str, object]:
    return asdict(record)


def _generated_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_included_in_aggregate(record: SeasonAblationRecord) -> bool:
    return (
        record.metrics_comparable
        and record.control_status == "ok"
        and record.treatment_status == "ok"
        and record.metric_status == "ok"
    )


def _excluded_reason(record: SeasonAblationRecord) -> str:
    if record.error_stage and record.error_message:
        return f"{record.error_stage}: {record.error_message}"
    if record.error_stage:
        return record.error_stage
    if record.metric_status != "ok":
        return f"metric_status={record.metric_status}"
    if record.control_status != "ok":
        return f"control_status={record.control_status}"
    if record.treatment_status != "ok":
        return f"treatment_status={record.treatment_status}"
    if not record.metrics_comparable:
        return "metrics_comparable=False"
    return "not included"


def _aggregate_json(records: list[SeasonAblationRecord], aggregate: SeasonAblationRecord) -> dict[str, object]:
    included = [record for record in records if _is_included_in_aggregate(record)]
    metrics = {
        "control_baseline_avg_points": aggregate.control_baseline_avg_points,
        "treatment_baseline_avg_points": aggregate.treatment_baseline_avg_points,
        "baseline_avg_points": aggregate.baseline_avg_points,
        "baseline_avg_points_equal": aggregate.baseline_avg_points_equal,
        "control_rf_avg_points": aggregate.control_rf_avg_points,
        "treatment_rf_avg_points": aggregate.treatment_rf_avg_points,
        "rf_avg_points_delta": aggregate.rf_avg_points_delta,
        "control_player_r2": aggregate.control_player_r2,
        "treatment_player_r2": aggregate.treatment_player_r2,
        "player_r2_delta": aggregate.player_r2_delta,
        "control_player_corr": aggregate.control_player_corr,
        "treatment_player_corr": aggregate.treatment_player_corr,
        "player_corr_delta": aggregate.player_corr_delta,
        "rf_minus_baseline_control": aggregate.rf_minus_baseline_control,
        "rf_minus_baseline_treatment": aggregate.rf_minus_baseline_treatment,
    }
    return {
        "included_seasons": [record.season for record in included],
        "excluded_seasons": [
            {"season": record.season, "reason": _excluded_reason(record)}
            for record in records
            if record not in included
        ],
        "aggregation_method": "mean",
        "metrics": metrics,
    }


def _config_json(config: FootyStatsPPGAblationConfig, resolved_output_root: Path) -> dict[str, object]:
    return {
        "project_root": str(config.project_root),
        "resolved_project_root": str(config.project_root.resolve()),
        "output_root": str(config.output_root),
        "resolved_output_root": str(resolved_output_root.resolve()),
        "seasons": list(config.seasons),
        "start_round": config.start_round,
        "budget": config.budget,
        "current_year": config.current_year,
        "resolved_current_year": config.resolved_current_year,
        "fixture_mode": "none",
        "control_footystats_mode": "none",
        "treatment_footystats_mode": "ppg",
        "footystats_evaluation_scope": "historical_candidate",
        "footystats_league_slug": config.footystats_league_slug,
        "force": config.force,
    }


def write_reports(result: FootyStatsPPGAblationResult) -> None:
    result.resolved_output_root.mkdir(parents=True, exist_ok=True)
    rows = [_csv_row(record) for record in [*result.seasons, result.aggregate]]
    csv_path = result.resolved_output_root / "ppg_ablation.csv"
    json_path = result.resolved_output_root / "ppg_ablation.json"
    csv_tmp = result.resolved_output_root / ".ppg_ablation.csv.tmp"
    json_tmp = result.resolved_output_root / ".ppg_ablation.json.tmp"
    csv_backup = result.resolved_output_root / ".ppg_ablation.csv.bak"
    json_backup = result.resolved_output_root / ".ppg_ablation.json.bak"
    csv_backed_up = False
    json_backed_up = False

    report = {
        "generated_at_utc": _generated_at_utc(),
        "config": _config_json(result.config, result.resolved_output_root),
        "seasons": [_json_record(record) for record in result.seasons],
        "aggregate": _aggregate_json(result.seasons, result.aggregate),
    }

    try:
        pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_tmp, index=False, na_rep="")
        json_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        csv_backup.unlink(missing_ok=True)
        json_backup.unlink(missing_ok=True)

        try:
            if csv_path.exists():
                csv_path.replace(csv_backup)
                csv_backed_up = True
            if json_path.exists():
                json_path.replace(json_backup)
                json_backed_up = True

            csv_tmp.replace(csv_path)
            json_tmp.replace(json_path)
        except BaseException:
            csv_path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)
            if csv_backed_up:
                csv_backup.replace(csv_path)
            if json_backed_up:
                json_backup.replace(json_path)
            raise
        else:
            csv_backup.unlink(missing_ok=True)
            json_backup.unlink(missing_ok=True)
    finally:
        csv_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    print(
        "FootyStats PPG ablation report is not implemented yet "
        f"for seasons {', '.join(str(season) for season in config.seasons)}."
    )
    return 2
