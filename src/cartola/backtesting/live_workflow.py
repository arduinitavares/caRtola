from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from cartola.backtesting.config import FootyStatsMode
from cartola.backtesting.market_capture import (
    CARTOLA_STATUS_ENDPOINT,
    LiveCaptureMetadata,
    MarketCaptureConfig,
    capture_market_round,
    fetch_cartola_json,
    load_valid_live_capture,
)
from cartola.backtesting.recommendation import (
    RecommendationConfig,
    RecommendationResult,
    _validate_output_root,
    run_recommendation,
)

CapturePolicy = Literal["fresh", "missing", "skip"]
WorkflowStatus = Literal["ok", "failed"]
ErrorStage = Literal["status_fetch", "capture_validation", "capture", "recommendation", "workflow_metadata"]
Clock = Callable[[], datetime]
WORKFLOW_VERSION = "live_workflow_v1"


@dataclass(frozen=True)
class LiveWorkflowConfig:
    season: int
    budget: float = 100.0
    project_root: Path = Path(".")
    output_root: Path = Path("data/08_reporting/recommendations")
    footystats_mode: FootyStatsMode = "ppg"
    footystats_league_slug: str = "brazil-serie-a"
    footystats_dir: Path = Path("data/footystats")
    current_year: int | None = None
    capture_policy: CapturePolicy = "fresh"
    allow_finalized_live_data: bool = False
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class LiveWorkflowResult:
    recommendation: RecommendationResult | None
    workflow_metadata: dict[str, object]
    output_path: Path | None


def _runtime_current_year() -> int:
    return datetime.now(UTC).year


def _resolved_current_year(config: LiveWorkflowConfig) -> int:
    return config.current_year if config.current_year is not None else _runtime_current_year()


def _run_started_at(now: Clock) -> tuple[str, str]:
    current = now().astimezone(UTC)
    compact = current.strftime("%Y%m%dT%H%M%S%fZ")
    iso = current.isoformat().replace("+00:00", "Z")
    return iso, f"run_started_at={compact}"


def _capture_age_seconds(captured_at_utc: str, now: Clock) -> float:
    captured = datetime.fromisoformat(captured_at_utc.replace("Z", "+00:00"))
    return max(0.0, (now().astimezone(UTC) - captured).total_seconds())


def _validate_current_year(config: LiveWorkflowConfig) -> int:
    current_year = _resolved_current_year(config)
    if config.season != current_year:
        raise ValueError(f"live workflow requires season {config.season} to equal current_year {current_year}")
    return current_year


def _live_workflow_link(
    *,
    config: LiveWorkflowConfig,
    run_started_at_utc: str,
    output_run_id: str,
    target_round: int,
    capture: LiveCaptureMetadata,
    capture_age_seconds: float,
) -> dict[str, object]:
    recommendation_output_path = (
        config.project_root
        / config.output_root
        / str(config.season)
        / f"round-{target_round}"
        / "live"
        / "runs"
        / output_run_id
    )
    return {
        "workflow_version": WORKFLOW_VERSION,
        "run_started_at_utc": run_started_at_utc,
        "capture_policy": config.capture_policy,
        "season": config.season,
        "current_year": _resolved_current_year(config),
        "target_round": target_round,
        "budget": float(config.budget),
        "footystats_mode": config.footystats_mode,
        "footystats_league_slug": config.footystats_league_slug,
        "capture_csv_path": str(capture.csv_path),
        "capture_metadata_path": str(capture.metadata_path),
        "capture_csv_sha256": capture.csv_sha256,
        "capture_captured_at_utc": capture.captured_at_utc,
        "capture_age_seconds": capture_age_seconds,
        "capture_status_mercado": capture.status_mercado,
        "capture_deadline_timestamp": capture.deadline_timestamp,
        "capture_deadline_parse_status": capture.deadline_parse_status,
        "recommendation_output_path": str(recommendation_output_path),
    }


def _workflow_metadata(
    *,
    live_workflow: dict[str, object],
    recommendation: RecommendationResult | None,
    status: WorkflowStatus,
    error_stage: ErrorStage | None = None,
    error: Exception | None = None,
) -> dict[str, object]:
    metadata = dict(live_workflow)
    output_path = Path(str(live_workflow["recommendation_output_path"]))
    metadata.update(
        {
            "recommendation_summary_path": str(output_path / "recommendation_summary.json"),
            "recommendation_metadata_path": str(output_path / "run_metadata.json"),
            "recommended_squad_path": str(output_path / "recommended_squad.csv"),
            "candidate_predictions_path": str(output_path / "candidate_predictions.csv"),
            "selected_count": None if recommendation is None else recommendation.summary.get("selected_count"),
            "predicted_points": None if recommendation is None else recommendation.summary.get("predicted_points"),
            "budget_used": None if recommendation is None else recommendation.summary.get("budget_used"),
            "status": status,
            "error_stage": error_stage,
            "error_type": None if error is None else type(error).__name__,
            "error_message": None if error is None else str(error),
        }
    )
    return metadata


def _write_workflow_metadata(output_path: Path, metadata: dict[str, object]) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "live_workflow_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _capture_fresh(config: LiveWorkflowConfig) -> tuple[int, LiveCaptureMetadata]:
    result = capture_market_round(
        MarketCaptureConfig(
            season=config.season,
            auto=True,
            force=True,
            current_year=config.current_year,
            project_root=config.project_root,
            timeout_seconds=config.timeout_seconds,
        )
    )
    capture = load_valid_live_capture(
        project_root=config.project_root,
        season=config.season,
        target_round=result.target_round,
    )
    return result.target_round, capture


def _int_payload_field(payload: dict[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _active_open_round(config: LiveWorkflowConfig) -> int:
    response = fetch_cartola_json(CARTOLA_STATUS_ENDPOINT, config.timeout_seconds)
    target_round = _int_payload_field(response.payload, "rodada_atual")
    if target_round <= 0:
        raise ValueError("rodada_atual must be a positive integer")
    status_mercado = _int_payload_field(response.payload, "status_mercado")
    if status_mercado != 1:
        raise ValueError(f"Cartola market is not open: rodada_atual={target_round} status_mercado={status_mercado}")
    return target_round


def _capture_missing(config: LiveWorkflowConfig) -> tuple[int, LiveCaptureMetadata]:
    target_round = _active_open_round(config)
    try:
        capture = load_valid_live_capture(
            project_root=config.project_root,
            season=config.season,
            target_round=target_round,
        )
        return target_round, capture
    except FileNotFoundError:
        result = capture_market_round(
            MarketCaptureConfig(
                season=config.season,
                auto=True,
                force=False,
                current_year=config.current_year,
                project_root=config.project_root,
                timeout_seconds=config.timeout_seconds,
            )
        )
        capture = load_valid_live_capture(
            project_root=config.project_root,
            season=config.season,
            target_round=result.target_round,
        )
        return result.target_round, capture


def _capture_skip(config: LiveWorkflowConfig) -> tuple[int, LiveCaptureMetadata]:
    target_round = _active_open_round(config)
    capture = load_valid_live_capture(
        project_root=config.project_root,
        season=config.season,
        target_round=target_round,
    )
    return target_round, capture


def _resolve_capture(config: LiveWorkflowConfig) -> tuple[int, LiveCaptureMetadata]:
    if config.capture_policy == "fresh":
        return _capture_fresh(config)
    if config.capture_policy == "missing":
        return _capture_missing(config)
    if config.capture_policy == "skip":
        return _capture_skip(config)
    raise ValueError(f"Unsupported capture policy: {config.capture_policy}")


def _assert_archive_available(output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"recommendation archive already exists: {output_path}")


def run_live_round(config: LiveWorkflowConfig, *, now: Clock = lambda: datetime.now(UTC)) -> LiveWorkflowResult:
    _validate_current_year(config)
    run_started_at_utc, output_run_id = _run_started_at(now)

    target_round, capture = _resolve_capture(config)
    capture_age = _capture_age_seconds(capture.captured_at_utc, now)
    live_workflow = _live_workflow_link(
        config=config,
        run_started_at_utc=run_started_at_utc,
        output_run_id=output_run_id,
        target_round=target_round,
        capture=capture,
        capture_age_seconds=capture_age,
    )
    recommendation_config = RecommendationConfig(
        season=config.season,
        target_round=target_round,
        mode="live",
        budget=config.budget,
        project_root=config.project_root,
        output_root=config.output_root,
        footystats_mode=config.footystats_mode,
        footystats_league_slug=config.footystats_league_slug,
        footystats_dir=config.footystats_dir,
        current_year=config.current_year,
        allow_finalized_live_data=config.allow_finalized_live_data,
        output_run_id=output_run_id,
        live_workflow=live_workflow,
    )
    _validate_output_root(recommendation_config)
    _assert_archive_available(recommendation_config.output_path)
    try:
        recommendation = run_recommendation(recommendation_config)
    except Exception as exc:
        metadata = _workflow_metadata(
            live_workflow=live_workflow,
            recommendation=None,
            status="failed",
            error_stage="recommendation",
            error=exc,
        )
        _write_workflow_metadata(recommendation_config.output_path, metadata)
        raise
    metadata = _workflow_metadata(live_workflow=live_workflow, recommendation=recommendation, status="ok")
    _write_workflow_metadata(recommendation_config.output_path, metadata)
    return LiveWorkflowResult(
        recommendation=recommendation,
        workflow_metadata=metadata,
        output_path=recommendation_config.output_path,
    )
