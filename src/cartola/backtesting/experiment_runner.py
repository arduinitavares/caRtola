from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping, Sequence, SupportsFloat, SupportsInt, cast

import pandas as pd

from cartola.backtesting.budgeting import BUDGET_POLICY_MOVING, normalize_budget_policy
from cartola.backtesting.experiment_config import (
    ChildRunSpec,
    ExperimentGroup,
    build_child_run_specs,
    config_hash,
    experiment_id,
)
from cartola.backtesting.experiment_index import (
    ExperimentIndex,
    artifact_pointer_payload,
    sha256_json,
    sha256_optional_file,
    source_hash_summary,
)
from cartola.backtesting.experiment_metrics import (
    calibration_slope_intercept,
    promotion_status,
    top_k_rows_by_round,
)
from cartola.backtesting.experiment_reports import build_experiment_html_reports
from cartola.backtesting.experiment_signatures import (
    ComparabilityError,
    candidate_pool_signature,
    compare_signature_sets,
    raw_cartola_source_identity,
    solver_status_signature,
)
from cartola.backtesting.experiment_tracking import (
    ExperimentTracker,
    NoOpExperimentTracker,
    TrackerStatus,
    TrackerWarning,
)
from cartola.backtesting.model_registry import model_n_jobs_for_metadata
from cartola.backtesting.runner import CSV_FLOAT_FORMAT, BacktestResult, run_backtest_for_experiment
from cartola.backtesting.scoring_contract import SCORING_CONTRACT_VERSION


@dataclass(frozen=True)
class ExperimentRunResult:
    experiment_id: str
    output_path: Path
    ranked_summary: pd.DataFrame
    metadata: dict[str, object]


@dataclass(frozen=True)
class ExperimentProgressEvent:
    event_type: Literal[
        "experiment_started",
        "child_started",
        "child_finished",
        "child_failed",
        "experiment_finished",
    ]
    experiment_id: str
    output_path: Path
    total_children: int
    completed_children: int
    child_index: int | None = None
    child_id: str | None = None
    season: int | None = None
    model_id: str | None = None
    feature_pack: str | None = None
    fixture_mode: str | None = None
    elapsed_seconds: float | None = None
    child_duration_seconds: float | None = None
    phase: str | None = None
    message: str | None = None


ExperimentProgressCallback = Callable[[ExperimentProgressEvent], None]


def run_model_experiment(
    *,
    group: ExperimentGroup,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    project_root: Path,
    output_root: Path,
    started_at_utc: str,
    models: tuple[str, ...] | None = None,
    exclude_models: tuple[str, ...] = (),
    profile_runtime: bool = False,
    progress_callback: ExperimentProgressCallback | None = None,
    tracker: ExperimentTracker | None = None,
) -> ExperimentRunResult:
    experiment_started = perf_counter()
    identity_specs = build_child_run_specs(
        group=group,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        project_root=project_root,
        output_root=output_root,
        current_year=current_year,
        jobs=jobs,
        models=models,
        exclude_models=exclude_models,
        profile_runtime=profile_runtime,
    )
    total_children = len(identity_specs)
    matrix_hash = config_hash({"child_runs": [spec.config_identity for spec in identity_specs]})
    run_id = experiment_id(group=group, started_at_utc=started_at_utc, matrix_hash=matrix_hash)
    output_path = project_root / output_root / run_id
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True)
    tracker = tracker or NoOpExperimentTracker()
    index = _experiment_index(project_root)
    index_warnings: list[str] = []
    try:
        index.initialize()
    except Exception as exc:
        index_warnings.append(f"initialize: {type(exc).__name__}: {exc}")

    specs = build_child_run_specs(
        group=group,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        project_root=project_root,
        output_root=output_root / run_id,
        current_year=current_year,
        jobs=jobs,
        models=models,
        exclude_models=exclude_models,
        profile_runtime=profile_runtime,
    )
    _emit_progress(
        progress_callback,
        ExperimentProgressEvent(
            event_type="experiment_started",
            experiment_id=run_id,
            output_path=output_path,
            total_children=total_children,
            completed_children=0,
            elapsed_seconds=0.0,
        ),
    )
    raw_sources = {
        str(season): raw_cartola_source_identity(project_root=project_root, season=season) for season in seasons
    }
    _safe_index_write(
        index,
        "upsert_experiment",
        _experiment_index_row(
            experiment_id_value=run_id,
            group=group,
            started_at_utc=started_at_utc,
            finished_at_utc=None,
            status="running",
            output_path=output_path,
            matrix_hash=matrix_hash,
            seasons=seasons,
            start_round=start_round,
            budget=budget,
            current_year=current_year,
            jobs=jobs,
            child_run_count=total_children,
            completed_child_run_count=0,
            failed_child_run_count=0,
            project_root=project_root,
            tracker=tracker,
        ),
        index_warnings,
    )

    child_runs: list[dict[str, object]] = []
    per_season_rows: list[dict[str, object]] = []
    prediction_metric_rows: list[dict[str, object]] = []
    calibration_decile_rows: list[dict[str, object]] = []
    candidate_pool_signatures: dict[str, dict[str, str]] = {}
    solver_status_signatures: dict[str, dict[str, str]] = {}
    comparability_partitions: dict[str, list[str]] = {}
    experiment_status: Literal["ok", "failed"] = "failed"
    tracker_finalized = False

    try:
        tracker.start_experiment(
            experiment_name=f"cartola-{group}",
            run_name=run_id,
            params={
                "group": group,
                "start_round": start_round,
                "budget": budget,
                "initial_budget": budget,
                "budget_policy": BUDGET_POLICY_MOVING,
                "current_year": current_year,
                "jobs": jobs,
                "models": list(models) if models is not None else None,
                "exclude_models": list(exclude_models),
                "profile_runtime": profile_runtime,
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
            },
            tags={
                "experiment_id": run_id,
                "matrix_hash": matrix_hash,
                "git.commit": _git_value(project_root, "rev-parse", "HEAD"),
                "git.branch": _git_value(project_root, "branch", "--show-current"),
                "git.dirty": _git_dirty(project_root),
                "python.version": sys.version,
                "uv.lock.hash": sha256_optional_file(project_root / "uv.lock"),
                "cartola.version": _package_version("cartola"),
                "pandas.version": _package_version("pandas"),
                "numpy.version": _package_version("numpy"),
                "scikit-learn.version": _package_version("scikit-learn"),
                "plotly.version": _package_version("plotly"),
                "mlflow.version": _package_version("mlflow"),
                "platform": platform.platform(),
            },
        )
        for child_index, spec in enumerate(specs, start=1):
            child_id = _child_id(spec)
            child_started = perf_counter()
            _emit_progress(
                progress_callback,
                _progress_event(
                    "child_started",
                    run_id=run_id,
                    output_path=output_path,
                    total_children=total_children,
                    completed_children=len(child_runs),
                    child_index=child_index,
                    child_id=child_id,
                    spec=spec,
                    elapsed_seconds=child_started - experiment_started,
                ),
            )
            tracker.start_child(
                run_name=f"season={spec.season} model={spec.model_id} feature_pack={spec.feature_pack}",
                params=_child_params(spec),
                tags={
                    "experiment_id": run_id,
                    "child_run_id": child_id,
                    "output_path": _relative_path(spec.output_path, project_root=project_root),
                    "comparability_partition": _comparability_partition(spec),
                },
            )
            try:
                result = run_backtest_for_experiment(spec.backtest_config, primary_model_id=spec.model_id)
            except Exception as exc:
                failed_at = perf_counter()
                _emit_progress(
                    progress_callback,
                    _progress_event(
                        "child_failed",
                        run_id=run_id,
                        output_path=output_path,
                        total_children=total_children,
                        completed_children=len(child_runs),
                        child_index=child_index,
                        child_id=child_id,
                        spec=spec,
                        elapsed_seconds=failed_at - experiment_started,
                        child_duration_seconds=failed_at - child_started,
                        phase="child_run",
                        message=str(exc),
                    ),
                )
                tracker.end_child(status="failed")
                metadata = _metadata(
                    status="failed",
                    experiment_id_value=run_id,
                    started_at_utc=started_at_utc,
                    group=group,
                    seasons=seasons,
                    start_round=start_round,
                    budget=budget,
                    current_year=current_year,
                    jobs=jobs,
                    matrix_hash=matrix_hash,
                    child_runs=child_runs,
                    raw_sources=raw_sources,
                    candidate_pool_signatures=candidate_pool_signatures,
                    solver_status_signatures=solver_status_signatures,
                    tracking_warnings=[asdict(warning) for warning in tracker.warnings],
                    index_warnings=index_warnings,
                    failure={"phase": "child_run", "message": str(exc), "child_id": child_id},
                )
                _upsert_failed_experiment_row(
                    index=index,
                    index_warnings=index_warnings,
                    experiment_id_value=run_id,
                    group=group,
                    started_at_utc=started_at_utc,
                    output_path=output_path,
                    matrix_hash=matrix_hash,
                    seasons=seasons,
                    start_round=start_round,
                    budget=budget,
                    current_year=current_year,
                    jobs=jobs,
                    child_run_count=total_children,
                    completed_child_run_count=len(child_runs),
                    failed_child_run_count=1,
                    project_root=project_root,
                    tracker=tracker,
                )
                tracker_finalized = _safe_finalize_tracker(tracker, status="failed")
                metadata = _metadata_with_current_warnings(metadata, tracker=tracker, index_warnings=index_warnings)
                _write_failure_artifacts(output_path, metadata)
                raise
            child_runs.append(_child_record(spec, result, child_id=child_id))
            try:
                child_candidate_signatures = _candidate_signatures_by_round(result.player_predictions)
                child_solver_signature = solver_status_signature(
                    result.round_results,
                    primary_model_id=spec.model_id,
                )
                candidate_pool_signatures[child_id] = child_candidate_signatures
                solver_status_signatures[child_id] = child_solver_signature
                comparability_partitions.setdefault(_comparability_partition(spec), []).append(child_id)
                per_season_rows.extend(_primary_summary_rows(spec, result, child_id=child_id))
                prediction_metric_rows.extend(_prediction_metric_rows(spec, result, child_id=child_id))
                calibration_decile_rows.extend(_calibration_decile_rows(spec, result, child_id=child_id))
                child_row = _child_index_row(
                    spec=spec,
                    result=result,
                    child_id=child_id,
                    experiment_id_value=run_id,
                    project_root=project_root,
                    raw_source_identity=raw_sources[str(spec.season)],
                    candidate_pool_signature_hash=sha256_json(child_candidate_signatures),
                    solver_status_signature_hash=sha256_json(child_solver_signature),
                    comparable=True,
                    tracker=tracker,
                )
                _safe_index_write(index, "upsert_child_run", child_row, index_warnings)
                pointer_path = _write_child_artifact_pointers(
                    project_root=project_root,
                    child_id=child_id,
                    output_path=spec.output_path,
                )
                tracker.log_child_metrics(_child_metrics(child_row))
                tracker.log_child_artifacts(
                    [
                        spec.output_path / "summary.csv",
                        spec.output_path / "diagnostics.csv",
                        spec.output_path / "run_metadata.json",
                        pointer_path,
                        spec.output_path / "player_predictions.csv",
                        spec.output_path / "selected_players.csv",
                    ]
                )
                tracker.end_child(status="ok")
            except ComparabilityError as exc:
                failed_at = perf_counter()
                _emit_progress(
                    progress_callback,
                    _progress_event(
                        "child_failed",
                        run_id=run_id,
                        output_path=output_path,
                        total_children=total_children,
                        completed_children=max(0, len(child_runs) - 1),
                        child_index=child_index,
                        child_id=child_id,
                        spec=spec,
                        elapsed_seconds=failed_at - experiment_started,
                        child_duration_seconds=failed_at - child_started,
                        phase="comparability",
                        message=str(exc),
                    ),
                )
                tracker.end_child(status="failed")
                metadata = _metadata(
                    status="failed",
                    experiment_id_value=run_id,
                    started_at_utc=started_at_utc,
                    group=group,
                    seasons=seasons,
                    start_round=start_round,
                    budget=budget,
                    current_year=current_year,
                    jobs=jobs,
                    matrix_hash=matrix_hash,
                    child_runs=child_runs,
                    raw_sources=raw_sources,
                    candidate_pool_signatures=candidate_pool_signatures,
                    solver_status_signatures=solver_status_signatures,
                    tracking_warnings=[asdict(warning) for warning in tracker.warnings],
                    index_warnings=index_warnings,
                    failure={"phase": "comparability", "message": str(exc), "child_id": child_id},
                )
                _upsert_failed_experiment_row(
                    index=index,
                    index_warnings=index_warnings,
                    experiment_id_value=run_id,
                    group=group,
                    started_at_utc=started_at_utc,
                    output_path=output_path,
                    matrix_hash=matrix_hash,
                    seasons=seasons,
                    start_round=start_round,
                    budget=budget,
                    current_year=current_year,
                    jobs=jobs,
                    child_run_count=total_children,
                    completed_child_run_count=max(0, len(child_runs) - 1),
                    failed_child_run_count=1,
                    project_root=project_root,
                    tracker=tracker,
                )
                tracker_finalized = _safe_finalize_tracker(tracker, status="failed")
                metadata = _metadata_with_current_warnings(metadata, tracker=tracker, index_warnings=index_warnings)
                _write_failure_artifacts(output_path, metadata)
                raise
            except Exception as exc:
                failed_at = perf_counter()
                _emit_progress(
                    progress_callback,
                    _progress_event(
                        "child_failed",
                        run_id=run_id,
                        output_path=output_path,
                        total_children=total_children,
                        completed_children=max(0, len(child_runs) - 1),
                        child_index=child_index,
                        child_id=child_id,
                        spec=spec,
                        elapsed_seconds=failed_at - experiment_started,
                        child_duration_seconds=failed_at - child_started,
                        phase="child_post_processing",
                        message=str(exc),
                    ),
                )
                tracker.end_child(status="failed")
                metadata = _metadata(
                    status="failed",
                    experiment_id_value=run_id,
                    started_at_utc=started_at_utc,
                    group=group,
                    seasons=seasons,
                    start_round=start_round,
                    budget=budget,
                    current_year=current_year,
                    jobs=jobs,
                    matrix_hash=matrix_hash,
                    child_runs=child_runs,
                    raw_sources=raw_sources,
                    candidate_pool_signatures=candidate_pool_signatures,
                    solver_status_signatures=solver_status_signatures,
                    tracking_warnings=[asdict(warning) for warning in tracker.warnings],
                    index_warnings=index_warnings,
                    failure={"phase": "child_post_processing", "message": str(exc), "child_id": child_id},
                )
                _upsert_failed_experiment_row(
                    index=index,
                    index_warnings=index_warnings,
                    experiment_id_value=run_id,
                    group=group,
                    started_at_utc=started_at_utc,
                    output_path=output_path,
                    matrix_hash=matrix_hash,
                    seasons=seasons,
                    start_round=start_round,
                    budget=budget,
                    current_year=current_year,
                    jobs=jobs,
                    child_run_count=total_children,
                    completed_child_run_count=max(0, len(child_runs) - 1),
                    failed_child_run_count=1,
                    project_root=project_root,
                    tracker=tracker,
                )
                tracker_finalized = _safe_finalize_tracker(tracker, status="failed")
                metadata = _metadata_with_current_warnings(metadata, tracker=tracker, index_warnings=index_warnings)
                _write_failure_artifacts(output_path, metadata)
                raise
            _emit_progress(
                progress_callback,
                _progress_event(
                    "child_finished",
                    run_id=run_id,
                    output_path=output_path,
                    total_children=total_children,
                    completed_children=len(child_runs),
                    child_index=child_index,
                    child_id=child_id,
                    spec=spec,
                    elapsed_seconds=perf_counter() - experiment_started,
                    child_duration_seconds=perf_counter() - child_started,
                ),
            )

        try:
            _check_candidate_comparability(candidate_pool_signatures, comparability_partitions)
            _check_solver_status_comparability(solver_status_signatures, comparability_partitions)
        except ComparabilityError as exc:
            metadata = _metadata(
                status="failed",
                experiment_id_value=run_id,
                started_at_utc=started_at_utc,
                group=group,
                seasons=seasons,
                start_round=start_round,
                budget=budget,
                current_year=current_year,
                jobs=jobs,
                matrix_hash=matrix_hash,
                child_runs=child_runs,
                raw_sources=raw_sources,
                candidate_pool_signatures=candidate_pool_signatures,
                solver_status_signatures=solver_status_signatures,
                tracking_warnings=[asdict(warning) for warning in tracker.warnings],
                index_warnings=index_warnings,
                failure={"phase": "comparability", "message": str(exc)},
            )
            _upsert_failed_experiment_row(
                index=index,
                index_warnings=index_warnings,
                experiment_id_value=run_id,
                group=group,
                started_at_utc=started_at_utc,
                output_path=output_path,
                matrix_hash=matrix_hash,
                seasons=seasons,
                start_round=start_round,
                budget=budget,
                current_year=current_year,
                jobs=jobs,
                child_run_count=total_children,
                completed_child_run_count=len(child_runs),
                failed_child_run_count=0,
                project_root=project_root,
                tracker=tracker,
            )
            tracker_finalized = _safe_finalize_tracker(tracker, status="failed")
            metadata = _metadata_with_current_warnings(metadata, tracker=tracker, index_warnings=index_warnings)
            _write_failure_artifacts(output_path, metadata)
            raise

        per_season_summary = pd.DataFrame(per_season_rows)
        prediction_metrics = pd.DataFrame(prediction_metric_rows)
        calibration_deciles = pd.DataFrame(calibration_decile_rows)
        ranked_summary = _rank_summary(per_season_summary, prediction_metrics)
        metadata = _metadata(
            status="ok",
            experiment_id_value=run_id,
            started_at_utc=started_at_utc,
            group=group,
            seasons=seasons,
            start_round=start_round,
            budget=budget,
            current_year=current_year,
            jobs=jobs,
            matrix_hash=matrix_hash,
            child_runs=child_runs,
            raw_sources=raw_sources,
            candidate_pool_signatures=candidate_pool_signatures,
            solver_status_signatures=solver_status_signatures,
            tracking_warnings=[asdict(warning) for warning in tracker.warnings],
            index_warnings=index_warnings,
            failure=None,
        )
        _write_success_artifacts(
            output_path=output_path,
            metadata=metadata,
            ranked_summary=ranked_summary,
            per_season_summary=per_season_summary,
            prediction_metrics=prediction_metrics,
            calibration_deciles=calibration_deciles,
        )
        _safe_index_write(
            index,
            "upsert_experiment",
            _experiment_index_row(
                experiment_id_value=run_id,
                group=group,
                started_at_utc=started_at_utc,
                finished_at_utc=_utc_now_id(),
                status="ok",
                output_path=output_path,
                matrix_hash=matrix_hash,
                seasons=seasons,
                start_round=start_round,
                budget=budget,
                current_year=current_year,
                jobs=jobs,
                child_run_count=total_children,
                completed_child_run_count=len(child_runs),
                failed_child_run_count=0,
                project_root=project_root,
                tracker=tracker,
            ),
            index_warnings,
        )
        tracker.log_parent_artifacts(
            [
                output_path / "ranked_summary.csv",
                output_path / "per_season_summary.csv",
                output_path / "prediction_metrics.csv",
                output_path / "calibration_deciles.csv",
                output_path / "comparability_report.json",
                output_path / "experiment_metadata.json",
                output_path / "comparison_report.md",
                output_path / "calibration_plots.html",
                output_path / "squad_performance_comparison.html",
            ]
        )
        tracker_finalized = _safe_finalize_tracker(tracker, status="ok")
        metadata = _metadata_with_current_warnings(metadata, tracker=tracker, index_warnings=index_warnings)
        _write_json(output_path / "experiment_metadata.json", metadata)
        _emit_progress(
            progress_callback,
            ExperimentProgressEvent(
                event_type="experiment_finished",
                experiment_id=run_id,
                output_path=output_path,
                total_children=total_children,
                completed_children=len(child_runs),
                elapsed_seconds=perf_counter() - experiment_started,
            ),
        )
        experiment_status = "ok"
        return ExperimentRunResult(
            experiment_id=run_id,
            output_path=output_path,
            ranked_summary=ranked_summary,
            metadata=metadata,
        )
    finally:
        if not tracker_finalized:
            _safe_finalize_tracker(tracker, status="ok" if experiment_status == "ok" else "failed")


def _emit_progress(
    callback: ExperimentProgressCallback | None,
    event: ExperimentProgressEvent,
) -> None:
    if callback is not None:
        callback(event)


def _progress_event(
    event_type: Literal["child_started", "child_finished", "child_failed"],
    *,
    run_id: str,
    output_path: Path,
    total_children: int,
    completed_children: int,
    child_index: int,
    child_id: str,
    spec: ChildRunSpec,
    elapsed_seconds: float,
    child_duration_seconds: float | None = None,
    phase: str | None = None,
    message: str | None = None,
) -> ExperimentProgressEvent:
    return ExperimentProgressEvent(
        event_type=event_type,
        experiment_id=run_id,
        output_path=output_path,
        total_children=total_children,
        completed_children=completed_children,
        child_index=child_index,
        child_id=child_id,
        season=spec.season,
        model_id=spec.model_id,
        feature_pack=spec.feature_pack,
        fixture_mode=spec.fixture_mode,
        elapsed_seconds=elapsed_seconds,
        child_duration_seconds=child_duration_seconds,
        phase=phase,
        message=message,
    )


def _child_id(spec: ChildRunSpec) -> str:
    return f"season={spec.season}/model={spec.model_id}/feature_pack={spec.feature_pack}"


def _comparability_partition(spec: ChildRunSpec) -> str:
    return f"season={spec.season}"


def _child_record(spec: ChildRunSpec, result: BacktestResult, *, child_id: str) -> dict[str, object]:
    return {
        "child_id": child_id,
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "fixture_mode": spec.fixture_mode,
        "feature_augmentation_mode": spec.backtest_config.feature_augmentation_mode,
        "output_path": str(spec.output_path),
        "model_n_jobs_effective": _model_n_jobs_for_child(spec, result),
        "strategy_roles": {
            "baseline": "baseline",
            spec.model_id: "primary_model",
            "price": "price",
        },
        "metadata": asdict(result.metadata),
    }


def _experiment_index(project_root: Path) -> ExperimentIndex:
    return ExperimentIndex(project_root / "data" / "08_reporting" / "experiments" / "experiment_index.sqlite")


def _utc_now_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _git_executable() -> str | None:
    return shutil.which("git")


def _git_value(project_root: Path, *args: str) -> str | None:
    git = _git_executable()
    if git is None:
        return None
    try:
        # Read-only Git metadata command with a resolved executable path.
        completed = subprocess.run(  # nosec B603
            [git, *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty(project_root: Path) -> bool:
    git = _git_executable()
    if git is None:
        return False
    try:
        # Read-only Git status command with a resolved executable path.
        completed = subprocess.run(  # nosec B603
            [git, "status", "--short"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(completed.stdout.strip())


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _safe_index_write(
    index: ExperimentIndex,
    method_name: Literal["upsert_experiment", "upsert_child_run"],
    row: Mapping[str, object],
    warnings: list[str],
) -> None:
    try:
        getattr(index, method_name)(row)
    except Exception as exc:
        warnings.append(f"{method_name}: {type(exc).__name__}: {exc}")


def _safe_finalize_tracker(tracker: ExperimentTracker, *, status: TrackerStatus) -> bool:
    try:
        tracker.end_experiment(status=status)
    except Exception as exc:
        tracker.warnings.append(
            TrackerWarning(
                phase="end_experiment",
                message=f"{type(exc).__name__}: {exc}",
            )
        )
    return True


def _upsert_failed_experiment_row(
    *,
    index: ExperimentIndex,
    index_warnings: list[str],
    experiment_id_value: str,
    group: ExperimentGroup,
    started_at_utc: str,
    output_path: Path,
    matrix_hash: str,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    child_run_count: int,
    completed_child_run_count: int,
    failed_child_run_count: int,
    project_root: Path,
    tracker: ExperimentTracker,
) -> None:
    _safe_index_write(
        index,
        "upsert_experiment",
        _experiment_index_row(
            experiment_id_value=experiment_id_value,
            group=group,
            started_at_utc=started_at_utc,
            finished_at_utc=_utc_now_id(),
            status="failed",
            output_path=output_path,
            matrix_hash=matrix_hash,
            seasons=seasons,
            start_round=start_round,
            budget=budget,
            current_year=current_year,
            jobs=jobs,
            child_run_count=child_run_count,
            completed_child_run_count=completed_child_run_count,
            failed_child_run_count=failed_child_run_count,
            project_root=project_root,
            tracker=tracker,
        ),
        index_warnings,
    )


def _experiment_index_row(
    *,
    experiment_id_value: str,
    group: ExperimentGroup,
    started_at_utc: str,
    finished_at_utc: str | None,
    status: str,
    output_path: Path,
    matrix_hash: str,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    child_run_count: int,
    completed_child_run_count: int,
    failed_child_run_count: int,
    project_root: Path,
    tracker: ExperimentTracker,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id_value,
        "group": group,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "status": status,
        "output_path": _relative_path(output_path, project_root=project_root),
        "matrix_hash": matrix_hash,
        "seasons": list(seasons),
        "start_round": start_round,
        "budget": budget,
        "budget_policy": BUDGET_POLICY_MOVING,
        "current_year": current_year,
        "jobs": jobs,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "git_branch": _git_value(project_root, "branch", "--show-current"),
        "git_dirty": _git_dirty(project_root),
        "python_version": sys.version,
        "uv_lock_hash": sha256_optional_file(project_root / "uv.lock"),
        "mlflow_enabled": tracker.__class__.__name__ == "MLflowExperimentTracker",
        "mlflow_status": _mlflow_status_from_tracker(tracker),
        "mlflow_parent_run_id": tracker.parent_run_id,
        "warning_count": len(tracker.warnings),
        "child_run_count": child_run_count,
        "completed_child_run_count": completed_child_run_count,
        "failed_child_run_count": failed_child_run_count,
    }


def _child_index_row(
    *,
    spec: ChildRunSpec,
    result: BacktestResult,
    child_id: str,
    experiment_id_value: str,
    project_root: Path,
    raw_source_identity: Mapping[str, object],
    candidate_pool_signature_hash: str | None,
    solver_status_signature_hash: str | None,
    comparable: bool,
    tracker: ExperimentTracker,
) -> dict[str, object]:
    primary_summary = result.summary[result.summary["strategy"] == spec.model_id]
    summary_row = primary_summary.iloc[0].to_dict() if not primary_summary.empty else {}
    prediction_rows = _prediction_metric_rows(spec, result, child_id=child_id)
    prediction_by_scope = {str(row["metric_scope"]): row for row in prediction_rows}
    candidate_metrics = prediction_by_scope.get("candidate_pool", {})
    selected_metrics = prediction_by_scope.get("selected_players", {})
    top50_metrics = prediction_by_scope.get("top50_candidates", {})
    return {
        "experiment_id": experiment_id_value,
        "child_run_id": child_id,
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "fixture_mode": spec.fixture_mode,
        "budget_policy": result.metadata.budget_policy,
        "footystats_mode": spec.backtest_config.footystats_mode,
        "matchup_context_mode": spec.backtest_config.matchup_context_mode,
        "feature_augmentation_mode": spec.backtest_config.feature_augmentation_mode,
        "output_path": _relative_path(spec.output_path, project_root=project_root),
        "status": "ok",
        "wall_clock_seconds": result.metadata.wall_clock_seconds,
        "backtest_jobs": spec.jobs,
        "backtest_workers_effective": result.metadata.backtest_workers_effective,
        "model_n_jobs_effective": _model_n_jobs_for_child(spec, result),
        "total_actual_points": _float_or_none(summary_row.get("total_actual_points")),
        "avg_actual_points": _float_or_none(summary_row.get("average_actual_points")),
        "total_predicted_points": _float_or_none(summary_row.get("total_predicted_points")),
        "prediction_mae": _float_or_none(candidate_metrics.get("mae")),
        "prediction_rmse": _float_or_none(candidate_metrics.get("rmse")),
        "prediction_r2": _float_or_none(candidate_metrics.get("r2")),
        "prediction_pearson": _float_or_none(candidate_metrics.get("pearson")),
        "prediction_spearman": _float_or_none(candidate_metrics.get("spearman")),
        "selected_calibration_slope": _float_or_none(selected_metrics.get("calibration_slope")),
        "top50_spearman": _float_or_none(top50_metrics.get("spearman")),
        "optimal_round_count": int(result.round_results["solver_status"].eq("Optimal").sum()),
        "skipped_round_count": int(result.round_results["solver_status"].ne("Optimal").sum()),
        "candidate_pool_signature_hash": candidate_pool_signature_hash,
        "solver_status_signature_hash": solver_status_signature_hash,
        "comparability_partition": _comparability_partition(spec),
        "comparable_within_partition": comparable,
        "ineligibility_reason": None,
        "source_hash_summary": source_hash_summary(raw_source_identity),
        "mlflow_child_run_id": tracker.child_run_id,
    }


def _model_n_jobs_for_child(spec: ChildRunSpec, result: BacktestResult) -> int | None:
    return (
        result.metadata.model_n_jobs_effective
        if model_n_jobs_for_metadata(spec.model_id, requested_n_jobs=spec.jobs) is not None
        else None
    )


def _child_metrics(row: Mapping[str, object]) -> dict[str, float | int | None]:
    return {
        "squad/actual_points_total": _float_or_none(row.get("total_actual_points")),
        "squad/actual_points_mean": _float_or_none(row.get("avg_actual_points")),
        "squad/predicted_points_total": _float_or_none(row.get("total_predicted_points")),
        "prediction/candidate_pool/mae": _float_or_none(row.get("prediction_mae")),
        "prediction/candidate_pool/rmse": _float_or_none(row.get("prediction_rmse")),
        "prediction/candidate_pool/r2": _float_or_none(row.get("prediction_r2")),
        "prediction/candidate_pool/pearson": _float_or_none(row.get("prediction_pearson")),
        "prediction/candidate_pool/spearman": _float_or_none(row.get("prediction_spearman")),
        "prediction/selected_players/calibration_slope": _float_or_none(row.get("selected_calibration_slope")),
        "prediction/top50/spearman": _float_or_none(row.get("top50_spearman")),
        "runtime/wall_clock_seconds": _float_or_none(row.get("wall_clock_seconds")),
        "rounds/optimal_count": _int_or_none(row.get("optimal_round_count")),
        "rounds/skipped_count": _int_or_none(row.get("skipped_round_count")),
    }


def _child_params(spec: ChildRunSpec) -> dict[str, object]:
    return {
        "group": spec.group,
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "fixture_mode": spec.fixture_mode,
        "budget_policy": BUDGET_POLICY_MOVING,
        "footystats_mode": spec.backtest_config.footystats_mode,
        "matchup_context_mode": spec.backtest_config.matchup_context_mode,
        "feature_augmentation_mode": spec.backtest_config.feature_augmentation_mode,
        "start_round": spec.backtest_config.start_round,
        "budget": spec.backtest_config.budget,
        "initial_budget": spec.backtest_config.budget,
        "current_year": spec.backtest_config.current_year,
        "jobs": spec.jobs,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        **{f"model/{key}": value for key, value in spec.model_parameters.items()},
    }


def _write_child_artifact_pointers(*, project_root: Path, child_id: str, output_path: Path) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    pointer_payload = artifact_pointer_payload(
        project_root=project_root,
        child_run_id=child_id,
        output_path=output_path,
        artifact_paths=[
            output_path / "player_predictions.csv",
            output_path / "selected_players.csv",
        ],
    )
    pointer_path = output_path / "artifact_pointers.json"
    pointer_path.write_text(json.dumps(pointer_payload, indent=2, sort_keys=True), encoding="utf-8")
    return pointer_path


def _relative_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _mlflow_status_from_tracker(tracker: ExperimentTracker) -> str:
    if tracker.__class__.__name__ != "MLflowExperimentTracker":
        return "disabled"
    if not tracker.warnings:
        return "ok"
    return "partial"


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(cast(SupportsFloat, value))


def _int_or_none(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(cast(SupportsInt, value))


def _candidate_signatures_by_round(player_predictions: pd.DataFrame) -> dict[str, str]:
    if player_predictions.empty:
        return {}
    return {
        _round_key(round_number): candidate_pool_signature(round_frame)
        for round_number, round_frame in player_predictions.groupby("rodada", sort=True)
    }


def _primary_summary_rows(spec: ChildRunSpec, result: BacktestResult, *, child_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary = result.summary[result.summary["strategy"] == spec.model_id]
    for row in summary.to_dict(orient="records"):
        rows.append(
            {
                "child_id": child_id,
                "season": spec.season,
                "model_id": spec.model_id,
                "feature_pack": spec.feature_pack,
                "fixture_mode": spec.fixture_mode,
                "budget_policy": result.metadata.budget_policy,
                **row,
            }
        )
    return rows


def _prediction_metric_rows(spec: ChildRunSpec, result: BacktestResult, *, child_id: str) -> list[dict[str, object]]:
    score_column = f"{spec.model_id}_score"
    scopes = [
        ("candidate_pool", None, result.player_predictions, score_column),
        (
            "selected_players",
            None,
            _selected_players_for_model(result.selected_players, model_id=spec.model_id),
            "predicted_points",
        ),
        (
            "top25_candidates",
            25,
            top_k_rows_by_round(result.player_predictions, score_column=score_column, k=25),
            score_column,
        ),
        (
            "top50_candidates",
            50,
            top_k_rows_by_round(result.player_predictions, score_column=score_column, k=50),
            score_column,
        ),
    ]
    return [
        _prediction_metric_row(
            spec,
            child_id=child_id,
            budget_policy=result.metadata.budget_policy,
            metric_scope=metric_scope,
            k=k,
            frame=frame,
            predicted_column=predicted_column,
        )
        for metric_scope, k, frame, predicted_column in scopes
    ]


def _prediction_metric_row(
    spec: ChildRunSpec,
    *,
    child_id: str,
    budget_policy: str,
    metric_scope: str,
    k: int | None,
    frame: pd.DataFrame,
    predicted_column: str,
) -> dict[str, object]:
    paired, warning = _paired_prediction_values(frame, predicted_column=predicted_column)
    metrics = _prediction_metrics(paired)
    calibration = calibration_slope_intercept(paired["predicted"], paired["actual"])
    return {
        "child_id": child_id,
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "fixture_mode": spec.fixture_mode,
        "budget_policy": normalize_budget_policy(budget_policy),
        "metric_scope": metric_scope,
        "k": k,
        "observed_count": len(paired),
        **metrics,
        "calibration_intercept": calibration["calibration_intercept"],
        "calibration_slope": calibration["calibration_slope"],
        "warning": warning or calibration["warning"],
    }


def _paired_prediction_values(
    frame: pd.DataFrame,
    *,
    predicted_column: str,
    actual_column: str = "pontuacao",
) -> tuple[pd.DataFrame, str | None]:
    if frame.empty:
        return pd.DataFrame({"predicted": pd.Series(dtype="float64"), "actual": pd.Series(dtype="float64")}), None

    missing_columns = [column for column in (predicted_column, actual_column) if column not in frame.columns]
    if missing_columns:
        return (
            pd.DataFrame({"predicted": pd.Series(dtype="float64"), "actual": pd.Series(dtype="float64")}),
            f"missing_columns:{','.join(missing_columns)}",
        )

    paired = pd.DataFrame(
        {
            "predicted": pd.to_numeric(frame[predicted_column], errors="coerce"),
            "actual": pd.to_numeric(frame[actual_column], errors="coerce"),
        }
    ).dropna()
    return paired.reset_index(drop=True), None


def _prediction_metrics(paired: pd.DataFrame) -> dict[str, float | None]:
    if paired.empty:
        return {
            "mae": None,
            "rmse": None,
            "r2": None,
            "pearson": None,
            "spearman": None,
        }

    predicted = paired["predicted"].astype(float)
    actual = paired["actual"].astype(float)
    residual = predicted - actual
    mae = float(residual.abs().mean())
    rmse = float(math.sqrt(float((residual**2).mean())))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": _r2_score(predicted, actual),
        "pearson": _correlation(predicted, actual, method="pearson"),
        "spearman": _correlation(predicted, actual, method="spearman"),
    }


def _r2_score(predicted: pd.Series, actual: pd.Series) -> float | None:
    if len(actual) < 2 or actual.nunique() == 1:
        return None
    actual_mean = actual.mean()
    ss_res = float(((actual - predicted) ** 2).sum())
    ss_tot = float(((actual - actual_mean) ** 2).sum())
    if ss_tot == 0:
        return None
    return 1 - (ss_res / ss_tot)


def _correlation(predicted: pd.Series, actual: pd.Series, *, method: Literal["pearson", "spearman"]) -> float | None:
    if len(predicted) < 2 or predicted.nunique() == 1 or actual.nunique() == 1:
        return None
    value = predicted.corr(actual, method=method)
    if pd.isna(value):
        return None
    return float(value)


def _selected_players_for_model(selected_players: pd.DataFrame, *, model_id: str) -> pd.DataFrame:
    if selected_players.empty or "strategy" not in selected_players.columns:
        return selected_players.copy()
    return selected_players[selected_players["strategy"] == model_id].copy()


def _calibration_decile_rows(spec: ChildRunSpec, result: BacktestResult, *, child_id: str) -> list[dict[str, object]]:
    score_column = f"{spec.model_id}_score"
    paired, _warning = _paired_prediction_values(result.player_predictions, predicted_column=score_column)
    if paired.empty:
        return []

    paired = paired.assign(_stable_order=range(len(paired)))
    ranked = paired.sort_values(
        by=["predicted", "_stable_order"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["decile"] = ((ranked.index.to_series() * 10) // len(ranked)) + 1
    rows: list[dict[str, object]] = []
    for decile, decile_frame in ranked.groupby("decile", sort=True):
        decile_number = int(str(decile))
        residual = decile_frame["actual"] - decile_frame["predicted"]
        rows.append(
            {
                "child_id": child_id,
                "season": spec.season,
                "model_id": spec.model_id,
                "feature_pack": spec.feature_pack,
                "fixture_mode": spec.fixture_mode,
                "budget_policy": normalize_budget_policy(result.metadata.budget_policy),
                "decile": decile_number,
                "row_count": len(decile_frame),
                "predicted_mean": float(decile_frame["predicted"].mean()),
                "actual_mean": float(decile_frame["actual"].mean()),
                "residual_mean": float(residual.mean()),
            }
        )
    return rows


def _check_candidate_comparability(
    candidate_pool_signatures: Mapping[str, Mapping[str, str]],
    comparability_partitions: Mapping[str, Sequence[str]],
) -> None:
    for partition_id, child_ids in comparability_partitions.items():
        rounds = sorted(
            {
                round_id
                for child_id in child_ids
                for round_id in candidate_pool_signatures.get(child_id, {})
            }
        )
        for round_id in rounds:
            compare_signature_sets(
                f"Candidate pool signatures for {partition_id} rodada={round_id}",
                {
                    child_id: candidate_pool_signatures.get(child_id, {}).get(round_id)
                    for child_id in child_ids
                },
            )


def _check_solver_status_comparability(
    solver_status_signatures: Mapping[str, Mapping[str, str]],
    comparability_partitions: Mapping[str, Sequence[str]],
) -> None:
    for partition_id, child_ids in comparability_partitions.items():
        compare_signature_sets(
            f"Solver-status signatures for {partition_id}",
            {child_id: solver_status_signatures.get(child_id) for child_id in child_ids},
        )


def _rank_summary(per_season_summary: pd.DataFrame, prediction_metrics: pd.DataFrame) -> pd.DataFrame:
    if per_season_summary.empty:
        ranked = pd.DataFrame(columns=pd.Index(_RANKED_SUMMARY_COLUMNS))
        ranked.insert(0, "rank", pd.Series(dtype="int64"))
        return ranked

    per_season_summary = _with_normalized_budget_policy(per_season_summary)
    prediction_metrics = _with_normalized_budget_policy(prediction_metrics)
    baseline_by_season = _baseline_actual_points_by_season(per_season_summary)
    top50_spearman_baseline = _baseline_metric_by_season(
        prediction_metrics,
        metric_scope="top50_candidates",
        metric_column="spearman",
    )
    rows = [
        _aggregate_summary_row(
            group_frame,
            prediction_metrics=prediction_metrics,
            baseline_by_season=baseline_by_season,
            top50_spearman_baseline=top50_spearman_baseline,
        )
        for _group_key, group_frame in per_season_summary.groupby(
            ["model_id", "feature_pack", "fixture_mode", "budget_policy"],
            sort=False,
        )
    ]
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        by=[
            "promotion_eligible",
            "aggregate_delta",
            "total_actual_points",
            "model_id",
            "feature_pack",
            "fixture_mode",
            "budget_policy",
        ],
        ascending=[False, False, False, True, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", pd.Series(range(1, len(ranked) + 1), dtype="int64"))
    return ranked.loc[:, ["rank", *_RANKED_SUMMARY_COLUMNS]]


_RANKED_SUMMARY_COLUMNS = [
    "model_id",
    "feature_pack",
    "fixture_mode",
    "budget_policy",
    "seasons_evaluated",
    "total_rounds",
    "total_actual_points",
    "average_actual_points",
    "total_predicted_points",
    "average_predicted_points",
    "worst_min_budget",
    "worst_max_budget_drawdown",
    "total_budget_constrained_rounds",
    "baseline_total_actual_points",
    "aggregate_delta",
    "average_actual_delta_per_round",
    "improved_seasons",
    "worst_season_avg_delta",
    "selected_calibration_slope",
    "top50_spearman_delta",
    "promotion_eligible",
    "promotion_reason",
]


def _aggregate_summary_row(
    group_frame: pd.DataFrame,
    *,
    prediction_metrics: pd.DataFrame,
    baseline_by_season: Mapping[tuple[int, str, str], float],
    top50_spearman_baseline: Mapping[tuple[int, str, str], float],
) -> dict[str, object]:
    first = group_frame.iloc[0]
    model_id = str(first["model_id"])
    feature_pack = str(first["feature_pack"])
    fixture_mode = str(first["fixture_mode"])
    budget_policy = normalize_budget_policy(first.get("budget_policy"))
    total_rounds = int(group_frame["rounds"].sum())
    total_actual_points = float(group_frame["total_actual_points"].sum())
    total_predicted_points = float(group_frame["total_predicted_points"].sum())
    worst_min_budget = _column_min_or_none(group_frame, "min_budget")
    worst_max_budget_drawdown = _column_max_or_none(group_frame, "max_budget_drawdown")
    total_budget_constrained_rounds = _column_int_sum_or_none(group_frame, "budget_constrained_rounds")
    season_deltas = _season_deltas(group_frame, baseline_by_season=baseline_by_season)
    aggregate_delta = _sum_or_none([delta for delta, _rounds in season_deltas])
    baseline_total_actual_points = _sum_or_none(
        [
            baseline_by_season[
                (int(row["season"]), str(row["fixture_mode"]), normalize_budget_policy(row.get("budget_policy")))
            ]
            for row in group_frame.to_dict(orient="records")
            if (int(row["season"]), str(row["fixture_mode"]), normalize_budget_policy(row.get("budget_policy")))
            in baseline_by_season
        ]
    )
    average_actual_delta_per_round = None if aggregate_delta is None or total_rounds == 0 else aggregate_delta / total_rounds
    season_average_deltas = [delta / rounds for delta, rounds in season_deltas if rounds > 0]
    worst_season_avg_delta = min(season_average_deltas) if season_average_deltas else None
    selected_calibration_slope = _mean_metric(
        prediction_metrics,
        model_id=model_id,
        feature_pack=feature_pack,
        fixture_mode=fixture_mode,
        budget_policy=budget_policy,
        metric_scope="selected_players",
        metric_column="calibration_slope",
    )
    top50_spearman = _mean_metric(
        prediction_metrics,
        model_id=model_id,
        feature_pack=feature_pack,
        fixture_mode=fixture_mode,
        budget_policy=budget_policy,
        metric_scope="top50_candidates",
        metric_column="spearman",
    )
    baseline_top50_spearman = _mean_baseline_metric(group_frame, top50_spearman_baseline)
    top50_spearman_delta = (
        None if top50_spearman is None or baseline_top50_spearman is None else top50_spearman - baseline_top50_spearman
    )
    promotion = promotion_status(
        aggregate_delta=aggregate_delta,
        improved_seasons=sum(1 for delta, _rounds in season_deltas if delta > 0),
        worst_season_avg_delta=worst_season_avg_delta,
        selected_calibration_slope=selected_calibration_slope,
        top50_spearman_delta=top50_spearman_delta,
        comparable=True,
    )
    return {
        "model_id": model_id,
        "feature_pack": feature_pack,
        "fixture_mode": fixture_mode,
        "budget_policy": budget_policy,
        "seasons_evaluated": int(group_frame["season"].nunique()),
        "total_rounds": total_rounds,
        "total_actual_points": total_actual_points,
        "average_actual_points": None if total_rounds == 0 else total_actual_points / total_rounds,
        "total_predicted_points": total_predicted_points,
        "average_predicted_points": None if total_rounds == 0 else total_predicted_points / total_rounds,
        "worst_min_budget": worst_min_budget,
        "worst_max_budget_drawdown": worst_max_budget_drawdown,
        "total_budget_constrained_rounds": total_budget_constrained_rounds,
        "baseline_total_actual_points": baseline_total_actual_points,
        "aggregate_delta": aggregate_delta,
        "average_actual_delta_per_round": average_actual_delta_per_round,
        "improved_seasons": sum(1 for delta, _rounds in season_deltas if delta > 0),
        "worst_season_avg_delta": worst_season_avg_delta,
        "selected_calibration_slope": selected_calibration_slope,
        "top50_spearman_delta": top50_spearman_delta,
        "promotion_eligible": bool(promotion["eligible"]),
        "promotion_reason": str(promotion["reason"]),
    }


def _baseline_actual_points_by_season(per_season_summary: pd.DataFrame) -> dict[tuple[int, str, str], float]:
    baseline = per_season_summary[
        per_season_summary["model_id"].eq("random_forest") & per_season_summary["feature_pack"].eq("ppg")
    ]
    return {
        (int(row["season"]), str(row["fixture_mode"]), normalize_budget_policy(row.get("budget_policy"))): float(
            row["total_actual_points"]
        )
        for row in baseline.to_dict(orient="records")
    }


def _baseline_metric_by_season(
    prediction_metrics: pd.DataFrame,
    *,
    metric_scope: str,
    metric_column: str,
) -> dict[tuple[int, str, str], float]:
    if prediction_metrics.empty or metric_column not in prediction_metrics.columns:
        return {}
    baseline = prediction_metrics[
        prediction_metrics["model_id"].eq("random_forest")
        & prediction_metrics["feature_pack"].eq("ppg")
        & prediction_metrics["metric_scope"].eq(metric_scope)
    ]
    values: dict[tuple[int, str, str], float] = {}
    for row in baseline.to_dict(orient="records"):
        value = row[metric_column]
        if not pd.isna(value):
            values[
                (int(row["season"]), str(row["fixture_mode"]), normalize_budget_policy(row.get("budget_policy")))
            ] = float(value)
    return values


def _season_deltas(
    group_frame: pd.DataFrame,
    *,
    baseline_by_season: Mapping[tuple[int, str, str], float],
) -> list[tuple[float, int]]:
    deltas: list[tuple[float, int]] = []
    for row in group_frame.to_dict(orient="records"):
        baseline = baseline_by_season.get(
            (int(row["season"]), str(row["fixture_mode"]), normalize_budget_policy(row.get("budget_policy")))
        )
        if baseline is not None:
            deltas.append((float(row["total_actual_points"]) - baseline, int(row["rounds"])))
    return deltas


def _sum_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values))


def _column_min_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min())


def _column_max_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _column_int_sum_or_none(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return int(values.sum())


def _with_normalized_budget_policy(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "budget_policy" not in result.columns:
        result["budget_policy"] = normalize_budget_policy(None)
    else:
        result["budget_policy"] = result["budget_policy"].map(normalize_budget_policy)
    return result


def _mean_metric(
    prediction_metrics: pd.DataFrame,
    *,
    model_id: str,
    feature_pack: str,
    fixture_mode: str,
    budget_policy: str,
    metric_scope: str,
    metric_column: str,
) -> float | None:
    if prediction_metrics.empty or metric_column not in prediction_metrics.columns:
        return None
    values = prediction_metrics[
        prediction_metrics["model_id"].eq(model_id)
        & prediction_metrics["feature_pack"].eq(feature_pack)
        & prediction_metrics["fixture_mode"].eq(fixture_mode)
        & prediction_metrics["budget_policy"].eq(normalize_budget_policy(budget_policy))
        & prediction_metrics["metric_scope"].eq(metric_scope)
    ][metric_column].dropna()
    if values.empty:
        return None
    return float(values.mean())


def _mean_baseline_metric(
    group_frame: pd.DataFrame,
    baseline_by_season: Mapping[tuple[int, str, str], float],
) -> float | None:
    values = [
        baseline_by_season[
            (int(row["season"]), str(row["fixture_mode"]), normalize_budget_policy(row.get("budget_policy")))
        ]
        for row in group_frame.to_dict(orient="records")
        if (int(row["season"]), str(row["fixture_mode"]), normalize_budget_policy(row.get("budget_policy")))
        in baseline_by_season
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _metadata(
    *,
    status: str,
    experiment_id_value: str,
    started_at_utc: str,
    group: ExperimentGroup,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    matrix_hash: str,
    child_runs: list[dict[str, object]],
    raw_sources: Mapping[str, object],
    candidate_pool_signatures: Mapping[str, object],
    solver_status_signatures: Mapping[str, object],
    failure: Mapping[str, object] | None,
    tracking_warnings: Sequence[Mapping[str, object]] | None = None,
    index_warnings: Sequence[str] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "status": status,
        "experiment_id": experiment_id_value,
        "experiment_started_at_utc": started_at_utc,
        "group": group,
        "seasons": list(seasons),
        "start_round": start_round,
        "budget": budget,
        "budget_policy": BUDGET_POLICY_MOVING,
        "initial_budget": budget,
        "current_year": current_year,
        "jobs": jobs,
        "matrix_hash": matrix_hash,
        "child_runs": child_runs,
        "raw_sources": dict(raw_sources),
        "candidate_pool_signatures": dict(candidate_pool_signatures),
        "solver_status_signatures": dict(solver_status_signatures),
        "tracking_warnings": [dict(warning) for warning in tracking_warnings or []],
        "index_warnings": list(index_warnings or []),
    }
    if failure is not None:
        metadata["failure"] = dict(failure)
    return metadata


def _metadata_with_current_warnings(
    metadata: Mapping[str, object],
    *,
    tracker: ExperimentTracker,
    index_warnings: Sequence[str],
) -> dict[str, object]:
    updated = dict(metadata)
    updated["tracking_warnings"] = [asdict(warning) for warning in tracker.warnings]
    updated["index_warnings"] = list(index_warnings)
    return updated


def _write_success_artifacts(
    output_path: Path,
    metadata: Mapping[str, object],
    ranked_summary: pd.DataFrame,
    per_season_summary: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    calibration_deciles: pd.DataFrame,
) -> None:
    _write_json(output_path / "experiment_metadata.json", metadata)
    ranked_summary.to_csv(output_path / "ranked_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    per_season_summary.to_csv(output_path / "per_season_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    prediction_metrics.to_csv(output_path / "prediction_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    calibration_deciles.to_csv(output_path / "calibration_deciles.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    _write_json(output_path / "comparability_report.json", {"status": "ok"})
    (output_path / "comparison_report.md").write_text("# Model Feature Experiment\n\nStatus: ok\n", encoding="utf-8")
    build_experiment_html_reports(output_path)


def _write_failure_artifacts(output_path: Path, metadata: Mapping[str, object]) -> None:
    _write_json(output_path / "experiment_metadata.json", metadata)
    failure = metadata.get("failure")
    _write_json(
        output_path / "comparability_report.json",
        {
            "status": "failed",
            "failure": failure,
        },
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, default=str, sort_keys=True, indent=2), encoding="utf-8")


def _round_key(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
