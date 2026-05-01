from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Literal, SupportsFloat, cast

import pandas as pd

from cartola.backtesting.experiment_config import config_hash
from cartola.backtesting.experiment_metrics import calibration_slope_intercept, top_k_rows_by_round
from cartola.backtesting.experiment_signatures import (
    candidate_pool_signature,
    compare_signature_sets,
    solver_status_signature,
)
from cartola.backtesting.ridge_tuning_config import (
    PRIMARY_INCUMBENT_CANDIDATE_ID,
    SECONDARY_CONTROL_CANDIDATE_ID,
    RidgeTuningSpec,
    build_ridge_tuning_specs,
)
from cartola.backtesting.ridge_tuning_metrics import rank_tuning_summary
from cartola.backtesting.runner import CSV_FLOAT_FORMAT, BacktestResult, run_backtest_for_experiment


@dataclass(frozen=True)
class RidgeTuningRunResult:
    experiment_id: str
    output_path: Path
    ranked_summary: pd.DataFrame
    metadata: dict[str, object]


@dataclass(frozen=True)
class RidgeTuningProgressEvent:
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
    stage: str | None = None
    season: int | None = None
    candidate_id: str | None = None
    feature_pack: str | None = None
    alpha: float | None = None
    elapsed_seconds: float | None = None
    child_duration_seconds: float | None = None
    phase: str | None = None
    message: str | None = None


RidgeTuningProgressCallback = Callable[[RidgeTuningProgressEvent], None]


def run_ridge_tuning(
    *,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    project_root: Path,
    output_root: Path,
    started_at_utc: str,
    progress_callback: RidgeTuningProgressCallback | None = None,
    skip_final_rerun: bool = False,
) -> RidgeTuningRunResult:
    experiment_started = perf_counter()
    identity_specs = build_ridge_tuning_specs(
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        project_root=project_root,
        output_root=output_root,
        current_year=current_year,
        jobs=jobs,
        stage="screen",
    )
    matrix_hash = config_hash({"child_runs": [spec.config_identity for spec in identity_specs]})
    experiment_id = f"group=ridge-alpha-tuning__started_at={started_at_utc}__matrix={matrix_hash[:12]}"
    output_path = project_root / output_root / experiment_id
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True)

    screen_specs = build_ridge_tuning_specs(
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        project_root=project_root,
        output_root=output_root / experiment_id,
        current_year=current_year,
        jobs=jobs,
        stage="screen",
    )
    child_run_count = len(screen_specs)
    _write_generation_manifest(
        output_path=output_path,
        experiment_id=experiment_id,
        started_at_utc=started_at_utc,
        matrix_hash=matrix_hash,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        current_year=current_year,
        jobs=jobs,
        screen_specs=screen_specs,
        skip_final_rerun=skip_final_rerun,
    )
    _emit_progress(
        progress_callback,
        RidgeTuningProgressEvent(
            event_type="experiment_started",
            experiment_id=experiment_id,
            output_path=output_path,
            total_children=child_run_count,
            completed_children=0,
            elapsed_seconds=0.0,
            phase="screen",
        ),
    )

    run_state = _RunState()
    final_specs: list[RidgeTuningSpec] = []
    reproducibility_mismatches: list[dict[str, object]] = []

    try:
        _run_specs(
            screen_specs,
            experiment_id=experiment_id,
            output_path=output_path,
            total_children=child_run_count,
            completed_offset=0,
            child_index_offset=0,
            experiment_started=experiment_started,
            progress_callback=progress_callback,
            state=run_state,
        )
        _check_comparability(run_state)

        screen_per_season = pd.DataFrame(run_state.per_season_rows)
        screen_prediction_metrics = pd.DataFrame(run_state.prediction_metric_rows)
        screen_ranked = _rank_for_stage(
            screen_per_season,
            screen_prediction_metrics,
            stage="screen",
            final_reproducibility_by_candidate=_all_reproducible(screen_per_season),
        )

        if skip_final_rerun:
            ranked_summary = _mark_final_rerun_skipped(screen_ranked)
            promotion_report = {
                "recommendation": "keep_incumbent",
                "reason": "final_rerun_skipped",
                "promoted_candidate_id": None,
                "stage": "screen",
                "reproducibility_mismatches": [],
            }
        else:
            selected_candidate_ids = _final_candidate_ids(screen_ranked)
            final_specs = build_ridge_tuning_specs(
                seasons=seasons,
                start_round=start_round,
                budget=budget,
                project_root=project_root,
                output_root=output_root / experiment_id,
                current_year=current_year,
                jobs=jobs,
                stage="final",
                candidate_ids=selected_candidate_ids,
            )
            child_run_count += len(final_specs)
            _run_specs(
                final_specs,
                experiment_id=experiment_id,
                output_path=output_path,
                total_children=child_run_count,
                completed_offset=len(screen_specs),
                child_index_offset=len(screen_specs),
                experiment_started=experiment_started,
                progress_callback=progress_callback,
                state=run_state,
            )
            _check_comparability(run_state)

            all_per_season = pd.DataFrame(run_state.per_season_rows)
            all_prediction_metrics = pd.DataFrame(run_state.prediction_metric_rows)
            final_reproducibility = _final_reproducibility_by_candidate(all_per_season)
            reproducibility_mismatches = _final_reproducibility_mismatches(all_per_season)
            controls_reproducible = _comparison_controls_reproducible(final_reproducibility)
            ranking_reproducibility = (
                final_reproducibility
                if controls_reproducible
                else {spec.candidate_id: False for spec in final_specs}
            )
            ranked_summary = _rank_for_stage(
                all_per_season,
                all_prediction_metrics,
                stage="final",
                final_reproducibility_by_candidate=ranking_reproducibility,
            )
            promotion_report = _promotion_report_from_final_ranked(
                ranked_summary,
                final_reproducibility_by_candidate=ranking_reproducibility,
                reproducibility_mismatches=reproducibility_mismatches,
                controls_reproducible=controls_reproducible,
            )

        metadata = _metadata(
            status="ok",
            experiment_id=experiment_id,
            started_at_utc=started_at_utc,
            seasons=seasons,
            start_round=start_round,
            budget=budget,
            current_year=current_year,
            jobs=jobs,
            matrix_hash=matrix_hash,
            child_runs=run_state.child_runs,
            candidate_pool_signatures=run_state.candidate_pool_signatures,
            solver_status_signatures=run_state.solver_status_signatures,
            final_candidate_ids=[spec.candidate_id for spec in final_specs],
            reproducibility_mismatches=reproducibility_mismatches,
            failure=None,
        )
        _write_success_artifacts(
            output_path=output_path,
            ranked_summary=ranked_summary,
            per_season_summary=pd.DataFrame(run_state.per_season_rows),
            prediction_metrics=pd.DataFrame(run_state.prediction_metric_rows),
            calibration_deciles=pd.DataFrame(run_state.calibration_decile_rows),
            metadata=metadata,
            promotion_report=promotion_report,
            comparability_report=_comparability_report(run_state),
        )
        _emit_progress(
            progress_callback,
            RidgeTuningProgressEvent(
                event_type="experiment_finished",
                experiment_id=experiment_id,
                output_path=output_path,
                total_children=child_run_count,
                completed_children=len(run_state.child_runs),
                elapsed_seconds=perf_counter() - experiment_started,
            ),
        )
        return RidgeTuningRunResult(
            experiment_id=experiment_id,
            output_path=output_path,
            ranked_summary=ranked_summary,
            metadata=metadata,
        )
    except Exception as exc:
        metadata = _metadata(
            status="failed",
            experiment_id=experiment_id,
            started_at_utc=started_at_utc,
            seasons=seasons,
            start_round=start_round,
            budget=budget,
            current_year=current_year,
            jobs=jobs,
            matrix_hash=matrix_hash,
            child_runs=run_state.child_runs,
            candidate_pool_signatures=run_state.candidate_pool_signatures,
            solver_status_signatures=run_state.solver_status_signatures,
            final_candidate_ids=[spec.candidate_id for spec in final_specs],
            reproducibility_mismatches=reproducibility_mismatches,
            failure={"phase": "run", "message": str(exc), "type": type(exc).__name__},
        )
        _write_failure_artifacts(output_path, metadata)
        raise


@dataclass
class _RunState:
    child_runs: list[dict[str, object]]
    per_season_rows: list[dict[str, object]]
    prediction_metric_rows: list[dict[str, object]]
    calibration_decile_rows: list[dict[str, object]]
    candidate_pool_signatures: dict[str, dict[str, str]]
    solver_status_signatures: dict[str, dict[str, str]]
    comparability_partitions: dict[str, list[str]]

    def __init__(self) -> None:
        self.child_runs = []
        self.per_season_rows = []
        self.prediction_metric_rows = []
        self.calibration_decile_rows = []
        self.candidate_pool_signatures = {}
        self.solver_status_signatures = {}
        self.comparability_partitions = {}


def _run_specs(
    specs: Sequence[RidgeTuningSpec],
    *,
    experiment_id: str,
    output_path: Path,
    total_children: int,
    completed_offset: int,
    child_index_offset: int,
    experiment_started: float,
    progress_callback: RidgeTuningProgressCallback | None,
    state: _RunState,
) -> None:
    for relative_index, spec in enumerate(specs, start=1):
        child_index = child_index_offset + relative_index
        child_id = _child_id(spec)
        child_started = perf_counter()
        _emit_progress(
            progress_callback,
            _progress_event(
                "child_started",
                experiment_id=experiment_id,
                output_path=output_path,
                total_children=total_children,
                completed_children=completed_offset + relative_index - 1,
                child_index=child_index,
                child_id=child_id,
                spec=spec,
                elapsed_seconds=child_started - experiment_started,
            ),
        )
        try:
            result = run_backtest_for_experiment(
                spec.backtest_config,
                primary_model_id="ridge",
                model_params={"alpha": spec.alpha},
            )
            child_candidate_signatures = _candidate_signatures_by_round(result.player_predictions)
            child_solver_signature = solver_status_signature(result.round_results, primary_model_id="ridge")
            state.child_runs.append(_child_record(spec, result, child_id=child_id))
            state.candidate_pool_signatures[child_id] = child_candidate_signatures
            state.solver_status_signatures[child_id] = child_solver_signature
            state.comparability_partitions.setdefault(_comparability_partition(spec), []).append(child_id)
            state.per_season_rows.extend(_primary_summary_rows(spec, result, child_id=child_id))
            state.prediction_metric_rows.extend(_prediction_metric_rows(spec, result, child_id=child_id))
            state.calibration_decile_rows.extend(_calibration_decile_rows(spec, result, child_id=child_id))
        except Exception as exc:
            failed_at = perf_counter()
            _emit_progress(
                progress_callback,
                _progress_event(
                    "child_failed",
                    experiment_id=experiment_id,
                    output_path=output_path,
                    total_children=total_children,
                    completed_children=len(state.child_runs),
                    child_index=child_index,
                    child_id=child_id,
                    spec=spec,
                    elapsed_seconds=failed_at - experiment_started,
                    child_duration_seconds=failed_at - child_started,
                    phase="child_run",
                    message=str(exc),
                ),
            )
            raise
        finished_at = perf_counter()
        _emit_progress(
            progress_callback,
            _progress_event(
                "child_finished",
                experiment_id=experiment_id,
                output_path=output_path,
                total_children=total_children,
                completed_children=len(state.child_runs),
                child_index=child_index,
                child_id=child_id,
                spec=spec,
                elapsed_seconds=finished_at - experiment_started,
                child_duration_seconds=finished_at - child_started,
            ),
        )


def _emit_progress(
    callback: RidgeTuningProgressCallback | None,
    event: RidgeTuningProgressEvent,
) -> None:
    if callback is not None:
        callback(event)


def _progress_event(
    event_type: Literal["child_started", "child_finished", "child_failed"],
    *,
    experiment_id: str,
    output_path: Path,
    total_children: int,
    completed_children: int,
    child_index: int,
    child_id: str,
    spec: RidgeTuningSpec,
    elapsed_seconds: float,
    child_duration_seconds: float | None = None,
    phase: str | None = None,
    message: str | None = None,
) -> RidgeTuningProgressEvent:
    return RidgeTuningProgressEvent(
        event_type=event_type,
        experiment_id=experiment_id,
        output_path=output_path,
        total_children=total_children,
        completed_children=completed_children,
        child_index=child_index,
        child_id=child_id,
        stage=spec.stage,
        season=spec.season,
        candidate_id=spec.candidate_id,
        feature_pack=spec.feature_pack,
        alpha=spec.alpha,
        elapsed_seconds=elapsed_seconds,
        child_duration_seconds=child_duration_seconds,
        phase=phase,
        message=message,
    )


def _child_id(spec: RidgeTuningSpec) -> str:
    return (
        f"stage={spec.stage}/season={spec.season}/candidate={spec.candidate_id}/"
        f"feature_pack={spec.feature_pack}/alpha={spec.alpha:g}"
    )


def _comparability_partition(spec: RidgeTuningSpec) -> str:
    return f"stage={spec.stage}/season={spec.season}"


def _child_record(spec: RidgeTuningSpec, result: BacktestResult, *, child_id: str) -> dict[str, object]:
    return {
        "child_id": child_id,
        "stage": spec.stage,
        "season": spec.season,
        "candidate_id": spec.candidate_id,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "alpha": spec.alpha,
        "model_params_hash": spec.model_params_hash,
        "tuning_generation_hash": spec.tuning_generation_hash,
        "output_path": str(spec.output_path),
        "metadata": asdict(result.metadata),
    }


def _primary_summary_rows(spec: RidgeTuningSpec, result: BacktestResult, *, child_id: str) -> list[dict[str, object]]:
    summary = result.summary[result.summary["strategy"] == spec.model_id] if "strategy" in result.summary.columns else result.summary
    rows: list[dict[str, object]] = []
    for row in summary.to_dict(orient="records"):
        rows.append(
            {
                "child_id": child_id,
                "stage": spec.stage,
                "candidate_id": spec.candidate_id,
                "season": spec.season,
                "model_id": spec.model_id,
                "feature_pack": spec.feature_pack,
                "alpha": spec.alpha,
                "model_params_json": json.dumps(dict(spec.model_parameters), sort_keys=True),
                "model_params_hash": spec.model_params_hash,
                "tuning_generation_hash": spec.tuning_generation_hash,
                **row,
            }
        )
    return rows


def _prediction_metric_rows(spec: RidgeTuningSpec, result: BacktestResult, *, child_id: str) -> list[dict[str, object]]:
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
            metric_scope=metric_scope,
            k=k,
            frame=frame,
            predicted_column=predicted_column,
        )
        for metric_scope, k, frame, predicted_column in scopes
    ]


def _prediction_metric_row(
    spec: RidgeTuningSpec,
    *,
    child_id: str,
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
        "stage": spec.stage,
        "candidate_id": spec.candidate_id,
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "alpha": spec.alpha,
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


def _calibration_decile_rows(spec: RidgeTuningSpec, result: BacktestResult, *, child_id: str) -> list[dict[str, object]]:
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
        residual = decile_frame["actual"] - decile_frame["predicted"]
        rows.append(
            {
                "child_id": child_id,
                "stage": spec.stage,
                "candidate_id": spec.candidate_id,
                "season": spec.season,
                "model_id": spec.model_id,
                "feature_pack": spec.feature_pack,
                "alpha": spec.alpha,
                "decile": int(str(decile)),
                "row_count": len(decile_frame),
                "predicted_mean": float(decile_frame["predicted"].mean()),
                "actual_mean": float(decile_frame["actual"].mean()),
                "residual_mean": float(residual.mean()),
            }
        )
    return rows


def _candidate_signatures_by_round(player_predictions: pd.DataFrame) -> dict[str, str]:
    if player_predictions.empty:
        return {}
    return {
        _round_key(round_number): candidate_pool_signature(round_frame)
        for round_number, round_frame in player_predictions.groupby("rodada", sort=True)
    }


def _round_key(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _check_comparability(state: _RunState) -> None:
    for partition_id, child_ids in state.comparability_partitions.items():
        rounds = sorted(
            {
                round_id
                for child_id in child_ids
                for round_id in state.candidate_pool_signatures.get(child_id, {})
            }
        )
        for round_id in rounds:
            compare_signature_sets(
                f"Candidate pool signatures for {partition_id} rodada={round_id}",
                {
                    child_id: state.candidate_pool_signatures.get(child_id, {}).get(round_id)
                    for child_id in child_ids
                },
            )
        compare_signature_sets(
            f"Solver-status signatures for {partition_id}",
            {child_id: state.solver_status_signatures.get(child_id) for child_id in child_ids},
        )


def _rank_for_stage(
    per_season_summary: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    *,
    stage: str,
    final_reproducibility_by_candidate: Mapping[str, bool],
) -> pd.DataFrame:
    if "stage" not in per_season_summary.columns:
        return rank_tuning_summary(
            per_season_summary,
            prediction_metrics,
            primary_incumbent_candidate_id=PRIMARY_INCUMBENT_CANDIDATE_ID,
            final_reproducibility_by_candidate=final_reproducibility_by_candidate,
        )

    stage_per_season = per_season_summary[per_season_summary["stage"].eq(stage)].copy()
    stage_prediction_metrics = (
        prediction_metrics[prediction_metrics["stage"].eq(stage)].copy()
        if "stage" in prediction_metrics.columns
        else prediction_metrics.copy()
    )
    return rank_tuning_summary(
        stage_per_season,
        stage_prediction_metrics,
        primary_incumbent_candidate_id=PRIMARY_INCUMBENT_CANDIDATE_ID,
        final_reproducibility_by_candidate=final_reproducibility_by_candidate,
    )


def _all_reproducible(per_season_summary: pd.DataFrame) -> dict[str, bool]:
    if "candidate_id" not in per_season_summary.columns:
        return {}
    return {str(candidate_id): True for candidate_id in per_season_summary["candidate_id"].dropna().unique()}


def _mark_final_rerun_skipped(ranked_summary: pd.DataFrame) -> pd.DataFrame:
    safe_ranked = ranked_summary.copy()
    if "promotion_eligible" in safe_ranked.columns:
        safe_ranked["promotion_eligible"] = False
    if "promotion_reason" in safe_ranked.columns:
        safe_ranked["promotion_reason"] = "final_rerun_skipped"
    return safe_ranked


def _final_candidate_ids(screen_ranked: pd.DataFrame) -> set[str]:
    candidate_ids = [PRIMARY_INCUMBENT_CANDIDATE_ID, SECONDARY_CONTROL_CANDIDATE_ID]
    if not screen_ranked.empty:
        eligible_challengers = screen_ranked[
            screen_ranked["promotion_eligible"].eq(True)
            & ~screen_ranked["candidate_id"].isin(
                [PRIMARY_INCUMBENT_CANDIDATE_ID, SECONDARY_CONTROL_CANDIDATE_ID],
            )
        ].sort_values(
            by=["aggregate_delta_vs_primary_incumbent", "total_actual_points", "candidate_id"],
            ascending=[False, False, True],
            na_position="last",
            kind="mergesort",
        )
        candidate_ids.extend(str(candidate_id) for candidate_id in eligible_challengers["candidate_id"].head(2))
    return set(dict.fromkeys(candidate_ids))


def _final_reproducibility_by_candidate(per_season_summary: pd.DataFrame) -> dict[str, bool]:
    if per_season_summary.empty or "stage" not in per_season_summary.columns:
        return {}
    totals = (
        per_season_summary.groupby(["stage", "candidate_id", "season"], sort=False)["total_actual_points"]
        .sum()
        .reset_index()
    )
    screen_totals = {
        (str(row["candidate_id"]), int(row["season"])): float(row["total_actual_points"])
        for row in totals[totals["stage"].eq("screen")].to_dict(orient="records")
    }
    reproducibility_by_candidate = {
        str(candidate_id): True for candidate_id in totals[totals["stage"].eq("final")]["candidate_id"].dropna().unique()
    }
    for row in totals[totals["stage"].eq("final")].to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        season = int(row["season"])
        final_total = float(row["total_actual_points"])
        screen_total = screen_totals.get((candidate_id, season))
        if screen_total is None or abs(final_total - screen_total) > 0.01:
            reproducibility_by_candidate[candidate_id] = False
    return reproducibility_by_candidate


def _final_reproducibility_mismatches(per_season_summary: pd.DataFrame) -> list[dict[str, object]]:
    if per_season_summary.empty or "stage" not in per_season_summary.columns:
        return []
    totals = (
        per_season_summary.groupby(["stage", "candidate_id", "season"], sort=False)["total_actual_points"]
        .sum()
        .reset_index()
    )
    screen_totals = {
        (str(row["candidate_id"]), int(row["season"])): float(row["total_actual_points"])
        for row in totals[totals["stage"].eq("screen")].to_dict(orient="records")
    }
    mismatches: list[dict[str, object]] = []
    for row in totals[totals["stage"].eq("final")].to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        season = int(row["season"])
        final_total = float(row["total_actual_points"])
        screen_total = screen_totals.get((candidate_id, season))
        absolute_delta = None if screen_total is None else abs(final_total - screen_total)
        if screen_total is None or absolute_delta is None or absolute_delta > 0.01:
            mismatches.append(
                {
                    "candidate_id": candidate_id,
                    "season": season,
                    "screen_total_actual_points": screen_total,
                    "final_total_actual_points": final_total,
                    "absolute_delta": absolute_delta,
                }
            )
    return mismatches


def _comparison_controls_reproducible(final_reproducibility_by_candidate: Mapping[str, bool]) -> bool:
    return (
        final_reproducibility_by_candidate.get(PRIMARY_INCUMBENT_CANDIDATE_ID) is True
        and final_reproducibility_by_candidate.get(SECONDARY_CONTROL_CANDIDATE_ID) is True
    )


def _promotion_report_from_final_ranked(
    ranked_summary: pd.DataFrame,
    *,
    final_reproducibility_by_candidate: Mapping[str, bool],
    reproducibility_mismatches: Sequence[Mapping[str, object]],
    controls_reproducible: bool,
) -> dict[str, object]:
    if ranked_summary.empty:
        return {
            "recommendation": "keep_incumbent",
            "reason": "no_final_candidates",
            "promoted_candidate_id": None,
            "stage": "final",
            "final_reproducibility_by_candidate": dict(final_reproducibility_by_candidate),
            "reproducibility_mismatches": [dict(mismatch) for mismatch in reproducibility_mismatches],
        }

    if not controls_reproducible:
        top = ranked_summary.iloc[0].to_dict()
        return {
            "recommendation": "keep_incumbent",
            "reason": "comparison_controls_non_reproducible",
            "promoted_candidate_id": None,
            "stage": "final",
            "top_candidate_id": str(top.get("candidate_id")),
            "final_reproducibility_by_candidate": dict(final_reproducibility_by_candidate),
            "reproducibility_mismatches": [dict(mismatch) for mismatch in reproducibility_mismatches],
        }

    eligible = ranked_summary[
        ranked_summary["promotion_eligible"].eq(True)
        & ~ranked_summary["candidate_id"].isin([PRIMARY_INCUMBENT_CANDIDATE_ID])
    ]
    if eligible.empty:
        top = ranked_summary.iloc[0].to_dict()
        return {
            "recommendation": "keep_incumbent",
            "reason": str(top.get("promotion_reason", "no_promotable_final_candidate")),
            "promoted_candidate_id": None,
            "stage": "final",
            "top_candidate_id": str(top.get("candidate_id")),
            "final_reproducibility_by_candidate": dict(final_reproducibility_by_candidate),
            "reproducibility_mismatches": [dict(mismatch) for mismatch in reproducibility_mismatches],
        }

    promoted = eligible.iloc[0].to_dict()
    return {
        "recommendation": "promote_candidate",
        "reason": str(promoted["promotion_reason"]),
        "promoted_candidate_id": str(promoted["candidate_id"]),
        "stage": "final",
        "aggregate_delta_vs_primary_incumbent": _float_or_none(promoted.get("aggregate_delta_vs_primary_incumbent")),
        "final_reproducibility_by_candidate": dict(final_reproducibility_by_candidate),
        "reproducibility_mismatches": [dict(mismatch) for mismatch in reproducibility_mismatches],
    }


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(cast(SupportsFloat, value))


def _write_generation_manifest(
    *,
    output_path: Path,
    experiment_id: str,
    started_at_utc: str,
    matrix_hash: str,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    screen_specs: Sequence[RidgeTuningSpec],
    skip_final_rerun: bool,
) -> None:
    payload = {
        "experiment_id": experiment_id,
        "started_at_utc": started_at_utc,
        "matrix_hash": matrix_hash,
        "seasons": list(seasons),
        "start_round": start_round,
        "budget": budget,
        "current_year": current_year,
        "jobs": jobs,
        "skip_final_rerun": skip_final_rerun,
        "candidate_count": len({spec.candidate_id for spec in screen_specs}),
        "child_run_count": len(screen_specs),
        "children": [dict(spec.config_identity) for spec in screen_specs],
    }
    _write_json(output_path / "tuning_generation_manifest.json", payload)


def _metadata(
    *,
    status: str,
    experiment_id: str,
    started_at_utc: str,
    seasons: tuple[int, ...],
    start_round: int,
    budget: float,
    current_year: int,
    jobs: int,
    matrix_hash: str,
    child_runs: Sequence[Mapping[str, object]],
    candidate_pool_signatures: Mapping[str, object],
    solver_status_signatures: Mapping[str, object],
    final_candidate_ids: Sequence[str],
    reproducibility_mismatches: Sequence[Mapping[str, object]],
    failure: Mapping[str, object] | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "status": status,
        "experiment_id": experiment_id,
        "experiment_started_at_utc": started_at_utc,
        "group": "ridge-alpha-tuning",
        "seasons": list(seasons),
        "start_round": start_round,
        "budget": budget,
        "current_year": current_year,
        "jobs": jobs,
        "matrix_hash": matrix_hash,
        "child_runs": [dict(child_run) for child_run in child_runs],
        "candidate_pool_signatures": dict(candidate_pool_signatures),
        "solver_status_signatures": dict(solver_status_signatures),
        "final_candidate_ids": list(final_candidate_ids),
        "reproducibility_mismatches": [dict(mismatch) for mismatch in reproducibility_mismatches],
    }
    if failure is not None:
        metadata["failure"] = dict(failure)
    return metadata


def _comparability_report(state: _RunState) -> dict[str, object]:
    return {
        "status": "ok",
        "partitions": {partition: list(child_ids) for partition, child_ids in state.comparability_partitions.items()},
        "candidate_pool_signature_count": len(state.candidate_pool_signatures),
        "solver_status_signature_count": len(state.solver_status_signatures),
    }


def _write_success_artifacts(
    *,
    output_path: Path,
    ranked_summary: pd.DataFrame,
    per_season_summary: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    calibration_deciles: pd.DataFrame,
    metadata: Mapping[str, object],
    promotion_report: Mapping[str, object],
    comparability_report: Mapping[str, object],
) -> None:
    ranked_summary.to_csv(output_path / "ranked_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    per_season_summary.to_csv(output_path / "per_season_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    prediction_metrics.to_csv(output_path / "prediction_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    calibration_deciles.to_csv(output_path / "calibration_deciles.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    _write_json(output_path / "comparability_report.json", comparability_report)
    _write_json(output_path / "promotion_report.json", promotion_report)
    _write_json(output_path / "experiment_metadata.json", metadata)
    (output_path / "comparison_report.md").write_text(
        _comparison_report_markdown(
            status="ok",
            promotion_report=promotion_report,
            ranked_summary=ranked_summary,
        ),
        encoding="utf-8",
    )
    (output_path / "calibration_plots.html").write_text(
        _summary_html(
            title="Ridge tuning calibration plots",
            status="ok",
            promotion_report=promotion_report,
            ranked_summary=ranked_summary,
        ),
        encoding="utf-8",
    )
    (output_path / "squad_performance_comparison.html").write_text(
        _summary_html(
            title="Ridge tuning squad performance comparison",
            status="ok",
            promotion_report=promotion_report,
            ranked_summary=ranked_summary,
        ),
        encoding="utf-8",
    )


def _comparison_report_markdown(
    *,
    status: str,
    promotion_report: Mapping[str, object],
    ranked_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Ridge Tuning Experiment",
        "",
        f"Status: {status}",
        f"Recommendation: {promotion_report.get('recommendation')}",
        f"Reason: {promotion_report.get('reason')}",
        "",
        "## Ranked Summary",
        "",
    ]
    rows = _ranked_summary_rows(ranked_summary)
    if not rows:
        lines.append("No ranked candidates.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| rank | candidate_id | total_actual_points | promotion_eligible | promotion_reason |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {rank} | {candidate_id} | {total_actual_points} | {promotion_eligible} | {promotion_reason} |".format(
                rank=row["rank"],
                candidate_id=row["candidate_id"],
                total_actual_points=row["total_actual_points"],
                promotion_eligible=row["promotion_eligible"],
                promotion_reason=row["promotion_reason"],
            )
        )
    return "\n".join(lines) + "\n"


def _summary_html(
    *,
    title: str,
    status: str,
    promotion_report: Mapping[str, object],
    ranked_summary: pd.DataFrame,
) -> str:
    rows = _ranked_summary_rows(ranked_summary)
    table_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row['rank']))}</td>"
        f"<td>{escape(str(row['candidate_id']))}</td>"
        f"<td>{escape(str(row['total_actual_points']))}</td>"
        f"<td>{escape(str(row['promotion_eligible']))}</td>"
        f"<td>{escape(str(row['promotion_reason']))}</td>"
        "</tr>"
        for row in rows
    )
    if not table_rows:
        table_rows = "<tr><td colspan=\"5\">No ranked candidates.</td></tr>"
    return (
        "<!doctype html>\n"
        f"<title>{escape(title)}</title>\n"
        f"<h1>{escape(title)}</h1>\n"
        f"<p>Status: {escape(status)}</p>\n"
        f"<p>Recommendation: {escape(str(promotion_report.get('recommendation')))}</p>\n"
        f"<p>Reason: {escape(str(promotion_report.get('reason')))}</p>\n"
        "<table>\n"
        "<thead><tr><th>rank</th><th>candidate_id</th><th>total_actual_points</th>"
        "<th>promotion_eligible</th><th>promotion_reason</th></tr></thead>\n"
        f"<tbody>{table_rows}</tbody>\n"
        "</table>\n"
    )


def _ranked_summary_rows(ranked_summary: pd.DataFrame) -> list[dict[str, object]]:
    if ranked_summary.empty:
        return []
    columns = [
        "rank",
        "candidate_id",
        "total_actual_points",
        "promotion_eligible",
        "promotion_reason",
    ]
    available_columns = [column for column in columns if column in ranked_summary.columns]
    rows = ranked_summary.loc[:, available_columns].head(5).to_dict(orient="records")
    return [
        {
            "rank": row.get("rank", ""),
            "candidate_id": row.get("candidate_id", ""),
            "total_actual_points": _report_value(row.get("total_actual_points")),
            "promotion_eligible": row.get("promotion_eligible", ""),
            "promotion_reason": row.get("promotion_reason", ""),
        }
        for row in rows
    ]


def _report_value(value: object) -> object:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return CSV_FLOAT_FORMAT % value
    return value


def _write_failure_artifacts(output_path: Path, metadata: Mapping[str, object]) -> None:
    _write_json(output_path / "experiment_metadata.json", metadata)
    _write_json(
        output_path / "comparability_report.json",
        {
            "status": "failed",
            "failure": metadata.get("failure"),
        },
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True), encoding="utf-8")
