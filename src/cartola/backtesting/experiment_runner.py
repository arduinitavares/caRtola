from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import pandas as pd

from cartola.backtesting.experiment_config import (
    ChildRunSpec,
    ExperimentGroup,
    build_child_run_specs,
    config_hash,
    experiment_id,
)
from cartola.backtesting.experiment_metrics import (
    calibration_slope_intercept,
    promotion_status,
    top_k_rows_by_round,
)
from cartola.backtesting.experiment_signatures import (
    ComparabilityError,
    candidate_pool_signature,
    compare_signature_sets,
    raw_cartola_source_identity,
    solver_status_signature,
)
from cartola.backtesting.model_registry import model_n_jobs_for_metadata
from cartola.backtesting.runner import CSV_FLOAT_FORMAT, BacktestResult, run_backtest_for_experiment


@dataclass(frozen=True)
class ExperimentRunResult:
    experiment_id: str
    output_path: Path
    ranked_summary: pd.DataFrame
    metadata: dict[str, object]


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
) -> ExperimentRunResult:
    identity_specs = build_child_run_specs(
        group=group,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        project_root=project_root,
        output_root=output_root,
        current_year=current_year,
        jobs=jobs,
    )
    matrix_hash = config_hash({"child_runs": [spec.config_identity for spec in identity_specs]})
    run_id = experiment_id(group=group, started_at_utc=started_at_utc, matrix_hash=matrix_hash)
    output_path = project_root / output_root / run_id
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True)

    specs = build_child_run_specs(
        group=group,
        seasons=seasons,
        start_round=start_round,
        budget=budget,
        project_root=project_root,
        output_root=output_root / run_id,
        current_year=current_year,
        jobs=jobs,
    )
    raw_sources = {
        str(season): raw_cartola_source_identity(project_root=project_root, season=season) for season in seasons
    }

    child_runs: list[dict[str, object]] = []
    per_season_rows: list[dict[str, object]] = []
    prediction_metric_rows: list[dict[str, object]] = []
    calibration_decile_rows: list[dict[str, object]] = []
    candidate_pool_signatures: dict[str, dict[str, str]] = {}
    solver_status_signatures: dict[str, dict[str, str]] = {}
    comparability_partitions: dict[str, list[str]] = {}

    for spec in specs:
        child_id = _child_id(spec)
        try:
            result = run_backtest_for_experiment(spec.backtest_config, primary_model_id=spec.model_id)
        except Exception as exc:
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
                failure={"phase": "child_run", "message": str(exc), "child_id": child_id},
            )
            _write_failure_artifacts(output_path, metadata)
            raise
        child_runs.append(_child_record(spec, result, child_id=child_id))
        try:
            candidate_pool_signatures[child_id] = _candidate_signatures_by_round(result.player_predictions)
            solver_status_signatures[child_id] = solver_status_signature(
                result.round_results,
                primary_model_id=spec.model_id,
            )
            comparability_partitions.setdefault(_comparability_partition(spec), []).append(child_id)
            per_season_rows.extend(_primary_summary_rows(spec, result, child_id=child_id))
            prediction_metric_rows.extend(_prediction_metric_rows(spec, result, child_id=child_id))
            calibration_decile_rows.extend(_calibration_decile_rows(spec, result, child_id=child_id))
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
                failure={"phase": "comparability", "message": str(exc), "child_id": child_id},
            )
            _write_failure_artifacts(output_path, metadata)
            raise

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
            failure={"phase": "comparability", "message": str(exc)},
        )
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
        failure=None,
    )
    _write_success_artifacts(
        output_path,
        metadata,
        ranked_summary,
        per_season_summary,
        prediction_metrics,
        calibration_deciles,
    )

    return ExperimentRunResult(
        experiment_id=run_id,
        output_path=output_path,
        ranked_summary=ranked_summary,
        metadata=metadata,
    )


def _child_id(spec: ChildRunSpec) -> str:
    return f"season={spec.season}/model={spec.model_id}/feature_pack={spec.feature_pack}"


def _comparability_partition(spec: ChildRunSpec) -> str:
    return f"season={spec.season}"


def _child_record(spec: ChildRunSpec, result: BacktestResult, *, child_id: str) -> dict[str, object]:
    model_n_jobs_effective = (
        result.metadata.model_n_jobs_effective
        if model_n_jobs_for_metadata(spec.model_id, requested_n_jobs=spec.jobs) is not None
        else None
    )
    return {
        "child_id": child_id,
        "season": spec.season,
        "model_id": spec.model_id,
        "feature_pack": spec.feature_pack,
        "fixture_mode": spec.fixture_mode,
        "output_path": str(spec.output_path),
        "model_n_jobs_effective": model_n_jobs_effective,
        "strategy_roles": {
            "baseline": "baseline",
            spec.model_id: "primary_model",
            "price": "price",
        },
        "metadata": asdict(result.metadata),
    }


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
        for _group_key, group_frame in per_season_summary.groupby(["model_id", "feature_pack", "fixture_mode"], sort=False)
    ]
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        by=["promotion_eligible", "aggregate_delta", "total_actual_points", "model_id", "feature_pack", "fixture_mode"],
        ascending=[False, False, False, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", pd.Series(range(1, len(ranked) + 1), dtype="int64"))
    return ranked.loc[:, ["rank", *_RANKED_SUMMARY_COLUMNS]]


_RANKED_SUMMARY_COLUMNS = [
    "model_id",
    "feature_pack",
    "fixture_mode",
    "seasons_evaluated",
    "total_rounds",
    "total_actual_points",
    "average_actual_points",
    "total_predicted_points",
    "average_predicted_points",
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
    baseline_by_season: Mapping[tuple[int, str], float],
    top50_spearman_baseline: Mapping[tuple[int, str], float],
) -> dict[str, object]:
    first = group_frame.iloc[0]
    model_id = str(first["model_id"])
    feature_pack = str(first["feature_pack"])
    fixture_mode = str(first["fixture_mode"])
    total_rounds = int(group_frame["rounds"].sum())
    total_actual_points = float(group_frame["total_actual_points"].sum())
    total_predicted_points = float(group_frame["total_predicted_points"].sum())
    season_deltas = _season_deltas(group_frame, baseline_by_season=baseline_by_season)
    aggregate_delta = _sum_or_none([delta for delta, _rounds in season_deltas])
    baseline_total_actual_points = _sum_or_none(
        [
            baseline_by_season[(int(row["season"]), str(row["fixture_mode"]))]
            for row in group_frame.to_dict(orient="records")
            if (int(row["season"]), str(row["fixture_mode"])) in baseline_by_season
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
        metric_scope="selected_players",
        metric_column="calibration_slope",
    )
    top50_spearman = _mean_metric(
        prediction_metrics,
        model_id=model_id,
        feature_pack=feature_pack,
        fixture_mode=fixture_mode,
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
        "seasons_evaluated": int(group_frame["season"].nunique()),
        "total_rounds": total_rounds,
        "total_actual_points": total_actual_points,
        "average_actual_points": None if total_rounds == 0 else total_actual_points / total_rounds,
        "total_predicted_points": total_predicted_points,
        "average_predicted_points": None if total_rounds == 0 else total_predicted_points / total_rounds,
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


def _baseline_actual_points_by_season(per_season_summary: pd.DataFrame) -> dict[tuple[int, str], float]:
    baseline = per_season_summary[
        per_season_summary["model_id"].eq("random_forest") & per_season_summary["feature_pack"].eq("ppg")
    ]
    return {
        (int(row["season"]), str(row["fixture_mode"])): float(row["total_actual_points"])
        for row in baseline.to_dict(orient="records")
    }


def _baseline_metric_by_season(
    prediction_metrics: pd.DataFrame,
    *,
    metric_scope: str,
    metric_column: str,
) -> dict[tuple[int, str], float]:
    if prediction_metrics.empty or metric_column not in prediction_metrics.columns:
        return {}
    baseline = prediction_metrics[
        prediction_metrics["model_id"].eq("random_forest")
        & prediction_metrics["feature_pack"].eq("ppg")
        & prediction_metrics["metric_scope"].eq(metric_scope)
    ]
    values: dict[tuple[int, str], float] = {}
    for row in baseline.to_dict(orient="records"):
        value = row[metric_column]
        if not pd.isna(value):
            values[(int(row["season"]), str(row["fixture_mode"]))] = float(value)
    return values


def _season_deltas(
    group_frame: pd.DataFrame,
    *,
    baseline_by_season: Mapping[tuple[int, str], float],
) -> list[tuple[float, int]]:
    deltas: list[tuple[float, int]] = []
    for row in group_frame.to_dict(orient="records"):
        baseline = baseline_by_season.get((int(row["season"]), str(row["fixture_mode"])))
        if baseline is not None:
            deltas.append((float(row["total_actual_points"]) - baseline, int(row["rounds"])))
    return deltas


def _sum_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values))


def _mean_metric(
    prediction_metrics: pd.DataFrame,
    *,
    model_id: str,
    feature_pack: str,
    fixture_mode: str,
    metric_scope: str,
    metric_column: str,
) -> float | None:
    if prediction_metrics.empty or metric_column not in prediction_metrics.columns:
        return None
    values = prediction_metrics[
        prediction_metrics["model_id"].eq(model_id)
        & prediction_metrics["feature_pack"].eq(feature_pack)
        & prediction_metrics["fixture_mode"].eq(fixture_mode)
        & prediction_metrics["metric_scope"].eq(metric_scope)
    ][metric_column].dropna()
    if values.empty:
        return None
    return float(values.mean())


def _mean_baseline_metric(
    group_frame: pd.DataFrame,
    baseline_by_season: Mapping[tuple[int, str], float],
) -> float | None:
    values = [
        baseline_by_season[(int(row["season"]), str(row["fixture_mode"]))]
        for row in group_frame.to_dict(orient="records")
        if (int(row["season"]), str(row["fixture_mode"])) in baseline_by_season
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
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "status": status,
        "experiment_id": experiment_id_value,
        "experiment_started_at_utc": started_at_utc,
        "group": group,
        "seasons": list(seasons),
        "start_round": start_round,
        "budget": budget,
        "current_year": current_year,
        "jobs": jobs,
        "matrix_hash": matrix_hash,
        "child_runs": child_runs,
        "raw_sources": dict(raw_sources),
        "candidate_pool_signatures": dict(candidate_pool_signatures),
        "solver_status_signatures": dict(solver_status_signatures),
    }
    if failure is not None:
        metadata["failure"] = dict(failure)
    return metadata


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
    (output_path / "calibration_plots.html").write_text("<!doctype html><title>Calibration plots</title>\n", encoding="utf-8")
    (output_path / "squad_performance_comparison.html").write_text(
        "<!doctype html><title>Squad performance comparison</title>\n",
        encoding="utf-8",
    )


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
