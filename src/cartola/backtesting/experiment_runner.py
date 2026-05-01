from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from cartola.backtesting.experiment_config import (
    ChildRunSpec,
    ExperimentGroup,
    build_child_run_specs,
    config_hash,
    experiment_id,
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
        candidate_pool_signatures[child_id] = _candidate_signatures_by_round(result.player_predictions)
        solver_status_signatures[child_id] = solver_status_signature(
            result.round_results,
            primary_model_id=spec.model_id,
        )
        comparability_partitions.setdefault(_comparability_partition(spec), []).append(child_id)
        per_season_rows.extend(_primary_summary_rows(spec, result, child_id=child_id))

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
    ranked_summary = _rank_summary(per_season_summary)
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
    _write_success_artifacts(output_path, metadata, ranked_summary, per_season_summary)

    return ExperimentRunResult(
        experiment_id=run_id,
        output_path=output_path,
        ranked_summary=ranked_summary,
        metadata=metadata,
    )


def _child_id(spec: ChildRunSpec) -> str:
    return f"season={spec.season}/model={spec.model_id}/feature_pack={spec.feature_pack}"


def _comparability_partition(spec: ChildRunSpec) -> str:
    return f"season={spec.season}/feature_pack={spec.feature_pack}"


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


def _rank_summary(per_season_summary: pd.DataFrame) -> pd.DataFrame:
    if per_season_summary.empty:
        ranked = per_season_summary.copy()
        ranked.insert(0, "rank", pd.Series(dtype="int64"))
        return ranked

    sort_columns = [
        column
        for column in ("total_actual_points", "average_actual_points", "model_id", "feature_pack", "season")
        if column in per_season_summary.columns
    ]
    ascending = [False if column in {"total_actual_points", "average_actual_points"} else True for column in sort_columns]
    ranked = per_season_summary.sort_values(
        by=sort_columns,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", pd.Series(range(1, len(ranked) + 1), dtype="int64"))
    return ranked


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
) -> None:
    _write_json(output_path / "experiment_metadata.json", metadata)
    ranked_summary.to_csv(output_path / "ranked_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    per_season_summary.to_csv(output_path / "per_season_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    pd.DataFrame(columns=pd.Index(["child_id", "season", "model_id", "feature_pack"])).to_csv(
        output_path / "prediction_metrics.csv",
        index=False,
    )
    pd.DataFrame(columns=pd.Index(["child_id", "season", "model_id", "feature_pack", "decile"])).to_csv(
        output_path / "calibration_deciles.csv",
        index=False,
    )
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
