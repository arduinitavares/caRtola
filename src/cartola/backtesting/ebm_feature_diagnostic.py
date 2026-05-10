from __future__ import annotations

import html
import importlib
import inspect
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd


class EbmDependencyError(RuntimeError):
    """Raised when InterpretML is unavailable or incompatible."""


class EbmDiagnosticInvalid(RuntimeError):
    """Raised when EBM diagnostic source artifacts are invalid."""

    def __init__(self, message: str, *, report: pd.DataFrame | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class SeasonFold:
    fold_id: str
    train_seasons: tuple[int, ...]
    validation_season: int
    inner_validation_mode: str = "disabled_full_outer_train"


@dataclass(frozen=True)
class EbmDiagnosticConfig:
    experiment_path: Path
    seasons: tuple[int, ...]
    model_id: str
    feature_pack: str
    fixture_mode: str


@dataclass(frozen=True)
class EbmDiagnosticResult:
    output_path: Path
    decision: dict[str, object]


@dataclass(frozen=True)
class SourceChildContext:
    source_experiment_id: str
    season: int
    model_id: str
    feature_pack: str
    fixture_mode: str
    matchup_context_mode: str
    footystats_mode: str
    budget_policy: str
    scoring_contract_version: str
    score_column: str
    child_path: Path
    source_prediction_provenance_status: str

    def as_row(self) -> dict[str, object]:
        return {
            "source_experiment_id": self.source_experiment_id,
            "requested_season": self.season,
            "season": self.season,
            "model_id": self.model_id,
            "feature_pack": self.feature_pack,
            "fixture_mode": self.fixture_mode,
            "matchup_context_mode": self.matchup_context_mode,
            "footystats_mode": self.footystats_mode,
            "budget_policy": self.budget_policy,
            "scoring_contract_version": self.scoring_contract_version,
            "primary_score_column": self.score_column,
            "child_path": str(self.child_path),
            "source_prediction_provenance_status": self.source_prediction_provenance_status,
            "discovery_only": True,
            "match_status": "matched",
            "conflicting_child_paths": [],
            "missing_metadata_fields": [],
        }


@dataclass(frozen=True)
class DiagnosticDataset:
    context: SourceChildContext
    valid_rows: pd.DataFrame
    invalid_rows: pd.DataFrame
    raw_feature_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    coach_row_count: int


@dataclass(frozen=True)
class EbmRuntimeInfo:
    available: bool
    version: str | None
    constructor_signature: str
    fit_signature: str
    supports_explicit_validation: bool


@dataclass(frozen=True)
class EBMFitResult:
    model: Any
    predictions: pd.Series
    fit_seconds: float
    fit_row_count: int
    target_type: str
    fold_id: str
    validation_season: int


_PREDICTIVE_METRIC_COLUMNS = (
    "discovery_only",
    "target_type",
    "prediction_type",
    "fold_id",
    "validation_season",
    "shared_evaluation_row_count",
    "mae",
    "rmse",
    "spearman",
    "top50_spearman",
    "calibration_slope",
    "mean_prediction_bias",
)

_SOURCE_CONTEXT_COLUMNS = (
    "discovery_only",
    "source_experiment_id",
    "requested_season",
    "season",
    "model_id",
    "feature_pack",
    "fixture_mode",
    "matchup_context_mode",
    "footystats_mode",
    "budget_policy",
    "scoring_contract_version",
    "primary_score_column",
    "match_status",
    "child_path",
    "conflicting_child_paths",
    "missing_metadata_fields",
    "source_prediction_provenance_status",
)

_FOLD_ASSIGNMENT_COLUMNS = (
    "discovery_only",
    "fold_id",
    "validation_season",
    "train_seasons",
    "inner_validation_mode",
    "train_row_count",
    "validation_row_count",
)

_INVALID_EBM_ROW_COLUMNS = (
    "discovery_only",
    "season",
    "rodada",
    "id_atleta",
    "apelido",
    "posicao",
    "invalid_reason",
    "pontuacao",
    "entrou_em_campo",
)

_INVALID_DIAGNOSTIC_REPORT_COLUMNS = (
    "discovery_only",
    "scope",
    "severity",
    "reason_type",
    "message",
    "artifact_path",
    "season",
    "model_id",
    "feature_pack",
)

_FEATURE_IMPORTANCE_COLUMNS = (
    "discovery_only",
    "target_type",
    "fold_id",
    "validation_season",
    "feature_name",
    "importance_rank",
    "importance_score",
)

_FEATURE_SHAPE_SUMMARY_COLUMNS = (
    "discovery_only",
    "target_type",
    "feature_name",
    "fold_id",
    "validation_season",
    "importance_rank",
    "importance_score",
    "effect_min",
    "effect_max",
    "effect_range",
    "term_support_extraction_status",
    "largest_positive_bin_lower",
    "largest_positive_bin_upper",
    "largest_positive_bin_row_support",
    "largest_positive_bin_round_support",
    "largest_negative_bin_lower",
    "largest_negative_bin_upper",
    "largest_negative_bin_row_support",
    "largest_negative_bin_round_support",
    "monotonicity_hint",
    "row_support",
    "season_support",
    "fold_candidate_signal",
)

_PAIRWISE_INTERACTION_COLUMNS = (
    "discovery_only",
    "target_type",
    "interaction_name",
    "feature_a",
    "feature_b",
    "fold_id",
    "validation_season",
    "importance_rank",
    "importance_score",
    "effect_range",
    "term_support_extraction_status",
    "max_effect_cell_row_support",
    "max_effect_cell_round_support",
    "min_effect_cell_row_support",
    "min_effect_cell_round_support",
    "row_support",
    "season_support",
    "fold_candidate_signal",
)

_CANDIDATE_HYPOTHESIS_COLUMNS = (
    "discovery_only",
    "target_type",
    "candidate_type",
    "term_name",
    "feature_a",
    "feature_b",
    "fold_signal_count",
    "validation_seasons_with_signal",
    "total_row_support",
    "min_bin_or_cell_row_support",
    "min_bin_or_cell_round_support",
    "effect_range_median",
    "direction_summary",
    "failed_validation_seasons",
    "candidate_hypothesis_flag",
    "candidate_scope",
)

_CSV_ARTIFACT_SCHEMAS = {
    "source_context.csv": _SOURCE_CONTEXT_COLUMNS,
    "fold_assignments.csv": _FOLD_ASSIGNMENT_COLUMNS,
    "predictive_metrics.csv": _PREDICTIVE_METRIC_COLUMNS,
    "feature_importance_by_fold.csv": _FEATURE_IMPORTANCE_COLUMNS,
    "feature_shape_summary.csv": _FEATURE_SHAPE_SUMMARY_COLUMNS,
    "pairwise_interactions.csv": _PAIRWISE_INTERACTION_COLUMNS,
    "candidate_hypotheses.csv": _CANDIDATE_HYPOTHESIS_COLUMNS,
    "invalid_ebm_rows.csv": _INVALID_EBM_ROW_COLUMNS,
    "invalid_diagnostic_report.csv": _INVALID_DIAGNOSTIC_REPORT_COLUMNS,
}

_MAIN_EFFECT_AGGREGATION_COLUMNS = (
    "target_type",
    "feature_name",
    "fold_id",
    "validation_season",
    "effect_range",
    "largest_positive_bin_row_support",
    "largest_positive_bin_round_support",
    "largest_negative_bin_row_support",
    "largest_negative_bin_round_support",
    "monotonicity_hint",
    "row_support",
    "fold_candidate_signal",
)

_PAIRWISE_AGGREGATION_COLUMNS = (
    "target_type",
    "interaction_name",
    "feature_a",
    "feature_b",
    "fold_id",
    "validation_season",
    "effect_range",
    "max_effect_cell_row_support",
    "max_effect_cell_round_support",
    "min_effect_cell_row_support",
    "min_effect_cell_round_support",
    "row_support",
    "fold_candidate_signal",
)

_MAIN_EFFECT_SUPPORT_COLUMNS = (
    "row_support",
    "largest_positive_bin_row_support",
    "largest_negative_bin_row_support",
    "largest_positive_bin_round_support",
    "largest_negative_bin_round_support",
)

_PAIRWISE_SUPPORT_COLUMNS = (
    "row_support",
    "max_effect_cell_row_support",
    "min_effect_cell_row_support",
    "max_effect_cell_round_support",
    "min_effect_cell_round_support",
)

_MIN_CANDIDATE_TOTAL_ROW_SUPPORT = 1000
_MIN_CANDIDATE_BIN_OR_CELL_ROW_SUPPORT = 50
_MIN_CANDIDATE_BIN_OR_CELL_ROUND_SUPPORT = 5


def build_season_folds(seasons: tuple[int, ...]) -> tuple[SeasonFold, ...]:
    if len(seasons) != len(set(seasons)):
        duplicate_seasons = tuple(season for season in sorted(set(seasons)) if seasons.count(season) > 1)
        raise EbmDiagnosticInvalid(f"Duplicate seasons are not allowed: {', '.join(map(str, duplicate_seasons))}")

    sorted_seasons = tuple(sorted(seasons))
    if len(sorted_seasons) < 3:
        raise EbmDiagnosticInvalid("At least three seasons are required to build EBM diagnostic folds")

    folds: list[SeasonFold] = []
    for fold_index, validation_index in enumerate(range(2, len(sorted_seasons))):
        folds.append(
            SeasonFold(
                fold_id=_season_fold_id(fold_index),
                train_seasons=sorted_seasons[:validation_index],
                validation_season=sorted_seasons[validation_index],
            )
        )
    return tuple(folds)


def compute_predictive_metrics(rows: pd.DataFrame, *, fold_id: str, validation_season: int) -> pd.DataFrame:
    required_columns = (
        "rodada",
        "target_actual_points",
        "source_model_score",
        "predicted_actual_points",
        "predicted_source_residual",
    )
    missing_columns = tuple(column for column in required_columns if column not in rows.columns)
    if missing_columns:
        raise EbmDiagnosticInvalid(f"Missing required metric columns: {', '.join(missing_columns)}")

    source_model_score = _numeric_series(rows, "source_model_score")
    actual_points = _numeric_series(rows, "target_actual_points")
    predicted_actual_points = _numeric_series(rows, "predicted_actual_points")
    predicted_source_residual = _numeric_series(rows, "predicted_source_residual")
    residual_corrected_score = source_model_score + predicted_source_residual
    shared_valid_mask = (
        actual_points.map(_is_finite_number)
        & source_model_score.map(_is_finite_number)
        & predicted_actual_points.map(_is_finite_number)
        & predicted_source_residual.map(_is_finite_number)
        & residual_corrected_score.map(_is_finite_number)
    )
    shared_rows = rows.loc[shared_valid_mask]
    prediction_specs = (
        ("actual_points", "source_model", source_model_score),
        ("actual_points", "actual_points", predicted_actual_points),
        (
            "source_residual",
            "residual_corrected",
            residual_corrected_score,
        ),
    )

    metric_rows: list[dict[str, object]] = []
    for target_type, prediction_type, predicted in prediction_specs:
        paired_rows = _paired_predictive_metric_rows(shared_rows, predicted.loc[shared_valid_mask])
        errors = paired_rows["predicted"] - paired_rows["actual"]
        metric_rows.append(
            {
                "discovery_only": True,
                "target_type": target_type,
                "prediction_type": prediction_type,
                "fold_id": fold_id,
                "validation_season": validation_season,
                "shared_evaluation_row_count": len(paired_rows),
                "mae": _mean_absolute_error(errors),
                "rmse": _root_mean_squared_error(errors),
                "spearman": _spearman(paired_rows["actual"], paired_rows["predicted"]),
                "top50_spearman": _top50_spearman(paired_rows),
                "calibration_slope": _calibration_slope(paired_rows),
                "mean_prediction_bias": _mean_prediction_bias(errors),
            }
        )

    return pd.DataFrame(metric_rows, columns=pd.Index(_PREDICTIVE_METRIC_COLUMNS))


def inspect_ebm_runtime(
    *, ebm_class: type[Any] | None, package_version: str | None
) -> EbmRuntimeInfo:
    if ebm_class is None:
        raise EbmDependencyError(
            "InterpretML is required for EBM diagnostics. Install the optional diagnostic dependencies."
        )
    constructor_signature = str(inspect.signature(ebm_class))
    fit_inspection = inspect.signature(ebm_class.fit)
    fit_signature = str(fit_inspection)
    fit_parameters = fit_inspection.parameters
    supports_explicit_validation = "X_val" in fit_parameters and "y_val" in fit_parameters
    return EbmRuntimeInfo(
        available=True,
        version=package_version,
        constructor_signature=constructor_signature,
        fit_signature=fit_signature,
        supports_explicit_validation=supports_explicit_validation,
    )


def fit_ebm_fold_target(
    *,
    ebm_class: type[Any],
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    target_type: str,
    fold_id: str,
    validation_season: int,
    random_seed: int,
) -> EBMFitResult:
    constructor_params: dict[str, object] = {
        "interactions": 0,
        "outer_bags": 8,
        "inner_bags": 0,
        "max_rounds": 20000,
        "random_state": random_seed,
        "n_jobs": -1,
        "objective": "rmse",
        "validation_size": 0.0,
        "early_stopping_rounds": 100,
    }
    model = ebm_class(**_filter_constructor_params(ebm_class, constructor_params))
    training_features = train_rows.loc[:, feature_columns]
    training_target = train_rows[target_column]
    validation_features = validation_rows.loc[:, feature_columns]

    started = perf_counter()
    model.fit(training_features, training_target)
    fit_seconds = perf_counter() - started
    raw_predictions = model.predict(validation_features)
    prediction_values = np.asarray(raw_predictions, dtype="float64")
    predictions = pd.Series(prediction_values, index=validation_rows.index)
    return EBMFitResult(
        model=model,
        predictions=predictions,
        fit_seconds=fit_seconds,
        fit_row_count=int(len(train_rows)),
        target_type=target_type,
        fold_id=fold_id,
        validation_season=validation_season,
    )


def assign_continuous_bins(values: pd.Series, *, learned_edges: tuple[float, ...]) -> pd.Series:
    edges = _validated_learned_edges(learned_edges)
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(-1, index=values.index, dtype="int64")
    non_missing_mask = numeric.notna()
    non_missing = numeric.loc[non_missing_mask].to_numpy(dtype=float)
    result.loc[non_missing_mask] = np.searchsorted(
        edges,
        non_missing,
        side="right",
    )
    return result.astype("int64")


def compute_interaction_cell_support(
    frame: pd.DataFrame,
    *,
    feature_a_bin: str,
    feature_b_bin: str,
) -> dict[tuple[int, int], dict[str, int]]:
    required_columns = tuple(dict.fromkeys(("rodada", feature_a_bin, feature_b_bin)))
    missing_columns = tuple(column for column in required_columns if column not in frame.columns)
    if missing_columns:
        raise EbmDiagnosticInvalid(f"Missing required interaction support columns: {', '.join(missing_columns)}")
    _validate_interaction_bin_column(frame, feature_a_bin)
    _validate_interaction_bin_column(frame, feature_b_bin)

    support: dict[tuple[int, int], dict[str, int]] = {}
    grouped = frame.groupby([feature_a_bin, feature_b_bin], sort=True)
    for raw_key, group in grouped:
        bin_a, bin_b = cast("tuple[object, object]", raw_key)
        support[(int(cast("Any", bin_a)), int(cast("Any", bin_b)))] = {
            "row_support": int(len(group)),
            "round_support": int(group["rodada"].nunique(dropna=True)),
        }
    return support


def aggregate_candidate_hypotheses(
    *,
    feature_shape_summary: pd.DataFrame,
    pairwise_interactions: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        *_aggregate_main_effect_hypotheses(feature_shape_summary),
        *_aggregate_pairwise_hypotheses(pairwise_interactions),
    ]
    return pd.DataFrame(rows, columns=pd.Index(_CANDIDATE_HYPOTHESIS_COLUMNS))


def build_ebm_feature_diagnostic(
    *,
    experiment_path: Path,
    output_path: Path,
    seasons: tuple[int, ...],
    model_id: str,
    feature_pack: str,
    fixture_mode: str,
    current_year: int,
    max_interactions: int,
    min_validation_rows: int,
    random_seed: int,
    profile_runtime: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    ebm_class: type[Any] | None = None,
) -> EbmDiagnosticResult:
    started = perf_counter()
    progress_messages: list[str] = []

    def progress(message: str) -> None:
        progress_messages.append(message)
        _emit_progress(progress_callback, message)

    progress("START source validation")
    config = EbmDiagnosticConfig(
        experiment_path=experiment_path,
        seasons=seasons,
        model_id=model_id,
        feature_pack=feature_pack,
        fixture_mode=fixture_mode,
    )
    try:
        contexts, source_report = resolve_source_children(config)
    except EbmDiagnosticInvalid as exc:
        source_context = exc.report if exc.report is not None else pd.DataFrame()
        return _write_build_result(
            output_path=output_path,
            started=started,
            diagnostic_status="invalid",
            diagnostic_phase="source_validation_failed",
            current_year=current_year,
            model_id=model_id,
            feature_pack=feature_pack,
            fixture_mode=fixture_mode,
            experiment_path=experiment_path,
            max_interactions=max_interactions,
            min_validation_rows=min_validation_rows,
            random_seed=random_seed,
            profile_runtime=profile_runtime,
            source_child_count=0,
            ebm_runtime=None,
            runtime_error=None,
            source_context=source_context,
            fold_assignments=pd.DataFrame(),
            predictive_metrics=pd.DataFrame(),
            feature_importance=pd.DataFrame(),
            feature_shape_summary=pd.DataFrame(),
            pairwise_interactions=pd.DataFrame(),
            candidate_hypotheses=pd.DataFrame(),
            invalid_rows=pd.DataFrame(),
            invalid_report=_invalid_report_frame(
                _invalid_report_row(
                    reason_type="source_context",
                    message=str(exc),
                    artifact_path=str(experiment_path / "experiment_metadata.json"),
                    model_id=model_id,
                    feature_pack=feature_pack,
                )
            ),
            candidate_count=0,
            progress_callback=progress_callback,
            progress_messages=progress_messages,
            completion_message=f"COMPLETE invalid EBM diagnostic source validation: output_path={output_path}",
        )

    source_context = pd.DataFrame([context.as_row() for context in contexts]) if contexts else source_report
    resolved_ebm_class, package_version = _resolve_ebm_class(ebm_class)
    try:
        ebm_runtime = inspect_ebm_runtime(ebm_class=resolved_ebm_class, package_version=package_version)
    except EbmDependencyError as exc:
        progress(f"INVALID dependency unavailable: {exc}")
        return _write_build_result(
            output_path=output_path,
            started=started,
            diagnostic_status="invalid",
            diagnostic_phase="dependency_unavailable",
            current_year=current_year,
            model_id=model_id,
            feature_pack=feature_pack,
            fixture_mode=fixture_mode,
            experiment_path=experiment_path,
            max_interactions=max_interactions,
            min_validation_rows=min_validation_rows,
            random_seed=random_seed,
            profile_runtime=profile_runtime,
            source_child_count=len(contexts),
            ebm_runtime=None,
            runtime_error=str(exc),
            source_context=source_context,
            fold_assignments=pd.DataFrame(),
            predictive_metrics=pd.DataFrame(),
            feature_importance=pd.DataFrame(),
            feature_shape_summary=pd.DataFrame(),
            pairwise_interactions=pd.DataFrame(),
            candidate_hypotheses=pd.DataFrame(),
            invalid_rows=pd.DataFrame(),
            invalid_report=_invalid_report_frame(
                _invalid_report_row(
                    reason_type="dependency",
                    message=str(exc),
                    artifact_path="interpret.glassbox.ExplainableBoostingRegressor",
                    model_id=model_id,
                    feature_pack=feature_pack,
                )
            ),
            candidate_count=0,
            progress_callback=progress_callback,
            progress_messages=progress_messages,
            completion_message=f"COMPLETE invalid EBM diagnostic dependency unavailable: output_path={output_path}",
        )
    active_ebm_class = cast("type[Any]", resolved_ebm_class)
    if not _ebm_class_supports_validation_disable(active_ebm_class):
        message = (
            "Selected EBM class does not expose a supported validation disable constructor parameter "
            "or **kwargs for disabled_full_outer_train mode"
        )
        progress(f"INVALID dependency compatibility: {message}")
        return _write_build_result(
            output_path=output_path,
            started=started,
            diagnostic_status="invalid",
            diagnostic_phase="dependency_unavailable",
            current_year=current_year,
            model_id=model_id,
            feature_pack=feature_pack,
            fixture_mode=fixture_mode,
            experiment_path=experiment_path,
            max_interactions=max_interactions,
            min_validation_rows=min_validation_rows,
            random_seed=random_seed,
            profile_runtime=profile_runtime,
            source_child_count=len(contexts),
            ebm_runtime=ebm_runtime,
            runtime_error=message,
            source_context=source_context,
            fold_assignments=pd.DataFrame(),
            predictive_metrics=pd.DataFrame(),
            feature_importance=pd.DataFrame(),
            feature_shape_summary=pd.DataFrame(),
            pairwise_interactions=pd.DataFrame(),
            candidate_hypotheses=pd.DataFrame(),
            invalid_rows=pd.DataFrame(),
            invalid_report=_invalid_report_frame(
                _invalid_report_row(
                    reason_type="dependency",
                    message=message,
                    artifact_path="interpret.glassbox.ExplainableBoostingRegressor",
                    model_id=model_id,
                    feature_pack=feature_pack,
                )
            ),
            candidate_count=0,
            progress_callback=progress_callback,
            progress_messages=progress_messages,
            completion_message=f"COMPLETE invalid EBM diagnostic dependency compatibility: output_path={output_path}",
        )

    datasets: list[DiagnosticDataset] = []
    for context in contexts:
        progress(f"START dataset load: season={context.season} child_path={context.child_path}")
        try:
            metadata = _read_json_object(context.child_path / "run_metadata.json", artifact_name="run_metadata.json")
            feature_columns = _feature_columns_from_metadata(metadata, context=context)
            predictions = _read_player_predictions(context.child_path / "player_predictions.csv")
            datasets.append(
                prepare_diagnostic_dataset(
                    context,
                    predictions,
                    feature_columns=feature_columns,
                )
            )
        except EbmDiagnosticInvalid as exc:
            return _write_build_result(
                output_path=output_path,
                started=started,
                diagnostic_status="invalid",
                diagnostic_phase="full_pipeline",
                current_year=current_year,
                model_id=model_id,
                feature_pack=feature_pack,
                fixture_mode=fixture_mode,
                experiment_path=experiment_path,
                max_interactions=max_interactions,
                min_validation_rows=min_validation_rows,
                random_seed=random_seed,
                profile_runtime=profile_runtime,
                source_child_count=len(contexts),
                ebm_runtime=ebm_runtime,
                runtime_error=None,
                source_context=source_context,
                fold_assignments=pd.DataFrame(),
                predictive_metrics=pd.DataFrame(),
                feature_importance=pd.DataFrame(),
                feature_shape_summary=pd.DataFrame(),
                pairwise_interactions=pd.DataFrame(),
                candidate_hypotheses=pd.DataFrame(),
                invalid_rows=_combined_invalid_rows(datasets),
                invalid_report=_invalid_report_frame(
                    _invalid_report_row(
                        reason_type="schema",
                        message=str(exc),
                        artifact_path=str(context.child_path),
                        season=context.season,
                        model_id=model_id,
                        feature_pack=feature_pack,
                    )
                ),
                candidate_count=0,
                progress_callback=progress_callback,
                progress_messages=progress_messages,
                completion_message=f"COMPLETE invalid EBM diagnostic dataset load: output_path={output_path}",
            )

    invalid_rows = _combined_invalid_rows(datasets)
    invalid_report_rows = [
        _invalid_report_row(
            reason_type="invalid_row_rate",
            message=(
                f"Child season={dataset.context.season} has invalid_row_rate="
                f"{_child_invalid_row_rate(dataset):.6f}, above threshold=0.005000"
            ),
            artifact_path=str(dataset.context.child_path / "player_predictions.csv"),
            season=dataset.context.season,
            model_id=model_id,
            feature_pack=feature_pack,
        )
        for dataset in datasets
        if _child_invalid_row_rate(dataset) > 0.005
    ]
    try:
        _require_consistent_raw_feature_columns(datasets)
    except EbmDiagnosticInvalid as exc:
        invalid_report_rows.append(
            _invalid_report_row(
                reason_type="feature_columns",
                message=str(exc),
                artifact_path=str(experiment_path),
                model_id=model_id,
                feature_pack=feature_pack,
            )
        )
    if invalid_report_rows:
        return _write_build_result(
            output_path=output_path,
            started=started,
            diagnostic_status="invalid",
            diagnostic_phase="full_pipeline",
            current_year=current_year,
            model_id=model_id,
            feature_pack=feature_pack,
            fixture_mode=fixture_mode,
            experiment_path=experiment_path,
            max_interactions=max_interactions,
            min_validation_rows=min_validation_rows,
            random_seed=random_seed,
            profile_runtime=profile_runtime,
            source_child_count=len(contexts),
            ebm_runtime=ebm_runtime,
            runtime_error=None,
            source_context=source_context,
            fold_assignments=pd.DataFrame(),
            predictive_metrics=pd.DataFrame(),
            feature_importance=pd.DataFrame(),
            feature_shape_summary=pd.DataFrame(),
            pairwise_interactions=pd.DataFrame(),
            candidate_hypotheses=pd.DataFrame(),
            invalid_rows=invalid_rows,
            invalid_report=_invalid_report_frame(*invalid_report_rows),
            candidate_count=0,
            progress_callback=progress_callback,
            progress_messages=progress_messages,
            completion_message=f"COMPLETE invalid EBM diagnostic dataset validation: output_path={output_path}",
        )

    combined_rows = _combined_valid_rows(datasets)
    feature_columns = _combined_feature_columns(datasets)
    if combined_rows.empty or not feature_columns:
        return _write_build_result(
            output_path=output_path,
            started=started,
            diagnostic_status="invalid",
            diagnostic_phase="full_pipeline",
            current_year=current_year,
            model_id=model_id,
            feature_pack=feature_pack,
            fixture_mode=fixture_mode,
            experiment_path=experiment_path,
            max_interactions=max_interactions,
            min_validation_rows=min_validation_rows,
            random_seed=random_seed,
            profile_runtime=profile_runtime,
            source_child_count=len(contexts),
            ebm_runtime=ebm_runtime,
            runtime_error=None,
            source_context=source_context,
            fold_assignments=pd.DataFrame(),
            predictive_metrics=pd.DataFrame(),
            feature_importance=pd.DataFrame(),
            feature_shape_summary=pd.DataFrame(),
            pairwise_interactions=pd.DataFrame(),
            candidate_hypotheses=pd.DataFrame(),
            invalid_rows=invalid_rows,
            invalid_report=_invalid_report_frame(
                _invalid_report_row(
                    reason_type="data",
                    message="No valid EBM diagnostic rows or feature columns were available after dataset preparation",
                    artifact_path=str(experiment_path),
                    model_id=model_id,
                    feature_pack=feature_pack,
                )
            ),
            candidate_count=0,
            progress_callback=progress_callback,
            progress_messages=progress_messages,
            completion_message=f"COMPLETE invalid EBM diagnostic empty dataset: output_path={output_path}",
        )

    try:
        folds = build_season_folds(seasons)
    except EbmDiagnosticInvalid as exc:
        return _write_build_result(
            output_path=output_path,
            started=started,
            diagnostic_status="invalid",
            diagnostic_phase="full_pipeline",
            current_year=current_year,
            model_id=model_id,
            feature_pack=feature_pack,
            fixture_mode=fixture_mode,
            experiment_path=experiment_path,
            max_interactions=max_interactions,
            min_validation_rows=min_validation_rows,
            random_seed=random_seed,
            profile_runtime=profile_runtime,
            source_child_count=len(contexts),
            ebm_runtime=ebm_runtime,
            runtime_error=None,
            source_context=source_context,
            fold_assignments=pd.DataFrame(),
            predictive_metrics=pd.DataFrame(),
            feature_importance=pd.DataFrame(),
            feature_shape_summary=pd.DataFrame(),
            pairwise_interactions=pd.DataFrame(),
            candidate_hypotheses=pd.DataFrame(),
            invalid_rows=invalid_rows,
            invalid_report=_invalid_report_frame(
                _invalid_report_row(
                    reason_type="folds",
                    message=str(exc),
                    artifact_path=str(experiment_path),
                    model_id=model_id,
                    feature_pack=feature_pack,
                )
            ),
            candidate_count=0,
            progress_callback=progress_callback,
            progress_messages=progress_messages,
            completion_message=f"COMPLETE invalid EBM diagnostic folds unavailable: output_path={output_path}",
        )

    fold_rows: list[dict[str, object]] = []
    metric_frames: list[pd.DataFrame] = []
    for fold in folds:
        train_rows = combined_rows.loc[combined_rows["season"].isin(fold.train_seasons)].copy()
        validation_rows = combined_rows.loc[combined_rows["season"].eq(fold.validation_season)].copy()
        fold_rows.append(
            {
                "discovery_only": True,
                "fold_id": fold.fold_id,
                "validation_season": fold.validation_season,
                "train_seasons": ",".join(str(value) for value in fold.train_seasons),
                "inner_validation_mode": fold.inner_validation_mode,
                "train_row_count": int(len(train_rows)),
                "validation_row_count": int(len(validation_rows)),
            }
        )
        if len(train_rows) < min_validation_rows or len(validation_rows) < min_validation_rows:
            progress(
                f"SKIP fold={fold.fold_id} pass=main_effect train_rows={len(train_rows)} "
                f"validation_rows={len(validation_rows)}"
            )
            invalid_report_rows.append(
                _invalid_report_row(
                    reason_type="insufficient_rows",
                    message=(
                        f"Fold {fold.fold_id} has train_rows={len(train_rows)} and "
                        f"validation_rows={len(validation_rows)}, below min_validation_rows={min_validation_rows}"
                    ),
                    artifact_path=str(experiment_path),
                    season=fold.validation_season,
                    model_id=model_id,
                    feature_pack=feature_pack,
                )
            )
            continue

        for target_type, target_column, prediction_column in (
            ("actual_points", "target_actual_points", "predicted_actual_points"),
            ("source_residual", "target_source_residual", "predicted_source_residual"),
        ):
            progress(f"START fold={fold.fold_id} target={target_type} pass=main_effect")
            fit_result = fit_ebm_fold_target(
                ebm_class=active_ebm_class,
                train_rows=train_rows,
                validation_rows=validation_rows,
                feature_columns=feature_columns,
                target_column=target_column,
                target_type=target_type,
                fold_id=fold.fold_id,
                validation_season=fold.validation_season,
                random_seed=random_seed,
            )
            validation_rows.loc[:, prediction_column] = fit_result.predictions
        metric_frames.append(
            compute_predictive_metrics(
                validation_rows,
                fold_id=fold.fold_id,
                validation_season=fold.validation_season,
            )
        )

    fold_assignments = pd.DataFrame(fold_rows, columns=pd.Index(_FOLD_ASSIGNMENT_COLUMNS))
    predictive_metrics = (
        pd.concat(metric_frames, ignore_index=True)
        if metric_frames
        else pd.DataFrame(columns=pd.Index(_PREDICTIVE_METRIC_COLUMNS))
    )
    feature_importance = pd.DataFrame(columns=pd.Index(_FEATURE_IMPORTANCE_COLUMNS))
    feature_shape_summary = pd.DataFrame(columns=pd.Index(_FEATURE_SHAPE_SUMMARY_COLUMNS))
    pairwise_interactions = pd.DataFrame(columns=pd.Index(_PAIRWISE_INTERACTION_COLUMNS))
    candidate_hypotheses = aggregate_candidate_hypotheses(
        feature_shape_summary=feature_shape_summary,
        pairwise_interactions=pairwise_interactions,
    )
    diagnostic_status = "diagnostic_complete" if not invalid_report_rows and not predictive_metrics.empty else "invalid"
    if predictive_metrics.empty and not invalid_report_rows:
        invalid_report_rows.append(
            _invalid_report_row(
                reason_type="metrics",
                message="No predictive metrics were produced by the EBM diagnostic pipeline",
                artifact_path=str(experiment_path),
                model_id=model_id,
                feature_pack=feature_pack,
            )
        )

    return _write_build_result(
        output_path=output_path,
        started=started,
        diagnostic_status=diagnostic_status,
        diagnostic_phase="full_pipeline",
        current_year=current_year,
        model_id=model_id,
        feature_pack=feature_pack,
        fixture_mode=fixture_mode,
        experiment_path=experiment_path,
        max_interactions=max_interactions,
        min_validation_rows=min_validation_rows,
        random_seed=random_seed,
        profile_runtime=profile_runtime,
        source_child_count=len(contexts),
        ebm_runtime=ebm_runtime,
        runtime_error=None,
        source_context=source_context,
        fold_assignments=fold_assignments,
        predictive_metrics=predictive_metrics,
        feature_importance=feature_importance,
        feature_shape_summary=feature_shape_summary,
        pairwise_interactions=pairwise_interactions,
        candidate_hypotheses=candidate_hypotheses,
        invalid_rows=invalid_rows,
        invalid_report=_invalid_report_frame(*invalid_report_rows),
        candidate_count=len(candidate_hypotheses),
        progress_callback=progress_callback,
        progress_messages=progress_messages,
        completion_message=f"COMPLETE EBM diagnostic full pipeline: output_path={output_path}",
    )


def _write_build_result(
    *,
    output_path: Path,
    started: float,
    diagnostic_status: str,
    diagnostic_phase: str,
    current_year: int,
    model_id: str,
    feature_pack: str,
    fixture_mode: str,
    experiment_path: Path,
    max_interactions: int,
    min_validation_rows: int,
    random_seed: int,
    profile_runtime: bool,
    source_child_count: int,
    ebm_runtime: EbmRuntimeInfo | None,
    runtime_error: str | None,
    source_context: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    predictive_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    feature_shape_summary: pd.DataFrame,
    pairwise_interactions: pd.DataFrame,
    candidate_hypotheses: pd.DataFrame,
    invalid_rows: pd.DataFrame,
    invalid_report: pd.DataFrame,
    candidate_count: int,
    progress_callback: Callable[[str], None] | None,
    progress_messages: list[str],
    completion_message: str,
) -> EbmDiagnosticResult:
    write_message = f"START artifact write: output_path={output_path}"
    progress_messages.append(write_message)
    _emit_progress(progress_callback, write_message)
    manifest_progress_messages = [*progress_messages, completion_message]
    decision: dict[str, object] = {
        "discovery_only": True,
        "diagnostic_status": diagnostic_status,
        "diagnostic_phase": diagnostic_phase,
        "inner_validation_mode": "disabled_full_outer_train",
        "position_handling": "one_hot",
        "candidate_count": candidate_count,
        "source_experiment_path": str(experiment_path),
        "output_path": str(output_path),
    }
    manifest: dict[str, object] = {
        "discovery_only": True,
        "diagnostic_status": diagnostic_status,
        "diagnostic_phase": diagnostic_phase,
        "current_year": current_year,
        "model_id": model_id,
        "feature_pack": feature_pack,
        "fixture_mode": fixture_mode,
        "position_handling": "one_hot",
        "inner_validation_mode": "disabled_full_outer_train",
        "total_wall_clock_seconds": perf_counter() - started,
        "source_child_count": source_child_count,
        "max_interactions": max_interactions,
        "min_validation_rows": min_validation_rows,
        "random_seed": random_seed,
        "profile_runtime": profile_runtime,
        "progress_logging": {
            "enabled": progress_callback is not None,
            "event_count": len(manifest_progress_messages),
            "messages": manifest_progress_messages,
        },
        "ebm_runtime": _ebm_runtime_manifest(ebm_runtime, runtime_error=runtime_error),
    }
    write_ebm_diagnostic_artifacts(
        output_path=output_path,
        manifest=manifest,
        source_context=source_context,
        fold_assignments=fold_assignments,
        predictive_metrics=predictive_metrics,
        feature_importance=feature_importance,
        feature_shape_summary=feature_shape_summary,
        pairwise_interactions=pairwise_interactions,
        candidate_hypotheses=candidate_hypotheses,
        invalid_rows=invalid_rows,
        invalid_report=invalid_report,
        decision=decision,
    )
    progress_messages.append(completion_message)
    _emit_progress(progress_callback, completion_message)
    return EbmDiagnosticResult(output_path=output_path, decision=decision)


def _resolve_ebm_class(ebm_class: type[Any] | None) -> tuple[type[Any] | None, str | None]:
    if ebm_class is not None:
        return ebm_class, "injected"
    return _load_default_ebm_class()


def _load_default_ebm_class() -> tuple[type[Any] | None, str | None]:
    try:
        interpret_module = importlib.import_module("interpret")
        glassbox_module = importlib.import_module("interpret.glassbox")
    except ImportError:
        return None, None
    loaded_ebm_class = getattr(glassbox_module, "ExplainableBoostingRegressor", None)
    if not isinstance(loaded_ebm_class, type):
        return None, _module_version(interpret_module)
    return loaded_ebm_class, _module_version(interpret_module)


def _module_version(module: object) -> str | None:
    version = getattr(module, "__version__", None)
    if version is None:
        return None
    return str(version)


def _ebm_class_supports_validation_disable(ebm_class: type[Any]) -> bool:
    constructor_parameters = inspect.signature(ebm_class).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in constructor_parameters.values()):
        return True
    return any(parameter in constructor_parameters for parameter in ("validation_size", "validation_fraction"))


def _ebm_runtime_manifest(
    runtime_info: EbmRuntimeInfo | None,
    *,
    runtime_error: str | None,
) -> dict[str, object]:
    if runtime_info is None:
        return {
            "available": False,
            "version": None,
            "constructor_signature": "",
            "fit_signature": "",
            "supports_explicit_validation": False,
            "error": runtime_error or "",
        }
    return {
        "available": runtime_info.available,
        "version": runtime_info.version,
        "constructor_signature": runtime_info.constructor_signature,
        "fit_signature": runtime_info.fit_signature,
        "supports_explicit_validation": runtime_info.supports_explicit_validation,
        "error": runtime_error or "",
    }


def _feature_columns_from_metadata(
    metadata: dict[str, Any],
    *,
    context: SourceChildContext,
) -> tuple[str, ...]:
    raw_feature_columns = metadata.get("feature_columns")
    if raw_feature_columns:
        return _deduplicate_feature_columns(
            _metadata_feature_column_list(
                raw_feature_columns,
                field_name="feature_columns",
                context=context,
            )
        )

    split_columns = [
        *_metadata_feature_column_list(
            metadata.get("footystats_feature_columns"),
            field_name="footystats_feature_columns",
            context=context,
        ),
        *_metadata_feature_column_list(
            metadata.get("matchup_context_feature_columns"),
            field_name="matchup_context_feature_columns",
            context=context,
        ),
    ]
    if split_columns:
        return _deduplicate_feature_columns(tuple(split_columns))
    raise EbmDiagnosticInvalid(
        f"run_metadata.json missing usable feature columns for season={context.season}: "
        f"{context.child_path / 'run_metadata.json'}"
    )


def _metadata_feature_column_list(
    value: object,
    *,
    field_name: str,
    context: SourceChildContext,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise EbmDiagnosticInvalid(
            f"run_metadata.json field {field_name} must be a list for season={context.season}: "
            f"{context.child_path / 'run_metadata.json'}"
        )
    columns: list[str] = []
    for index, column in enumerate(value):
        if not isinstance(column, str) or not column.strip():
            raise EbmDiagnosticInvalid(
                f"run_metadata.json field {field_name}[{index}] must be a non-empty string "
                f"for season={context.season}: {context.child_path / 'run_metadata.json'}"
            )
        columns.append(column)
    return tuple(columns)


def _deduplicate_feature_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    deduplicated: list[str] = []
    for column in columns:
        if column not in deduplicated:
            deduplicated.append(column)
    return tuple(deduplicated)


def _read_player_predictions(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise EbmDiagnosticInvalid(f"Unable to read source child player_predictions.csv: {path}") from exc


def _combined_feature_columns(datasets: Sequence[DiagnosticDataset]) -> tuple[str, ...]:
    feature_columns: list[str] = []
    for dataset in datasets:
        for column in dataset.feature_columns:
            if column not in feature_columns:
                feature_columns.append(column)
    return tuple(feature_columns)


def _combined_valid_rows(datasets: Sequence[DiagnosticDataset]) -> pd.DataFrame:
    feature_columns = _combined_feature_columns(datasets)
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        rows = dataset.valid_rows.copy()
        for column in feature_columns:
            if column not in rows.columns:
                if not column.startswith("posicao_"):
                    raise EbmDiagnosticInvalid(
                        f"Feature column {column} missing from prepared child rows for season={dataset.context.season}"
                    )
                rows[column] = 0.0
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _combined_invalid_rows(datasets: Sequence[DiagnosticDataset]) -> pd.DataFrame:
    frames = [dataset.invalid_rows for dataset in datasets if not dataset.invalid_rows.empty]
    if not frames:
        return pd.DataFrame(columns=pd.Index(_INVALID_EBM_ROW_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _invalid_report_row(
    *,
    reason_type: str,
    message: str,
    artifact_path: str,
    severity: str = "error",
    scope: str = "diagnostic",
    season: int | str = "",
    model_id: str = "",
    feature_pack: str = "",
) -> dict[str, object]:
    return {
        "discovery_only": True,
        "scope": scope,
        "severity": severity,
        "reason_type": reason_type,
        "message": message,
        "artifact_path": artifact_path,
        "season": season,
        "model_id": model_id,
        "feature_pack": feature_pack,
    }


def _invalid_report_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=pd.Index(_INVALID_DIAGNOSTIC_REPORT_COLUMNS))


def _child_invalid_row_rate(dataset: DiagnosticDataset) -> float:
    loaded_rows = len(dataset.valid_rows) + len(dataset.invalid_rows)
    if loaded_rows == 0:
        return 0.0
    return float(len(dataset.invalid_rows) / loaded_rows)


def _require_consistent_raw_feature_columns(datasets: Sequence[DiagnosticDataset]) -> None:
    if not datasets:
        return
    expected_columns = datasets[0].raw_feature_columns
    mismatches = [
        f"season={dataset.context.season} columns={','.join(dataset.raw_feature_columns)}"
        for dataset in datasets[1:]
        if dataset.raw_feature_columns != expected_columns
    ]
    if not mismatches:
        return
    raise EbmDiagnosticInvalid(
        "Source child raw feature columns must be identical across seasons. "
        f"expected={','.join(expected_columns)} mismatches={'; '.join(mismatches)}"
    )


def write_ebm_diagnostic_artifacts(
    *,
    output_path: Path,
    manifest: dict[str, object],
    source_context: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    predictive_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    feature_shape_summary: pd.DataFrame,
    pairwise_interactions: pd.DataFrame,
    candidate_hypotheses: pd.DataFrame,
    invalid_rows: pd.DataFrame,
    invalid_report: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_payload = {**manifest, "discovery_only": True}
    decision_payload = {**decision, "discovery_only": True}

    (output_path / "ebm_diagnostic_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "ebm_diagnostic_decision.json").write_text(
        json.dumps(decision_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    csv_artifacts = {
        "source_context.csv": source_context,
        "fold_assignments.csv": fold_assignments,
        "predictive_metrics.csv": predictive_metrics,
        "feature_importance_by_fold.csv": feature_importance,
        "feature_shape_summary.csv": feature_shape_summary,
        "pairwise_interactions.csv": pairwise_interactions,
        "candidate_hypotheses.csv": candidate_hypotheses,
        "invalid_ebm_rows.csv": invalid_rows,
        "invalid_diagnostic_report.csv": invalid_report,
    }
    for artifact_name, frame in csv_artifacts.items():
        _write_csv(output_path / artifact_name, frame, columns=_CSV_ARTIFACT_SCHEMAS[artifact_name])
    (output_path / "ebm_feature_diagnostic.html").write_text(
        _html_report(decision=decision_payload, manifest=manifest_payload),
        encoding="utf-8",
    )


def resolve_source_children(config: EbmDiagnosticConfig) -> tuple[tuple[SourceChildContext, ...], pd.DataFrame]:
    metadata_path = config.experiment_path / "experiment_metadata.json"
    ranked_summary_path = config.experiment_path / "ranked_summary.csv"
    if not ranked_summary_path.is_file():
        raise EbmDiagnosticInvalid(f"Missing source experiment ranked_summary.csv: {ranked_summary_path}")
    parent_metadata = _read_json_object(metadata_path, artifact_name="experiment_metadata.json")
    source_experiment_id = _required_str(
        parent_metadata,
        "experiment_id",
        artifact_name="experiment_metadata.json",
        field_path="experiment_id",
    )
    child_runs_value = _required_field(
        parent_metadata,
        "child_runs",
        artifact_name="experiment_metadata.json",
        field_path="child_runs",
    )
    if not isinstance(child_runs_value, list):
        raise EbmDiagnosticInvalid("experiment_metadata.json field child_runs must be a list")

    child_runs = _child_run_entries(child_runs_value)
    contexts: list[SourceChildContext] = []
    report_rows: list[dict[str, object]] = []
    for season in config.seasons:
        matches = [
            (index, child)
            for index, child in enumerate(child_runs)
            if _child_matches_config(child, child_index=index, config=config, season=season)
        ]
        if not matches:
            report_rows.append(_unmatched_row(source_experiment_id, config=config, season=season))
            continue
        if len(matches) > 1:
            report_rows.append(
                _duplicate_row(
                    source_experiment_id,
                    config=config,
                    season=season,
                    matches=matches,
                )
            )
            continue
        index, child = matches[0]
        contexts.append(
            _source_child_context(
                source_experiment_id,
                child,
                child_index=index,
                config=config,
                season=season,
            )
        )

    if report_rows:
        report = pd.DataFrame(report_rows)
        if any(row["match_status"] == "duplicate" for row in report_rows):
            raise EbmDiagnosticInvalid("Duplicate source child matches", report=report)
        raise EbmDiagnosticInvalid("Missing source child matches", report=report)
    return tuple(contexts), pd.DataFrame()


def prepare_diagnostic_dataset(
    context: SourceChildContext,
    predictions: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> DiagnosticDataset:
    required_columns = (
        "rodada",
        "id_atleta",
        "apelido",
        "id_clube",
        "posicao",
        "status",
        "pontuacao",
        "entrou_em_campo",
        "preco_pre_rodada",
        context.score_column,
    )
    missing_columns = tuple(column for column in required_columns if column not in predictions.columns)
    if missing_columns:
        raise EbmDiagnosticInvalid(f"Missing required prediction columns: {', '.join(missing_columns)}")

    prepared = predictions.copy()
    if "season" not in prepared.columns:
        prepared["season"] = context.season
    else:
        prepared["season"] = prepared["season"].fillna(context.season)
    prepared["source_model_score"] = pd.to_numeric(prepared[context.score_column], errors="coerce")
    prepared["target_actual_points"] = pd.NA
    prepared["invalid_reason"] = ""

    coach_mask = prepared["posicao"] == "tec"
    coach_row_count = int(coach_mask.sum())
    player_rows = prepared.loc[~coach_mask].copy()

    actual_points = pd.to_numeric(player_rows["pontuacao"], errors="coerce")
    source_scores = pd.to_numeric(player_rows["source_model_score"], errors="coerce")
    target_values: list[float | None] = []
    invalid_reasons: list[str] = []
    for index, row in player_rows.iterrows():
        row_actual_points = actual_points.loc[index]
        source_score = source_scores.loc[index]
        reasons: list[str] = []
        if not _is_finite_number(source_score):
            reasons.append("invalid_source_model_score")

        raw_points = row["pontuacao"]
        if _is_finite_number(row_actual_points):
            target_values.append(float(row_actual_points))
        elif pd.isna(raw_points) and _entered_field_is_false(row["entrou_em_campo"]):
            target_values.append(0.0)
        else:
            target_values.append(None)
            if pd.isna(raw_points):
                reasons.append("missing_actual_points_for_entered_player")
            else:
                reasons.append("invalid_actual_points")
        invalid_reasons.append(";".join(reasons))

    player_rows["source_model_score"] = source_scores
    player_rows["target_actual_points"] = target_values
    player_rows["invalid_reason"] = invalid_reasons
    player_rows["target_source_residual"] = (
        pd.to_numeric(player_rows["target_actual_points"], errors="coerce") - player_rows["source_model_score"]
    )
    invalid_rows = player_rows.loc[player_rows["invalid_reason"] != ""].reset_index(drop=True)
    valid_rows = player_rows.loc[player_rows["invalid_reason"] == ""].copy()
    valid_rows["target_actual_points"] = pd.to_numeric(valid_rows["target_actual_points"], errors="raise")
    valid_rows["source_model_score"] = pd.to_numeric(valid_rows["source_model_score"], errors="raise")

    resolved_feature_columns = _prepare_diagnostic_features(valid_rows, feature_columns)
    return DiagnosticDataset(
        context=context,
        valid_rows=valid_rows.reset_index(drop=True),
        invalid_rows=invalid_rows,
        raw_feature_columns=feature_columns,
        feature_columns=resolved_feature_columns,
        coach_row_count=coach_row_count,
    )


def _prepare_diagnostic_features(
    valid_rows: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[str, ...]:
    identity_columns = frozenset(("id_atleta", "id_clube", "apelido", "season", "rodada"))
    excluded_columns = identity_columns | frozenset(
        column for column in feature_columns if _is_excluded_feature_column(column)
    )
    missing_feature_columns = tuple(
        column for column in feature_columns if column not in excluded_columns and column not in valid_rows.columns
    )
    if missing_feature_columns:
        raise EbmDiagnosticInvalid(f"Missing feature columns: {', '.join(missing_feature_columns)}")

    resolved_columns: list[str] = []
    for column in feature_columns:
        if column in excluded_columns:
            continue
        if column == "posicao":
            for position in sorted(valid_rows["posicao"].dropna().astype(str).unique()):
                dummy_column = f"posicao_{position}"
                valid_rows[dummy_column] = (valid_rows["posicao"].astype(str) == position).astype(float)
                resolved_columns.append(dummy_column)
            continue

        numeric_feature = pd.to_numeric(valid_rows[column], errors="coerce")
        if not numeric_feature.map(_is_finite_number).all():
            raise EbmDiagnosticInvalid(f"Feature column {column} must be numeric and finite for valid rows")
        valid_rows[column] = numeric_feature
        resolved_columns.append(column)
    return tuple(resolved_columns)


def _is_excluded_feature_column(column: str) -> bool:
    normalized = column.strip().lower()
    if normalized in {
        "pontuacao",
        "entrou_em_campo",
        "target_actual_points",
        "target_source_residual",
        "source_model_score",
        "actual_points",
        "actual_points_with_captain",
        "actual_points_with_capitao",
    }:
        return True
    if normalized.endswith("_score"):
        return True
    if normalized.startswith("predicted_") or normalized.endswith("_prediction"):
        return True
    if "capitao" in normalized or "captain" in normalized:
        return True
    if normalized.startswith("scout_") or normalized.endswith("_scout"):
        return True
    if normalized.startswith("pontuacao"):
        return True
    return False


def _aggregate_main_effect_hypotheses(feature_shape_summary: pd.DataFrame) -> list[dict[str, object]]:
    if feature_shape_summary.empty:
        return []
    _require_dataframe_columns(
        feature_shape_summary,
        columns=_MAIN_EFFECT_AGGREGATION_COLUMNS,
        artifact_name="feature_shape_summary",
    )

    rows: list[dict[str, object]] = []
    grouped = feature_shape_summary.groupby(["target_type", "feature_name"], sort=True, dropna=False)
    for raw_key, group in grouped:
        target_type, feature_name = cast("tuple[object, object]", raw_key)
        term_name = str(feature_name)
        _validate_unique_fold_rows(group, term_name=term_name)
        signal_mask = group["fold_candidate_signal"].eq(True)
        signals = group.loc[signal_mask]
        if signals.empty:
            continue

        _validate_numeric_signal_support(signals, columns=_MAIN_EFFECT_SUPPORT_COLUMNS, term_name=term_name)
        direction_values = _normalized_direction_values(signals["monotonicity_hint"])
        total_row_support = _sum_numeric_int(signals, "row_support")
        min_bin_or_cell_row_support = _min_numeric_int(
            signals,
            (
                "largest_positive_bin_row_support",
                "largest_negative_bin_row_support",
            ),
        )
        min_bin_or_cell_round_support = _min_numeric_int(
            signals,
            (
                "largest_positive_bin_round_support",
                "largest_negative_bin_round_support",
            ),
        )
        fold_signal_count = _fold_signal_count(signals)
        validation_season_signal_count = _validation_season_signal_count(signals)
        candidate_hypothesis_flag = (
            str(target_type) == "source_residual"
            and fold_signal_count >= 2
            and validation_season_signal_count >= 2
            and total_row_support >= _MIN_CANDIDATE_TOTAL_ROW_SUPPORT
            and min_bin_or_cell_row_support >= _MIN_CANDIDATE_BIN_OR_CELL_ROW_SUPPORT
            and min_bin_or_cell_round_support >= _MIN_CANDIDATE_BIN_OR_CELL_ROUND_SUPPORT
            and _directions_compatible(direction_values)
        )
        rows.append(
            {
                "discovery_only": True,
                "target_type": str(target_type),
                "candidate_type": "main_effect",
                "term_name": term_name,
                "feature_a": term_name,
                "feature_b": "",
                "fold_signal_count": fold_signal_count,
                "validation_seasons_with_signal": _joined_sorted_values(signals["validation_season"]),
                "total_row_support": total_row_support,
                "min_bin_or_cell_row_support": min_bin_or_cell_row_support,
                "min_bin_or_cell_round_support": min_bin_or_cell_round_support,
                "effect_range_median": _median_numeric_float(signals, "effect_range"),
                "direction_summary": ",".join(sorted(direction_values)),
                "failed_validation_seasons": _joined_sorted_values(group.loc[~signal_mask, "validation_season"]),
                "candidate_hypothesis_flag": bool(candidate_hypothesis_flag),
                "candidate_scope": "human_review_only",
            }
        )
    return rows


def _aggregate_pairwise_hypotheses(pairwise_interactions: pd.DataFrame) -> list[dict[str, object]]:
    if pairwise_interactions.empty:
        return []
    _require_dataframe_columns(
        pairwise_interactions,
        columns=_PAIRWISE_AGGREGATION_COLUMNS,
        artifact_name="pairwise_interactions",
    )

    rows: list[dict[str, object]] = []
    grouped = pairwise_interactions.groupby(
        ["target_type", "interaction_name", "feature_a", "feature_b"],
        sort=True,
        dropna=False,
    )
    for raw_key, group in grouped:
        target_type, interaction_name, feature_a, feature_b = cast("tuple[object, object, object, object]", raw_key)
        term_name = str(interaction_name)
        _validate_unique_fold_rows(group, term_name=term_name)
        signal_mask = group["fold_candidate_signal"].eq(True)
        signals = group.loc[signal_mask]
        if signals.empty:
            continue

        _validate_numeric_signal_support(signals, columns=_PAIRWISE_SUPPORT_COLUMNS, term_name=term_name)
        total_row_support = _sum_numeric_int(signals, "row_support")
        min_bin_or_cell_row_support = _min_numeric_int(
            signals,
            (
                "max_effect_cell_row_support",
                "min_effect_cell_row_support",
            ),
        )
        min_bin_or_cell_round_support = _min_numeric_int(
            signals,
            (
                "max_effect_cell_round_support",
                "min_effect_cell_round_support",
            ),
        )
        fold_signal_count = _fold_signal_count(signals)
        validation_season_signal_count = _validation_season_signal_count(signals)
        candidate_hypothesis_flag = (
            str(target_type) == "source_residual"
            and fold_signal_count >= 2
            and validation_season_signal_count >= 2
            and total_row_support >= _MIN_CANDIDATE_TOTAL_ROW_SUPPORT
            and min_bin_or_cell_row_support >= _MIN_CANDIDATE_BIN_OR_CELL_ROW_SUPPORT
            and min_bin_or_cell_round_support >= _MIN_CANDIDATE_BIN_OR_CELL_ROUND_SUPPORT
        )
        rows.append(
            {
                "discovery_only": True,
                "target_type": str(target_type),
                "candidate_type": "interaction",
                "term_name": term_name,
                "feature_a": str(feature_a),
                "feature_b": str(feature_b),
                "fold_signal_count": fold_signal_count,
                "validation_seasons_with_signal": _joined_sorted_values(signals["validation_season"]),
                "total_row_support": total_row_support,
                "min_bin_or_cell_row_support": min_bin_or_cell_row_support,
                "min_bin_or_cell_round_support": min_bin_or_cell_round_support,
                "effect_range_median": _median_numeric_float(signals, "effect_range"),
                "direction_summary": "interaction_mixed",
                "failed_validation_seasons": _joined_sorted_values(group.loc[~signal_mask, "validation_season"]),
                "candidate_hypothesis_flag": bool(candidate_hypothesis_flag),
                "candidate_scope": "human_review_only",
            }
        )
    return rows


def _require_dataframe_columns(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    artifact_name: str,
) -> None:
    missing_columns = tuple(column for column in columns if column not in frame.columns)
    if missing_columns:
        raise EbmDiagnosticInvalid(f"Missing required {artifact_name} columns: {', '.join(missing_columns)}")


def _write_csv(path: Path, frame: pd.DataFrame, *, columns: tuple[str, ...]) -> None:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    output["discovery_only"] = True
    extra_columns = [column for column in output.columns if column not in columns]
    output = output.loc[:, [*columns, *extra_columns]]
    output.to_csv(path, index=False)


def _html_report(*, decision: dict[str, object], manifest: dict[str, object]) -> str:
    decision_status = str(decision.get("diagnostic_status", "unknown"))
    decision_json = html.escape(json.dumps(decision, indent=2, sort_keys=True))
    manifest_json = html.escape(json.dumps(manifest, indent=2, sort_keys=True))
    escaped_status = html.escape(decision_status)
    return (
        "<!doctype html>"
        "<html>"
        "<head><meta charset='utf-8'><title>EBM Feature Diagnostic</title></head>"
        "<body>"
        "<h1>EBM Feature Diagnostic</h1>"
        "<p><strong>discovery_only=true</strong></p>"
        f"<p><strong>diagnostic_status={escaped_status}</strong></p>"
        f"<h2>Decision</h2><pre>{decision_json}</pre>"
        f"<h2>Manifest</h2><pre>{manifest_json}</pre>"
        "</body>"
        "</html>"
    )


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _validate_unique_fold_rows(group: pd.DataFrame, *, term_name: str) -> None:
    duplicated_mask = group.duplicated(subset=["fold_id", "validation_season"], keep=False)
    if not bool(duplicated_mask.any()):
        return

    duplicated_pairs = (
        group.loc[duplicated_mask, ["fold_id", "validation_season"]]
        .drop_duplicates()
        .sort_values(["fold_id", "validation_season"], key=lambda values: values.astype(str))
    )
    pair_summaries = [
        f"fold_id={row['fold_id']} validation_season={row['validation_season']}"
        for _, row in duplicated_pairs.iterrows()
    ]
    pair_summary = "; ".join(pair_summaries)
    raise EbmDiagnosticInvalid(
        f"Duplicate fold rows for {term_name}: {pair_summary}. "
        f"Duplicate signal rows for {term_name}: {pair_summary}"
    )


def _validate_numeric_signal_support(
    signals: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    term_name: str,
) -> None:
    for column in columns:
        numeric = pd.to_numeric(signals[column], errors="coerce")
        invalid_mask = ~numeric.map(_is_finite_number)
        if bool(invalid_mask.any()):
            invalid_rows = signals.loc[invalid_mask, ["fold_id", "validation_season"]]
            invalid_context = _joined_fold_validation_pairs(invalid_rows)
            raise EbmDiagnosticInvalid(
                f"Invalid numeric support for {term_name}: {column} contains missing, non-numeric, "
                f"or non-finite values at {invalid_context}"
            )


def _fold_signal_count(signals: pd.DataFrame) -> int:
    return int(signals["fold_id"].nunique(dropna=True))


def _validation_season_signal_count(signals: pd.DataFrame) -> int:
    return int(signals["validation_season"].nunique(dropna=True))


def _joined_fold_validation_pairs(rows: pd.DataFrame) -> str:
    pair_rows = rows.drop_duplicates().sort_values(
        ["fold_id", "validation_season"],
        key=lambda values: values.astype(str),
    )
    return "; ".join(
        f"fold_id={row['fold_id']} validation_season={row['validation_season']}" for _, row in pair_rows.iterrows()
    )


def _sum_numeric_int(frame: pd.DataFrame, column: str) -> int:
    values = _finite_numeric_values(frame[column])
    if values.empty:
        return 0
    return int(values.sum())


def _min_numeric_int(frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
    values = pd.concat([_finite_numeric_values(frame[column]) for column in columns], ignore_index=True)
    if values.empty:
        return 0
    return int(values.min())


def _median_numeric_float(frame: pd.DataFrame, column: str) -> float:
    values = _finite_numeric_values(frame[column])
    if values.empty:
        return float("nan")
    return float(values.median())


def _finite_numeric_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.loc[numeric.map(_is_finite_number)]


def _joined_sorted_values(values: pd.Series) -> str:
    normalized = {_normalized_sortable_value(value) for value in values.dropna()}
    sorted_values = sorted(normalized, key=_sortable_value_key)
    return ",".join(str(value) for value in sorted_values)


def _normalized_sortable_value(value: object) -> int | str:
    try:
        numeric_value = float(cast("Any", value))
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(numeric_value) and numeric_value.is_integer():
        return int(numeric_value)
    if math.isfinite(numeric_value):
        return str(numeric_value)
    return str(value)


def _sortable_value_key(value: int | str) -> tuple[int, int | str]:
    if isinstance(value, int):
        return (0, value)
    return (1, value)


def _normalized_direction_values(values: pd.Series) -> set[str]:
    directions: set[str] = set()
    for value in values.dropna():
        direction = str(value).strip().lower()
        if direction and direction != "nan":
            directions.add(direction)
    return directions


def _directions_compatible(directions: set[str]) -> bool:
    if not directions:
        return False
    if directions & {"mixed", "unstable"}:
        return False
    monotone_directions = {"increasing", "decreasing"}
    shaped_directions = {"u_shaped", "inverted_u"}
    if directions & monotone_directions and directions & shaped_directions:
        return False
    contradictory_direction_sets = (
        monotone_directions,
        shaped_directions,
    )
    return not any(contradictory_directions.issubset(directions) for contradictory_directions in contradictory_direction_sets)


def _season_fold_id(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"fold_{index + 1}"


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _paired_predictive_metric_rows(rows: pd.DataFrame, predicted: pd.Series) -> pd.DataFrame:
    actual = _numeric_series(rows, "target_actual_points")
    valid_mask = actual.map(_is_finite_number) & predicted.map(_is_finite_number)
    paired_rows = pd.DataFrame(
        {
            "rodada": rows["rodada"],
            "actual": actual,
            "predicted": predicted,
        }
    )
    if "id_atleta" in rows.columns:
        paired_rows["id_atleta"] = rows["id_atleta"]
    return paired_rows.loc[valid_mask].reset_index(drop=True)


def _mean_absolute_error(errors: pd.Series) -> float:
    if errors.empty:
        return float("nan")
    return float(errors.abs().mean())


def _root_mean_squared_error(errors: pd.Series) -> float:
    if errors.empty:
        return float("nan")
    return float(np.sqrt((errors**2).mean()))


def _spearman(actual: pd.Series, predicted: pd.Series) -> float:
    if len(actual) < 2 or actual.nunique(dropna=True) < 2 or predicted.nunique(dropna=True) < 2:
        return float("nan")
    correlation = actual.corr(predicted, method="spearman")
    if pd.isna(correlation):
        return float("nan")
    return float(correlation)


def _top50_spearman(paired_rows: pd.DataFrame) -> float:
    round_spearman_values: list[float] = []
    for _, round_rows in paired_rows.groupby("rodada"):
        if len(round_rows) < 50:
            continue
        sort_columns = ["predicted"]
        ascending = [False]
        if "id_atleta" in round_rows.columns:
            sort_columns.append("id_atleta")
            ascending.append(True)
        top_rows = round_rows.sort_values(sort_columns, ascending=ascending, kind="mergesort").head(50)
        spearman = _spearman(top_rows["actual"], top_rows["predicted"])
        if not math.isnan(spearman):
            round_spearman_values.append(spearman)
    if not round_spearman_values:
        return float("nan")
    return float(sum(round_spearman_values) / len(round_spearman_values))


def _calibration_slope(paired_rows: pd.DataFrame) -> float:
    if len(paired_rows) < 2 or paired_rows["predicted"].nunique(dropna=True) < 2:
        return float("nan")
    coefficients = np.polyfit(
        paired_rows["predicted"].to_numpy(dtype=float),
        paired_rows["actual"].to_numpy(dtype=float),
        1,
    )
    return float(coefficients[0])


def _mean_prediction_bias(errors: pd.Series) -> float:
    if errors.empty:
        return float("nan")
    return float(errors.mean())


def _filter_constructor_params(ebm_class: type[Any], params: dict[str, object]) -> dict[str, object]:
    signature = inspect.signature(ebm_class)
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(params)

    accepted = set(parameters)
    aliases = {
        "validation_size": "validation_fraction",
        "early_stopping_rounds": "early_stopping_run_length",
    }
    filtered: dict[str, object] = {}
    for key, value in params.items():
        if key in accepted:
            filtered[key] = value
            continue
        alias = aliases.get(key)
        if alias is not None and alias in accepted:
            filtered[alias] = value
    return filtered


def _validated_learned_edges(learned_edges: tuple[float, ...]) -> np.ndarray:
    try:
        edges = np.asarray(learned_edges, dtype=float)
    except (TypeError, ValueError) as exc:
        raise EbmDiagnosticInvalid("learned_edges must be numeric, finite, and sorted nondecreasing") from exc
    if not bool(np.isfinite(edges).all()):
        raise EbmDiagnosticInvalid("learned_edges must contain only finite values")
    if bool((np.diff(edges) < 0).any()):
        raise EbmDiagnosticInvalid("learned_edges must be sorted nondecreasing")
    return edges


def _validate_interaction_bin_column(frame: pd.DataFrame, column: str) -> None:
    values = frame[column]
    if bool(values.isna().any()):
        raise EbmDiagnosticInvalid(f"Interaction bin column contains NaN values: {column}")
    numeric = pd.to_numeric(values, errors="coerce")
    if bool(numeric.isna().any()):
        raise EbmDiagnosticInvalid(f"Interaction bin column must contain numeric integral values: {column}")
    numeric_values = numeric.to_numpy(dtype=float)
    if not bool(np.isclose(numeric_values, np.round(numeric_values)).all()):
        raise EbmDiagnosticInvalid(f"Interaction bin column must contain integral values: {column}")


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(cast("Any", value)))
    except (TypeError, ValueError):
        return False


def _entered_field_is_false(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no", "n"}
    if isinstance(value, bool):
        return not value
    try:
        return float(cast("Any", value)) == 0.0
    except (TypeError, ValueError):
        return False


def _child_run_entries(child_runs: Sequence[object]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, child_run in enumerate(child_runs):
        if not isinstance(child_run, dict):
            raise EbmDiagnosticInvalid(f"experiment_metadata.json child_runs[{index}] must be an object")
        entries.append(cast("dict[str, Any]", child_run))
    return entries


def _child_matches_config(
    child: dict[str, Any],
    *,
    child_index: int,
    config: EbmDiagnosticConfig,
    season: int,
) -> bool:
    if not (
        _optional_int(child.get("season")) == season
        and child.get("model_id") == config.model_id
        and child.get("feature_pack") == config.feature_pack
        and child.get("fixture_mode") == config.fixture_mode
    ):
        return False
    metadata = child.get("metadata")
    if not isinstance(metadata, dict):
        raise EbmDiagnosticInvalid(f"experiment_metadata.json child_runs[{child_index}].metadata must be an object")
    return metadata.get("budget_policy") == "moving"


def _source_child_context(
    source_experiment_id: str,
    child: dict[str, Any],
    *,
    child_index: int,
    config: EbmDiagnosticConfig,
    season: int,
) -> SourceChildContext:
    child_path = f"child_runs[{child_index}]"
    metadata = _required_object(
        child,
        "metadata",
        artifact_name="experiment_metadata.json",
        field_path=f"{child_path}.metadata",
    )
    output_path = _required_str(
        child,
        "output_path",
        artifact_name="experiment_metadata.json",
        field_path=f"{child_path}.output_path",
    )
    resolved_child_path = _resolve_child_path(config.experiment_path, output_path)
    model_id = _required_str(
        child,
        "model_id",
        artifact_name="experiment_metadata.json",
        field_path=f"{child_path}.model_id",
    )
    feature_pack = _required_str(
        child,
        "feature_pack",
        artifact_name="experiment_metadata.json",
        field_path=f"{child_path}.feature_pack",
    )
    fixture_mode = _required_str(
        child,
        "fixture_mode",
        artifact_name="experiment_metadata.json",
        field_path=f"{child_path}.fixture_mode",
    )
    parent_values: dict[str, object] = {
        "season": season,
        "model_id": model_id,
        "feature_pack": feature_pack,
        "fixture_mode": fixture_mode,
        "matchup_context_mode": _required_str(
            metadata,
            "matchup_context_mode",
            artifact_name="experiment_metadata.json",
            field_path=f"{child_path}.metadata.matchup_context_mode",
        ),
        "footystats_mode": _required_str(
            metadata,
            "footystats_mode",
            artifact_name="experiment_metadata.json",
            field_path=f"{child_path}.metadata.footystats_mode",
        ),
        "budget_policy": _required_str(
            metadata,
            "budget_policy",
            artifact_name="experiment_metadata.json",
            field_path=f"{child_path}.metadata.budget_policy",
        ),
        "scoring_contract_version": _required_str(
            metadata,
            "scoring_contract_version",
            artifact_name="experiment_metadata.json",
            field_path=f"{child_path}.metadata.scoring_contract_version",
        ),
    }
    _require_matching_parent_metadata(metadata, parent_values, child_path=child_path)
    score_column = f"{model_id}_score"
    _verify_source_prediction_provenance(
        child_path=resolved_child_path,
        parent_values=parent_values,
        score_column=score_column,
    )
    return SourceChildContext(
        source_experiment_id=source_experiment_id,
        season=season,
        model_id=model_id,
        feature_pack=feature_pack,
        fixture_mode=fixture_mode,
        matchup_context_mode=str(parent_values["matchup_context_mode"]),
        footystats_mode=str(parent_values["footystats_mode"]),
        budget_policy=str(parent_values["budget_policy"]),
        scoring_contract_version=str(parent_values["scoring_contract_version"]),
        score_column=score_column,
        child_path=resolved_child_path,
        source_prediction_provenance_status="verified",
    )


def _verify_source_prediction_provenance(
    *,
    child_path: Path,
    parent_values: dict[str, object],
    score_column: str,
) -> None:
    metadata_path = child_path / "run_metadata.json"
    predictions_path = child_path / "player_predictions.csv"
    round_results_path = child_path / "round_results.csv"
    if not metadata_path.is_file():
        raise EbmDiagnosticInvalid(f"Missing source child run_metadata.json: {metadata_path}")
    if not predictions_path.is_file():
        raise EbmDiagnosticInvalid(f"Missing source child player_predictions.csv: {predictions_path}")
    if not round_results_path.is_file():
        raise EbmDiagnosticInvalid(f"Missing source child round_results.csv: {round_results_path}")

    child_metadata = _read_json_object(metadata_path, artifact_name="run_metadata.json")
    _require_child_metadata_matches_parent(child_metadata, parent_values, metadata_path=metadata_path)
    try:
        predictions = pd.read_csv(predictions_path, nrows=0)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise EbmDiagnosticInvalid(f"Unable to read source child player_predictions.csv: {predictions_path}") from exc
    if score_column not in predictions.columns:
        raise EbmDiagnosticInvalid(f"Missing score column in player_predictions.csv: {score_column}")


def _require_child_metadata_matches_parent(
    child_metadata: dict[str, Any],
    parent_values: dict[str, object],
    *,
    metadata_path: Path,
) -> None:
    required_fields = (
        "season",
        "fixture_mode",
        "matchup_context_mode",
        "footystats_mode",
        "budget_policy",
        "scoring_contract_version",
    )
    for field in required_fields:
        actual = _required_field(child_metadata, field, artifact_name="run_metadata.json", field_path=field)
        expected = parent_values[field]
        if actual != expected:
            raise EbmDiagnosticInvalid(
                f"run_metadata.json field {field}={actual!r} disagrees with parent metadata {expected!r}: "
                f"{metadata_path}"
            )
    for field in ("model_id", "feature_pack"):
        if field in child_metadata and child_metadata[field] != parent_values[field]:
            raise EbmDiagnosticInvalid(
                f"run_metadata.json field {field}={child_metadata[field]!r} disagrees with parent metadata "
                f"{parent_values[field]!r}: {metadata_path}"
            )


def _require_matching_parent_metadata(
    metadata: dict[str, Any],
    parent_values: dict[str, object],
    *,
    child_path: str,
) -> None:
    for field in ("season", "model_id", "feature_pack", "fixture_mode"):
        if field in metadata and metadata[field] != parent_values[field]:
            raise EbmDiagnosticInvalid(
                f"experiment_metadata.json {child_path}.metadata.{field}={metadata[field]!r} "
                f"disagrees with child_runs field {parent_values[field]!r}"
            )


def _read_json_object(path: Path, *, artifact_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EbmDiagnosticInvalid(f"Missing {artifact_name}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EbmDiagnosticInvalid(f"Invalid JSON in {artifact_name}: {path}") from exc
    if not isinstance(payload, dict):
        raise EbmDiagnosticInvalid(f"{artifact_name} must contain an object: {path}")
    return payload


def _required_field(
    payload: dict[str, Any],
    field: str,
    *,
    artifact_name: str,
    field_path: str,
) -> object:
    if field not in payload:
        raise EbmDiagnosticInvalid(f"{artifact_name} missing required field: {field_path}")
    return payload[field]


def _required_object(
    payload: dict[str, Any],
    field: str,
    *,
    artifact_name: str,
    field_path: str,
) -> dict[str, Any]:
    value = _required_field(payload, field, artifact_name=artifact_name, field_path=field_path)
    if not isinstance(value, dict):
        raise EbmDiagnosticInvalid(f"{artifact_name} field {field_path} must be an object")
    return cast("dict[str, Any]", value)


def _required_str(
    payload: dict[str, Any],
    field: str,
    *,
    artifact_name: str,
    field_path: str,
) -> str:
    value = _required_field(payload, field, artifact_name=artifact_name, field_path=field_path)
    if not isinstance(value, str):
        raise EbmDiagnosticInvalid(f"{artifact_name} field {field_path} must be a string")
    return value


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _resolve_child_path(experiment_path: Path, output_path: str) -> Path:
    path = Path(output_path)
    if path.is_absolute():
        return path
    project_path = _project_root_from_experiment_path(experiment_path) / path
    if project_path.exists():
        return project_path
    return experiment_path / path


def _project_root_from_experiment_path(experiment_path: Path) -> Path:
    for parent in (experiment_path, *experiment_path.parents):
        if parent.name == "08_reporting" and parent.parent.name == "data":
            return parent.parent.parent
    return experiment_path.parent


def _unmatched_row(source_experiment_id: str, *, config: EbmDiagnosticConfig, season: int) -> dict[str, object]:
    return {
        **_requested_row(source_experiment_id, config=config, season=season),
        "match_status": "missing",
        "child_path": "",
        "source_prediction_provenance_status": "unverified",
        "conflicting_child_paths": [],
        "missing_metadata_fields": [],
    }


def _duplicate_row(
    source_experiment_id: str,
    *,
    config: EbmDiagnosticConfig,
    season: int,
    matches: list[tuple[int, dict[str, Any]]],
) -> dict[str, object]:
    return {
        **_requested_row(source_experiment_id, config=config, season=season),
        "match_status": "duplicate",
        "child_path": "",
        "source_prediction_provenance_status": "unverified",
        "conflicting_child_paths": [
            str(_resolve_child_path(config.experiment_path, str(match.get("output_path", "")))) for _, match in matches
        ],
        "missing_metadata_fields": [],
    }


def _requested_row(source_experiment_id: str, *, config: EbmDiagnosticConfig, season: int) -> dict[str, object]:
    return {
        "source_experiment_id": source_experiment_id,
        "requested_season": season,
        "season": season,
        "model_id": config.model_id,
        "feature_pack": config.feature_pack,
        "fixture_mode": config.fixture_mode,
        "matchup_context_mode": "",
        "footystats_mode": "",
        "budget_policy": "moving",
        "scoring_contract_version": "",
        "primary_score_column": f"{config.model_id}_score",
        "discovery_only": True,
    }
