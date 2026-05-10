from __future__ import annotations

import inspect
import json
import math
from collections.abc import Sequence
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
            "season": self.season,
            "model_id": self.model_id,
            "feature_pack": self.feature_pack,
            "fixture_mode": self.fixture_mode,
            "matchup_context_mode": self.matchup_context_mode,
            "footystats_mode": self.footystats_mode,
            "budget_policy": self.budget_policy,
            "scoring_contract_version": self.scoring_contract_version,
            "score_column": self.score_column,
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


def resolve_source_children(config: EbmDiagnosticConfig) -> tuple[tuple[SourceChildContext, ...], pd.DataFrame]:
    metadata_path = config.experiment_path / "experiment_metadata.json"
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
        feature_columns=resolved_feature_columns,
        coach_row_count=coach_row_count,
    )


def _prepare_diagnostic_features(
    valid_rows: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[str, ...]:
    identity_columns = frozenset(("id_atleta", "id_clube", "apelido", "season", "rodada"))
    missing_feature_columns = tuple(
        column for column in feature_columns if column not in identity_columns and column not in valid_rows.columns
    )
    if missing_feature_columns:
        raise EbmDiagnosticInvalid(f"Missing feature columns: {', '.join(missing_feature_columns)}")

    resolved_columns: list[str] = []
    for column in feature_columns:
        if column in identity_columns:
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
    if not metadata_path.is_file():
        raise EbmDiagnosticInvalid(f"Missing source child run_metadata.json: {metadata_path}")
    if not predictions_path.is_file():
        raise EbmDiagnosticInvalid(f"Missing source child player_predictions.csv: {predictions_path}")

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
        "season": season,
        "model_id": config.model_id,
        "feature_pack": config.feature_pack,
        "fixture_mode": config.fixture_mode,
        "matchup_context_mode": "",
        "footystats_mode": "",
        "budget_policy": "moving",
        "scoring_contract_version": "",
        "score_column": f"{config.model_id}_score",
        "discovery_only": True,
    }
