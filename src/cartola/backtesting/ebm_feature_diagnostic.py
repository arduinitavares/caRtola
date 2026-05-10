from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class EbmDependencyError(RuntimeError):
    """Raised when InterpretML is unavailable or incompatible."""


class EbmDiagnosticInvalid(RuntimeError):
    """Raised when EBM diagnostic source artifacts are invalid."""

    def __init__(self, message: str, *, report: pd.DataFrame | None = None) -> None:
        super().__init__(message)
        self.report = report


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
class EbmRuntimeInfo:
    available: bool
    version: str | None
    constructor_signature: str
    fit_signature: str
    supports_explicit_validation: bool


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


def _child_run_entries(child_runs: list[object]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, child_run in enumerate(child_runs):
        if not isinstance(child_run, dict):
            raise EbmDiagnosticInvalid(f"experiment_metadata.json child_runs[{index}] must be an object")
        entries.append(child_run)
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
    return value


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
