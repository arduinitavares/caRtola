from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

CONTRACT_UNVERIFIED = "CONTRACT_UNVERIFIED"

CARTOLA_STATUS_ENDPOINT = "https://api.cartola.globo.com/mercado/status"
CARTOLA_MARKET_ENDPOINT = "https://api.cartola.globo.com/atletas/mercado"
CARTOLA_SCHEMES_ENDPOINT = "https://api.cartola.globo.com/esquemas"

APPROVED_PROFILE: dict[str, str] = {
    "model_id": "xgboost_depth2_l2_heavy",
    "footystats_mode": "ppg_xg",
    "fixture_mode": "none",
    "matchup_context_mode": "none",
    "scoring_contract_version": "cartola_standard_2026_v1",
}

JsonValue = dict[str, Any] | list[Any]
Fetch = Callable[[str, float], JsonValue]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class SubmissionConfig:
    project_root: Path = Path(".")
    recommendation_path: Path | None = None
    submission_plan: Path | None = None
    timeout_seconds: float = 30.0
    confirm_submit: bool = False
    confirm_payload_sha256: str | None = None
    allow_non_approved_model: bool = False
    override_reason: str | None = None
    safety_margin_seconds: int = 120


@dataclass(frozen=True)
class SquadSubmissionResult:
    attempt_directory: Path | None
    submission_plan_path: Path | None
    submission_result_path: Path | None
    payload_sha256: str | None
    status: str


@dataclass(frozen=True)
class RecommendationArtifact:
    path: Path
    selected: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]
    live_workflow_metadata: dict[str, Any] | None
    source_artifact_hashes: dict[str, str]

    @property
    def season(self) -> int:
        return int(self.summary["season"])

    @property
    def target_round(self) -> int:
        return int(self.summary["target_round"])


class SquadSubmissionError(ValueError):
    pass


class ContractUnverifiedError(SquadSubmissionError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    canonical_payload = {
        **payload,
        "atletas": [int(athlete_id) for athlete_id in payload["atletas"]],
        "capitao": int(payload["capitao"]),
        "esquema": int(payload["esquema"]),
    }
    return json.dumps(
        canonical_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload_bytes(payload)).hexdigest()


def fetch_public_json(url: str, timeout_seconds: float) -> JsonValue:
    import requests  # type: ignore[import-untyped]

    response = requests.get(url, timeout=timeout_seconds)
    if response.status_code != 200:
        raise SquadSubmissionError(
            f"Cartola public request failed: url={url} status={response.status_code}",
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise SquadSubmissionError(f"Cartola public response is not valid JSON: url={url}") from exc
    if not isinstance(payload, (dict, list)):
        raise SquadSubmissionError(f"Cartola public JSON payload must be an object or array: url={url}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SquadSubmissionError(f"Unable to read JSON artifact: path={path}") from exc
    except json.JSONDecodeError as exc:
        raise SquadSubmissionError(f"Invalid JSON artifact: path={path}") from exc
    if not isinstance(payload, dict):
        raise SquadSubmissionError(f"JSON artifact must be an object: path={path}")
    return payload


def _resolve_project_path(project_root: Path, value: Path) -> Path:
    project_root_resolved = project_root.resolve()
    resolved = value.resolve() if value.is_absolute() else (project_root_resolved / value).resolve()
    if not resolved.is_relative_to(project_root_resolved):
        raise SquadSubmissionError(f"Path must resolve inside project_root: path={value}")
    return resolved


def _validate_recommendation_path(project_root: Path, recommendation_path: Path) -> Path:
    path = _resolve_project_path(project_root, recommendation_path)
    project_root_resolved = project_root.resolve()
    relative_parts = path.relative_to(project_root_resolved).parts
    forbidden_components = {
        "backtests",
        "experiments",
        "policy_simulations",
        "blend_diagnostics",
        "oracle_discovery",
        "ebm_diagnostics",
    }
    if forbidden_components.intersection(relative_parts):
        raise SquadSubmissionError(
            "recommendation_path must be a canonical live recommendation artifact path, not a research output path",
        )
    if len(relative_parts) != 8 or relative_parts[:3] != ("data", "08_reporting", "recommendations"):
        raise SquadSubmissionError(
            "recommendation_path must be a canonical live recommendation path: "
            "data/08_reporting/recommendations/<season>/round-<round>/live/runs/run_started_at=...",
        )
    season_part = relative_parts[3]
    round_part = relative_parts[4]
    mode_part = relative_parts[5]
    runs_part = relative_parts[6]
    run_id_part = relative_parts[7]
    if (
        not season_part.isdecimal()
        or not round_part.startswith("round-")
        or not round_part.removeprefix("round-").isdecimal()
        or mode_part != "live"
        or runs_part != "runs"
        or not run_id_part.startswith("run_started_at=")
        or run_id_part == "run_started_at="
    ):
        raise SquadSubmissionError(
            "recommendation_path must be a canonical live recommendation path: "
            "data/08_reporting/recommendations/<season>/round-<round>/live/runs/run_started_at=...",
        )
    return path


def _resolve_required_artifact_file(project_root: Path, path: Path) -> Path:
    try:
        return _resolve_project_path(project_root, path)
    except SquadSubmissionError as exc:
        raise SquadSubmissionError(f"Artifact file must resolve inside project_root: path={path}") from exc


def _resolve_optional_artifact_file(project_root: Path, path: Path) -> Path | None:
    if not path.exists() and not path.is_symlink():
        return None
    return _resolve_required_artifact_file(project_root, path)


def load_recommendation_artifact(*, project_root: Path, recommendation_path: Path) -> RecommendationArtifact:
    path = _validate_recommendation_path(project_root, recommendation_path)
    selected_path = _resolve_required_artifact_file(project_root, path / "recommended_squad.csv")
    summary_path = _resolve_required_artifact_file(project_root, path / "recommendation_summary.json")
    metadata_path = _resolve_required_artifact_file(project_root, path / "run_metadata.json")
    live_workflow_path = _resolve_optional_artifact_file(project_root, path / "live_workflow_metadata.json")

    try:
        selected = pd.read_csv(selected_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise SquadSubmissionError(f"Unable to read recommended squad artifact: path={selected_path}") from exc

    summary = _read_json_object(summary_path)
    metadata = _read_json_object(metadata_path)
    live_workflow_metadata = _read_json_object(live_workflow_path) if live_workflow_path is not None else None

    source_artifact_hashes = {
        "recommended_squad.csv": _sha256_file(selected_path),
        "recommendation_summary.json": _sha256_file(summary_path),
        "run_metadata.json": _sha256_file(metadata_path),
    }
    if live_workflow_path is not None:
        source_artifact_hashes["live_workflow_metadata.json"] = _sha256_file(live_workflow_path)

    return RecommendationArtifact(
        path=path,
        selected=selected,
        summary=summary,
        metadata=metadata,
        live_workflow_metadata=live_workflow_metadata,
        source_artifact_hashes=source_artifact_hashes,
    )


def run_submission(
    config: SubmissionConfig,
    *,
    fetch: Fetch = fetch_public_json,
    clock: Clock = utc_now,
) -> SquadSubmissionResult:
    del fetch, clock
    if config.confirm_submit:
        raise ContractUnverifiedError(CONTRACT_UNVERIFIED)
    raise SquadSubmissionError("recommendation_path is required for Phase 1 plan generation")
