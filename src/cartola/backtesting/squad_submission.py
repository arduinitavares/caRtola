from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd

CONTRACT_UNVERIFIED = "CONTRACT_UNVERIFIED"

CARTOLA_STATUS_ENDPOINT = "https://api.cartola.globo.com/mercado/status"
CARTOLA_MARKET_ENDPOINT = "https://api.cartola.globo.com/atletas/mercado"
CARTOLA_SCHEMES_ENDPOINT = "https://api.cartola.globo.com/esquemas"

POSITION_ID_TO_CODE = {1: "gol", 2: "lat", 3: "zag", 4: "mei", 5: "ata", 6: "tec"}

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


@dataclass(frozen=True)
class FormationScheme:
    scheme_id: int
    name: str
    position_counts: dict[str, int]


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


def _int_value(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise SquadSubmissionError(f"{field_name} must be an integer, not a boolean")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            return int(numeric_value)
        raise SquadSubmissionError(f"{field_name} must be an integer")
    if isinstance(value, str):
        stripped = value.strip()
        signless = stripped.removeprefix("-")
        if stripped and signless.isdecimal():
            return int(stripped)
    raise SquadSubmissionError(f"{field_name} must be an integer")


def _float_value(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise SquadSubmissionError(f"{field_name} must be a finite number, not a boolean")
    if isinstance(value, Real):
        numeric_value = float(value)
    elif isinstance(value, str):
        try:
            numeric_value = float(value.strip())
        except ValueError as exc:
            raise SquadSubmissionError(f"{field_name} must be a finite number") from exc
    else:
        raise SquadSubmissionError(f"{field_name} must be a finite number")
    if not math.isfinite(numeric_value):
        raise SquadSubmissionError(f"{field_name} must be a finite number")
    return numeric_value


def _strip_accents(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).strip())
    return "".join(character for character in normalized if not unicodedata.combining(character)).lower()


def _string_value(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise SquadSubmissionError(f"{field_name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise SquadSubmissionError(f"{field_name} must not be empty")
    return stripped


def parse_schemes(payload: JsonValue) -> dict[str, FormationScheme]:
    if isinstance(payload, list):
        scheme_rows = payload
    elif isinstance(payload, dict):
        scheme_rows = payload.get("esquemas")
        if not isinstance(scheme_rows, list):
            raise SquadSubmissionError("Cartola schemes payload must contain an esquemas list")
    else:
        raise SquadSubmissionError("Cartola schemes payload must be an object or list")

    schemes: dict[str, FormationScheme] = {}
    for index, row in enumerate(scheme_rows):
        if not isinstance(row, dict):
            raise SquadSubmissionError(f"Cartola scheme row must be an object: index={index}")
        row = cast("dict[str, object]", row)
        raw_name = row.get("nome", row.get("esquema"))
        name = _string_value(raw_name, f"scheme[{index}].nome")
        raw_scheme_id = row.get("esquema_id", row.get("id"))
        scheme_id = _int_value(raw_scheme_id, f"scheme[{index}].esquema_id")
        raw_positions = row.get("posicoes")
        if not isinstance(raw_positions, dict):
            raise SquadSubmissionError(f"Cartola scheme positions must be an object: formation={name}")
        position_counts = {
            _strip_accents(position_code): _int_value(count, f"scheme[{index}].posicoes.{position_code}")
            for position_code, count in raw_positions.items()
        }
        schemes[name] = FormationScheme(scheme_id=scheme_id, name=name, position_counts=position_counts)
    return schemes


def _market_position_map(market_payload: JsonValue) -> dict[int, str]:
    position_map = dict(POSITION_ID_TO_CODE)
    if not isinstance(market_payload, dict):
        return position_map

    raw_positions = market_payload.get("posicoes")
    if not isinstance(raw_positions, dict):
        return position_map

    for raw_key, raw_position in raw_positions.items():
        if not isinstance(raw_position, dict):
            continue
        raw_position = cast("dict[str, object]", raw_position)
        raw_position_id = raw_position.get("id", raw_key)
        raw_position_code = raw_position.get("abreviacao", raw_position.get("slug", raw_position.get("nome")))
        if raw_position_code is None:
            continue
        position_id = _int_value(raw_position_id, f"market.posicoes.{raw_key}.id")
        position_map[position_id] = _strip_accents(raw_position_code)
    return position_map


def _market_athlete_index(market_payload: JsonValue) -> dict[int, dict[str, object]]:
    if not isinstance(market_payload, dict):
        raise SquadSubmissionError("Cartola market payload must be an object")
    raw_athletes = market_payload.get("atletas")
    if not isinstance(raw_athletes, list):
        raise SquadSubmissionError("Cartola market payload must contain an atletas list")

    athlete_index: dict[int, dict[str, object]] = {}
    for index, raw_athlete in enumerate(raw_athletes):
        if not isinstance(raw_athlete, dict):
            raise SquadSubmissionError(f"Cartola market athlete row must be an object: index={index}")
        raw_athlete = cast("dict[str, object]", raw_athlete)
        athlete_id = _int_value(raw_athlete.get("atleta_id"), f"market.atletas[{index}].atleta_id")
        if athlete_id in athlete_index:
            raise SquadSubmissionError(f"Duplicate athlete in Cartola market payload: atleta_id={athlete_id}")
        athlete_index[athlete_id] = raw_athlete
    return athlete_index


def _selected_position_counts(selected: pd.DataFrame) -> dict[str, int]:
    if "posicao" not in selected.columns:
        raise SquadSubmissionError("recommended squad artifact must contain a posicao column")

    counts: dict[str, int] = {}
    for raw_position in selected["posicao"].tolist():
        position_code = _strip_accents(raw_position)
        counts[position_code] = counts.get(position_code, 0) + 1
    return counts


def _normalize_position_counts(position_counts: dict[str, int], expected_counts: dict[str, int]) -> dict[str, int]:
    position_keys = set(position_counts) | set(expected_counts)
    return {position: position_counts.get(position, 0) for position in sorted(position_keys)}


def _selected_athlete_rows(selected: pd.DataFrame) -> list[dict[str, object]]:
    selected_rows = cast("list[dict[str, object]]", selected.to_dict("records"))
    seen_athlete_ids: set[int] = set()
    for index, selected_row in enumerate(selected_rows):
        athlete_id = _int_value(selected_row.get("id_atleta"), f"selected[{index}].id_atleta")
        if athlete_id in seen_athlete_ids:
            raise SquadSubmissionError(f"Duplicate selected athlete in recommendation artifact: id_atleta={athlete_id}")
        seen_athlete_ids.add(athlete_id)
    return selected_rows


def _validate_open_status(
    artifact: RecommendationArtifact,
    status_payload: JsonValue,
    now: datetime,
    safety_margin_seconds: int,
) -> tuple[int, int]:
    if not isinstance(status_payload, dict):
        raise SquadSubmissionError("Cartola market status payload must be an object")

    market_season = _int_value(status_payload.get("temporada"), "status.temporada")
    market_round = _int_value(status_payload.get("rodada_atual"), "status.rodada_atual")
    if market_season != artifact.season:
        raise SquadSubmissionError(
            f"Cartola market season does not match recommendation artifact: "
            f"market_season={market_season} artifact_season={artifact.season}",
        )
    if market_round != artifact.target_round:
        raise SquadSubmissionError(
            f"Cartola market round does not match recommendation artifact: "
            f"market_round={market_round} artifact_round={artifact.target_round}",
        )

    status_mercado = _int_value(status_payload.get("status_mercado"), "status.status_mercado")
    if status_mercado != 1:
        raise SquadSubmissionError(f"Cartola market is not open: status_mercado={status_mercado}")
    if status_payload.get("game_over") is True:
        raise SquadSubmissionError("Cartola market is not open: game_over=true")

    raw_deadline = status_payload.get("fechamento")
    if not isinstance(raw_deadline, dict):
        raise SquadSubmissionError("Cartola market is not open: fechamento.timestamp is required")
    raw_deadline = cast("dict[str, object]", raw_deadline)
    deadline_timestamp = _float_value(raw_deadline.get("timestamp"), "status.fechamento.timestamp")
    if deadline_timestamp - now.timestamp() < safety_margin_seconds:
        raise SquadSubmissionError("Cartola market is not open: deadline is inside the safety margin")

    return market_round, market_season


def _status_is_playable(market_row: dict[str, object]) -> bool:
    if "status_id" in market_row:
        try:
            return _int_value(market_row.get("status_id"), "market.status_id") == 7
        except SquadSubmissionError:
            return False

    raw_status = market_row.get("status")
    if isinstance(raw_status, dict):
        status_row = cast("dict[str, object]", raw_status)
        if _strip_accents(status_row.get("nome")) == "provavel":
            return True
    return False


def _artifact_formation(artifact: RecommendationArtifact) -> str:
    raw_formation = artifact.summary.get("formation")
    if raw_formation is None:
        raw_formation = artifact.metadata.get("formation")
    return _string_value(raw_formation, "recommendation formation")


def validate_artifact_against_public_market(
    artifact: RecommendationArtifact,
    status_payload: JsonValue,
    schemes_payload: JsonValue,
    market_payload: JsonValue,
    *,
    now: datetime,
    safety_margin_seconds: int,
) -> dict[str, object]:
    market_round, market_season = _validate_open_status(artifact, status_payload, now, safety_margin_seconds)
    schemes = parse_schemes(schemes_payload)
    formation = _artifact_formation(artifact)
    scheme = schemes.get(formation)
    if scheme is None:
        raise SquadSubmissionError(f"Recommendation formation is not available in Cartola schemes: formation={formation}")

    selected_position_counts = _selected_position_counts(artifact.selected)
    normalized_selected_position_counts = _normalize_position_counts(selected_position_counts, scheme.position_counts)
    normalized_scheme_position_counts = _normalize_position_counts(scheme.position_counts, selected_position_counts)
    if normalized_selected_position_counts != normalized_scheme_position_counts:
        raise SquadSubmissionError(
            f"Recommendation formation counts do not match Cartola scheme: "
            f"formation={formation} "
            f"selected={normalized_selected_position_counts} scheme={normalized_scheme_position_counts}",
        )

    selected_rows = _selected_athlete_rows(artifact.selected)
    market_positions = _market_position_map(market_payload)
    market_athletes = _market_athlete_index(market_payload)
    not_comparable_fields: set[str] = set()
    has_club_column = "id_clube" in artifact.selected.columns
    if not has_club_column:
        not_comparable_fields.add("id_clube")

    for index, selected_row in enumerate(selected_rows):
        athlete_id = _int_value(selected_row.get("id_atleta"), f"selected[{index}].id_atleta")
        market_row = market_athletes.get(athlete_id)
        if market_row is None:
            raise SquadSubmissionError(f"Selected athlete is missing from current Cartola market: id_atleta={athlete_id}")

        selected_name = _string_value(selected_row.get("apelido"), f"selected[{index}].apelido")
        market_name = _string_value(market_row.get("apelido"), f"market.atletas[{athlete_id}].apelido")
        if selected_name != market_name:
            raise SquadSubmissionError(
                f"Selected athlete nickname drift: id_atleta={athlete_id} selected={selected_name} market={market_name}",
            )

        selected_position = _strip_accents(selected_row.get("posicao"))
        market_position_id = _int_value(market_row.get("posicao_id"), f"market.atletas[{athlete_id}].posicao_id")
        market_position = market_positions.get(market_position_id)
        if market_position != selected_position:
            raise SquadSubmissionError(
                f"Selected athlete position drift: id_atleta={athlete_id} "
                f"selected={selected_position} market={market_position}",
            )

        if has_club_column:
            selected_club_id = _int_value(selected_row.get("id_clube"), f"selected[{index}].id_clube")
            market_club_id = _int_value(market_row.get("clube_id"), f"market.atletas[{athlete_id}].clube_id")
            if selected_club_id != market_club_id:
                raise SquadSubmissionError(
                    f"Selected athlete club drift: id_atleta={athlete_id} "
                    f"selected={selected_club_id} market={market_club_id}",
                )

        selected_price = _float_value(selected_row.get("preco_pre_rodada"), f"selected[{index}].preco_pre_rodada")
        market_price = _float_value(market_row.get("preco_num"), f"market.atletas[{athlete_id}].preco_num")
        if abs(selected_price - market_price) > 0.01:
            raise SquadSubmissionError(
                f"Selected athlete price drift: id_atleta={athlete_id} "
                f"selected={selected_price:.2f} market={market_price:.2f}",
            )

        if not _status_is_playable(market_row):
            raise SquadSubmissionError(f"Selected athlete status drift: id_atleta={athlete_id}")

    return {
        "market_round": market_round,
        "market_season": market_season,
        "formation": formation,
        "formation_scheme_id": scheme.scheme_id,
        "selected_position_counts": normalized_selected_position_counts,
        "account_budget_verified": False,
        "not_comparable_fields": sorted(not_comparable_fields),
    }


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
