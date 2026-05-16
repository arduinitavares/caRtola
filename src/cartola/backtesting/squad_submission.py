from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

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
