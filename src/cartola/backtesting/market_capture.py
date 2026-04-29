from __future__ import annotations

import hashlib  # noqa: F401
import json  # noqa: F401
import shutil  # noqa: F401
import uuid  # noqa: F401
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.data import load_round_file  # noqa: F401

CARTOLA_STATUS_ENDPOINT = "https://api.cartola.globo.com/mercado/status"
CARTOLA_MARKET_ENDPOINT = "https://api.cartola.globo.com/atletas/mercado"
CAPTURE_VERSION = "market_capture_v1"

REQUIRED_ATHLETE_FIELDS: tuple[str, ...] = (
    "atleta_id",
    "apelido",
    "clube_id",
    "posicao_id",
    "status_id",
    "preco_num",
    "media_num",
    "jogos_num",
)

OPTIONAL_ATHLETE_FIELDS: tuple[str, ...] = (
    "slug",
    "nome",
    "foto",
    "apelido_abreviado",
    "minimo_para_valorizar",
)

RAW_OUTPUT_COLUMNS: tuple[str, ...] = (
    "atletas.rodada_id",
    "atletas.status_id",
    "atletas.posicao_id",
    "atletas.atleta_id",
    "atletas.apelido",
    "atletas.slug",
    "atletas.clube_id",
    "atletas.clube.id.full.name",
    "atletas.preco_num",
    "atletas.pontos_num",
    "atletas.media_num",
    "atletas.jogos_num",
    "atletas.variacao_num",
    "atletas.entrou_em_campo",
    "atletas.minimo_para_valorizar",
    "atletas.apelido_abreviado",
    "atletas.nome",
    "atletas.foto",
    *DEFAULT_SCOUT_COLUMNS,
)


@dataclass(frozen=True)
class MarketCaptureConfig:
    season: int
    target_round: int | None = None
    auto: bool = False
    force: bool = False
    current_year: int | None = None
    project_root: Path = Path(".")
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class CapturedJsonResponse:
    payload: dict[str, Any]
    status_code: int
    final_url: str
    body_sha256: str


@dataclass(frozen=True)
class MarketCaptureResult:
    csv_path: Path
    metadata_path: Path
    target_round: int
    athlete_count: int
    status_mercado: int
    deadline_timestamp: int | None
    deadline_parse_status: str
    reused_existing: bool = False


Fetch = Callable[[str, float], CapturedJsonResponse]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_z(now: Clock) -> str:
    return now().astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _runtime_current_year() -> int:
    return datetime.now(UTC).year


def _int_field(payload: dict[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if value is None:
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return parsed


def deadline_metadata(status_payload: dict[str, Any]) -> tuple[int | None, str]:
    fechamento = status_payload.get("fechamento")
    if not isinstance(fechamento, dict) or "timestamp" not in fechamento:
        return None, "missing"
    try:
        timestamp = int(fechamento["timestamp"])
    except (TypeError, ValueError):
        return None, "invalid"
    return timestamp, "ok"


def _resolved_current_year(config: MarketCaptureConfig) -> int:
    return config.current_year if config.current_year is not None else _runtime_current_year()


def _target_round_from_status(config: MarketCaptureConfig, status_payload: dict[str, Any]) -> int:
    rodada_atual = _int_field(status_payload, "rodada_atual")
    if rodada_atual <= 0:
        raise ValueError("rodada_atual must be a positive integer")

    if config.auto:
        if config.target_round is not None:
            raise ValueError("--auto and --target-round are mutually exclusive")
        return rodada_atual

    if config.target_round is None:
        raise ValueError("target_round is required unless auto=True")
    if config.target_round != rodada_atual:
        raise ValueError(
            f"target_round {config.target_round} does not match mercado/status rodada_atual {rodada_atual}"
        )
    return config.target_round


def _club_map(market_payload: dict[str, Any]) -> dict[int, str]:
    clubes = market_payload.get("clubes")
    if not isinstance(clubes, dict):
        raise ValueError("market payload clubes must be an object")

    result: dict[int, str] = {}
    for key, value in clubes.items():
        if not isinstance(value, dict):
            raise ValueError(f"club payload must be an object: {key!r}")
        club_id = _int_field(value, "id")
        club_name = value.get("nome")
        if not isinstance(club_name, str) or not club_name.strip():
            raise ValueError(f"club {club_id} must have nome")
        result[club_id] = club_name
    return result


def _athletes(market_payload: dict[str, Any]) -> list[dict[str, Any]]:
    athletes = market_payload.get("atletas")
    if not isinstance(athletes, list) or not athletes:
        raise ValueError("market payload atletas must be a non-empty list")
    if not all(isinstance(row, dict) for row in athletes):
        raise ValueError("every athlete payload must be an object")
    return athletes


def _required_athlete_value(athlete: dict[str, Any], field_name: str) -> Any:
    if field_name not in athlete:
        raise ValueError(f"athlete payload missing required field {field_name!r}")
    return athlete[field_name]


def build_live_market_frame(market_payload: dict[str, Any], *, target_round: int) -> pd.DataFrame:
    clubs = _club_map(market_payload)
    rows: list[dict[str, Any]] = []
    for athlete in _athletes(market_payload):
        for field_name in REQUIRED_ATHLETE_FIELDS:
            _required_athlete_value(athlete, field_name)

        club_id = int(athlete["clube_id"])
        if club_id not in clubs:
            raise ValueError(f"athlete clube_id {club_id} has no matching club payload")

        row: dict[str, Any] = {
            "atletas.rodada_id": target_round,
            "atletas.status_id": athlete["status_id"],
            "atletas.posicao_id": athlete["posicao_id"],
            "atletas.atleta_id": athlete["atleta_id"],
            "atletas.apelido": athlete["apelido"],
            "atletas.slug": athlete.get("slug"),
            "atletas.clube_id": club_id,
            "atletas.clube.id.full.name": clubs[club_id],
            "atletas.preco_num": athlete["preco_num"],
            "atletas.pontos_num": 0.0,
            "atletas.media_num": athlete["media_num"],
            "atletas.jogos_num": athlete["jogos_num"],
            "atletas.variacao_num": 0.0,
            "atletas.entrou_em_campo": False,
            "atletas.minimo_para_valorizar": athlete.get("minimo_para_valorizar"),
            "atletas.apelido_abreviado": athlete.get("apelido_abreviado"),
            "atletas.nome": athlete.get("nome"),
            "atletas.foto": athlete.get("foto"),
        }
        for scout in DEFAULT_SCOUT_COLUMNS:
            row[scout] = 0
        rows.append(row)

    return pd.DataFrame(rows, columns=list(RAW_OUTPUT_COLUMNS))
