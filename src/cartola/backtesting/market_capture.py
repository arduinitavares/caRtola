from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.data import load_round_file

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


def _validate_config_year(config: MarketCaptureConfig) -> int:
    current_year = _resolved_current_year(config)
    if config.season != current_year:
        raise ValueError(f"season {config.season} must equal current_year {current_year}")
    return current_year


def _validate_open_market(status_payload: dict[str, Any]) -> int:
    status_mercado = _int_field(status_payload, "status_mercado")
    if status_mercado != 1:
        rodada_atual = status_payload.get("rodada_atual")
        raise ValueError(
            f"Cartola market is not open: rodada_atual={rodada_atual} status_mercado {status_mercado}"
        )
    return status_mercado


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

    return pd.DataFrame(rows, columns=pd.Index(RAW_OUTPUT_COLUMNS))


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_cartola_json(url: str, timeout_seconds: float) -> CapturedJsonResponse:
    import requests  # type: ignore[import-untyped]

    response = requests.get(url, timeout=timeout_seconds)
    body = response.content
    if response.status_code != 200:
        raise ValueError(f"Cartola request failed: url={url} status={response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"Cartola response is not valid JSON: url={url}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Cartola JSON payload must be an object: url={url}")
    return CapturedJsonResponse(
        payload=payload,
        status_code=response.status_code,
        final_url=str(response.url),
        body_sha256=_sha256_bytes(body),
    )


def _final_csv_path(config: MarketCaptureConfig, target_round: int) -> Path:
    return config.project_root / "data" / "01_raw" / str(config.season) / f"rodada-{target_round}.csv"


def _final_metadata_path(config: MarketCaptureConfig, target_round: int) -> Path:
    return config.project_root / "data" / "01_raw" / str(config.season) / f"rodada-{target_round}.capture.json"


def _metadata(
    *,
    config: MarketCaptureConfig,
    current_year: int,
    target_round: int,
    captured_at_utc: str,
    status_response: CapturedJsonResponse,
    market_response: CapturedJsonResponse,
    status_mercado: int,
    deadline_timestamp: int | None,
    deadline_parse_status: str,
    athlete_count: int,
    csv_path: Path,
    csv_sha256: str,
) -> dict[str, Any]:
    return {
        "capture_version": CAPTURE_VERSION,
        "season": config.season,
        "current_year": current_year,
        "target_round": target_round,
        "captured_at_utc": captured_at_utc,
        "status_endpoint": CARTOLA_STATUS_ENDPOINT,
        "status_final_url": status_response.final_url,
        "status_http_status": status_response.status_code,
        "status_response_sha256": status_response.body_sha256,
        "market_endpoint": CARTOLA_MARKET_ENDPOINT,
        "market_final_url": market_response.final_url,
        "market_http_status": market_response.status_code,
        "market_response_sha256": market_response.body_sha256,
        "rodada_atual": _int_field(status_response.payload, "rodada_atual"),
        "status_mercado": status_mercado,
        "deadline_timestamp": deadline_timestamp,
        "deadline_parse_status": deadline_parse_status,
        "athlete_count": athlete_count,
        "csv_path": str(csv_path),
        "csv_sha256": csv_sha256,
    }


def _write_temp_csv_and_metadata(
    *,
    temp_dir: Path,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    temp_dir.mkdir(parents=True, exist_ok=False)
    temp_csv = temp_dir / "round.csv"
    temp_metadata = temp_dir / "capture.json"
    frame.to_csv(temp_csv, index=False)
    metadata = dict(metadata)
    metadata["csv_sha256"] = _sha256_file(temp_csv)
    temp_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return temp_csv, temp_metadata


def _validate_previous_capture(
    final_csv: Path,
    final_metadata: Path,
    *,
    config: MarketCaptureConfig,
    target_round: int,
) -> None:
    if not final_csv.exists() or not final_metadata.exists():
        raise ValueError("destination is not a previous valid live capture")
    try:
        metadata = json.loads(final_metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("destination is not a previous valid live capture") from exc

    if metadata.get("capture_version") != CAPTURE_VERSION:
        raise ValueError("destination is not a previous valid live capture")
    if metadata.get("season") != config.season or metadata.get("target_round") != target_round:
        raise ValueError("destination is not a previous valid live capture")
    if Path(str(metadata.get("csv_path"))) != final_csv:
        raise ValueError("destination is not a previous valid live capture")
    if metadata.get("csv_sha256") != _sha256_file(final_csv):
        raise ValueError("destination is not a previous valid live capture")


def _validate_temp_capture(temp_csv: Path, temp_metadata: Path, *, target_round: int, final_csv: Path) -> None:
    loaded = load_round_file(temp_csv)
    rounds = sorted(pd.to_numeric(loaded["rodada"], errors="raise").astype(int).unique().tolist())
    if rounds != [target_round]:
        raise ValueError(f"generated CSV rodada mismatch: {rounds}")

    metadata = json.loads(temp_metadata.read_text(encoding="utf-8"))
    if Path(str(metadata.get("csv_path"))) != final_csv:
        raise ValueError("capture metadata csv_path does not point to final CSV")
    if metadata.get("csv_sha256") != _sha256_file(temp_csv):
        raise ValueError("capture metadata csv_sha256 does not match generated CSV")


def _publish_pair(
    *,
    temp_csv: Path,
    temp_metadata: Path,
    final_csv: Path,
    final_metadata: Path,
    force: bool,
    config: MarketCaptureConfig,
    target_round: int,
) -> None:
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    if final_csv.exists() or final_metadata.exists():
        if not force:
            raise FileExistsError(f"destination already exists: {final_csv}")
        _validate_previous_capture(final_csv, final_metadata, config=config, target_round=target_round)

    backup_csv = final_csv.with_name(f"{final_csv.name}.bak-{uuid.uuid4().hex}") if final_csv.exists() else None
    backup_metadata = (
        final_metadata.with_name(f"{final_metadata.name}.bak-{uuid.uuid4().hex}")
        if final_metadata.exists()
        else None
    )
    publication_completed = False
    rollback_completed = False
    csv_backed_up = False
    metadata_backed_up = False
    csv_published = False
    metadata_published = False
    try:
        if backup_csv is not None:
            final_csv.replace(backup_csv)
            csv_backed_up = True
        if backup_metadata is not None:
            final_metadata.replace(backup_metadata)
            metadata_backed_up = True
        temp_metadata.replace(final_metadata)
        metadata_published = True
        temp_csv.replace(final_csv)
        csv_published = True
        publication_completed = True
    except Exception:
        if final_csv.exists() and (csv_published or csv_backed_up):
            final_csv.unlink()
        if final_metadata.exists() and (metadata_published or metadata_backed_up):
            final_metadata.unlink()
        if metadata_backed_up and backup_metadata is not None and backup_metadata.exists():
            backup_metadata.replace(final_metadata)
        if csv_backed_up and backup_csv is not None and backup_csv.exists():
            backup_csv.replace(final_csv)
        rollback_completed = True
        raise
    finally:
        if publication_completed or rollback_completed:
            if backup_csv is not None and backup_csv.exists():
                backup_csv.unlink()
            if backup_metadata is not None and backup_metadata.exists():
                backup_metadata.unlink()


def capture_market_round(
    config: MarketCaptureConfig,
    *,
    fetch: Fetch = fetch_cartola_json,
    now: Clock = _utc_now,
) -> MarketCaptureResult:
    current_year = _validate_config_year(config)
    status_response = fetch(CARTOLA_STATUS_ENDPOINT, config.timeout_seconds)
    target_round = _target_round_from_status(config, status_response.payload)
    status_mercado = _validate_open_market(status_response.payload)
    final_csv = _final_csv_path(config, target_round)
    final_metadata = _final_metadata_path(config, target_round)

    if config.auto and not config.force and final_csv.exists() and final_metadata.exists():
        _validate_previous_capture(final_csv, final_metadata, config=config, target_round=target_round)
        metadata = json.loads(final_metadata.read_text(encoding="utf-8"))
        return MarketCaptureResult(
            csv_path=final_csv,
            metadata_path=final_metadata,
            target_round=target_round,
            athlete_count=int(metadata["athlete_count"]),
            status_mercado=int(metadata["status_mercado"]),
            deadline_timestamp=metadata["deadline_timestamp"],
            deadline_parse_status=str(metadata["deadline_parse_status"]),
            reused_existing=True,
        )

    market_response = fetch(CARTOLA_MARKET_ENDPOINT, config.timeout_seconds)
    frame = build_live_market_frame(market_response.payload, target_round=target_round)
    deadline_timestamp, deadline_parse_status = deadline_metadata(status_response.payload)

    season_dir = final_csv.parent
    temp_dir = season_dir / f".tmp-market-capture-{uuid.uuid4().hex}"
    try:
        season_dir.mkdir(parents=True, exist_ok=True)
        placeholder_metadata = _metadata(
            config=config,
            current_year=current_year,
            target_round=target_round,
            captured_at_utc=_utc_now_z(now),
            status_response=status_response,
            market_response=market_response,
            status_mercado=status_mercado,
            deadline_timestamp=deadline_timestamp,
            deadline_parse_status=deadline_parse_status,
            athlete_count=len(frame),
            csv_path=final_csv,
            csv_sha256="",
        )
        temp_csv, temp_metadata = _write_temp_csv_and_metadata(
            temp_dir=temp_dir,
            frame=frame,
            metadata=placeholder_metadata,
        )
        _validate_temp_capture(temp_csv, temp_metadata, target_round=target_round, final_csv=final_csv)
        _publish_pair(
            temp_csv=temp_csv,
            temp_metadata=temp_metadata,
            final_csv=final_csv,
            final_metadata=final_metadata,
            force=config.force,
            config=config,
            target_round=target_round,
        )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return MarketCaptureResult(
        csv_path=final_csv,
        metadata_path=final_metadata,
        target_round=target_round,
        athlete_count=len(frame),
        status_mercado=status_mercado,
        deadline_timestamp=deadline_timestamp,
        deadline_parse_status=deadline_parse_status,
    )
