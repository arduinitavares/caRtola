from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import requests

CARTOLA_FIXTURE_ENDPOINT = "https://api.cartola.globo.com/partidas/{round_number}"
CARTOLA_DEADLINE_ENDPOINT = "https://api.cartola.globo.com/mercado/status"
FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS = 300
CAPTURE_VERSION = "fixture_capture_v1"
Fetch = Callable[[str], Any]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class CapturedResponse:
    payload: dict[str, Any]
    http_date_header: str
    http_date_utc: datetime
    status_code: int
    final_url: str


@dataclass(frozen=True)
class CaptureResult:
    capture_dir: Path
    captured_at_utc: datetime
    deadline_at_utc: datetime
    fixture_rows: list[dict[str, Any]]


def parse_http_date_utc(value: str) -> datetime:
    if not value.endswith(" GMT"):
        raise ValueError(f"HTTP Date must be an RFC 7231 GMT value: {value!r}")

    try:
        parsed = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT")
    except ValueError as exc:
        raise ValueError(f"HTTP Date must be an RFC 7231 GMT value: {value!r}") from exc

    return parsed.replace(tzinfo=UTC)


def cartola_fixture_rows(payload: dict[str, Any], *, round_number: int) -> list[dict[str, Any]]:
    payload_round = _integer_field(payload, "rodada", context="Fixture payload rodada")
    if payload_round != round_number:
        raise ValueError(f"Fixture payload rodada {payload_round} does not match requested round {round_number}")

    partidas = payload.get("partidas")
    if not isinstance(partidas, list):
        raise ValueError("Fixture payload partidas must be a list")

    rows: list[dict[str, Any]] = []
    for partida in partidas:
        if not isinstance(partida, dict):
            raise ValueError("Fixture payload partidas entries must be objects")
        valida = partida.get("valida")
        if not isinstance(valida, bool):
            raise ValueError("Fixture payload partida valida must be a boolean")
        _integer_field(partida, "timestamp", context="Fixture payload partida timestamp")
        if not valida:
            continue

        home_id = _integer_field(partida, "clube_casa_id", context="Cartola club IDs")
        away_id = _integer_field(partida, "clube_visitante_id", context="Cartola club IDs")
        rows.append(
            {
                "rodada": round_number,
                "id_clube_home": home_id,
                "id_clube_away": away_id,
                "data": _fixture_date(partida.get("partida_data")),
            }
        )

    return rows


def cartola_deadline_at(payload: dict[str, Any], *, season: int, round_number: int) -> datetime:
    payload_season = _integer_field(payload, "temporada", context="Deadline payload temporada")
    if payload_season != season:
        raise ValueError(f"Deadline payload temporada {payload_season} does not match requested season {season}")

    payload_round = _integer_field(payload, "rodada_atual", context="Deadline payload rodada_atual")
    if payload_round != round_number:
        raise ValueError(f"Deadline payload rodada_atual {payload_round} does not match requested round {round_number}")
    _integer_field(payload, "status_mercado", context="Deadline payload status_mercado")

    fechamento = payload.get("fechamento")
    if not isinstance(fechamento, dict):
        raise ValueError("Deadline payload fechamento must be an object")

    for field in ("ano", "mes", "dia", "hora", "minuto"):
        _integer_field(fechamento, field, context=f"Deadline payload fechamento.{field}")
    timestamp = _integer_field(fechamento, "timestamp", context="Deadline payload fechamento.timestamp")
    return datetime.fromtimestamp(timestamp, tz=UTC)


def capture_cartola_snapshot(
    *,
    project_root: str | Path = ".",
    season: int,
    round_number: int,
    source: str = "cartola_api",
    fetch: Fetch | None = None,
    now: Clock | None = None,
) -> CaptureResult:
    if source != "cartola_api":
        raise ValueError(f"Unsupported fixture snapshot source: {source!r}")

    clock = now or _utc_now
    requester = fetch or _fetch_url
    capture_started_at_utc = _to_utc(clock(), field_name="capture_started_at_utc")

    fixture_endpoint = CARTOLA_FIXTURE_ENDPOINT.format(round_number=round_number)
    deadline_endpoint = CARTOLA_DEADLINE_ENDPOINT
    fixture_response = _capture_response(requester, fixture_endpoint)
    deadline_response = _capture_response(requester, deadline_endpoint)

    captured_at_utc = _to_utc(clock(), field_name="captured_at_utc")
    fixture_rows = cartola_fixture_rows(fixture_response.payload, round_number=round_number)
    deadline_at_utc = cartola_deadline_at(deadline_response.payload, season=season, round_number=round_number)
    if captured_at_utc >= deadline_at_utc:
        raise ValueError("captured_at_utc must be strictly before deadline_at_utc")

    observed_skews = [
        abs((captured_at_utc - fixture_response.http_date_utc).total_seconds()),
        abs((captured_at_utc - deadline_response.http_date_utc).total_seconds()),
    ]
    max_observed_skew = max(observed_skews)
    if max_observed_skew > FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS:
        raise ValueError(
            "HTTP Date clock skew exceeds "
            f"{FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS} seconds: {max_observed_skew:.3f}"
        )

    root = Path(project_root)
    round_dir = root / "data" / "01_raw" / "fixtures_snapshots" / str(season) / f"rodada-{round_number}"
    capture_dir = round_dir / f"captured_at={_directory_timestamp(captured_at_utc)}"
    tmp_dir = round_dir / f".tmp-{uuid.uuid4().hex}"
    capture_metadata = {
        "capture_started_at_utc": _iso_utc_z(capture_started_at_utc),
        "captured_at_utc": _iso_utc_z(captured_at_utc),
        "fixture_http_date_header": fixture_response.http_date_header,
        "fixture_http_date_utc": _iso_utc_z(fixture_response.http_date_utc),
        "fixture_http_status": fixture_response.status_code,
        "fixture_final_url": fixture_response.final_url,
        "deadline_http_date_header": deadline_response.http_date_header,
        "deadline_http_date_utc": _iso_utc_z(deadline_response.http_date_utc),
        "deadline_http_status": deadline_response.status_code,
        "deadline_final_url": deadline_response.final_url,
        "clock_skew_tolerance_seconds": FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS,
        "max_observed_clock_skew_seconds": max_observed_skew,
        "source": source,
        "season": season,
        "rodada": round_number,
        "capture_version": CAPTURE_VERSION,
    }

    if capture_dir.exists():
        raise FileExistsError(f"Fixture snapshot capture directory already exists: {capture_dir}")

    try:
        round_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir()
        _write_json(tmp_dir / "fixtures.json", fixture_response.payload)
        _write_json(tmp_dir / "deadline.json", deadline_response.payload)
        _write_json(tmp_dir / "capture.json", capture_metadata)
        tmp_dir.rename(capture_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    return CaptureResult(
        capture_dir=capture_dir,
        captured_at_utc=captured_at_utc,
        deadline_at_utc=deadline_at_utc,
        fixture_rows=fixture_rows,
    )


def _integer_field(payload: dict[str, Any], field: str, *, context: str) -> int:
    value = payload.get(field)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    raise ValueError(f"{context} must be an integer")


def _fixture_date(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid partida_data: {value!r}")
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid partida_data: {value!r}") from exc


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _fetch_url(url: str) -> Any:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response


def _capture_response(fetch: Fetch, url: str) -> CapturedResponse:
    response = fetch(url)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Cartola response payload must be a JSON object: {url}")

    http_date_header = _http_date_header(response)
    return CapturedResponse(
        payload=payload,
        http_date_header=http_date_header,
        http_date_utc=parse_http_date_utc(http_date_header),
        status_code=int(response.status_code),
        final_url=str(response.url),
    )


def _http_date_header(response: Any) -> str:
    headers = response.headers
    value = headers.get("Date")
    if value is None:
        value = headers.get("date")
    if not isinstance(value, str) or not value:
        raise ValueError("HTTP Date header is required for strict fixture snapshot capture")
    return value


def _to_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _iso_utc_z(value: datetime) -> str:
    return _to_utc(value, field_name="timestamp").isoformat().replace("+00:00", "Z")


def _directory_timestamp(value: datetime) -> str:
    return _to_utc(value, field_name="captured_at_utc").strftime("%Y-%m-%dT%H-%M-%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
