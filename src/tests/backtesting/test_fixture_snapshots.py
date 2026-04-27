from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from cartola.backtesting.fixture_snapshots import (
    FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS,
    capture_cartola_snapshot,
    cartola_deadline_at,
    cartola_fixture_rows,
    parse_http_date_utc,
)

FROZEN_FIXTURE_PAYLOAD = {
    "rodada": 12,
    "partidas": [
        {
            "clube_casa_id": 262,
            "clube_visitante_id": 277,
            "partida_data": "2026-06-01 19:00:00",
            "timestamp": 1780340400,
            "valida": True,
            "placar_oficial_mandante": 3,
            "placar_oficial_visitante": 1,
        },
        {
            "clube_casa_id": 275,
            "clube_visitante_id": 284,
            "partida_data": "2026-06-01 21:30:00",
            "timestamp": 1780349400,
            "valida": False,
        },
    ],
}

FROZEN_DEADLINE_PAYLOAD = {
    "temporada": 2026,
    "rodada_atual": 12,
    "status_mercado": 1,
    "fechamento": {
        "dia": 1,
        "mes": 6,
        "ano": 2026,
        "hora": 18,
        "minuto": 59,
        "timestamp": 1780340340,
    },
}


def test_parse_http_date_utc_accepts_rfc_7231_gmt() -> None:
    parsed = parse_http_date_utc("Mon, 01 Jun 2026 18:00:00 GMT")

    assert parsed == datetime(2026, 6, 1, 18, 0, tzinfo=UTC)


def test_parse_http_date_utc_rejects_non_gmt_values() -> None:
    with pytest.raises(ValueError, match="HTTP Date"):
        parse_http_date_utc("2026-06-01T18:00:00Z")


def test_cartola_fixture_rows_extracts_only_valid_matches() -> None:
    rows = cartola_fixture_rows(FROZEN_FIXTURE_PAYLOAD, round_number=12)

    assert rows == [
        {
            "rodada": 12,
            "id_clube_home": 262,
            "id_clube_away": 277,
            "data": "2026-06-01",
        }
    ]


def test_cartola_fixture_rows_rejects_round_mismatch() -> None:
    with pytest.raises(ValueError, match="Fixture payload rodada"):
        cartola_fixture_rows(FROZEN_FIXTURE_PAYLOAD, round_number=13)


def test_cartola_fixture_rows_rejects_non_integer_club_ids() -> None:
    payload = {
        "rodada": 12,
        "partidas": [
            {
                "clube_casa_id": "abc",
                "clube_visitante_id": 277,
                "partida_data": "2026-06-01 19:00:00",
                "timestamp": 1780340400,
                "valida": True,
            }
        ],
    }

    with pytest.raises(ValueError, match="Cartola club IDs"):
        cartola_fixture_rows(payload, round_number=12)


def test_cartola_fixture_rows_rejects_valid_match_missing_timestamp() -> None:
    payload = {
        "rodada": 12,
        "partidas": [
            {
                "clube_casa_id": 262,
                "clube_visitante_id": 277,
                "partida_data": "2026-06-01 19:00:00",
                "valida": True,
            }
        ],
    }

    with pytest.raises(ValueError, match="partida timestamp"):
        cartola_fixture_rows(payload, round_number=12)


def test_cartola_fixture_rows_rejects_missing_or_non_bool_valida() -> None:
    missing_valida = {
        "rodada": 12,
        "partidas": [
            {
                "clube_casa_id": 262,
                "clube_visitante_id": 277,
                "partida_data": "2026-06-01 19:00:00",
                "timestamp": 1780340400,
            }
        ],
    }
    non_bool_valida = {
        "rodada": 12,
        "partidas": [
            {
                "clube_casa_id": 262,
                "clube_visitante_id": 277,
                "partida_data": "2026-06-01 19:00:00",
                "timestamp": 1780340400,
                "valida": "true",
            }
        ],
    }

    with pytest.raises(ValueError, match="partida valida"):
        cartola_fixture_rows(missing_valida, round_number=12)
    with pytest.raises(ValueError, match="partida valida"):
        cartola_fixture_rows(non_bool_valida, round_number=12)


def test_cartola_deadline_at_uses_fechamento_timestamp() -> None:
    deadline = cartola_deadline_at(FROZEN_DEADLINE_PAYLOAD, season=2026, round_number=12)

    assert deadline == datetime.fromtimestamp(1780340340, tz=UTC)
    assert FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS == 300


def test_cartola_deadline_at_rejects_wrong_round() -> None:
    with pytest.raises(ValueError, match="rodada_atual"):
        cartola_deadline_at(FROZEN_DEADLINE_PAYLOAD, season=2026, round_number=13)


def test_cartola_deadline_at_rejects_missing_status_mercado() -> None:
    payload = {
        "temporada": 2026,
        "rodada_atual": 12,
        "fechamento": FROZEN_DEADLINE_PAYLOAD["fechamento"],
    }

    with pytest.raises(ValueError, match="status_mercado"):
        cartola_deadline_at(payload, season=2026, round_number=12)


def test_cartola_deadline_at_rejects_missing_fechamento_component() -> None:
    fechamento = dict(FROZEN_DEADLINE_PAYLOAD["fechamento"])
    del fechamento["hora"]
    payload = {
        "temporada": 2026,
        "rodada_atual": 12,
        "status_mercado": 1,
        "fechamento": fechamento,
    }

    with pytest.raises(ValueError, match="fechamento.hora"):
        cartola_deadline_at(payload, season=2026, round_number=12)


HTTP_DATE = "Mon, 01 Jun 2026 18:00:00 GMT"


def test_capture_cartola_snapshot_writes_atomic_snapshot(tmp_path: Path) -> None:
    captured_at = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)

    def fake_fetch(url: str) -> object:
        if url.endswith("/partidas/12"):
            return _response(FROZEN_FIXTURE_PAYLOAD, url)
        return _response(FROZEN_DEADLINE_PAYLOAD, url)

    result = capture_cartola_snapshot(
        project_root=tmp_path,
        season=2026,
        round_number=12,
        source="cartola_api",
        fetch=fake_fetch,
        now=lambda: captured_at,
    )

    assert result.capture_dir.name == "captured_at=2026-06-01T18-00-00Z"
    assert (result.capture_dir / "capture.json").exists()
    assert (result.capture_dir / "fixtures.json").exists()
    assert (result.capture_dir / "deadline.json").exists()
    assert not list((tmp_path / "data/01_raw/fixtures_snapshots/2026/rodada-12").glob(".tmp-*"))


def test_capture_cartola_snapshot_failure_leaves_no_valid_directory(tmp_path: Path) -> None:
    captured_at = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)

    def fake_fetch(url: str) -> object:
        if url.endswith("/partidas/12"):
            return _response(FROZEN_FIXTURE_PAYLOAD, url)
        raise RuntimeError("network failed")

    with pytest.raises(RuntimeError, match="network failed"):
        capture_cartola_snapshot(
            project_root=tmp_path,
            season=2026,
            round_number=12,
            source="cartola_api",
            fetch=fake_fetch,
            now=lambda: captured_at,
        )

    round_dir = tmp_path / "data/01_raw/fixtures_snapshots/2026/rodada-12"
    assert not list(round_dir.glob("captured_at=*"))


def _response(payload: dict[str, object], url: str) -> object:
    class Response:
        status_code = 200
        headers = {"Date": HTTP_DATE}

        def __init__(self) -> None:
            self.url = url

        def json(self) -> dict[str, object]:
            return payload

        def raise_for_status(self) -> None:
            return None

    return Response()
