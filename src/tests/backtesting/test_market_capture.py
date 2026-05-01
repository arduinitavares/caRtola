from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.data import load_round_file
from cartola.backtesting.market_capture import (
    CAPTURE_VERSION,
    CapturedJsonResponse,
    Fetch,
    MarketCaptureConfig,
    build_live_market_frame,
    capture_market_round,
    deadline_metadata,
)


def _status_payload(*, rodada_atual: int = 14, status_mercado: int = 1) -> dict[str, object]:
    return {
        "rodada_atual": rodada_atual,
        "status_mercado": status_mercado,
        "fechamento": {"timestamp": 1777748340},
    }


def _market_payload(*, stale_round: int = 13) -> dict[str, object]:
    return {
        "clubes": {
            "264": {"id": 264, "nome": "Corinthians"},
            "276": {"id": 276, "nome": "Sao Paulo"},
        },
        "atletas": [
            {
                "atleta_id": 100652,
                "apelido": "Yuri Alberto",
                "slug": "yuri-alberto",
                "nome": "Yuri Alberto",
                "foto": "https://example.invalid/yuri.png",
                "apelido_abreviado": "Yuri",
                "clube_id": 264,
                "posicao_id": 5,
                "status_id": 7,
                "preco_num": 10.45,
                "pontos_num": 9.9,
                "media_num": 2.49,
                "jogos_num": 9,
                "variacao_num": -0.39,
                "rodada_id": stale_round,
                "entrou_em_campo": True,
                "minimo_para_valorizar": 6.12,
                "scout": {"G": 2, "A": 1, "DS": 4},
            },
            {
                "atleta_id": 80287,
                "apelido": "Luciano",
                "slug": "luciano",
                "nome": "Luciano",
                "foto": "https://example.invalid/luciano.png",
                "apelido_abreviado": "Luc",
                "clube_id": 276,
                "posicao_id": 5,
                "status_id": 7,
                "preco_num": 15.0,
                "pontos_num": 11.8,
                "media_num": 5.35,
                "jogos_num": 11,
                "variacao_num": 0.67,
                "rodada_id": stale_round,
                "entrou_em_campo": True,
                "minimo_para_valorizar": 7.2,
                "scout": {"G": 1, "FS": 2},
            },
        ],
    }


def _captured(payload: dict[str, object], url: str) -> CapturedJsonResponse:
    return CapturedJsonResponse(
        payload=payload,
        status_code=200,
        final_url=url,
        body_sha256="payload-sha256",
    )


def _fetch_pair(
    status_payload: dict[str, object] | None = None,
    market_payload: dict[str, object] | None = None,
) -> Fetch:
    status_payload = _status_payload() if status_payload is None else status_payload
    market_payload = _market_payload() if market_payload is None else market_payload

    def fetch(url: str, timeout: float) -> object:
        if url.endswith("/mercado/status"):
            return _captured(status_payload, url)
        if url.endswith("/atletas/mercado"):
            return _captured(market_payload, url)
        raise AssertionError(f"unexpected URL {url}")

    return fetch


def test_build_live_market_frame_replaces_stale_round_and_sanitizes_outcomes() -> None:
    frame = build_live_market_frame(_market_payload(stale_round=13), target_round=14)

    assert frame["atletas.rodada_id"].tolist() == [14, 14]
    assert frame["atletas.pontos_num"].tolist() == [0.0, 0.0]
    assert frame["atletas.variacao_num"].tolist() == [0.0, 0.0]
    assert frame["atletas.entrou_em_campo"].tolist() == [False, False]
    assert frame["atletas.clube.id.full.name"].tolist() == ["Corinthians", "Sao Paulo"]
    assert frame["atletas.preco_num"].tolist() == [10.45, 15.0]
    for scout in DEFAULT_SCOUT_COLUMNS:
        assert scout in frame.columns
        assert frame[scout].tolist() == [0, 0]


def test_build_live_market_frame_loads_through_round_loader(tmp_path: Path) -> None:
    frame = build_live_market_frame(_market_payload(stale_round=13), target_round=14)
    csv_path = tmp_path / "rodada-14.csv"
    frame.to_csv(csv_path, index=False)

    loaded = load_round_file(csv_path)

    assert loaded["rodada"].tolist() == [14, 14]
    assert loaded["status"].tolist() == ["Provavel", "Provavel"]
    assert loaded["posicao"].tolist() == ["ata", "ata"]
    assert loaded["preco_pre_rodada"].tolist() == [10.45, 15.0]
    assert loaded["pontuacao"].tolist() == [0.0, 0.0]
    assert loaded["entrou_em_campo"].tolist() == [False, False]


def test_build_live_market_frame_rejects_missing_club_mapping() -> None:
    payload = _market_payload()
    del payload["clubes"]["264"]

    with pytest.raises(ValueError, match="no matching club payload"):
        build_live_market_frame(payload, target_round=14)


def test_build_live_market_frame_rejects_missing_required_athlete_field() -> None:
    payload = _market_payload()
    del payload["atletas"][0]["preco_num"]

    with pytest.raises(ValueError, match="preco_num"):
        build_live_market_frame(payload, target_round=14)


def test_deadline_metadata_reports_ok_missing_and_invalid() -> None:
    assert deadline_metadata(_status_payload()) == (1777748340, "ok")
    assert deadline_metadata({"rodada_atual": 14, "status_mercado": 1}) == (None, "missing")
    assert deadline_metadata({"fechamento": {"timestamp": "bad"}}) == (None, "invalid")


def test_fetch_cartola_json_records_response_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from cartola.backtesting.market_capture import fetch_cartola_json

    class Response:
        status_code = 200
        content = json.dumps({"ok": True}).encode()
        url = "https://api.cartola.globo.com/mercado/status?x=1"

        def json(self) -> dict[str, object]:
            return {"ok": True}

    def fake_get(url: str, timeout: float) -> Response:
        assert url == "https://api.cartola.globo.com/mercado/status"
        assert timeout == 12.0
        return Response()

    monkeypatch.setattr("requests.get", fake_get)

    captured = fetch_cartola_json("https://api.cartola.globo.com/mercado/status", 12.0)

    assert captured.payload == {"ok": True}
    assert captured.status_code == 200
    assert captured.final_url == "https://api.cartola.globo.com/mercado/status?x=1"
    assert captured.body_sha256 == "6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3"


def test_fetch_cartola_json_rejects_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    from cartola.backtesting.market_capture import fetch_cartola_json

    class Response:
        status_code = 500
        content = b"error"
        url = "https://api.cartola.globo.com/mercado/status"

        def json(self) -> dict[str, object]:
            return {"error": True}

    monkeypatch.setattr("requests.get", lambda url, timeout: Response())

    with pytest.raises(ValueError, match="status=500"):
        fetch_cartola_json("https://api.cartola.globo.com/mercado/status", 12.0)


def test_fetch_cartola_json_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from cartola.backtesting.market_capture import fetch_cartola_json

    class Response:
        status_code = 200
        content = b"not-json"
        url = "https://api.cartola.globo.com/mercado/status"

        def json(self) -> dict[str, object]:
            raise ValueError("bad json")

    monkeypatch.setattr("requests.get", lambda url, timeout: Response())

    with pytest.raises(ValueError, match="not valid JSON"):
        fetch_cartola_json("https://api.cartola.globo.com/mercado/status", 12.0)


def test_capture_refuses_wrong_current_year(tmp_path: Path) -> None:
    config = MarketCaptureConfig(
        season=2025,
        target_round=14,
        current_year=2026,
        project_root=tmp_path,
    )

    with pytest.raises(ValueError, match="season 2025 must equal current_year 2026"):
        capture_market_round(config, fetch=lambda url, timeout: _captured({}, url))


def test_capture_refuses_closed_market(tmp_path: Path) -> None:
    config = MarketCaptureConfig(
        season=2026,
        target_round=14,
        current_year=2026,
        project_root=tmp_path,
    )

    with pytest.raises(ValueError, match="status_mercado 2"):
        capture_market_round(config, fetch=_fetch_pair(status_payload=_status_payload(status_mercado=2)))


def test_capture_refuses_target_round_mismatch(tmp_path: Path) -> None:
    config = MarketCaptureConfig(
        season=2026,
        target_round=13,
        current_year=2026,
        project_root=tmp_path,
    )

    with pytest.raises(ValueError, match="does not match"):
        capture_market_round(config, fetch=_fetch_pair(status_payload=_status_payload(rodada_atual=14)))


def test_capture_publishes_csv_and_metadata_atomically_enough(tmp_path: Path) -> None:
    config = MarketCaptureConfig(
        season=2026,
        target_round=14,
        current_year=2026,
        project_root=tmp_path,
    )

    result = capture_market_round(
        config,
        fetch=_fetch_pair(),
        now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
    )

    assert result.csv_path == tmp_path / "data/01_raw/2026/rodada-14.csv"
    assert result.metadata_path == tmp_path / "data/01_raw/2026/rodada-14.capture.json"
    assert result.athlete_count == 2
    assert result.deadline_timestamp == 1777748340
    assert result.deadline_parse_status == "ok"
    assert result.csv_path.exists()
    assert result.metadata_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["capture_version"] == CAPTURE_VERSION
    assert metadata["csv_path"] == str(result.csv_path)
    assert metadata["csv_sha256"]
    assert metadata["status_endpoint"] == "https://api.cartola.globo.com/mercado/status"
    assert metadata["status_final_url"] == "https://api.cartola.globo.com/mercado/status"
    assert metadata["status_http_status"] == 200
    assert metadata["status_response_sha256"]
    assert metadata["market_endpoint"] == "https://api.cartola.globo.com/atletas/mercado"
    assert metadata["market_final_url"] == "https://api.cartola.globo.com/atletas/mercado"
    assert metadata["market_http_status"] == 200
    assert metadata["market_response_sha256"]
    assert metadata["deadline_parse_status"] == "ok"
    assert metadata["deadline_timestamp"] == 1777748340
    loaded = load_round_file(result.csv_path)
    assert loaded["rodada"].tolist() == [14, 14]
    assert not list((tmp_path / "data/01_raw/2026").glob(".tmp-market-capture-*"))


def test_capture_does_not_publish_when_generated_csv_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    final_csv = tmp_path / "data/01_raw/2026/rodada-14.csv"
    final_metadata = tmp_path / "data/01_raw/2026/rodada-14.capture.json"

    def fail_load_round_file(path: Path) -> object:
        raise ValueError("bad generated csv")

    monkeypatch.setattr("cartola.backtesting.market_capture.load_round_file", fail_load_round_file)

    with pytest.raises(ValueError, match="bad generated csv"):
        capture_market_round(config, fetch=_fetch_pair())

    assert not final_csv.exists()
    assert not final_metadata.exists()
    assert not list((tmp_path / "data/01_raw/2026").glob(".tmp-market-capture-*"))


def test_capture_refuses_existing_csv_without_force(tmp_path: Path) -> None:
    season_dir = tmp_path / "data/01_raw/2026"
    season_dir.mkdir(parents=True)
    (season_dir / "rodada-14.csv").write_text("not,a,capture\n", encoding="utf-8")
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        capture_market_round(config, fetch=_fetch_pair())


def test_capture_force_refuses_raw_file_without_capture_metadata(tmp_path: Path) -> None:
    season_dir = tmp_path / "data/01_raw/2026"
    season_dir.mkdir(parents=True)
    (season_dir / "rodada-14.csv").write_text("not,a,capture\n", encoding="utf-8")
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path, force=True)

    with pytest.raises(ValueError, match="not a previous valid live capture"):
        capture_market_round(config, fetch=_fetch_pair())


def test_load_valid_live_capture_returns_capture_metadata(tmp_path: Path) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    capture_market_round(config, fetch=_fetch_pair(), now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC))

    from cartola.backtesting.market_capture import load_valid_live_capture

    metadata = load_valid_live_capture(project_root=tmp_path, season=2026, target_round=14)

    assert metadata.csv_path == tmp_path / "data/01_raw/2026/rodada-14.csv"
    assert metadata.metadata_path == tmp_path / "data/01_raw/2026/rodada-14.capture.json"
    assert metadata.target_round == 14
    assert metadata.season == 2026
    assert metadata.status_mercado == 1
    assert metadata.deadline_timestamp == 1777748340
    assert metadata.deadline_parse_status == "ok"
    assert metadata.captured_at_utc == "2026-04-29T12:00:00Z"
    assert len(metadata.csv_sha256) == 64


def test_load_valid_live_capture_rejects_hash_mismatch(tmp_path: Path) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    capture_market_round(config, fetch=_fetch_pair(), now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC))
    csv_path = tmp_path / "data/01_raw/2026/rodada-14.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    from cartola.backtesting.market_capture import load_valid_live_capture

    with pytest.raises(ValueError, match="not a previous valid live capture"):
        load_valid_live_capture(project_root=tmp_path, season=2026, target_round=14)


def test_load_valid_live_capture_rejects_csv_path_mismatch(tmp_path: Path) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    capture_market_round(config, fetch=_fetch_pair(), now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC))
    metadata_path = tmp_path / "data/01_raw/2026/rodada-14.capture.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["csv_path"] = str(tmp_path / "data/01_raw/2026/rodada-13.csv")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from cartola.backtesting.market_capture import load_valid_live_capture

    with pytest.raises(ValueError, match="not a previous valid live capture"):
        load_valid_live_capture(project_root=tmp_path, season=2026, target_round=14)


def test_load_valid_live_capture_rejects_bad_captured_at(tmp_path: Path) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    capture_market_round(config, fetch=_fetch_pair(), now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC))
    metadata_path = tmp_path / "data/01_raw/2026/rodada-14.capture.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["captured_at_utc"] = "2026-04-29T12:00:00+00:00"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from cartola.backtesting.market_capture import load_valid_live_capture

    with pytest.raises(ValueError, match="not a previous valid live capture"):
        load_valid_live_capture(project_root=tmp_path, season=2026, target_round=14)


def test_load_valid_live_capture_rejects_closed_market_metadata(tmp_path: Path) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    capture_market_round(config, fetch=_fetch_pair(), now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC))
    metadata_path = tmp_path / "data/01_raw/2026/rodada-14.capture.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status_mercado"] = 2
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from cartola.backtesting.market_capture import load_valid_live_capture

    with pytest.raises(ValueError, match="not a previous valid live capture"):
        load_valid_live_capture(project_root=tmp_path, season=2026, target_round=14)


def test_capture_force_replaces_previous_valid_capture(tmp_path: Path) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    capture_market_round(config, fetch=_fetch_pair(), now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC))

    forced = capture_market_round(
        MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path, force=True),
        fetch=_fetch_pair(market_payload=_market_payload(stale_round=12)),
        now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
    )

    assert forced.csv_path.exists()
    assert json.loads(forced.metadata_path.read_text(encoding="utf-8"))["captured_at_utc"] == "2026-04-29T12:05:00Z"


def test_capture_force_restores_previous_capture_when_csv_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    initial = capture_market_round(
        config,
        fetch=_fetch_pair(),
        now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
    )
    old_csv_text = initial.csv_path.read_text(encoding="utf-8")
    old_metadata_text = initial.metadata_path.read_text(encoding="utf-8")
    old_metadata_hash = json.loads(old_metadata_text)["csv_sha256"]
    original_replace = Path.replace

    def fail_final_csv_replace(self: Path, target: str | Path) -> Path:
        if self.name == "round.csv" and Path(target) == initial.csv_path:
            raise RuntimeError("publish failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_final_csv_replace)

    with pytest.raises(RuntimeError, match="publish failed"):
        capture_market_round(
            MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path, force=True),
            fetch=_fetch_pair(market_payload=_market_payload(stale_round=12)),
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    restored_metadata_text = initial.metadata_path.read_text(encoding="utf-8")
    assert initial.csv_path.read_text(encoding="utf-8") == old_csv_text
    assert restored_metadata_text == old_metadata_text
    assert json.loads(restored_metadata_text)["csv_sha256"] == old_metadata_hash
    assert not list((tmp_path / "data/01_raw/2026").glob(".tmp-market-capture-*"))


def test_capture_force_preserves_backups_when_rollback_restoration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    initial = capture_market_round(
        config,
        fetch=_fetch_pair(),
        now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
    )
    season_dir = initial.csv_path.parent
    original_replace = Path.replace

    def fail_publish_and_rollback_csv(self: Path, target: str | Path) -> Path:
        target_path = Path(target)
        if self.name == "round.csv" and target_path == initial.csv_path:
            raise RuntimeError("publish failed")
        if self.name.startswith("rodada-14.csv.bak-") and target_path == initial.csv_path:
            raise RuntimeError("rollback failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_rollback_csv)

    with pytest.raises(RuntimeError):
        capture_market_round(
            MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path, force=True),
            fetch=_fetch_pair(market_payload=_market_payload(stale_round=12)),
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    assert list(season_dir.glob("*.bak-*"))
    assert not list(season_dir.glob(".tmp-market-capture-*"))


def test_capture_force_preserves_originals_when_csv_backup_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    initial = capture_market_round(
        config,
        fetch=_fetch_pair(),
        now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
    )
    old_csv_text = initial.csv_path.read_text(encoding="utf-8")
    old_metadata_text = initial.metadata_path.read_text(encoding="utf-8")
    original_replace = Path.replace

    def fail_csv_backup(self: Path, target: str | Path) -> Path:
        if self == initial.csv_path and Path(target).name.startswith("rodada-14.csv.bak-"):
            raise RuntimeError("backup failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_csv_backup)

    with pytest.raises(RuntimeError, match="backup failed"):
        capture_market_round(
            MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path, force=True),
            fetch=_fetch_pair(market_payload=_market_payload(stale_round=12)),
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    assert initial.csv_path.read_text(encoding="utf-8") == old_csv_text
    assert initial.metadata_path.read_text(encoding="utf-8") == old_metadata_text
    assert "2026-04-29T12:05:00Z" not in initial.metadata_path.read_text(encoding="utf-8")
    assert not list(initial.csv_path.parent.glob(".tmp-market-capture-*"))


def test_capture_auto_refuses_existing_capture_without_reusing_stale_data(tmp_path: Path) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    capture_market_round(config, fetch=_fetch_pair(), now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC))
    original_metadata = (tmp_path / "data/01_raw/2026/rodada-14.capture.json").read_text(encoding="utf-8")

    fetched_urls: list[str] = []

    def fetch(url: str, timeout: float) -> object:
        fetched_urls.append(url)
        return _fetch_pair(market_payload=_market_payload(stale_round=12))(url, timeout)

    with pytest.raises(FileExistsError, match="destination already exists"):
        capture_market_round(
            MarketCaptureConfig(season=2026, auto=True, current_year=2026, project_root=tmp_path),
            fetch=fetch,
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    assert any(url.endswith("/mercado/status") for url in fetched_urls)
    assert any(url.endswith("/atletas/mercado") for url in fetched_urls)
    assert (tmp_path / "data/01_raw/2026/rodada-14.capture.json").read_text(encoding="utf-8") == original_metadata
    assert not list((tmp_path / "data/01_raw/2026").glob(".tmp-market-capture-*"))


def _raw_round_csv(path: Path, round_number: int) -> None:
    rows: list[dict[str, object]] = []
    player_id = 1
    for posicao, count in {"gol": 2, "lat": 3, "zag": 3, "mei": 4, "ata": 4, "tec": 2}.items():
        for offset in range(count):
            row = {
                "atletas.rodada_id": round_number,
                "atletas.status_id": 7,
                "atletas.posicao_id": {"gol": 1, "lat": 2, "zag": 3, "mei": 4, "ata": 5, "tec": 6}[posicao],
                "atletas.atleta_id": player_id,
                "atletas.apelido": f"{posicao}-{offset}",
                "atletas.slug": f"{posicao}-{offset}",
                "atletas.clube_id": player_id,
                "atletas.clube.id.full.name": f"Club {player_id}",
                "atletas.preco_num": 5.0,
                "atletas.pontos_num": float(round_number + offset),
                "atletas.media_num": float(round_number + offset),
                "atletas.jogos_num": round_number - 1,
                "atletas.variacao_num": 0.0,
                "atletas.entrou_em_campo": True,
                "atletas.minimo_para_valorizar": 0.0,
                "atletas.apelido_abreviado": f"{posicao}{offset}",
                "atletas.nome": f"{posicao} {offset}",
                "atletas.foto": "",
            }
            for scout in DEFAULT_SCOUT_COLUMNS:
                row[scout] = 1 if scout == "DS" else 0
            rows.append(row)
            player_id += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def test_captured_round_can_feed_live_recommendation(tmp_path: Path) -> None:
    from cartola.backtesting.recommendation import RecommendationConfig, run_recommendation

    season_dir = tmp_path / "data/01_raw/2026"
    season_dir.mkdir(parents=True)
    _raw_round_csv(season_dir / "rodada-1.csv", 1)
    _raw_round_csv(season_dir / "rodada-2.csv", 2)

    # Expand the frozen market payload enough to satisfy the optimizer formation.
    payload = {"clubes": {}, "atletas": []}
    player_id = 1
    for posicao_id, count in {1: 2, 2: 3, 3: 3, 4: 4, 5: 4, 6: 2}.items():
        for offset in range(count):
            payload["clubes"][str(player_id)] = {"id": player_id, "nome": f"Club {player_id}"}
            payload["atletas"].append(
                {
                    "atleta_id": player_id,
                    "apelido": f"p{player_id}",
                    "slug": f"p{player_id}",
                    "nome": f"Player {player_id}",
                    "foto": "",
                    "apelido_abreviado": f"p{player_id}",
                    "clube_id": player_id,
                    "posicao_id": posicao_id,
                    "status_id": 7,
                    "preco_num": 5.0,
                    "media_num": 3.0 + offset,
                    "jogos_num": 2,
                    "minimo_para_valorizar": 0.0,
                }
            )
            player_id += 1

    capture_market_round(
        MarketCaptureConfig(season=2026, target_round=3, current_year=2026, project_root=tmp_path),
        fetch=_fetch_pair(status_payload=_status_payload(rodada_atual=3), market_payload=payload),
        now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
    )

    result = run_recommendation(
        RecommendationConfig(
            season=2026,
            target_round=3,
            mode="live",
            current_year=2026,
            project_root=tmp_path,
            footystats_mode="none",
        )
    )

    assert result.summary["mode"] == "live"
    assert result.summary["target_round"] == 3
    assert result.summary["selected_count"] == 12
