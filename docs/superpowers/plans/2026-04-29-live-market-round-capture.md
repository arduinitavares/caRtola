# Live Market Round Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe live Cartola market capture command that writes the current open-round CSV needed by `scripts/recommend_squad.py --mode live`.

**Architecture:** Add a focused `cartola.backtesting.market_capture` module for payload validation, live CSV generation, metadata, and publication safety. Add `scripts/capture_market_round.py` as a thin CLI wrapper. Keep recommendation prediction code unchanged; the capture command creates the missing canonical input file.

**Tech Stack:** Python 3.13, pandas, requests, pytest, existing `load_round_file()` / `load_season_data()` loader contract, existing `DEFAULT_SCOUT_COLUMNS` and raw Cartola column conventions.

---

## File Structure

- Create `src/cartola/backtesting/market_capture.py`
  - Owns Cartola market/status fetching, payload validation, live CSV row construction, metadata construction, SHA-256 helpers, temp publication, overwrite validation, and the public `capture_market_round()` API.
- Create `scripts/capture_market_round.py`
  - Thin argparse wrapper around `capture_market_round()`.
- Create `src/tests/backtesting/test_market_capture.py`
  - Unit and integration-style tests for transformation, validation, metadata, publication safety, overwrite rules, and recommendation compatibility.
- Create `src/tests/backtesting/test_capture_market_round_cli.py`
  - CLI parser and success/failure smoke tests.
- Modify `README.md`
  - Document capture-then-recommend live workflow and stop pointing live users at the old downloader.
- Modify `roadmap.md`
  - Mark the live capture item as delivered after implementation.

---

### Task 1: Add Frozen Payloads And Failing Transformation Tests

**Files:**
- Create: `src/tests/backtesting/test_market_capture.py`

- [ ] **Step 1: Add frozen Cartola payload fixtures**

Create `src/tests/backtesting/test_market_capture.py` with these imports and payload helpers:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.config import DEFAULT_SCOUT_COLUMNS
from cartola.backtesting.data import load_round_file
from cartola.backtesting.market_capture import build_live_market_frame


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
```

- [ ] **Step 2: Add failing test for live CSV sanitization**

Append:

```python
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
```

- [ ] **Step 3: Add failing test for loader compatibility**

Append:

```python
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
```

- [ ] **Step 4: Run tests to verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_market_capture.py::test_build_live_market_frame_replaces_stale_round_and_sanitizes_outcomes src/tests/backtesting/test_market_capture.py::test_build_live_market_frame_loads_through_round_loader -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cartola.backtesting.market_capture'`.

- [ ] **Step 5: Commit failing tests**

```bash
git add src/tests/backtesting/test_market_capture.py
git commit -m "test: define live market frame capture contract"
```

---

### Task 2: Implement Live Market Frame Construction

**Files:**
- Create: `src/cartola/backtesting/market_capture.py`
- Test: `src/tests/backtesting/test_market_capture.py`

- [ ] **Step 1: Add constants, dataclasses, and validation helpers**

Create `src/cartola/backtesting/market_capture.py`:

```python
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
import requests

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
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return parsed
```

- [ ] **Step 2: Implement deadline metadata and target-round selection**

Append:

```python
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
```

- [ ] **Step 3: Implement market frame construction**

Append:

```python
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
```

- [ ] **Step 4: Run Task 1 tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_market_capture.py::test_build_live_market_frame_replaces_stale_round_and_sanitizes_outcomes src/tests/backtesting/test_market_capture.py::test_build_live_market_frame_loads_through_round_loader -q
```

Expected: PASS.

- [ ] **Step 5: Add validation tests**

First update the import block in `src/tests/backtesting/test_market_capture.py`:

```python
from cartola.backtesting.market_capture import build_live_market_frame, deadline_metadata
```

Then append to `src/tests/backtesting/test_market_capture.py`:

```python
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
```

- [ ] **Step 6: Run all market capture tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_market_capture.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cartola/backtesting/market_capture.py src/tests/backtesting/test_market_capture.py
git commit -m "feat: build sanitized live market frames"
```

---

### Task 3: Add HTTP Fetching, Metadata, And Config Helpers

**Files:**
- Modify: `src/cartola/backtesting/market_capture.py`
- Modify: `src/tests/backtesting/test_market_capture.py`

- [ ] **Step 1: Add failing tests for HTTP fetch behavior**

Append:

```python
def test_fetch_cartola_json_records_response_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from cartola.backtesting.market_capture import fetch_cartola_json

    class Response:
        status_code = 200
        content = b'{"ok": true}'
        url = "https://api.cartola.globo.com/mercado/status?x=1"

        def json(self) -> dict[str, object]:
            return {"ok": True}

    def fake_get(url: str, timeout: float) -> Response:
        assert url == "https://api.cartola.globo.com/mercado/status"
        assert timeout == 12.0
        return Response()

    monkeypatch.setattr("cartola.backtesting.market_capture.requests.get", fake_get)

    captured = fetch_cartola_json("https://api.cartola.globo.com/mercado/status", 12.0)

    assert captured.payload == {"ok": True}
    assert captured.status_code == 200
    assert captured.final_url == "https://api.cartola.globo.com/mercado/status?x=1"
    assert captured.body_sha256 == "6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3"


def test_fetch_cartola_json_rejects_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    from cartola.backtesting.market_capture import fetch_cartola_json

    class Response:
        status_code = 500
        content = b'error'
        url = "https://api.cartola.globo.com/mercado/status"

        def json(self) -> dict[str, object]:
            return {"error": True}

    monkeypatch.setattr("cartola.backtesting.market_capture.requests.get", lambda url, timeout: Response())

    with pytest.raises(ValueError, match="status=500"):
        fetch_cartola_json("https://api.cartola.globo.com/mercado/status", 12.0)


def test_fetch_cartola_json_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from cartola.backtesting.market_capture import fetch_cartola_json

    class Response:
        status_code = 200
        content = b'not-json'
        url = "https://api.cartola.globo.com/mercado/status"

        def json(self) -> dict[str, object]:
            raise ValueError("bad json")

    monkeypatch.setattr("cartola.backtesting.market_capture.requests.get", lambda url, timeout: Response())

    with pytest.raises(ValueError, match="not valid JSON"):
        fetch_cartola_json("https://api.cartola.globo.com/mercado/status", 12.0)
```

- [ ] **Step 2: Run fetch tests to verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_market_capture.py::test_fetch_cartola_json_records_response_metadata src/tests/backtesting/test_market_capture.py::test_fetch_cartola_json_rejects_non_200 src/tests/backtesting/test_market_capture.py::test_fetch_cartola_json_rejects_invalid_json -q
```

Expected: FAIL because `fetch_cartola_json` is not implemented.

- [ ] **Step 3: Implement fetch and top-level validation helpers**

Append to `market_capture.py`:

```python
def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_cartola_json(url: str, timeout_seconds: float) -> CapturedJsonResponse:
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


def _validate_config_year(config: MarketCaptureConfig) -> int:
    current_year = _resolved_current_year(config)
    if config.season != current_year:
        raise ValueError(f"season {config.season} must equal current_year {current_year}")
    return current_year


def _validate_open_market(status_payload: dict[str, Any]) -> int:
    status_mercado = _int_field(status_payload, "status_mercado")
    if status_mercado != 1:
        rodada_atual = status_payload.get("rodada_atual")
        raise ValueError(f"Cartola market is not open: rodada_atual={rodada_atual} status_mercado {status_mercado}")
    return status_mercado
```

- [ ] **Step 4: Add metadata and temporary-file helpers**

Append:

```python
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
```

- [ ] **Step 5: Run fetch tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_market_capture.py::test_fetch_cartola_json_records_response_metadata src/tests/backtesting/test_market_capture.py::test_fetch_cartola_json_rejects_non_200 src/tests/backtesting/test_market_capture.py::test_fetch_cartola_json_rejects_invalid_json -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/market_capture.py src/tests/backtesting/test_market_capture.py
git commit -m "feat: add Cartola market fetch helpers"
```

---

### Task 4: Implement Publication Safety, Force Rules, And Auto Idempotence

**Files:**
- Modify: `src/cartola/backtesting/market_capture.py`
- Modify: `src/tests/backtesting/test_market_capture.py`

- [ ] **Step 1: Add publication tests**

First update the import block in `src/tests/backtesting/test_market_capture.py`:

```python
from cartola.backtesting.market_capture import (
    CAPTURE_VERSION,
    MarketCaptureConfig,
    build_live_market_frame,
    capture_market_round,
    deadline_metadata,
)
```

Then append:

```python
def _captured(payload: dict[str, object], url: str, *, status_code: int = 200) -> object:
    from cartola.backtesting.market_capture import CapturedJsonResponse

    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    return CapturedJsonResponse(
        payload=payload,
        status_code=status_code,
        final_url=url,
        body_sha256=__import__("hashlib").sha256(body).hexdigest(),
    )


def _fetch_pair(status_payload: dict[str, object] | None = None, market_payload: dict[str, object] | None = None):
    status_payload = _status_payload() if status_payload is None else status_payload
    market_payload = _market_payload() if market_payload is None else market_payload

    def fetch(url: str, timeout: float) -> object:
        if url.endswith("/mercado/status"):
            return _captured(status_payload, url)
        if url.endswith("/atletas/mercado"):
            return _captured(market_payload, url)
        raise AssertionError(f"unexpected URL {url}")

    return fetch


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
    assert metadata["csv_sha256"]
    loaded = load_round_file(result.csv_path)
    assert loaded["rodada"].tolist() == [14, 14]
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


def test_capture_auto_reuses_existing_valid_capture(tmp_path: Path) -> None:
    config = MarketCaptureConfig(season=2026, target_round=14, current_year=2026, project_root=tmp_path)
    capture_market_round(config, fetch=_fetch_pair(), now=lambda: datetime(2026, 4, 29, 12, 0, tzinfo=UTC))

    result = capture_market_round(
        MarketCaptureConfig(season=2026, auto=True, current_year=2026, project_root=tmp_path),
        fetch=_fetch_pair(),
    )

    assert result.reused_existing is True
```

- [ ] **Step 2: Implement previous-capture validation**

Append to `market_capture.py`:

```python
def _validate_previous_capture(final_csv: Path, final_metadata: Path, *, config: MarketCaptureConfig, target_round: int) -> None:
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
```

- [ ] **Step 3: Implement temp validation and publication**

Append:

```python
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
    backup_metadata = final_metadata.with_name(f"{final_metadata.name}.bak-{uuid.uuid4().hex}") if final_metadata.exists() else None
    try:
        if backup_csv is not None:
            final_csv.replace(backup_csv)
        if backup_metadata is not None:
            final_metadata.replace(backup_metadata)
        temp_metadata.replace(final_metadata)
        temp_csv.replace(final_csv)
    except Exception:
        if final_csv.exists():
            final_csv.unlink()
        if final_metadata.exists():
            final_metadata.unlink()
        if backup_metadata is not None and backup_metadata.exists():
            backup_metadata.replace(final_metadata)
        if backup_csv is not None and backup_csv.exists():
            backup_csv.replace(final_csv)
        raise
    finally:
        if backup_csv is not None and backup_csv.exists():
            backup_csv.unlink()
        if backup_metadata is not None and backup_metadata.exists():
            backup_metadata.unlink()
```

- [ ] **Step 4: Implement `capture_market_round()`**

Append:

```python
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
    temp_csv = temp_metadata = None
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
```

- [ ] **Step 5: Run publication tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_market_capture.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/market_capture.py src/tests/backtesting/test_market_capture.py
git commit -m "feat: safely publish live market captures"
```

---

### Task 5: Add CLI Wrapper

**Files:**
- Create: `scripts/capture_market_round.py`
- Create: `src/tests/backtesting/test_capture_market_round_cli.py`

- [ ] **Step 1: Add CLI tests**

Create `src/tests/backtesting/test_capture_market_round_cli.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

import scripts.capture_market_round as cli
from cartola.backtesting.market_capture import MarketCaptureResult


def test_parse_args_requires_target_or_auto() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--current-year", "2026"])


def test_parse_args_rejects_target_and_auto_together() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--target-round", "14", "--auto", "--current-year", "2026"])


def test_main_prints_capture_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_capture(config):
        assert config.season == 2026
        assert config.target_round == 14
        assert config.current_year == 2026
        assert config.force is False
        return MarketCaptureResult(
            csv_path=Path("data/01_raw/2026/rodada-14.csv"),
            metadata_path=Path("data/01_raw/2026/rodada-14.capture.json"),
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        )

    monkeypatch.setattr(cli, "capture_market_round", fake_capture)

    exit_code = cli.main(["--season", "2026", "--target-round", "14", "--current-year", "2026"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Captured live market round: data/01_raw/2026/rodada-14.csv" in captured.out
    assert "athletes=747" in captured.out
    assert "status_mercado=1" in captured.out
    assert "deadline_timestamp=1777748340" in captured.out
```

- [ ] **Step 2: Run CLI tests to verify failure**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_capture_market_round_cli.py -q
```

Expected: FAIL because `scripts/capture_market_round.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/capture_market_round.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cartola.backtesting.market_capture import MarketCaptureConfig, capture_market_round


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the open Cartola market round for live recommendations.")
    parser.add_argument("--season", type=_positive_int, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-round", type=_positive_int, default=None)
    target.add_argument("--auto", action="store_true")
    parser.add_argument("--current-year", type=_positive_int, default=None)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = capture_market_round(
        MarketCaptureConfig(
            season=args.season,
            target_round=args.target_round,
            auto=args.auto,
            force=args.force,
            current_year=args.current_year,
            project_root=args.project_root,
        )
    )
    reused = " reused_existing=true" if result.reused_existing else ""
    print(f"Captured live market round: {result.csv_path}{reused}")
    print(
        "metadata="
        f"{result.metadata_path} "
        f"target_round={result.target_round} "
        f"athletes={result.athlete_count} "
        f"status_mercado={result.status_mercado} "
        f"deadline_parse_status={result.deadline_parse_status} "
        f"deadline_timestamp={result.deadline_timestamp}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_capture_market_round_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/capture_market_round.py src/tests/backtesting/test_capture_market_round_cli.py
git commit -m "feat: add live market capture cli"
```

---

### Task 6: Add Recommendation Compatibility Test And Docs

**Files:**
- Modify: `src/tests/backtesting/test_market_capture.py`
- Modify: `README.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Add live recommendation compatibility test**

Append to `src/tests/backtesting/test_market_capture.py`:

```python
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
```

- [ ] **Step 2: Run compatibility test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_market_capture.py::test_captured_round_can_feed_live_recommendation -q
```

Expected: PASS.

- [ ] **Step 3: Update README live recommendation section**

In `README.md`, in the recommendation section before the `recommend_squad.py --mode live` command, add:

```markdown
Antes de gerar uma recomendação live, capture a rodada aberta do mercado:

```bash
uv run --frozen python scripts/capture_market_round.py \
  --season 2026 \
  --auto \
  --current-year 2026
```

Esse comando usa `mercado/status.rodada_atual` para nomear `data/01_raw/{season}/rodada-{rodada_atual}.csv`, sanitiza pontuação/scouts da rodada alvo e se recusa a sobrescrever dados existentes que não sejam uma captura live anterior.
```

- [ ] **Step 4: Update roadmap delivered list**

In `roadmap.md`, add this delivered bullet under "Single-round squad recommendation workflow" or immediately after it:

```markdown
- Live market round capture:
  - `scripts/capture_market_round.py` writes the open market round CSV for live recommendations,
  - validates current-year/open-market scope,
  - sanitizes target-round outcome fields,
  - publishes CSV and `.capture.json` with safe overwrite rules.
```

Then update the roadmap numbered list by marking live market capture as delivered and making the next item "Use capture plus recommendation for the next 2026 open round."

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_market_capture.py src/tests/backtesting/test_capture_market_round_cli.py src/tests/backtesting/test_recommendation.py src/tests/backtesting/test_recommend_squad_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tests/backtesting/test_market_capture.py README.md roadmap.md
git commit -m "docs: document live market capture workflow"
```

---

### Task 7: Full Verification And Real Open-Market Smoke

**Files:**
- No source changes expected.

- [ ] **Step 1: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest all pass.

- [ ] **Step 2: Run real capture smoke only if market is open**

Run:

```bash
uv run --frozen python scripts/capture_market_round.py \
  --season 2026 \
  --auto \
  --current-year 2026
```

Expected when Cartola market is open: command prints `Captured live market round:` and writes `data/01_raw/2026/rodada-{rodada_atual}.csv` plus `.capture.json`.

Expected when Cartola market is closed: command exits non-zero with a clear `status_mercado` message. This is acceptable and should be reported, not patched around.

- [ ] **Step 3: If capture succeeds, run live recommendation smoke**

Run with the captured round number:

```bash
uv run --frozen python scripts/recommend_squad.py \
  --season 2026 \
  --target-round 14 \
  --mode live \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

Expected: Rich success panel and recommendation outputs under `data/08_reporting/recommendations/2026/round-14/live/`.

- [ ] **Step 4: Review generated data before commit**

Run:

```bash
git status --short
```

If real smoke generated `data/01_raw/2026/rodada-14.csv` or `.capture.json`, decide explicitly whether to commit those live data artifacts. For normal feature implementation, do not commit live generated data unless the user asks.

- [ ] **Step 5: Commit any final docs/test-only cleanup**

If no cleanup is needed, skip this commit. If cleanup is needed:

```bash
git add README.md roadmap.md src/tests/backtesting/test_market_capture.py src/tests/backtesting/test_capture_market_round_cli.py
git commit -m "chore: finalize live market capture workflow"
```

---

## Self-Review Checklist

- Spec current-year validation is covered by Task 3.
- Exact output CSV schema is covered by Task 2 `RAW_OUTPUT_COLUMNS` and loader tests.
- Required Cartola payload fields and club join behavior are covered by Task 2 tests.
- HTTP non-200 and invalid JSON are implemented in Task 3; add focused tests if coverage reports miss these branches.
- Metadata response hashes, final URLs, status codes, CSV hash, deadline parse status, and capture version are covered by Tasks 3 and 4.
- Validate-before-publish and temp directory cleanup are covered by Task 4.
- Overwrite rules and `--force` safety are covered by Task 4.
- CLI shape and mutually exclusive target selection are covered by Task 5.
- Recommendation compatibility is covered by Task 6.
- Full repo verification is covered by Task 7.
