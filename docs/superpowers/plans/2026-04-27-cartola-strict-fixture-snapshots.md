# Cartola Strict Fixture Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict no-leakage fixture snapshot path where fixture features are accepted only from pre-lock Cartola API snapshots with validated provenance.

**Architecture:** Add a snapshot capture module for Cartola API payloads, a strict fixture module for manifest validation/generation/loading, and runner/CLI mode wiring. Keep current reconstructed 2025 fixtures as explicit exploratory data and make `fixture_mode="none"` the safe default.

**Tech Stack:** Python 3.13, pandas, requests, dataclasses, pathlib, pytest, Ruff, ty, Bandit, uv.

---

## File Structure

- Create `src/cartola/backtesting/fixture_snapshots.py`
  - Cartola API capture source.
  - RFC 7231 HTTP `Date` parsing and clock-skew validation.
  - Atomic snapshot directory writes.
  - Frozen-payload parsing helpers.

- Create `src/cartola/backtesting/strict_fixtures.py`
  - SHA-256 helpers.
  - Strict manifest validation.
  - Strict snapshot discovery.
  - Canonical fixture generation from snapshots.
  - Strict fixture loading.

- Modify `src/cartola/backtesting/config.py`
  - Add `FixtureMode` and `StrictAlignmentPolicy` literals.
  - Add `fixture_mode` and `strict_alignment_policy` to `BacktestConfig`.

- Modify `src/cartola/backtesting/data.py`
  - Keep exploratory `load_fixtures`.
  - Add no strict logic here except shared fixture normalization if needed.

- Modify `src/cartola/backtesting/runner.py`
  - Load fixtures based on mode.
  - Validate strict manifests and alignment scope.
  - Write `run_metadata.json`.
  - Preserve neutral fixture columns for `fixture_mode="none"`.

- Modify `src/cartola/backtesting/cli.py`
  - Add `--fixture-mode none|exploratory|strict`.
  - Add `--strict-alignment-policy fail|exclude_round`.
  - Print exploratory warning.

- Create `scripts/capture_fixture_snapshot.py`
  - CLI wrapper around strict capture.

- Create `scripts/generate_strict_fixtures.py`
  - CLI wrapper around strict fixture generation.

- Create `src/tests/backtesting/test_fixture_snapshots.py`
  - Capture, parser, HTTP date, and atomic write tests.

- Create `src/tests/backtesting/test_strict_fixtures.py`
  - Manifest, generation, strict loading tests.

- Modify `src/tests/backtesting/test_config.py`
- Modify `src/tests/backtesting/test_cli.py`
- Modify `src/tests/backtesting/test_runner.py`

---

## Shared Test Fixtures

Use these frozen payloads in `test_fixture_snapshots.py` and `test_strict_fixtures.py`.

```python
FROZEN_FIXTURE_PAYLOAD = {
    "rodada": 12,
    "clubes": {},
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

HTTP_DATE = "Mon, 01 Jun 2026 18:00:00 GMT"
```

---

### Task 1: Backtest Config Modes And Metadata Skeleton

**Files:**
- Modify: `src/cartola/backtesting/config.py`
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/cartola/backtesting/cli.py`
- Test: `src/tests/backtesting/test_config.py`
- Test: `src/tests/backtesting/test_cli.py`
- Test: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write failing config tests**

Append to `src/tests/backtesting/test_config.py`:

```python
def test_backtest_config_defaults_to_no_fixture_mode() -> None:
    from cartola.backtesting.config import BacktestConfig

    config = BacktestConfig()

    assert config.fixture_mode == "none"
    assert config.strict_alignment_policy == "fail"


def test_backtest_config_accepts_fixture_modes() -> None:
    from cartola.backtesting.config import BacktestConfig

    assert BacktestConfig(fixture_mode="exploratory").fixture_mode == "exploratory"
    assert BacktestConfig(fixture_mode="strict").fixture_mode == "strict"
    assert BacktestConfig(strict_alignment_policy="exclude_round").strict_alignment_policy == "exclude_round"
```

- [ ] **Step 2: Run config tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_config.py::test_backtest_config_defaults_to_no_fixture_mode src/tests/backtesting/test_config.py::test_backtest_config_accepts_fixture_modes -q
```

Expected: FAIL with `AttributeError` or dataclass constructor errors because the fields do not exist yet.

- [ ] **Step 3: Implement config fields**

Modify `src/cartola/backtesting/config.py` imports:

```python
from typing import Literal, Mapping
```

Add near constants:

```python
FixtureMode = Literal["none", "exploratory", "strict"]
StrictAlignmentPolicy = Literal["fail", "exclude_round"]
```

Add to `BacktestConfig`:

```python
    fixture_mode: FixtureMode = "none"
    strict_alignment_policy: StrictAlignmentPolicy = "fail"
```

- [ ] **Step 4: Run config tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_config.py::test_backtest_config_defaults_to_no_fixture_mode src/tests/backtesting/test_config.py::test_backtest_config_accepts_fixture_modes -q
```

Expected: PASS.

- [ ] **Step 5: Write failing CLI parser tests**

Append to `src/tests/backtesting/test_cli.py`:

```python
def test_cli_parses_fixture_mode_and_alignment_policy() -> None:
    from cartola.backtesting.cli import parse_args

    args = parse_args(
        [
            "--season",
            "2026",
            "--fixture-mode",
            "strict",
            "--strict-alignment-policy",
            "exclude_round",
        ]
    )

    assert args.fixture_mode == "strict"
    assert args.strict_alignment_policy == "exclude_round"
```

- [ ] **Step 6: Run CLI parser test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli.py::test_cli_parses_fixture_mode_and_alignment_policy -q
```

Expected: FAIL because arguments are not defined.

- [ ] **Step 7: Add CLI args**

Modify `src/cartola/backtesting/cli.py` in `parse_args`:

```python
    parser.add_argument("--fixture-mode", choices=("none", "exploratory", "strict"), default="none")
    parser.add_argument("--strict-alignment-policy", choices=("fail", "exclude_round"), default="fail")
```

Modify `BacktestConfig(...)` construction:

```python
        fixture_mode=args.fixture_mode,
        strict_alignment_policy=args.strict_alignment_policy,
```

- [ ] **Step 8: Run CLI parser test and verify it passes**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli.py::test_cli_parses_fixture_mode_and_alignment_policy -q
```

Expected: PASS.

- [ ] **Step 9: Write failing metadata test for fixture_mode none**

Append to `src/tests/backtesting/test_runner.py`:

```python
def test_run_backtest_writes_metadata_for_no_fixture_mode(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    run_backtest(config, season_df=season_df)

    metadata = pd.read_json(tmp_path / "data/08_reporting/backtests/2025/run_metadata.json", typ="series").to_dict()
    assert metadata["fixture_mode"] == "none"
    assert metadata["fixture_source_directory"] is None
    assert metadata["fixture_manifest_paths"] == []
    assert metadata["fixture_manifest_sha256"] == {}
    assert metadata["warnings"] == []
```

- [ ] **Step 10: Run metadata test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_run_backtest_writes_metadata_for_no_fixture_mode -q
```

Expected: FAIL because `run_metadata.json` is not written.

- [ ] **Step 11: Implement minimal run metadata for no fixture mode**

Modify `src/cartola/backtesting/runner.py` imports:

```python
import json
```

Add dataclass:

```python
@dataclass(frozen=True)
class BacktestMetadata:
    season: int
    start_round: int
    max_round: int
    fixture_mode: str
    strict_alignment_policy: str
    fixture_source_directory: str | None
    fixture_manifest_paths: list[str]
    fixture_manifest_sha256: dict[str, str]
    generator_versions: list[str]
    excluded_rounds: list[int]
    warnings: list[str]
```

Add field to `BacktestResult`:

```python
    metadata: BacktestMetadata
```

After `max_round = _max_round(data)`, add:

```python
    metadata = BacktestMetadata(
        season=config.season,
        start_round=config.start_round,
        max_round=max_round,
        fixture_mode=config.fixture_mode,
        strict_alignment_policy=config.strict_alignment_policy,
        fixture_source_directory=None,
        fixture_manifest_paths=[],
        fixture_manifest_sha256={},
        generator_versions=[],
        excluded_rounds=[],
        warnings=[],
    )
```

Pass `metadata` to `_write_outputs` and return it in `BacktestResult`.

Update `_write_outputs` signature:

```python
    metadata: BacktestMetadata,
) -> None:
```

At the end of `_write_outputs`:

```python
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 12: Run metadata test and verify it passes**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_run_backtest_writes_metadata_for_no_fixture_mode -q
```

Expected: PASS.

- [ ] **Step 13: Run task test set**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_config.py src/tests/backtesting/test_cli.py src/tests/backtesting/test_runner.py -q
```

Expected: PASS.

- [ ] **Step 14: Commit Task 1**

Run:

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/cli.py src/cartola/backtesting/runner.py src/tests/backtesting/test_config.py src/tests/backtesting/test_cli.py src/tests/backtesting/test_runner.py
git commit -m "feat: add fixture modes and backtest metadata"
```

---

### Task 2: Cartola Snapshot Capture

**Files:**
- Create: `src/cartola/backtesting/fixture_snapshots.py`
- Create: `src/tests/backtesting/test_fixture_snapshots.py`
- Create: `scripts/capture_fixture_snapshot.py`

- [ ] **Step 1: Write failing parser tests**

Create `src/tests/backtesting/test_fixture_snapshots.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cartola.backtesting.fixture_snapshots import (
    FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS,
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


def test_cartola_deadline_at_uses_fechamento_timestamp() -> None:
    deadline = cartola_deadline_at(FROZEN_DEADLINE_PAYLOAD, season=2026, round_number=12)

    assert deadline == datetime.fromtimestamp(1780340340, tz=UTC)
    assert FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS == 300


def test_cartola_deadline_at_rejects_wrong_round() -> None:
    with pytest.raises(ValueError, match="rodada_atual"):
        cartola_deadline_at(FROZEN_DEADLINE_PAYLOAD, season=2026, round_number=13)
```

- [ ] **Step 2: Run parser tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixture_snapshots.py -q
```

Expected: FAIL because `cartola.backtesting.fixture_snapshots` does not exist.

- [ ] **Step 3: Implement parser helpers**

Create `src/cartola/backtesting/fixture_snapshots.py`:

```python
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import requests

CARTOLA_FIXTURE_ENDPOINT = "https://api.cartola.globo.com/partidas/{round_number}"
CARTOLA_DEADLINE_ENDPOINT = "https://api.cartola.globo.com/mercado/status"
FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS = 300
CAPTURE_VERSION = "fixture_capture_v1"


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


def parse_http_date_utc(value: str) -> datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid HTTP Date header: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"HTTP Date header must include GMT timezone: {value!r}")
    if not value.endswith(" GMT"):
        raise ValueError(f"HTTP Date header must be RFC 7231 GMT format: {value!r}")
    return parsed.astimezone(UTC)


def cartola_fixture_rows(payload: dict[str, Any], *, round_number: int) -> list[dict[str, Any]]:
    if int(payload.get("rodada", -1)) != int(round_number):
        raise ValueError(f"Fixture payload rodada does not match requested round {round_number}")
    rows: list[dict[str, Any]] = []
    for partida in payload.get("partidas", []):
        if partida.get("valida") is not True:
            continue
        try:
            home_id = int(partida["clube_casa_id"])
            away_id = int(partida["clube_visitante_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Cartola club IDs must be present and parse as integers") from exc
        date_value = pd_datetime_date(str(partida["partida_data"]))
        rows.append(
            {
                "rodada": int(round_number),
                "id_clube_home": home_id,
                "id_clube_away": away_id,
                "data": date_value,
            }
        )
    return rows


def pd_datetime_date(value: str) -> str:
    from pandas import to_datetime

    return to_datetime(value, errors="raise").date().isoformat()


def cartola_deadline_at(payload: dict[str, Any], *, season: int, round_number: int) -> datetime:
    if int(payload.get("temporada", -1)) != int(season):
        raise ValueError(f"Deadline payload temporada does not match requested season {season}")
    if int(payload.get("rodada_atual", -1)) != int(round_number):
        raise ValueError(f"Deadline payload rodada_atual does not match requested round {round_number}")
    fechamento = payload.get("fechamento")
    if not isinstance(fechamento, dict):
        raise ValueError("Deadline payload missing fechamento object")
    try:
        timestamp = int(fechamento["timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Deadline payload missing numeric fechamento.timestamp") from exc
    return datetime.fromtimestamp(timestamp, tz=UTC)
```

- [ ] **Step 4: Run parser tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixture_snapshots.py -q
```

Expected: PASS for parser tests.

- [ ] **Step 5: Add failing capture atomicity tests**

Append to `src/tests/backtesting/test_fixture_snapshots.py`:

```python
from pathlib import Path

from cartola.backtesting.fixture_snapshots import capture_cartola_snapshot


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
```

- [ ] **Step 6: Run capture tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixture_snapshots.py::test_capture_cartola_snapshot_writes_atomic_snapshot src/tests/backtesting/test_fixture_snapshots.py::test_capture_cartola_snapshot_failure_leaves_no_valid_directory -q
```

Expected: FAIL because `capture_cartola_snapshot` is not implemented.

- [ ] **Step 7: Implement capture**

Append to `src/cartola/backtesting/fixture_snapshots.py`:

```python
def capture_cartola_snapshot(
    *,
    project_root: str | Path,
    season: int,
    round_number: int,
    source: str = "cartola_api",
    fetch: Callable[[str], Any] | None = None,
    now: Callable[[], datetime] | None = None,
) -> CaptureResult:
    if source != "cartola_api":
        raise ValueError(f"Unsupported fixture snapshot source: {source}")
    clock = now or (lambda: datetime.now(UTC))
    fetcher = fetch or _requests_fetch
    capture_started_at = clock()
    fixture_url = CARTOLA_FIXTURE_ENDPOINT.format(round_number=round_number)
    deadline_url = CARTOLA_DEADLINE_ENDPOINT
    fixture_response = _fetch_json(fetcher, fixture_url)
    deadline_response = _fetch_json(fetcher, deadline_url)
    captured_at = clock()
    deadline_at = cartola_deadline_at(deadline_response.payload, season=season, round_number=round_number)
    cartola_fixture_rows(fixture_response.payload, round_number=round_number)
    _validate_clock_skew(captured_at, fixture_response.http_date_utc, deadline_response.http_date_utc)
    if captured_at >= deadline_at:
        raise ValueError("Snapshot captured_at_utc must be strictly before deadline_at_utc")

    round_dir = Path(project_root) / "data" / "01_raw" / "fixtures_snapshots" / str(season) / f"rodada-{round_number}"
    final_dir = round_dir / f"captured_at={_dir_timestamp(captured_at)}"
    if final_dir.exists():
        raise FileExistsError(f"Snapshot directory already exists: {final_dir}")
    tmp_dir = round_dir / f".tmp-{uuid.uuid4().hex}"
    try:
        tmp_dir.mkdir(parents=True)
        _write_json(tmp_dir / "fixtures.json", fixture_response.payload)
        _write_json(tmp_dir / "deadline.json", deadline_response.payload)
        _write_json(
            tmp_dir / "capture.json",
            {
                "capture_started_at_utc": _iso_utc(capture_started_at),
                "captured_at_utc": _iso_utc(captured_at),
                "fixture_http_date_header": fixture_response.http_date_header,
                "fixture_http_date_utc": _iso_utc(fixture_response.http_date_utc),
                "fixture_http_status": fixture_response.status_code,
                "fixture_final_url": fixture_response.final_url,
                "deadline_http_date_header": deadline_response.http_date_header,
                "deadline_http_date_utc": _iso_utc(deadline_response.http_date_utc),
                "deadline_http_status": deadline_response.status_code,
                "deadline_final_url": deadline_response.final_url,
                "clock_skew_tolerance_seconds": FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS,
                "max_observed_clock_skew_seconds": max(
                    abs((captured_at - fixture_response.http_date_utc).total_seconds()),
                    abs((captured_at - deadline_response.http_date_utc).total_seconds()),
                ),
                "source": source,
                "season": int(season),
                "rodada": int(round_number),
                "capture_version": CAPTURE_VERSION,
            },
        )
        os.replace(tmp_dir, final_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return CaptureResult(capture_dir=final_dir, captured_at_utc=captured_at)


def _requests_fetch(url: str) -> requests.Response:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response


def _fetch_json(fetcher: Callable[[str], Any], url: str) -> CapturedResponse:
    response = fetcher(url)
    response.raise_for_status()
    date_header = response.headers.get("Date")
    if not date_header:
        raise ValueError("Missing HTTP Date header")
    return CapturedResponse(
        payload=dict(response.json()),
        http_date_header=str(date_header),
        http_date_utc=parse_http_date_utc(str(date_header)),
        status_code=int(response.status_code),
        final_url=str(response.url),
    )


def _validate_clock_skew(captured_at: datetime, fixture_date: datetime, deadline_date: datetime) -> None:
    for value in (captured_at, fixture_date, deadline_date):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Clock evidence must be timezone-aware")
    max_skew = max(abs((captured_at - fixture_date).total_seconds()), abs((captured_at - deadline_date).total_seconds()))
    if max_skew > FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS:
        raise ValueError("Clock skew exceeds strict capture tolerance")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _dir_timestamp(value: datetime) -> str:
    return _iso_utc(value).replace(":", "-")
```

- [ ] **Step 8: Run capture tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixture_snapshots.py -q
```

Expected: PASS.

- [ ] **Step 9: Add capture CLI script**

Create `scripts/capture_fixture_snapshot.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cartola.backtesting.fixture_snapshots import capture_cartola_snapshot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture strict pre-lock Cartola fixture snapshot.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--source", choices=("cartola_api",), default="cartola_api")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = capture_cartola_snapshot(
        project_root=args.project_root,
        season=args.season,
        round_number=args.round_number,
        source=args.source,
    )
    print(f"Captured fixture snapshot: {result.capture_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10: Run task tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixture_snapshots.py -q
uv run --frozen ty check
```

Expected: PASS.

- [ ] **Step 11: Commit Task 2**

Run:

```bash
git add src/cartola/backtesting/fixture_snapshots.py src/tests/backtesting/test_fixture_snapshots.py scripts/capture_fixture_snapshot.py
git commit -m "feat: capture strict Cartola fixture snapshots"
```

---

### Task 3: Strict Fixture Generation And Manifest Validation

**Files:**
- Create: `src/cartola/backtesting/strict_fixtures.py`
- Create: `src/tests/backtesting/test_strict_fixtures.py`
- Create: `scripts/generate_strict_fixtures.py`

- [ ] **Step 1: Write failing manifest validation tests**

Create `src/tests/backtesting/test_strict_fixtures.py`:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from cartola.backtesting.strict_fixtures import (
    generate_strict_fixture,
    load_strict_fixtures,
    sha256_file,
    validate_strict_manifest,
)


def test_validate_strict_manifest_rejects_missing_manifest(tmp_path: Path) -> None:
    fixture_file = tmp_path / "data/01_raw/fixtures_strict/2026/partidas-12.csv"
    fixture_file.parent.mkdir(parents=True)
    fixture_file.write_text("rodada,id_clube_home,id_clube_away,data\n12,262,277,2026-06-01\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        validate_strict_manifest(
            fixture_file,
            season=2026,
            round_number=12,
            source="cartola_api",
            project_root=tmp_path,
        )


def test_validate_strict_manifest_rejects_edited_fixture_hash(tmp_path: Path) -> None:
    fixture_file, manifest_file = _write_valid_strict_fixture(tmp_path)
    fixture_file.write_text("rodada,id_clube_home,id_clube_away,data\n12,262,277,2026-06-02\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical_fixture_sha256"):
        validate_strict_manifest(
            fixture_file,
            season=2026,
            round_number=12,
            source="cartola_api",
            project_root=tmp_path,
        )


def test_validate_strict_manifest_rejects_round_mismatch(tmp_path: Path) -> None:
    fixture_file, manifest_file = _write_valid_strict_fixture(tmp_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["rodada"] = 13
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="rodada"):
        validate_strict_manifest(
            fixture_file,
            season=2026,
            round_number=12,
            source="cartola_api",
            project_root=tmp_path,
        )
```

Include helper in the same test file:

```python
def _write_valid_strict_fixture(root: Path) -> tuple[Path, Path]:
    capture_dir = root / "data/01_raw/fixtures_snapshots/2026/rodada-12/captured_at=2026-06-01T18-00-00Z"
    capture_dir.mkdir(parents=True)
    capture = {
        "captured_at_utc": "2026-06-01T18:00:00Z",
        "source": "cartola_api",
        "season": 2026,
        "rodada": 12,
    }
    fixtures = {"rodada": 12, "partidas": []}
    deadline = {"temporada": 2026, "rodada_atual": 12, "fechamento": {"timestamp": 1780340340}}
    for name, payload in {"capture.json": capture, "fixtures.json": fixtures, "deadline.json": deadline}.items():
        (capture_dir / name).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    fixture_dir = root / "data/01_raw/fixtures_strict/2026"
    fixture_dir.mkdir(parents=True)
    fixture_file = fixture_dir / "partidas-12.csv"
    fixture_file.write_text("rodada,id_clube_home,id_clube_away,data\n12,262,277,2026-06-01\n", encoding="utf-8")
    manifest_file = fixture_dir / "partidas-12.manifest.json"
    manifest = {
        "mode": "strict",
        "season": 2026,
        "rodada": 12,
        "source": "cartola_api",
        "capture_metadata_path": str(capture_dir / "capture.json"),
        "capture_metadata_sha256": sha256_file(capture_dir / "capture.json"),
        "fixture_snapshot_path": str(capture_dir / "fixtures.json"),
        "fixture_snapshot_sha256": sha256_file(capture_dir / "fixtures.json"),
        "deadline_snapshot_path": str(capture_dir / "deadline.json"),
        "deadline_snapshot_sha256": sha256_file(capture_dir / "deadline.json"),
        "captured_at_utc": "2026-06-01T18:00:00Z",
        "deadline_at_utc": "2026-06-01T18:59:00Z",
        "deadline_source": "cartola_api_market_status",
        "fixture_endpoint": "https://api.cartola.globo.com/partidas/12",
        "fixture_final_url": "https://api.cartola.globo.com/partidas/12",
        "deadline_endpoint": "https://api.cartola.globo.com/mercado/status",
        "deadline_final_url": "https://api.cartola.globo.com/mercado/status",
        "generator_version": "fixture_snapshot_v1",
        "club_mapping_path": None,
        "club_mapping_sha256": None,
        "club_id_allowlist_path": None,
        "club_id_allowlist_sha256": None,
        "canonical_fixture_path": str(fixture_file),
        "canonical_fixture_sha256": sha256_file(fixture_file),
    }
    manifest_file.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return fixture_file, manifest_file
```

- [ ] **Step 2: Run manifest tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_strict_fixtures.py -q
```

Expected: FAIL because `strict_fixtures.py` does not exist.

- [ ] **Step 3: Implement manifest validator and hash helpers**

Create `src/cartola/backtesting/strict_fixtures.py`:

```python
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from cartola.backtesting.data import normalize_fixture_frame
from cartola.backtesting.fixture_snapshots import cartola_deadline_at, cartola_fixture_rows

GENERATOR_VERSION = "fixture_snapshot_v1"


@dataclass(frozen=True)
class StrictFixtureLoadResult:
    fixtures: pd.DataFrame
    manifest_paths: list[str]
    manifest_sha256: dict[str, str]
    generator_versions: list[str]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_strict_manifest(
    fixture_file: str | Path,
    *,
    season: int,
    round_number: int,
    source: str,
    project_root: str | Path,
) -> dict[str, Any]:
    fixture_path = Path(fixture_file)
    manifest_path = fixture_path.with_name(f"{fixture_path.stem}.manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Strict fixture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "mode",
        "season",
        "rodada",
        "source",
        "capture_metadata_path",
        "capture_metadata_sha256",
        "fixture_snapshot_path",
        "fixture_snapshot_sha256",
        "deadline_snapshot_path",
        "deadline_snapshot_sha256",
        "captured_at_utc",
        "deadline_at_utc",
        "canonical_fixture_path",
        "canonical_fixture_sha256",
        "generator_version",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Missing strict manifest fields: {missing}")
    if manifest["mode"] != "strict":
        raise ValueError("Strict fixture manifest mode must be 'strict'")
    if int(manifest["season"]) != int(season):
        raise ValueError("Strict fixture manifest season mismatch")
    if int(manifest["rodada"]) != int(round_number):
        raise ValueError("Strict fixture manifest rodada mismatch")
    if manifest["source"] != source:
        raise ValueError("Strict fixture manifest source mismatch")

    root = Path(project_root).resolve()
    canonical = _resolve_under_root(manifest["canonical_fixture_path"], root)
    if canonical != fixture_path.resolve():
        raise ValueError("Strict manifest canonical_fixture_path does not point to loaded file")
    if sha256_file(canonical) != manifest["canonical_fixture_sha256"]:
        raise ValueError("Strict manifest canonical_fixture_sha256 mismatch")

    for path_key, hash_key in [
        ("capture_metadata_path", "capture_metadata_sha256"),
        ("fixture_snapshot_path", "fixture_snapshot_sha256"),
        ("deadline_snapshot_path", "deadline_snapshot_sha256"),
    ]:
        resolved = _resolve_under_root(manifest[path_key], root)
        if sha256_file(resolved) != manifest[hash_key]:
            raise ValueError(f"Strict manifest {hash_key} mismatch")

    captured = _parse_manifest_utc(manifest["captured_at_utc"], "captured_at_utc")
    deadline = _parse_manifest_utc(manifest["deadline_at_utc"], "deadline_at_utc")
    if captured >= deadline:
        raise ValueError("Strict manifest captured_at_utc must be before deadline_at_utc")
    return manifest


def _resolve_under_root(path: str, root: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Strict manifest path escapes project_root: {path}")
    return resolved


def _parse_manifest_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"Strict manifest {field_name} must be ISO-8601 UTC ending in Z")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"Strict manifest {field_name} must be UTC")
    return parsed.astimezone(UTC)
```

- [ ] **Step 4: Run manifest tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_strict_fixtures.py -q
```

Expected: PASS for current tests.

- [ ] **Step 5: Add failing generation tests**

Append to `src/tests/backtesting/test_strict_fixtures.py`:

```python
def test_generate_strict_fixture_uses_latest_valid_snapshot(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at="2026-06-01T17-00-00Z", home_id=262, away_id=277)
    _write_snapshot(tmp_path, captured_at="2026-06-01T18-00-00Z", home_id=275, away_id=284)

    fixture_file = generate_strict_fixture(
        project_root=tmp_path,
        season=2026,
        round_number=12,
        source="cartola_api",
    )

    frame = pd.read_csv(fixture_file)
    assert frame[["id_clube_home", "id_clube_away"]].to_dict("records") == [
        {"id_clube_home": 275, "id_clube_away": 284}
    ]
    assert fixture_file.with_name("partidas-12.manifest.json").exists()


def test_generate_strict_fixture_refuses_overwrite_without_force(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at="2026-06-01T18-00-00Z", home_id=262, away_id=277)
    generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")

    with pytest.raises(FileExistsError):
        generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")
```

Add helper:

```python
def _write_snapshot(root: Path, *, captured_at: str, home_id: int, away_id: int) -> Path:
    dirname = f"captured_at={captured_at.replace(':', '-')}"
    capture_dir = root / "data/01_raw/fixtures_snapshots/2026/rodada-12" / dirname
    capture_dir.mkdir(parents=True)
    fixtures = {
        "rodada": 12,
        "partidas": [
            {
                "clube_casa_id": home_id,
                "clube_visitante_id": away_id,
                "partida_data": "2026-06-01 19:00:00",
                "timestamp": 1780340400,
                "valida": True,
            }
        ],
    }
    deadline = {
        "temporada": 2026,
        "rodada_atual": 12,
        "fechamento": {"timestamp": 1780343940},
    }
    capture = {
        "captured_at_utc": captured_at,
        "source": "cartola_api",
        "season": 2026,
        "rodada": 12,
        "fixture_final_url": "https://api.cartola.globo.com/partidas/12",
        "deadline_final_url": "https://api.cartola.globo.com/mercado/status",
    }
    for name, payload in {"capture.json": capture, "fixtures.json": fixtures, "deadline.json": deadline}.items():
        (capture_dir / name).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return capture_dir
```

- [ ] **Step 6: Run generation tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_strict_fixtures.py::test_generate_strict_fixture_uses_latest_valid_snapshot src/tests/backtesting/test_strict_fixtures.py::test_generate_strict_fixture_refuses_overwrite_without_force -q
```

Expected: FAIL because generation is not implemented.

- [ ] **Step 7: Implement generation**

Append to `src/cartola/backtesting/strict_fixtures.py`:

```python
def generate_strict_fixture(
    *,
    project_root: str | Path,
    season: int,
    round_number: int,
    source: str,
    captured_at: str | None = None,
    force: bool = False,
) -> Path:
    if source != "cartola_api":
        raise ValueError(f"Unsupported strict fixture source: {source}")
    root = Path(project_root)
    capture_dir = _select_snapshot(root, season=season, round_number=round_number, captured_at=captured_at)
    capture = json.loads((capture_dir / "capture.json").read_text(encoding="utf-8"))
    fixtures = json.loads((capture_dir / "fixtures.json").read_text(encoding="utf-8"))
    deadline = json.loads((capture_dir / "deadline.json").read_text(encoding="utf-8"))
    deadline_at = cartola_deadline_at(deadline, season=season, round_number=round_number)
    rows = cartola_fixture_rows(fixtures, round_number=round_number)
    captured = _parse_manifest_utc(capture["captured_at_utc"], "captured_at_utc")
    if captured >= deadline_at:
        raise ValueError("Selected snapshot is not before deadline")

    fixture_dir = root / "data" / "01_raw" / "fixtures_strict" / str(season)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_file = fixture_dir / f"partidas-{round_number}.csv"
    manifest_file = fixture_dir / f"partidas-{round_number}.manifest.json"
    if not force and (fixture_file.exists() or manifest_file.exists()):
        raise FileExistsError(f"Strict fixture already exists for round {round_number}")

    frame = pd.DataFrame(rows, columns=["rodada", "id_clube_home", "id_clube_away", "data"])
    tmp_fixture = fixture_dir / f".partidas-{round_number}.csv.tmp"
    tmp_manifest = fixture_dir / f".partidas-{round_number}.manifest.json.tmp"
    frame.to_csv(tmp_fixture, index=False)
    manifest = {
        "mode": "strict",
        "season": int(season),
        "rodada": int(round_number),
        "source": source,
        "capture_metadata_path": str(capture_dir / "capture.json"),
        "capture_metadata_sha256": sha256_file(capture_dir / "capture.json"),
        "fixture_snapshot_path": str(capture_dir / "fixtures.json"),
        "fixture_snapshot_sha256": sha256_file(capture_dir / "fixtures.json"),
        "deadline_snapshot_path": str(capture_dir / "deadline.json"),
        "deadline_snapshot_sha256": sha256_file(capture_dir / "deadline.json"),
        "captured_at_utc": capture["captured_at_utc"],
        "deadline_at_utc": deadline_at.isoformat().replace("+00:00", "Z"),
        "deadline_source": "cartola_api_market_status",
        "fixture_endpoint": f"https://api.cartola.globo.com/partidas/{round_number}",
        "fixture_final_url": capture.get("fixture_final_url", f"https://api.cartola.globo.com/partidas/{round_number}"),
        "deadline_endpoint": "https://api.cartola.globo.com/mercado/status",
        "deadline_final_url": capture.get("deadline_final_url", "https://api.cartola.globo.com/mercado/status"),
        "generator_version": GENERATOR_VERSION,
        "club_mapping_path": None,
        "club_mapping_sha256": None,
        "club_id_allowlist_path": None,
        "club_id_allowlist_sha256": None,
        "canonical_fixture_path": str(fixture_file),
        "canonical_fixture_sha256": sha256_file(tmp_fixture),
    }
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_fixture, fixture_file)
    os.replace(tmp_manifest, manifest_file)
    return fixture_file


def _select_snapshot(root: Path, *, season: int, round_number: int, captured_at: str | None) -> Path:
    round_dir = root / "data" / "01_raw" / "fixtures_snapshots" / str(season) / f"rodada-{round_number}"
    captures = [path for path in round_dir.glob("captured_at=*") if path.is_dir()]
    if captured_at is not None:
        matches = [path for path in captures if path.name == f"captured_at={captured_at.replace(':', '-')}"]
        if len(matches) != 1:
            raise FileNotFoundError(f"Explicit snapshot not found: {captured_at}")
        return matches[0]
    parsed: list[tuple[datetime, Path]] = []
    seen: set[datetime] = set()
    for path in captures:
        capture = json.loads((path / "capture.json").read_text(encoding="utf-8"))
        captured = _parse_manifest_utc(capture["captured_at_utc"], "captured_at_utc")
        if captured in seen:
            raise ValueError("Duplicate captured_at_utc snapshot values")
        seen.add(captured)
        parsed.append((captured, path))
    if not parsed:
        raise FileNotFoundError(f"No strict snapshot found for season={season} round={round_number}")
    return sorted(parsed, key=lambda item: item[0])[-1][1]
```

- [ ] **Step 8: Run strict fixture tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_strict_fixtures.py -q
uv run --frozen ty check
```

Expected: PASS.

- [ ] **Step 9: Add generation CLI script**

Create `scripts/generate_strict_fixtures.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cartola.backtesting.strict_fixtures import generate_strict_fixture


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict Cartola fixture CSV from captured snapshot.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--source", choices=("cartola_api",), default="cartola_api")
    parser.add_argument("--captured-at")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fixture_file = generate_strict_fixture(
        project_root=args.project_root,
        season=args.season,
        round_number=args.round_number,
        source=args.source,
        captured_at=args.captured_at,
        force=args.force,
    )
    print(f"Generated strict fixture: {fixture_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10: Commit Task 3**

Run:

```bash
git add src/cartola/backtesting/strict_fixtures.py src/tests/backtesting/test_strict_fixtures.py scripts/generate_strict_fixtures.py
git commit -m "feat: generate and validate strict fixture manifests"
```

---

### Task 4: Strict Fixture Loader

**Files:**
- Modify: `src/cartola/backtesting/strict_fixtures.py`
- Modify: `src/tests/backtesting/test_strict_fixtures.py`

- [ ] **Step 1: Write failing strict loader tests**

Append to `src/tests/backtesting/test_strict_fixtures.py`:

```python
def test_load_strict_fixtures_validates_required_rounds(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, captured_at="2026-06-01T18-00-00Z", home_id=262, away_id=277)
    generate_strict_fixture(project_root=tmp_path, season=2026, round_number=12, source="cartola_api")

    result = load_strict_fixtures(
        season=2026,
        project_root=tmp_path,
        required_rounds=[12],
        source="cartola_api",
    )

    assert result.fixtures["rodada"].tolist() == [12]
    assert len(result.manifest_paths) == 1
    assert result.generator_versions == ["fixture_snapshot_v1"]


def test_load_strict_fixtures_rejects_missing_required_round(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="partidas-12"):
        load_strict_fixtures(
            season=2026,
            project_root=tmp_path,
            required_rounds=[12],
            source="cartola_api",
        )
```

- [ ] **Step 2: Run loader tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_strict_fixtures.py::test_load_strict_fixtures_validates_required_rounds src/tests/backtesting/test_strict_fixtures.py::test_load_strict_fixtures_rejects_missing_required_round -q
```

Expected: FAIL because `load_strict_fixtures` is not implemented.

- [ ] **Step 3: Implement strict loader**

Append to `src/cartola/backtesting/strict_fixtures.py`:

```python
def load_strict_fixtures(
    *,
    season: int,
    project_root: str | Path,
    required_rounds: list[int],
    source: str = "cartola_api",
) -> StrictFixtureLoadResult:
    root = Path(project_root)
    fixture_dir = root / "data" / "01_raw" / "fixtures_strict" / str(season)
    frames: list[pd.DataFrame] = []
    manifest_paths: list[str] = []
    manifest_hashes: dict[str, str] = {}
    versions: set[str] = set()
    for round_number in sorted(set(int(value) for value in required_rounds)):
        fixture_file = fixture_dir / f"partidas-{round_number}.csv"
        if not fixture_file.exists():
            raise FileNotFoundError(f"Missing strict fixture file: {fixture_file}")
        manifest = validate_strict_manifest(
            fixture_file,
            season=season,
            round_number=round_number,
            source=source,
            project_root=root,
        )
        frame = normalize_fixture_frame(pd.read_csv(fixture_file), source=fixture_file)
        frames.append(frame)
        manifest_file = fixture_file.with_name(f"{fixture_file.stem}.manifest.json")
        manifest_paths.append(str(manifest_file))
        manifest_hashes[manifest_file.name] = sha256_file(manifest_file)
        versions.add(str(manifest["generator_version"]))
    fixtures = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=pd.Index(["rodada", "id_clube_home", "id_clube_away", "data"])
    )
    return StrictFixtureLoadResult(
        fixtures=fixtures,
        manifest_paths=manifest_paths,
        manifest_sha256=manifest_hashes,
        generator_versions=sorted(versions),
    )
```

- [ ] **Step 4: Run strict loader tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_strict_fixtures.py -q
uv run --frozen ty check
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add src/cartola/backtesting/strict_fixtures.py src/tests/backtesting/test_strict_fixtures.py
git commit -m "feat: load manifest-validated strict fixtures"
```

---

### Task 5: Runner And CLI Fixture Modes

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/cartola/backtesting/cli.py`
- Modify: `src/tests/backtesting/test_runner.py`
- Modify: `src/tests/backtesting/test_cli.py`

- [ ] **Step 1: Write failing runner tests for default none and exploratory opt-in**

Modify existing `test_run_backtest_uses_fixture_files_when_available` in `src/tests/backtesting/test_runner.py` so `BacktestConfig` uses `fixture_mode="exploratory"`.

Add:

```python
def test_run_backtest_default_none_ignores_exploratory_fixture_files(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    _write_tiny_fixture_files(tmp_path, range(1, 6))
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)

    result = run_backtest(config, season_df=season_df)

    assert result.metadata.fixture_mode == "none"
    assert result.player_predictions["is_home"].eq(0).all()
```

Add:

```python
def test_run_backtest_exploratory_writes_warning_metadata(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    _write_tiny_fixture_files(tmp_path, range(1, 6))
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, fixture_mode="exploratory")

    result = run_backtest(config, season_df=season_df)

    assert result.metadata.fixture_mode == "exploratory"
    assert result.metadata.warnings
    assert "not strict no-leakage" in result.metadata.warnings[0]
```

- [ ] **Step 2: Run runner tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_run_backtest_default_none_ignores_exploratory_fixture_files src/tests/backtesting/test_runner.py::test_run_backtest_exploratory_writes_warning_metadata -q
```

Expected: FAIL because current runner auto-loads exploratory files regardless of mode.

- [ ] **Step 3: Implement fixture mode loading**

Modify `src/cartola/backtesting/runner.py` imports:

```python
from cartola.backtesting.strict_fixtures import StrictFixtureLoadResult, load_strict_fixtures
```

Replace `fixture_data = ...` line with:

```python
    fixture_load = _load_fixture_data(config, data, fixtures)
    fixture_data = fixture_load.fixtures
```

Add dataclass:

```python
@dataclass(frozen=True)
class FixtureLoadForRun:
    fixtures: pd.DataFrame | None
    source_directory: str | None
    manifest_paths: list[str]
    manifest_sha256: dict[str, str]
    generator_versions: list[str]
    warnings: list[str]
```

Add helper:

```python
EXPLORATORY_WARNING = (
    "Exploratory fixture mode is not strict no-leakage data; files may be reconstructed from post-round evidence."
)


def _load_fixture_data(
    config: BacktestConfig,
    season_df: pd.DataFrame,
    explicit_fixtures: pd.DataFrame | None,
) -> FixtureLoadForRun:
    if config.fixture_mode == "none":
        return FixtureLoadForRun(None, None, [], {}, [], [])
    if config.fixture_mode == "exploratory":
        fixtures = explicit_fixtures.copy() if explicit_fixtures is not None else load_fixtures(
            config.season, project_root=config.project_root
        )
        return FixtureLoadForRun(
            fixtures=fixtures,
            source_directory=str(config.project_root / "data" / "01_raw" / "fixtures" / str(config.season)),
            manifest_paths=[],
            manifest_sha256={},
            generator_versions=[],
            warnings=[EXPLORATORY_WARNING],
        )
    if config.fixture_mode == "strict":
        required_rounds = _strict_required_rounds(season_df)
        loaded = load_strict_fixtures(
            season=config.season,
            project_root=config.project_root,
            required_rounds=required_rounds,
            source="cartola_api",
        )
        return FixtureLoadForRun(
            fixtures=loaded.fixtures,
            source_directory=str(config.project_root / "data" / "01_raw" / "fixtures_strict" / str(config.season)),
            manifest_paths=loaded.manifest_paths,
            manifest_sha256=loaded.manifest_sha256,
            generator_versions=loaded.generator_versions,
            warnings=[],
        )
    raise ValueError(f"Unknown fixture_mode: {config.fixture_mode}")


def _strict_required_rounds(season_df: pd.DataFrame) -> list[int]:
    if season_df.empty:
        return []
    max_round = int(pd.to_numeric(season_df["rodada"], errors="raise").max())
    return list(range(1, max_round + 1))
```

Use `fixture_load` fields when constructing `BacktestMetadata`.

- [ ] **Step 4: Run runner mode tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Add failing strict mode loader integration test**

Append to `src/tests/backtesting/test_runner.py`:

```python
def test_run_backtest_strict_requires_manifest_validated_fixtures(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100, fixture_mode="strict")

    with pytest.raises(FileNotFoundError, match="partidas-1"):
        run_backtest(config, season_df=season_df)
```

- [ ] **Step 6: Run strict integration test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_run_backtest_strict_requires_manifest_validated_fixtures -q
```

Expected: PASS if Task 5 Step 3 is implemented.

- [ ] **Step 7: Update CLI stdout warning behavior**

Modify `src/cartola/backtesting/cli.py` after `run_backtest(config)`:

```python
    result = run_backtest(config)
    for warning in result.metadata.warnings:
        print(f"WARNING: {warning}")
```

- [ ] **Step 8: Add CLI warning test**

Append to `src/tests/backtesting/test_cli.py`:

```python
def test_cli_prints_exploratory_warning(monkeypatch, capsys, tmp_path) -> None:
    from cartola.backtesting.cli import main
    from cartola.backtesting.runner import BacktestMetadata, BacktestResult
    import pandas as pd

    def fake_run_backtest(config):
        return BacktestResult(
            round_results=pd.DataFrame(),
            selected_players=pd.DataFrame(),
            player_predictions=pd.DataFrame(),
            summary=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            metadata=BacktestMetadata(
                season=2025,
                start_round=5,
                max_round=38,
                fixture_mode="exploratory",
                strict_alignment_policy="fail",
                fixture_source_directory=None,
                fixture_manifest_paths=[],
                fixture_manifest_sha256={},
                generator_versions=[],
                excluded_rounds=[],
                warnings=["Exploratory fixture mode is not strict no-leakage data."],
            ),
        )

    monkeypatch.setattr("cartola.backtesting.cli.run_backtest", fake_run_backtest)

    assert main(["--project-root", str(tmp_path), "--fixture-mode", "exploratory"]) == 0
    assert "WARNING: Exploratory fixture mode" in capsys.readouterr().out
```

- [ ] **Step 9: Run CLI and runner tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_cli.py src/tests/backtesting/test_runner.py -q
uv run --frozen ty check
```

Expected: PASS.

- [ ] **Step 10: Commit Task 5**

Run:

```bash
git add src/cartola/backtesting/runner.py src/cartola/backtesting/cli.py src/tests/backtesting/test_runner.py src/tests/backtesting/test_cli.py
git commit -m "feat: enforce fixture modes in backtests"
```

---

### Task 6: Strict Alignment Policy, Docs, And Final Verification

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/tests/backtesting/test_runner.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing exclude-round test**

Append to `src/tests/backtesting/test_runner.py`:

```python
def test_strict_alignment_policy_exclude_round_removes_invalid_round_before_training(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    fixtures = _tiny_fixtures(range(1, 6))
    fixtures = fixtures[fixtures["rodada"] != 3].copy()
    config = BacktestConfig(
        project_root=tmp_path,
        start_round=5,
        budget=100,
        fixture_mode="strict",
        strict_alignment_policy="exclude_round",
    )

    monkeypatch.setattr(
        "cartola.backtesting.runner.load_strict_fixtures",
        lambda **kwargs: type(
            "Loaded",
            (),
            {
                "fixtures": fixtures,
                "manifest_paths": ["manifest-1.json"],
                "manifest_sha256": {"manifest-1.json": "abc"},
                "generator_versions": ["fixture_snapshot_v1"],
            },
        )(),
    )

    result = run_backtest(config, season_df=season_df)

    assert 3 in result.metadata.excluded_rounds
    assert 3 not in set(result.player_predictions["rodada"].dropna().astype(int).tolist())
```

- [ ] **Step 2: Run exclude-round test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_strict_alignment_policy_exclude_round_removes_invalid_round_before_training -q
```

Expected: FAIL because `exclude_round` is not implemented.

- [ ] **Step 3: Implement exclude-round policy**

In `src/cartola/backtesting/runner.py`, change `_validate_fixture_alignment` to return excluded rounds:

```python
def _validate_fixture_alignment(
    fixtures: pd.DataFrame | None,
    season_df: pd.DataFrame,
    *,
    policy: str = "fail",
) -> list[int]:
    if fixtures is None:
        return []
    report = build_round_alignment_report(fixtures, season_df)
    invalid = report[~report["is_valid"].astype(bool)]
    if invalid.empty:
        return []
    details = invalid[["rodada", "missing_from_fixtures", "extra_in_fixtures"]].to_dict("records")
    if policy == "exclude_round":
        return [int(value) for value in invalid["rodada"].tolist()]
    raise ValueError(f"Fixture alignment failed: {details}")
```

After fixture load and before max round:

```python
    excluded_rounds = _validate_fixture_alignment(
        fixture_data,
        data,
        policy=config.strict_alignment_policy if config.fixture_mode == "strict" else "fail",
    )
    if excluded_rounds:
        data = data[~pd.to_numeric(data["rodada"], errors="raise").isin(excluded_rounds)].copy()
```

Remove the old `_validate_fixture_alignment(fixture_data, data)` call.

Use `excluded_rounds=excluded_rounds` in metadata.

- [ ] **Step 4: Run exclude-round test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_strict_alignment_policy_exclude_round_removes_invalid_round_before_training -q
```

Expected: PASS.

- [ ] **Step 5: Update README**

In `README.md`, update the backtest section to show explicit fixture modes:

````markdown
Por padrão, o backtest roda sem features de partidas:

```bash
uv run python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none
```

Para usar os arquivos reconstruídos de 2025, opte explicitamente pelo modo exploratório:

```bash
uv run python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory
```

O modo `strict` só aceita fixtures geradas a partir de snapshots pré-fechamento com manifesto válido:

```bash
uv run python -m cartola.backtesting.cli --season 2026 --fixture-mode strict
```
````

- [ ] **Step 6: Run full backtesting tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/ -q
```

Expected: all backtesting tests pass.

- [ ] **Step 7: Run default no-fixture backtest**

Run:

```bash
uv run --frozen --no-dev python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none
```

Expected: command exits 0 and writes `data/08_reporting/backtests/2025/run_metadata.json` with `"fixture_mode": "none"`.

- [ ] **Step 8: Run exploratory 2025 backtest**

Run:

```bash
uv run --frozen --no-dev python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory
```

Expected: command exits 0, prints `WARNING: Exploratory fixture mode...`, and writes `run_metadata.json` with `"fixture_mode": "exploratory"` and a non-empty `warnings` list.

- [ ] **Step 9: Run quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff clean, ty clean, Bandit no issues, pytest passes.

- [ ] **Step 10: Commit Task 6**

Run:

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py README.md
git commit -m "feat: add strict alignment policy and fixture docs"
```

---

## Self-Review Checklist

- [ ] Spec coverage: strict snapshots, Cartola endpoints, HTTP-date clock evidence, no-outcome generator boundary, strict manifests, fixture modes, metadata, strict validation range, and exploratory warnings are covered.
- [ ] No same-round outcome data enters strict generation; `strict_fixtures.py` does not import `load_season_data`, `played_club_set`, runner, metrics, or diagnostics.
- [ ] `fixture_mode="none"` is the safe default and does not auto-load exploratory files.
- [ ] `fixture_mode="exploratory"` requires explicit opt-in and writes/prints warning.
- [ ] Strict capture only supports active `cartola_api` round and checks `temporada`/`rodada_atual`.
- [ ] Manifest validation checks hashes, UTC JSON timestamps, RFC 7231 HTTP-date capture evidence, identity fields, and symlink-safe project-root containment.
- [ ] Missing strict fixture files in `1..M` fail strict loading.
- [ ] Final verification includes `uv run --frozen scripts/pyrepo-check --all`.
