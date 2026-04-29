# Strict Round Fixture Capture Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manual `scripts/capture_strict_round_fixture.py` command that captures strict pre-lock Cartola fixture evidence for the active/current round and generates the canonical strict fixture CSV and manifest from that exact snapshot.

**Architecture:** Tighten strict snapshot validation first, then add a small orchestration module that composes `capture_cartola_snapshot(...)` and `generate_strict_fixture(...)`. Keep the CLI thin: parse arguments, call the orchestration module, and print stable success/error output.

**Tech Stack:** Python 3.12, argparse, dataclasses, pathlib, pytest, requests, existing strict fixture snapshot/generator modules.

---

## File Structure

- Modify `src/cartola/backtesting/fixture_snapshots.py`
  - Enforce `status_mercado == 1` in shared deadline validation.
  - Add a strict active-round parser/fetch helper for `--auto`.

- Create `src/cartola/backtesting/strict_round_fixture_capture.py`
  - Owns orchestration and formatting helpers.
  - Validates current-year/source/round-selection contract.
  - Binds generation to `capture_result.captured_at_utc`.
  - Raises a structured capture error when snapshot capture fails after round resolution.
  - Raises a structured generation error when capture succeeds but generation fails.

- Create `scripts/capture_strict_round_fixture.py`
  - Thin CLI wrapper around `run_strict_round_fixture_capture(...)`.
  - Handles parse errors through argparse.
  - Prints stable labels and returns `1` for expected operational failures without traceback.

- Modify `src/tests/backtesting/test_fixture_snapshots.py`
  - Adds shared validation tests for closed market status and active-round parsing.

- Create `src/tests/backtesting/test_strict_round_fixture_capture.py`
  - Unit-tests orchestration sequencing without subprocess assertions.

- Create `src/tests/backtesting/test_capture_strict_round_fixture_cli.py`
  - Unit-tests parser and terminal output.

- Modify `README.md` and `roadmap.md`
  - Document the new manual command and roadmap status.

---

### Task 1: Enforce Open-Market Status In Shared Snapshot Validation

**Files:**
- Modify: `src/cartola/backtesting/fixture_snapshots.py`
- Test: `src/tests/backtesting/test_fixture_snapshots.py`

- [ ] **Step 1: Write failing deadline validation tests**

Add these tests below `test_cartola_deadline_at_rejects_missing_status_mercado` in `src/tests/backtesting/test_fixture_snapshots.py`:

```python
def test_cartola_deadline_at_rejects_closed_market_status() -> None:
    payload = {
        **FROZEN_DEADLINE_PAYLOAD,
        "status_mercado": 2,
    }

    with pytest.raises(ValueError, match="Cartola market is not open"):
        cartola_deadline_at(payload, season=2026, round_number=12)


def test_capture_cartola_snapshot_rejects_closed_market_status_for_explicit_round(tmp_path: Path) -> None:
    captured_at = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)
    closed_deadline_payload = {
        **FROZEN_DEADLINE_PAYLOAD,
        "status_mercado": 2,
    }

    def fake_fetch(url: str) -> object:
        if url.endswith("/partidas/12"):
            return _response(FROZEN_FIXTURE_PAYLOAD, url)
        return _response(closed_deadline_payload, url)

    with pytest.raises(ValueError, match="Cartola market is not open"):
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
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_fixture_snapshots.py::test_cartola_deadline_at_rejects_closed_market_status \
  src/tests/backtesting/test_fixture_snapshots.py::test_capture_cartola_snapshot_rejects_closed_market_status_for_explicit_round \
  -q
```

Expected: both tests fail because `cartola_deadline_at(...)` parses `status_mercado` but does not yet require it to equal `1`.

- [ ] **Step 3: Implement minimal shared status enforcement**

In `src/cartola/backtesting/fixture_snapshots.py`, replace this line in `cartola_deadline_at(...)`:

```python
    _integer_field(payload, "status_mercado", context="Deadline payload status_mercado")
```

with:

```python
    status_mercado = _integer_field(payload, "status_mercado", context="Deadline payload status_mercado")
    if status_mercado != 1:
        raise ValueError(
            "Cartola market is not open: "
            f"rodada_atual={payload_round} status_mercado={status_mercado}"
        )
```

- [ ] **Step 4: Verify the new tests pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_fixture_snapshots.py::test_cartola_deadline_at_rejects_closed_market_status \
  src/tests/backtesting/test_fixture_snapshots.py::test_capture_cartola_snapshot_rejects_closed_market_status_for_explicit_round \
  -q
```

Expected: `2 passed`.

- [ ] **Step 5: Verify existing strict snapshot and fixture tests still pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_fixture_snapshots.py \
  src/tests/backtesting/test_strict_fixtures.py \
  -q
```

Expected: all selected tests pass. If an existing test fixture used `status_mercado != 1`, update that fixture to `1` only when the test is not specifically about closed-market rejection.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cartola/backtesting/fixture_snapshots.py src/tests/backtesting/test_fixture_snapshots.py
git commit -m "fix: require open market for strict fixture snapshots"
```

---

### Task 2: Add A Strict Active-Round Helper For `--auto`

**Files:**
- Modify: `src/cartola/backtesting/fixture_snapshots.py`
- Test: `src/tests/backtesting/test_fixture_snapshots.py`

- [ ] **Step 1: Write failing tests for active-round parsing and fetching**

Update the import in `src/tests/backtesting/test_fixture_snapshots.py` to include the new helpers:

```python
from cartola.backtesting.fixture_snapshots import (
    FROZEN_CLOCK_SKEW_TOLERANCE_SECONDS,
    capture_cartola_snapshot,
    cartola_active_open_round,
    cartola_deadline_at,
    cartola_fixture_rows,
    fetch_cartola_active_open_round,
    parse_http_date_utc,
)
```

Add these tests after `test_cartola_deadline_at_rejects_closed_market_status`:

```python
def test_cartola_active_open_round_parses_open_market_round() -> None:
    assert cartola_active_open_round(FROZEN_DEADLINE_PAYLOAD) == 12


def test_cartola_active_open_round_rejects_non_positive_round() -> None:
    payload = {
        **FROZEN_DEADLINE_PAYLOAD,
        "rodada_atual": 0,
    }

    with pytest.raises(ValueError, match="rodada_atual must be a positive integer"):
        cartola_active_open_round(payload)


def test_cartola_active_open_round_rejects_closed_market() -> None:
    payload = {
        **FROZEN_DEADLINE_PAYLOAD,
        "status_mercado": 2,
    }

    with pytest.raises(ValueError, match="Cartola market is not open"):
        cartola_active_open_round(payload)


def test_fetch_cartola_active_open_round_uses_market_status_endpoint() -> None:
    urls: list[str] = []

    def fake_fetch(url: str) -> object:
        urls.append(url)
        return _response(FROZEN_DEADLINE_PAYLOAD, url)

    assert fetch_cartola_active_open_round(fetch=fake_fetch) == 12
    assert urls == ["https://api.cartola.globo.com/mercado/status"]
```

- [ ] **Step 2: Run the active-round tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_fixture_snapshots.py::test_cartola_active_open_round_parses_open_market_round \
  src/tests/backtesting/test_fixture_snapshots.py::test_cartola_active_open_round_rejects_non_positive_round \
  src/tests/backtesting/test_fixture_snapshots.py::test_cartola_active_open_round_rejects_closed_market \
  src/tests/backtesting/test_fixture_snapshots.py::test_fetch_cartola_active_open_round_uses_market_status_endpoint \
  -q
```

Expected: collection or import failure because the helpers do not exist yet.

- [ ] **Step 3: Add the helper functions**

In `src/cartola/backtesting/fixture_snapshots.py`, insert these functions immediately above `cartola_deadline_at(...)`:

```python
def cartola_active_open_round(payload: dict[str, Any]) -> int:
    rodada_atual = _integer_field(payload, "rodada_atual", context="Deadline payload rodada_atual")
    if rodada_atual <= 0:
        raise ValueError("rodada_atual must be a positive integer")

    status_mercado = _integer_field(payload, "status_mercado", context="Deadline payload status_mercado")
    if status_mercado != 1:
        raise ValueError(
            "Cartola market is not open: "
            f"rodada_atual={rodada_atual} status_mercado={status_mercado}"
        )

    return rodada_atual


def fetch_cartola_active_open_round(*, fetch: Fetch | None = None) -> int:
    requester = fetch or _fetch_url
    response = _capture_response(requester, CARTOLA_DEADLINE_ENDPOINT)
    return cartola_active_open_round(response.payload)
```

Then update `cartola_deadline_at(...)` so it reuses the shared helper:

```python
def cartola_deadline_at(payload: dict[str, Any], *, season: int, round_number: int) -> datetime:
    payload_season = _integer_field(payload, "temporada", context="Deadline payload temporada")
    if payload_season != season:
        raise ValueError(f"Deadline payload temporada {payload_season} does not match requested season {season}")

    payload_round = cartola_active_open_round(payload)
    if payload_round != round_number:
        raise ValueError(f"Deadline payload rodada_atual {payload_round} does not match requested round {round_number}")

    fechamento = payload.get("fechamento")
    if not isinstance(fechamento, dict):
        raise ValueError("Deadline payload fechamento must be an object")

    for field in ("ano", "mes", "dia", "hora", "minuto"):
        _integer_field(fechamento, field, context=f"Deadline payload fechamento.{field}")
    timestamp = _integer_field(fechamento, "timestamp", context="Deadline payload fechamento.timestamp")
    return datetime.fromtimestamp(timestamp, tz=UTC)
```

- [ ] **Step 4: Verify active-round tests pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_fixture_snapshots.py::test_cartola_active_open_round_parses_open_market_round \
  src/tests/backtesting/test_fixture_snapshots.py::test_cartola_active_open_round_rejects_non_positive_round \
  src/tests/backtesting/test_fixture_snapshots.py::test_cartola_active_open_round_rejects_closed_market \
  src/tests/backtesting/test_fixture_snapshots.py::test_fetch_cartola_active_open_round_uses_market_status_endpoint \
  -q
```

Expected: `4 passed`.

- [ ] **Step 5: Run the fixture snapshot suite**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_fixture_snapshots.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cartola/backtesting/fixture_snapshots.py src/tests/backtesting/test_fixture_snapshots.py
git commit -m "feat: add strict active round resolver"
```

---

### Task 3: Add The Strict Fixture Capture Orchestration Module

**Files:**
- Create: `src/cartola/backtesting/strict_round_fixture_capture.py`
- Test: `src/tests/backtesting/test_strict_round_fixture_capture.py`

- [ ] **Step 1: Write failing orchestration tests**

Create `src/tests/backtesting/test_strict_round_fixture_capture.py` with this content:

```python
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import cartola.backtesting.strict_round_fixture_capture as strict_capture
from cartola.backtesting.fixture_snapshots import CaptureResult
from cartola.backtesting.strict_fixtures import StrictFixtureLoadResult


CAPTURED_AT = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)
DEADLINE_AT = datetime(2026, 6, 1, 18, 59, tzinfo=UTC)


def _capture_result(project_root: Path, *, round_number: int = 12) -> CaptureResult:
    return CaptureResult(
        capture_dir=project_root
        / "data"
        / "01_raw"
        / "fixtures_snapshots"
        / "2026"
        / f"rodada-{round_number}"
        / "captured_at=2026-06-01T18-00-00Z",
        captured_at_utc=CAPTURED_AT,
        deadline_at_utc=DEADLINE_AT,
        fixture_rows=[],
    )


def _strict_result(project_root: Path, *, round_number: int = 12) -> StrictFixtureLoadResult:
    fixture_path = project_root / "data" / "01_raw" / "fixtures_strict" / "2026" / f"partidas-{round_number}.csv"
    manifest_path = fixture_path.with_suffix(".manifest.json")
    return StrictFixtureLoadResult(
        fixture_path=fixture_path,
        manifest_path=manifest_path,
        manifest={"mode": "strict"},
        captured_at_utc=CAPTURED_AT,
        deadline_at_utc=DEADLINE_AT,
        generator_version="fixture_snapshot_v1",
    )


def test_run_requires_exactly_one_round_selector(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        strict_capture.run_strict_round_fixture_capture(
            strict_capture.StrictRoundFixtureCaptureConfig(
                season=2026,
                current_year=2026,
                project_root=tmp_path,
            )
        )

    with pytest.raises(ValueError, match="exactly one"):
        strict_capture.run_strict_round_fixture_capture(
            strict_capture.StrictRoundFixtureCaptureConfig(
                season=2026,
                round_number=12,
                auto=True,
                current_year=2026,
                project_root=tmp_path,
            )
        )


def test_run_rejects_wrong_current_year_before_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strict_capture, "fetch_cartola_active_open_round", lambda **kwargs: (_ for _ in ()).throw(AssertionError("fetched status")))
    monkeypatch.setattr(strict_capture, "capture_cartola_snapshot", lambda **kwargs: (_ for _ in ()).throw(AssertionError("captured")))

    with pytest.raises(ValueError, match="season 2025 must equal current_year 2026"):
        strict_capture.run_strict_round_fixture_capture(
            strict_capture.StrictRoundFixtureCaptureConfig(
                season=2025,
                auto=True,
                current_year=2026,
                project_root=tmp_path,
            )
        )


def test_run_auto_resolves_active_round_and_generates_from_captured_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_calls: list[dict[str, object]] = []
    generate_calls: list[dict[str, object]] = []

    def fake_active_round(**kwargs: object) -> int:
        return 12

    def fake_capture(**kwargs: object) -> CaptureResult:
        capture_calls.append(kwargs)
        return _capture_result(tmp_path, round_number=12)

    def fake_generate(**kwargs: object) -> StrictFixtureLoadResult:
        generate_calls.append(kwargs)
        return _strict_result(tmp_path, round_number=12)

    monkeypatch.setattr(strict_capture, "fetch_cartola_active_open_round", fake_active_round)
    monkeypatch.setattr(strict_capture, "capture_cartola_snapshot", fake_capture)
    monkeypatch.setattr(strict_capture, "generate_strict_fixture", fake_generate)

    result = strict_capture.run_strict_round_fixture_capture(
        strict_capture.StrictRoundFixtureCaptureConfig(
            season=2026,
            auto=True,
            current_year=2026,
            project_root=tmp_path,
        )
    )

    assert capture_calls[0]["round_number"] == 12
    assert generate_calls[0]["round_number"] == 12
    assert generate_calls[0]["captured_at"] == CAPTURED_AT
    assert generate_calls[0]["force"] is False
    assert result.round_number == 12
    assert result.capture_dir.name == "captured_at=2026-06-01T18-00-00Z"


def test_run_explicit_round_passes_round_without_active_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_calls: list[dict[str, object]] = []

    monkeypatch.setattr(strict_capture, "fetch_cartola_active_open_round", lambda **kwargs: (_ for _ in ()).throw(AssertionError("auto fetch should not run")))
    monkeypatch.setattr(
        strict_capture,
        "capture_cartola_snapshot",
        lambda **kwargs: capture_calls.append(kwargs) or _capture_result(tmp_path, round_number=13),
    )
    monkeypatch.setattr(strict_capture, "generate_strict_fixture", lambda **kwargs: _strict_result(tmp_path, round_number=13))

    result = strict_capture.run_strict_round_fixture_capture(
        strict_capture.StrictRoundFixtureCaptureConfig(
            season=2026,
            round_number=13,
            current_year=2026,
            project_root=tmp_path,
        )
    )

    assert capture_calls[0]["round_number"] == 13
    assert result.round_number == 13


def test_run_force_generate_maps_only_to_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_calls: list[dict[str, object]] = []
    generate_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        strict_capture,
        "capture_cartola_snapshot",
        lambda **kwargs: capture_calls.append(kwargs) or _capture_result(tmp_path),
    )
    monkeypatch.setattr(
        strict_capture,
        "generate_strict_fixture",
        lambda **kwargs: generate_calls.append(kwargs) or _strict_result(tmp_path),
    )

    strict_capture.run_strict_round_fixture_capture(
        strict_capture.StrictRoundFixtureCaptureConfig(
            season=2026,
            round_number=12,
            current_year=2026,
            project_root=tmp_path,
            force_generate=True,
        )
    )

    assert "force" not in capture_calls[0]
    assert generate_calls[0]["force"] is True


def test_run_generation_failure_preserves_capture_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _capture_result(tmp_path)
    monkeypatch.setattr(strict_capture, "capture_cartola_snapshot", lambda **kwargs: capture)

    def fail_generate(**kwargs: object) -> StrictFixtureLoadResult:
        raise FileExistsError("Strict fixture target already exists")

    monkeypatch.setattr(strict_capture, "generate_strict_fixture", fail_generate)

    with pytest.raises(strict_capture.StrictFixtureGenerationError) as error_info:
        strict_capture.run_strict_round_fixture_capture(
            strict_capture.StrictRoundFixtureCaptureConfig(
                season=2026,
                round_number=12,
                current_year=2026,
                project_root=tmp_path,
            )
        )

    error = error_info.value
    assert error.capture_result == capture
    assert isinstance(error.original, FileExistsError)
    assert "Strict fixture target already exists" in str(error)


def test_run_capture_failure_preserves_attempted_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(strict_capture, "fetch_cartola_active_open_round", lambda **kwargs: 12)

    def fail_capture(**kwargs: object) -> CaptureResult:
        raise ValueError("Deadline payload rodada_atual 13 does not match requested round 12")

    monkeypatch.setattr(strict_capture, "capture_cartola_snapshot", fail_capture)
    monkeypatch.setattr(
        strict_capture,
        "generate_strict_fixture",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("generation should not run")),
    )

    with pytest.raises(strict_capture.StrictFixtureCaptureError) as error_info:
        strict_capture.run_strict_round_fixture_capture(
            strict_capture.StrictRoundFixtureCaptureConfig(
                season=2026,
                auto=True,
                current_year=2026,
                project_root=tmp_path,
            )
        )

    error = error_info.value
    assert error.round_number == 12
    assert isinstance(error.original, ValueError)
    assert "Deadline payload rodada_atual 13" in str(error)
```

- [ ] **Step 2: Run the orchestration tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_strict_round_fixture_capture.py -q
```

Expected: import failure because `cartola.backtesting.strict_round_fixture_capture` does not exist yet.

- [ ] **Step 3: Implement the orchestration module**

Create `src/cartola/backtesting/strict_round_fixture_capture.py` with this content:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from cartola.backtesting.fixture_snapshots import (
    CaptureResult,
    Fetch,
    capture_cartola_snapshot,
    fetch_cartola_active_open_round,
)
from cartola.backtesting.strict_fixtures import (
    STRICT_SOURCE,
    StrictFixtureLoadResult,
    generate_strict_fixture,
)


@dataclass(frozen=True)
class StrictRoundFixtureCaptureConfig:
    season: int
    round_number: int | None = None
    auto: bool = False
    current_year: int | None = None
    source: str = STRICT_SOURCE
    project_root: Path = Path(".")
    force_generate: bool = False


@dataclass(frozen=True)
class StrictRoundFixtureCaptureResult:
    season: int
    round_number: int
    capture_dir: Path
    fixture_path: Path
    manifest_path: Path
    captured_at_utc: datetime
    deadline_at_utc: datetime


class StrictFixtureGenerationError(RuntimeError):
    def __init__(self, *, capture_result: CaptureResult, original: Exception) -> None:
        self.capture_result = capture_result
        self.original = original
        super().__init__(f"{type(original).__name__}: {original}")


class StrictFixtureCaptureError(RuntimeError):
    def __init__(self, *, round_number: int, original: Exception) -> None:
        self.round_number = round_number
        self.original = original
        super().__init__(f"round {round_number}: {type(original).__name__}: {original}")


def run_strict_round_fixture_capture(
    config: StrictRoundFixtureCaptureConfig,
    *,
    fetch: Fetch | None = None,
    now: Any | None = None,
) -> StrictRoundFixtureCaptureResult:
    if config.source != STRICT_SOURCE:
        raise ValueError(f"Unsupported strict fixture source: {config.source!r}")

    current_year = config.current_year if config.current_year is not None else datetime.now(UTC).year
    if config.season != current_year:
        raise ValueError(f"season {config.season} must equal current_year {current_year}")

    round_number = _resolve_round_number(config, fetch=fetch)
    root = Path(config.project_root)
    try:
        capture_result = capture_cartola_snapshot(
            project_root=root,
            season=config.season,
            round_number=round_number,
            source=config.source,
            fetch=fetch,
            now=now,
        )
    except (FileExistsError, FileNotFoundError, ValueError, json.JSONDecodeError, requests.RequestException) as exc:
        raise StrictFixtureCaptureError(round_number=round_number, original=exc) from exc

    try:
        strict_result = generate_strict_fixture(
            project_root=root,
            season=config.season,
            round_number=round_number,
            source=config.source,
            captured_at=capture_result.captured_at_utc,
            force=config.force_generate,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise StrictFixtureGenerationError(capture_result=capture_result, original=exc) from exc

    return _result_from_parts(
        season=config.season,
        round_number=round_number,
        capture_result=capture_result,
        strict_result=strict_result,
    )


def format_project_path(project_root: str | Path, path: str | Path) -> str:
    root = Path(project_root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def format_utc_z(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _resolve_round_number(config: StrictRoundFixtureCaptureConfig, *, fetch: Fetch | None) -> int:
    selectors = int(config.auto) + int(config.round_number is not None)
    if selectors != 1:
        raise ValueError("exactly one of auto=True or round_number must be provided")
    if config.auto:
        return fetch_cartola_active_open_round(fetch=fetch)
    if config.round_number is None:
        raise ValueError("round_number is required when auto=False")
    if config.round_number <= 0:
        raise ValueError("round_number must be a positive integer")
    return config.round_number


def _result_from_parts(
    *,
    season: int,
    round_number: int,
    capture_result: CaptureResult,
    strict_result: StrictFixtureLoadResult,
) -> StrictRoundFixtureCaptureResult:
    return StrictRoundFixtureCaptureResult(
        season=season,
        round_number=round_number,
        capture_dir=capture_result.capture_dir,
        fixture_path=strict_result.fixture_path,
        manifest_path=strict_result.manifest_path,
        captured_at_utc=capture_result.captured_at_utc,
        deadline_at_utc=capture_result.deadline_at_utc,
    )
```

- [ ] **Step 4: Verify orchestration tests pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_strict_round_fixture_capture.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Run type-adjacent focused tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_strict_round_fixture_capture.py \
  src/tests/backtesting/test_fixture_snapshots.py \
  src/tests/backtesting/test_strict_fixtures.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cartola/backtesting/strict_round_fixture_capture.py src/tests/backtesting/test_strict_round_fixture_capture.py
git commit -m "feat: orchestrate strict round fixture capture"
```

---

### Task 4: Add The Manual Strict Fixture Capture CLI

**Files:**
- Create: `scripts/capture_strict_round_fixture.py`
- Test: `src/tests/backtesting/test_capture_strict_round_fixture_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `src/tests/backtesting/test_capture_strict_round_fixture_cli.py` with this content:

```python
from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cartola.backtesting.fixture_snapshots import CaptureResult
from cartola.backtesting.strict_round_fixture_capture import (
    StrictFixtureCaptureError,
    StrictFixtureGenerationError,
    StrictRoundFixtureCaptureResult,
)


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "capture_strict_round_fixture.py"
SPEC = importlib.util.spec_from_file_location("capture_strict_round_fixture", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _result(project_root: Path) -> StrictRoundFixtureCaptureResult:
    return StrictRoundFixtureCaptureResult(
        season=2026,
        round_number=12,
        capture_dir=project_root
        / "data"
        / "01_raw"
        / "fixtures_snapshots"
        / "2026"
        / "rodada-12"
        / "captured_at=2026-06-01T18-00-00Z",
        fixture_path=project_root / "data" / "01_raw" / "fixtures_strict" / "2026" / "partidas-12.csv",
        manifest_path=project_root
        / "data"
        / "01_raw"
        / "fixtures_strict"
        / "2026"
        / "partidas-12.manifest.json",
        captured_at_utc=datetime(2026, 6, 1, 18, 0, tzinfo=UTC),
        deadline_at_utc=datetime(2026, 6, 1, 18, 59, tzinfo=UTC),
    )


def test_parse_args_requires_auto_or_round() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--current-year", "2026"])


def test_parse_args_rejects_auto_and_round_together() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--auto", "--round", "12", "--current-year", "2026"])


def test_parse_args_rejects_non_cartola_source() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--season", "2026", "--auto", "--source", "thesportsdb", "--current-year", "2026"])


def test_main_prints_success_summary_with_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def fake_run(config):
        calls.append(config)
        return _result(tmp_path)

    monkeypatch.setattr(cli, "run_strict_round_fixture_capture", fake_run)

    exit_code = cli.main(
        [
            "--season",
            "2026",
            "--auto",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert calls[0].auto is True
    assert calls[0].force_generate is False
    output = capsys.readouterr().out
    assert "Strict fixture capture complete" in output
    assert "Season: 2026" in output
    assert "Round: 12" in output
    assert "Snapshot directory: data/01_raw/fixtures_snapshots/2026/rodada-12/captured_at=2026-06-01T18-00-00Z" in output
    assert "Strict fixture: data/01_raw/fixtures_strict/2026/partidas-12.csv" in output
    assert "Manifest: data/01_raw/fixtures_strict/2026/partidas-12.manifest.json" in output
    assert "Captured at UTC: 2026-06-01T18:00:00Z" in output
    assert "Deadline at UTC: 2026-06-01T18:59:00Z" in output


def test_main_maps_force_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run(config):
        calls.append(config)
        return _result(tmp_path)

    monkeypatch.setattr(cli, "run_strict_round_fixture_capture", fake_run)

    assert cli.main(
        [
            "--season",
            "2026",
            "--round",
            "12",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
            "--force-generate",
        ]
    ) == 0
    assert calls[0].round_number == 12
    assert calls[0].force_generate is True


def test_main_prints_generation_failure_with_retained_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture = CaptureResult(
        capture_dir=tmp_path
        / "data"
        / "01_raw"
        / "fixtures_snapshots"
        / "2026"
        / "rodada-12"
        / "captured_at=2026-06-01T18-00-00Z",
        captured_at_utc=datetime(2026, 6, 1, 18, 0, tzinfo=UTC),
        deadline_at_utc=datetime(2026, 6, 1, 18, 59, tzinfo=UTC),
        fixture_rows=[],
    )

    def fake_run(config):
        raise StrictFixtureGenerationError(
            capture_result=capture,
            original=FileExistsError("Strict fixture target already exists"),
        )

    monkeypatch.setattr(cli, "run_strict_round_fixture_capture", fake_run)

    exit_code = cli.main(
        [
            "--season",
            "2026",
            "--round",
            "12",
            "--current-year",
            "2026",
            "--project-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: strict fixture generation failed after snapshot capture" in captured.err
    assert "Retained snapshot directory: data/01_raw/fixtures_snapshots/2026/rodada-12/captured_at=2026-06-01T18-00-00Z" in captured.err
    assert "Captured at UTC: 2026-06-01T18:00:00Z" in captured.err
    assert "Deadline at UTC: 2026-06-01T18:59:00Z" in captured.err
    assert "Original error: FileExistsError: Strict fixture target already exists" in captured.err
    assert "Traceback" not in captured.err


def test_main_prints_capture_failure_with_attempted_round(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(config):
        raise StrictFixtureCaptureError(
            round_number=12,
            original=ValueError("Deadline payload rodada_atual 13 does not match requested round 12"),
        )

    monkeypatch.setattr(cli, "run_strict_round_fixture_capture", fake_run)

    exit_code = cli.main(["--season", "2026", "--auto", "--current-year", "2026"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: strict fixture capture failed for round 12" in captured.err
    assert "Original error: ValueError: Deadline payload rodada_atual 13 does not match requested round 12" in captured.err
    assert "Traceback" not in captured.err


def test_main_prints_operational_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "run_strict_round_fixture_capture",
        lambda config: (_ for _ in ()).throw(ValueError("season 2025 must equal current_year 2026")),
    )

    exit_code = cli.main(["--season", "2025", "--auto", "--current-year", "2026"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: season 2025 must equal current_year 2026" in captured.err
    assert "Traceback" not in captured.err
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_capture_strict_round_fixture_cli.py -q
```

Expected: import failure because `scripts/capture_strict_round_fixture.py` does not exist yet.

- [ ] **Step 3: Implement the CLI script**

Create `scripts/capture_strict_round_fixture.py` with this content:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import requests

from cartola.backtesting.strict_round_fixture_capture import (
    StrictFixtureCaptureError,
    StrictFixtureGenerationError,
    StrictRoundFixtureCaptureConfig,
    StrictRoundFixtureCaptureResult,
    format_project_path,
    format_utc_z,
    run_strict_round_fixture_capture,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the open Cartola fixture snapshot and generate a strict fixture."
    )
    parser.add_argument("--season", type=_positive_int, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--auto", action="store_true")
    selector.add_argument("--round", dest="round_number", type=_positive_int, default=None)
    parser.add_argument("--current-year", type=_positive_int, default=None)
    parser.add_argument("--source", choices=("cartola_api",), default="cartola_api")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--force-generate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = StrictRoundFixtureCaptureConfig(
        season=args.season,
        round_number=args.round_number,
        auto=args.auto,
        current_year=args.current_year,
        source=args.source,
        project_root=args.project_root,
        force_generate=args.force_generate,
    )

    try:
        result = run_strict_round_fixture_capture(config)
    except StrictFixtureCaptureError as error:
        _print_capture_error(error)
        return 1
    except StrictFixtureGenerationError as error:
        _print_generation_error(args.project_root, error)
        return 1
    except (ValueError, FileExistsError, FileNotFoundError, json.JSONDecodeError, requests.RequestException) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _print_success(args.project_root, result)
    return 0


def _print_capture_error(error: StrictFixtureCaptureError) -> None:
    print(f"error: strict fixture capture failed for round {error.round_number}", file=sys.stderr)
    print(f"Original error: {type(error.original).__name__}: {error.original}", file=sys.stderr)


def _print_success(project_root: Path, result: StrictRoundFixtureCaptureResult) -> None:
    print("Strict fixture capture complete")
    print(f"Season: {result.season}")
    print(f"Round: {result.round_number}")
    print(f"Snapshot directory: {format_project_path(project_root, result.capture_dir)}")
    print(f"Strict fixture: {format_project_path(project_root, result.fixture_path)}")
    print(f"Manifest: {format_project_path(project_root, result.manifest_path)}")
    print(f"Captured at UTC: {format_utc_z(result.captured_at_utc)}")
    print(f"Deadline at UTC: {format_utc_z(result.deadline_at_utc)}")


def _print_generation_error(project_root: Path, error: StrictFixtureGenerationError) -> None:
    capture = error.capture_result
    print("error: strict fixture generation failed after snapshot capture", file=sys.stderr)
    print(f"Retained snapshot directory: {format_project_path(project_root, capture.capture_dir)}", file=sys.stderr)
    print(f"Captured at UTC: {format_utc_z(capture.captured_at_utc)}", file=sys.stderr)
    print(f"Deadline at UTC: {format_utc_z(capture.deadline_at_utc)}", file=sys.stderr)
    print(f"Original error: {type(error.original).__name__}: {error.original}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify CLI tests pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_capture_strict_round_fixture_cli.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Run focused command-related tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_capture_strict_round_fixture_cli.py \
  src/tests/backtesting/test_strict_round_fixture_capture.py \
  src/tests/backtesting/test_fixture_snapshots.py \
  src/tests/backtesting/test_strict_fixtures.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/capture_strict_round_fixture.py src/tests/backtesting/test_capture_strict_round_fixture_cli.py
git commit -m "feat: add strict round fixture capture cli"
```

---

### Task 5: Document The Manual Strict Fixture Capture Workflow

**Files:**
- Modify: `README.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Add README documentation**

Find the operational/live recommendation area in `README.md` and add this command block near the live workflow command:

````markdown
### Capture strict fixture evidence

Before market lock, capture strict fixture evidence and generate the canonical strict fixture file:

```bash
uv run --frozen python scripts/capture_strict_round_fixture.py \
  --season 2026 \
  --auto \
  --current-year 2026
```

This command is separate from `scripts/run_live_round.py`. Live recommendations still default to `fixture_mode=none`; strict fixture capture is provenance work for future fixture-context integration.
````

- [ ] **Step 2: Update roadmap status**

In `roadmap.md`, update the strict fixture/live recommendation entry so it says:

```markdown
- [ ] Capture strict pre-lock fixture snapshots every live round with `scripts/capture_strict_round_fixture.py`.
  - Manual v1 command captures snapshot evidence and generates strict `fixtures_strict` CSV/manifest.
  - Future step: integrate strict fixtures into live recommendations as an explicit opt-in mode after several successful live captures.
```

If the roadmap already has equivalent wording, replace it with the block above instead of duplicating the item.

- [ ] **Step 3: Run documentation-adjacent checks**

Run:

```bash
rg "capture_strict_round_fixture|fixtures_strict|fixture_mode=none" README.md roadmap.md
```

Expected: the new command and the non-integration note are present.

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md roadmap.md
git commit -m "docs: document strict round fixture capture command"
```

---

### Task 6: Run The Full Verification Gate

**Files:**
- No source changes unless the gate exposes a real issue.

- [ ] **Step 1: Run the full repository gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: Ruff, ty, Bandit, and pytest all pass.

- [ ] **Step 2: If Ruff reports formatting/import issues, fix them mechanically**

Run the exact formatter/linter command reported by `scripts/pyrepo-check`, or run:

```bash
uv run --frozen ruff check . --fix
uv run --frozen ruff format .
```

Expected: only formatting/import ordering changes.

- [ ] **Step 3: Re-run the full gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: all checks pass.

- [ ] **Step 4: Commit gate fixes if any were needed**

If Step 2 changed files, run:

```bash
git add .
git commit -m "chore: satisfy strict fixture capture checks"
```

If Step 2 did not change files, do not create an empty commit.

---

## Self-Review Checklist

- Spec coverage:
  - Current-year guard: Task 3 orchestration tests and implementation.
  - Exactly one of `--auto`/`--round`: Task 3 module validation and Task 4 argparse tests.
  - `status_mercado == 1` for both modes: Task 1 shared deadline validation and Task 2 active helper.
  - Exact captured snapshot binding: Task 3 asserts `captured_at` is passed to generation.
  - Narrow force scope: Task 3 asserts `force` reaches only generation.
  - Capture failure after advisory auto resolution: Task 3 structured error and Task 4 terminal output.
  - Retained snapshot on generation failure: Task 3 structured error and Task 4 terminal output.
  - Project-root-relative paths and UTC `Z`: Task 3 formatting helpers and Task 4 output assertions.
  - No live recommendation integration: Task 5 documentation.

- Placeholder scan:
  - The plan avoids empty “add tests” instructions, deferred work markers, and undefined functions in later tasks.

- Type consistency:
  - `StrictRoundFixtureCaptureConfig`, `StrictRoundFixtureCaptureResult`, `StrictFixtureCaptureError`, and `StrictFixtureGenerationError` are defined in Task 3 before the CLI uses them in Task 4.
  - `format_project_path(...)` and `format_utc_z(...)` are defined in Task 3 before Task 4 imports them.
