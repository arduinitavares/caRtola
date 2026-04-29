# One-Command Live Round Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/run_live_round.py`, a one-command live workflow that captures or validates the open Cartola market round, runs live recommendation for that exact round, archives outputs, and links recommendation metadata to the capture.

**Architecture:** Add a shared live-capture validator to `market_capture.py`, extend `RecommendationConfig` with archived output and `live_workflow` metadata inputs, then add a thin orchestration module `live_workflow.py` plus CLI wrapper. The wrapper delegates raw-data writes to the existing safe capture primitive and delegates model work to `run_recommendation()`.

**Tech Stack:** Python 3.13, pandas, Rich for optional terminal formatting, pytest, Ruff, ty, Bandit, existing Cartola backtesting modules.

---

## File Structure

- Modify `src/cartola/backtesting/market_capture.py`
  - Add `LiveCaptureMetadata`.
  - Add public `load_valid_live_capture()`.
  - Refactor force-overwrite validation to use the shared helper.
- Modify `src/tests/backtesting/test_market_capture.py`
  - Add validator tests for valid captures, hash mismatch, path mismatch, bad timestamp, and closed-market metadata.
- Modify `src/cartola/backtesting/recommendation.py`
  - Add `output_run_id` and `live_workflow` to `RecommendationConfig`.
  - Append `runs/{output_run_id}` to archived output paths.
  - Include `live_workflow` in `run_metadata.json` before outputs are written.
  - Tighten output-root validation for absolute paths outside `project_root`.
- Modify `src/tests/backtesting/test_recommendation.py`
  - Add archived output path tests.
  - Add metadata-link tests.
  - Add absolute output-root rejection tests.
- Create `src/cartola/backtesting/live_workflow.py`
  - Implement capture policy orchestration.
  - Implement workflow metadata model/building/writing.
  - Implement `run_live_round()`.
- Create `src/tests/backtesting/test_live_workflow.py`
  - Unit-test fresh, missing, skip, failure, archive, metadata, and output-root behaviors.
- Create `scripts/run_live_round.py`
  - CLI parser and Rich/plain stable output labels.
- Create `src/tests/backtesting/test_run_live_round_cli.py`
  - CLI parser and terminal output tests.
- Modify `README.md`
  - Document one-command live workflow.
- Modify `roadmap.md`
  - Move this workflow into Delivered after implementation, or update the roadmap item from planned to available.

---

### Task 1: Shared Live Capture Validator

**Files:**
- Modify: `src/cartola/backtesting/market_capture.py`
- Modify: `src/tests/backtesting/test_market_capture.py`

- [ ] **Step 1: Write failing tests for public live-capture validation**

Append these tests near the existing force-overwrite capture tests in `src/tests/backtesting/test_market_capture.py`:

```python
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
```

Add hash, path, timestamp, and status rejection tests:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_returns_capture_metadata \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_rejects_hash_mismatch \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_rejects_csv_path_mismatch \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_rejects_bad_captured_at \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_rejects_closed_market_metadata \
  -q
```

Expected: FAIL because `load_valid_live_capture` is not defined.

- [ ] **Step 3: Implement `LiveCaptureMetadata` and shared validator**

In `src/cartola/backtesting/market_capture.py`, add this dataclass below `MarketCaptureResult`:

```python
@dataclass(frozen=True)
class LiveCaptureMetadata:
    csv_path: Path
    metadata_path: Path
    season: int
    target_round: int
    csv_sha256: str
    captured_at_utc: str
    status_mercado: int
    deadline_timestamp: int | None
    deadline_parse_status: str
```

Add `timedelta` to the existing datetime import:

```python
from datetime import UTC, datetime, timedelta
```

Add helpers near `_validate_previous_capture`:

```python
def _metadata_path_for(project_root: Path, season: int, target_round: int) -> Path:
    return project_root / "data" / "01_raw" / str(season) / f"rodada-{target_round}.capture.json"


def _csv_path_for(project_root: Path, season: int, target_round: int) -> Path:
    return project_root / "data" / "01_raw" / str(season) / f"rodada-{target_round}.csv"


def _parse_utc_z(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("destination is not a previous valid live capture")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("destination is not a previous valid live capture") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("destination is not a previous valid live capture")
    return value
```

Then implement the public helper:

```python
def load_valid_live_capture(*, project_root: Path, season: int, target_round: int) -> LiveCaptureMetadata:
    final_csv = _csv_path_for(project_root, season, target_round)
    final_metadata = _metadata_path_for(project_root, season, target_round)
    if not final_csv.exists() or not final_metadata.exists():
        raise FileNotFoundError(f"live capture files missing for season={season} target_round={target_round}")
    try:
        metadata = json.loads(final_metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("destination is not a previous valid live capture") from exc

    if metadata.get("capture_version") != CAPTURE_VERSION:
        raise ValueError("destination is not a previous valid live capture")
    if metadata.get("season") != season or metadata.get("target_round") != target_round:
        raise ValueError("destination is not a previous valid live capture")
    if Path(str(metadata.get("csv_path"))) != final_csv:
        raise ValueError("destination is not a previous valid live capture")
    csv_sha256 = _sha256_file(final_csv)
    if metadata.get("csv_sha256") != csv_sha256:
        raise ValueError("destination is not a previous valid live capture")
    captured_at_utc = _parse_utc_z(metadata.get("captured_at_utc"))
    status_mercado = _int_field(metadata, "status_mercado")
    if status_mercado != 1:
        raise ValueError("destination is not a previous valid live capture")
    deadline_parse_status = str(metadata.get("deadline_parse_status"))
    deadline_timestamp_value = metadata.get("deadline_timestamp")
    deadline_timestamp = None if deadline_timestamp_value is None else int(deadline_timestamp_value)
    return LiveCaptureMetadata(
        csv_path=final_csv,
        metadata_path=final_metadata,
        season=season,
        target_round=target_round,
        csv_sha256=csv_sha256,
        captured_at_utc=captured_at_utc,
        status_mercado=status_mercado,
        deadline_timestamp=deadline_timestamp,
        deadline_parse_status=deadline_parse_status,
    )
```

Refactor `_validate_previous_capture()` to use it:

```python
def _validate_previous_capture(
    final_csv: Path,
    final_metadata: Path,
    *,
    config: MarketCaptureConfig,
    target_round: int,
) -> None:
    try:
        metadata = load_valid_live_capture(
            project_root=config.project_root,
            season=config.season,
            target_round=target_round,
        )
    except FileNotFoundError as exc:
        raise ValueError("destination is not a previous valid live capture") from exc
    if metadata.csv_path != final_csv or metadata.metadata_path != final_metadata:
        raise ValueError("destination is not a previous valid live capture")
```

- [ ] **Step 4: Run validator tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_returns_capture_metadata \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_rejects_hash_mismatch \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_rejects_csv_path_mismatch \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_rejects_bad_captured_at \
  src/tests/backtesting/test_market_capture.py::test_load_valid_live_capture_rejects_closed_market_metadata \
  src/tests/backtesting/test_market_capture.py::test_capture_force_replaces_previous_valid_capture \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/market_capture.py src/tests/backtesting/test_market_capture.py
git commit -m "feat: expose live capture validation"
```

---

### Task 2: Recommendation Archive and Metadata Inputs

**Files:**
- Modify: `src/cartola/backtesting/recommendation.py`
- Modify: `src/tests/backtesting/test_recommendation.py`

- [ ] **Step 1: Write failing tests for archived output path and live workflow metadata**

Add to `src/tests/backtesting/test_recommendation.py` near `test_recommendation_config_output_path()`:

```python
def test_recommendation_config_output_path_includes_output_run_id() -> None:
    config = RecommendationConfig(
        season=2026,
        target_round=14,
        mode="live",
        project_root=Path("/tmp/cartola"),
        output_run_id="run_started_at=20260429T123456000000Z",
    )

    assert config.output_path == Path(
        "/tmp/cartola/data/08_reporting/recommendations/2026/round-14/live/runs/"
        "run_started_at=20260429T123456000000Z"
    )


def test_recommendation_config_rejects_output_run_id_with_path_separator() -> None:
    config = RecommendationConfig(
        season=2026,
        target_round=14,
        mode="live",
        output_run_id="../escape",
    )

    with pytest.raises(ValueError, match="output_run_id"):
        _validate_mode_scope(config)
```

Add a metadata test after an existing `run_recommendation` success test:

```python
def test_run_recommendation_writes_live_workflow_metadata_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_df = _season_frame(range(1, 4), target_round=3, live_target=True)
    monkeypatch.setattr("cartola.backtesting.recommendation.load_season_data", lambda *a, **k: season_df)
    live_workflow = {
        "capture_policy": "fresh",
        "target_round": 3,
        "capture_csv_path": str(tmp_path / "data/01_raw/2026/rodada-3.csv"),
        "capture_metadata_path": str(tmp_path / "data/01_raw/2026/rodada-3.capture.json"),
        "capture_csv_sha256": "a" * 64,
        "recommendation_output_path": str(
            tmp_path
            / "data/08_reporting/recommendations/2026/round-3/live/runs/run_started_at=20260429T123456000000Z"
        ),
    }
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        current_year=2026,
        footystats_mode="none",
        output_run_id="run_started_at=20260429T123456000000Z",
        live_workflow=live_workflow,
    )

    result = run_recommendation(config)

    metadata_path = config.output_path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["live_workflow"] == live_workflow
    assert result.metadata["live_workflow"] == live_workflow
```

Add absolute-root rejection:

```python
def test_run_recommendation_rejects_absolute_output_root_outside_project(tmp_path: Path) -> None:
    config = RecommendationConfig(
        season=2026,
        target_round=3,
        mode="live",
        project_root=tmp_path,
        output_root=Path("/tmp/outside-cartola-recommendations"),
        current_year=2026,
        footystats_mode="none",
    )

    with pytest.raises(ValueError, match="inside project_root"):
        run_recommendation(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_recommendation.py::test_recommendation_config_output_path_includes_output_run_id \
  src/tests/backtesting/test_recommendation.py::test_recommendation_config_rejects_output_run_id_with_path_separator \
  src/tests/backtesting/test_recommendation.py::test_run_recommendation_writes_live_workflow_metadata_link \
  src/tests/backtesting/test_recommendation.py::test_run_recommendation_rejects_absolute_output_root_outside_project \
  -q
```

Expected: FAIL because `output_run_id` and `live_workflow` are not fields yet.

- [ ] **Step 3: Extend `RecommendationConfig`**

In `src/cartola/backtesting/recommendation.py`, add fields:

```python
    output_run_id: str | None = None
    live_workflow: Mapping[str, object] | None = None
```

Update `output_path`:

```python
    @property
    def output_path(self) -> Path:
        base = self.project_root / self.output_root / str(self.season) / f"round-{self.target_round}" / self.mode
        if self.output_run_id is None:
            return base
        return base / "runs" / self.output_run_id
```

In `_validate_mode_scope()`, add:

```python
    if config.output_run_id is not None:
        run_id_path = Path(config.output_run_id)
        if run_id_path.name != config.output_run_id or run_id_path.is_absolute():
            raise ValueError(f"output_run_id must be a single path segment: {config.output_run_id!r}")
```

- [ ] **Step 4: Tighten output-root validation**

Replace `_validate_output_root()` with:

```python
def _validate_output_root(config: RecommendationConfig) -> None:
    project_root = config.project_root.resolve()
    protected_backtest_root = (project_root / "data" / "08_reporting" / "backtests").resolve()
    output_root = _resolve_output_root(config)
    if project_root != output_root and project_root not in output_root.parents:
        raise ValueError(f"Recommendation output_root must resolve inside project_root: output_root={config.output_root}")
    if output_root == protected_backtest_root or protected_backtest_root in output_root.parents:
        raise ValueError(
            "Recommendation output_root cannot be inside backtest reports: "
            f"output_root={config.output_root}"
        )
```

- [ ] **Step 5: Include `live_workflow` before metadata write**

In `_build_metadata()`, add:

```python
        "live_workflow": dict(config.live_workflow) if config.live_workflow is not None else None,
```

Keep this inside the returned metadata object before `_write_recommendation_outputs()` is called. Do not patch `run_metadata.json` later.

- [ ] **Step 6: Run recommendation tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_recommendation.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cartola/backtesting/recommendation.py src/tests/backtesting/test_recommendation.py
git commit -m "feat: archive recommendation outputs"
```

---

### Task 3: Core Live Workflow Module

**Files:**
- Create: `src/cartola/backtesting/live_workflow.py`
- Create: `src/tests/backtesting/test_live_workflow.py`

- [ ] **Step 1: Write failing tests for `fresh` policy**

Create `src/tests/backtesting/test_live_workflow.py` with:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cartola.backtesting.live_workflow import LiveWorkflowConfig, run_live_round
from cartola.backtesting.market_capture import LiveCaptureMetadata, MarketCaptureResult
from cartola.backtesting.recommendation import RecommendationConfig, RecommendationResult


def _capture_metadata(tmp_path: Path, *, round_number: int = 14) -> LiveCaptureMetadata:
    return LiveCaptureMetadata(
        csv_path=tmp_path / f"data/01_raw/2026/rodada-{round_number}.csv",
        metadata_path=tmp_path / f"data/01_raw/2026/rodada-{round_number}.capture.json",
        season=2026,
        target_round=round_number,
        csv_sha256="a" * 64,
        captured_at_utc="2026-04-29T12:00:00Z",
        status_mercado=1,
        deadline_timestamp=1777748340,
        deadline_parse_status="ok",
    )


def _recommendation_result(config: RecommendationConfig) -> RecommendationResult:
    summary = {
        "season": config.season,
        "target_round": config.target_round,
        "mode": config.mode,
        "budget": config.budget,
        "budget_used": 99.5,
        "predicted_points": 73.25,
        "selected_count": 12,
        "output_directory": str(config.output_path),
    }
    config.output_path.mkdir(parents=True, exist_ok=True)
    (config.output_path / "run_metadata.json").write_text(
        json.dumps({"live_workflow": config.live_workflow}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RecommendationResult(
        recommended_squad=None,
        candidate_predictions=None,
        summary=summary,
        metadata={"live_workflow": config.live_workflow},
    )


def test_run_live_round_fresh_captures_and_uses_capture_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_calls = []
    recommendation_calls = []
    metadata = _capture_metadata(tmp_path, round_number=14)

    def fake_capture(config, **kwargs):
        capture_calls.append(config)
        return MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        )

    def fake_load_capture(**kwargs):
        assert kwargs == {"project_root": tmp_path, "season": 2026, "target_round": 14}
        return metadata

    def fake_recommend(config):
        recommendation_calls.append(config)
        return _recommendation_result(config)

    monkeypatch.setattr("cartola.backtesting.live_workflow.capture_market_round", fake_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fake_recommend)

    result = run_live_round(
        LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
        now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
    )

    assert capture_calls[0].auto is True
    assert capture_calls[0].force is True
    assert recommendation_calls[0].target_round == 14
    assert recommendation_calls[0].mode == "live"
    assert recommendation_calls[0].output_run_id == "run_started_at=20260429T123456123456Z"
    assert recommendation_calls[0].live_workflow["capture_policy"] == "fresh"
    assert recommendation_calls[0].live_workflow["capture_csv_sha256"] == "a" * 64
    assert result.workflow_metadata["predicted_points"] == 73.25
    assert result.workflow_metadata["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_live_workflow.py::test_run_live_round_fresh_captures_and_uses_capture_round -q
```

Expected: FAIL because `live_workflow.py` does not exist.

- [ ] **Step 3: Implement core dataclasses and fresh policy**

Create `src/cartola/backtesting/live_workflow.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

from cartola.backtesting.market_capture import (
    LiveCaptureMetadata,
    MarketCaptureConfig,
    capture_market_round,
    load_valid_live_capture,
)
from cartola.backtesting.recommendation import RecommendationConfig, RecommendationResult, run_recommendation

CapturePolicy = Literal["fresh", "missing", "skip"]
WorkflowStatus = Literal["ok", "failed"]
ErrorStage = Literal["status_fetch", "capture_validation", "capture", "recommendation", "workflow_metadata"]
Clock = Callable[[], datetime]
WORKFLOW_VERSION = "live_workflow_v1"


@dataclass(frozen=True)
class LiveWorkflowConfig:
    season: int
    budget: float = 100.0
    project_root: Path = Path(".")
    output_root: Path = Path("data/08_reporting/recommendations")
    footystats_mode: Literal["none", "ppg", "ppg_xg"] = "ppg"
    footystats_league_slug: str = "brazil-serie-a"
    footystats_dir: Path = Path("data/footystats")
    current_year: int | None = None
    capture_policy: CapturePolicy = "fresh"
    allow_finalized_live_data: bool = False
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class LiveWorkflowResult:
    recommendation: RecommendationResult | None
    workflow_metadata: dict[str, object]
    output_path: Path | None


def _runtime_current_year() -> int:
    return datetime.now(UTC).year


def _resolved_current_year(config: LiveWorkflowConfig) -> int:
    return config.current_year if config.current_year is not None else _runtime_current_year()


def _run_started_at(now: Clock) -> tuple[str, str]:
    current = now().astimezone(UTC)
    compact = current.strftime("%Y%m%dT%H%M%S%fZ")
    iso = current.isoformat().replace("+00:00", "Z")
    return iso, f"run_started_at={compact}"


def _capture_age_seconds(captured_at_utc: str, now: Clock) -> float:
    captured = datetime.fromisoformat(captured_at_utc.replace("Z", "+00:00"))
    return max(0.0, (now().astimezone(UTC) - captured).total_seconds())


def _validate_current_year(config: LiveWorkflowConfig) -> int:
    current_year = _resolved_current_year(config)
    if config.season != current_year:
        raise ValueError(f"live workflow requires season {config.season} to equal current_year {current_year}")
    return current_year


def _live_workflow_link(
    *,
    config: LiveWorkflowConfig,
    run_started_at_utc: str,
    output_run_id: str,
    target_round: int,
    capture: LiveCaptureMetadata,
    capture_age_seconds: float,
) -> dict[str, object]:
    recommendation_output_path = (
        config.project_root
        / config.output_root
        / str(config.season)
        / f"round-{target_round}"
        / "live"
        / "runs"
        / output_run_id
    )
    return {
        "workflow_version": WORKFLOW_VERSION,
        "run_started_at_utc": run_started_at_utc,
        "capture_policy": config.capture_policy,
        "season": config.season,
        "current_year": _resolved_current_year(config),
        "target_round": target_round,
        "budget": float(config.budget),
        "footystats_mode": config.footystats_mode,
        "footystats_league_slug": config.footystats_league_slug,
        "capture_csv_path": str(capture.csv_path),
        "capture_metadata_path": str(capture.metadata_path),
        "capture_csv_sha256": capture.csv_sha256,
        "capture_captured_at_utc": capture.captured_at_utc,
        "capture_age_seconds": capture_age_seconds,
        "capture_status_mercado": capture.status_mercado,
        "capture_deadline_timestamp": capture.deadline_timestamp,
        "capture_deadline_parse_status": capture.deadline_parse_status,
        "recommendation_output_path": str(recommendation_output_path),
    }


def _workflow_metadata(
    *,
    live_workflow: dict[str, object],
    recommendation: RecommendationResult | None,
    status: WorkflowStatus,
    error_stage: ErrorStage | None = None,
    error: Exception | None = None,
) -> dict[str, object]:
    metadata = dict(live_workflow)
    output_path = Path(str(live_workflow["recommendation_output_path"]))
    metadata.update(
        {
            "recommendation_summary_path": str(output_path / "recommendation_summary.json"),
            "recommendation_metadata_path": str(output_path / "run_metadata.json"),
            "recommended_squad_path": str(output_path / "recommended_squad.csv"),
            "candidate_predictions_path": str(output_path / "candidate_predictions.csv"),
            "selected_count": None if recommendation is None else recommendation.summary.get("selected_count"),
            "predicted_points": None if recommendation is None else recommendation.summary.get("predicted_points"),
            "budget_used": None if recommendation is None else recommendation.summary.get("budget_used"),
            "status": status,
            "error_stage": error_stage,
            "error_type": None if error is None else type(error).__name__,
            "error_message": None if error is None else str(error),
        }
    )
    return metadata


def _write_workflow_metadata(output_path: Path, metadata: dict[str, object]) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "live_workflow_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _capture_fresh(config: LiveWorkflowConfig) -> tuple[int, LiveCaptureMetadata]:
    result = capture_market_round(
        MarketCaptureConfig(
            season=config.season,
            auto=True,
            force=True,
            current_year=config.current_year,
            project_root=config.project_root,
            timeout_seconds=config.timeout_seconds,
        )
    )
    capture = load_valid_live_capture(project_root=config.project_root, season=config.season, target_round=result.target_round)
    return result.target_round, capture


def run_live_round(config: LiveWorkflowConfig, *, now: Clock = lambda: datetime.now(UTC)) -> LiveWorkflowResult:
    _validate_current_year(config)
    run_started_at_utc, output_run_id = _run_started_at(now)
    if config.capture_policy != "fresh":
        raise ValueError(f"Unsupported capture policy: {config.capture_policy}")

    target_round, capture = _capture_fresh(config)
    capture_age = _capture_age_seconds(capture.captured_at_utc, now)
    live_workflow = _live_workflow_link(
        config=config,
        run_started_at_utc=run_started_at_utc,
        output_run_id=output_run_id,
        target_round=target_round,
        capture=capture,
        capture_age_seconds=capture_age,
    )
    recommendation_config = RecommendationConfig(
        season=config.season,
        target_round=target_round,
        mode="live",
        budget=config.budget,
        project_root=config.project_root,
        output_root=config.output_root,
        footystats_mode=config.footystats_mode,
        footystats_league_slug=config.footystats_league_slug,
        footystats_dir=config.footystats_dir,
        current_year=config.current_year,
        allow_finalized_live_data=config.allow_finalized_live_data,
        output_run_id=output_run_id,
        live_workflow=live_workflow,
    )
    recommendation = run_recommendation(recommendation_config)
    metadata = _workflow_metadata(live_workflow=live_workflow, recommendation=recommendation, status="ok")
    _write_workflow_metadata(recommendation_config.output_path, metadata)
    return LiveWorkflowResult(
        recommendation=recommendation,
        workflow_metadata=metadata,
        output_path=recommendation_config.output_path,
    )
```

- [ ] **Step 4: Run fresh policy test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_live_workflow.py::test_run_live_round_fresh_captures_and_uses_capture_round -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/live_workflow.py src/tests/backtesting/test_live_workflow.py
git commit -m "feat: orchestrate fresh live round workflow"
```

---

### Task 4: Missing and Skip Capture Policies

**Files:**
- Modify: `src/cartola/backtesting/live_workflow.py`
- Modify: `src/tests/backtesting/test_live_workflow.py`

- [ ] **Step 1: Write failing tests for missing/skip policies**

Append to `src/tests/backtesting/test_live_workflow.py`:

```python
def test_run_live_round_missing_reuses_valid_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    capture_calls = []
    status_calls = []
    recommendation_calls = []

    def fake_fetch_status(config):
        status_calls.append(config)
        return 14

    def fake_load_capture(**kwargs):
        return metadata

    def fake_capture(config, **kwargs):
        capture_calls.append(config)
        raise AssertionError("missing policy should not capture when valid capture exists")

    def fake_recommend(config):
        recommendation_calls.append(config)
        return _recommendation_result(config)

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.capture_market_round", fake_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fake_recommend)

    result = run_live_round(
        LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="missing"),
        now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
    )

    assert status_calls
    assert capture_calls == []
    assert recommendation_calls[0].target_round == 14
    assert result.workflow_metadata["capture_age_seconds"] == 300.0


def test_run_live_round_missing_captures_when_capture_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    capture_calls = []
    load_calls = 0

    def fake_fetch_status(config):
        return 14

    def fake_load_capture(**kwargs):
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            raise FileNotFoundError("live capture files missing for season=2026 target_round=14")
        return metadata

    def fake_capture(config, **kwargs):
        capture_calls.append(config)
        return MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        )

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.capture_market_round", fake_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", _recommendation_result)

    run_live_round(
        LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="missing"),
        now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
    )

    assert capture_calls[0].auto is True
    assert capture_calls[0].force is False


def test_run_live_round_skip_requires_valid_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_status(config):
        return 14

    def fake_load_capture(**kwargs):
        raise ValueError("destination is not a previous valid live capture")

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)

    with pytest.raises(ValueError, match="previous valid live capture"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="skip"),
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    assert not (tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs").exists()


def test_run_live_round_missing_fails_on_invalid_existing_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_calls = []

    def fake_fetch_status(config):
        return 14

    def fake_load_capture(**kwargs):
        raise ValueError("destination is not a previous valid live capture")

    def fake_capture(config, **kwargs):
        capture_calls.append(config)
        raise AssertionError("invalid existing capture must not be overwritten by missing policy")

    monkeypatch.setattr("cartola.backtesting.live_workflow._active_open_round", fake_fetch_status)
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", fake_load_capture)
    monkeypatch.setattr("cartola.backtesting.live_workflow.capture_market_round", fake_capture)

    with pytest.raises(ValueError, match="previous valid live capture"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026, capture_policy="missing"),
            now=lambda: datetime(2026, 4, 29, 12, 5, tzinfo=UTC),
        )

    assert capture_calls == []
    assert not (tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_live_workflow.py::test_run_live_round_missing_reuses_valid_capture \
  src/tests/backtesting/test_live_workflow.py::test_run_live_round_missing_captures_when_capture_is_absent \
  src/tests/backtesting/test_live_workflow.py::test_run_live_round_skip_requires_valid_capture \
  src/tests/backtesting/test_live_workflow.py::test_run_live_round_missing_fails_on_invalid_existing_capture \
  -q
```

Expected: FAIL because `_active_open_round`, `missing`, and `skip` are not implemented.

- [ ] **Step 3: Implement status fetch and policies**

In `live_workflow.py`, add:

```python
from typing import Any

from cartola.backtesting.market_capture import CARTOLA_STATUS_ENDPOINT, fetch_cartola_json
```

Add:

```python
def _int_payload_field(payload: dict[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _active_open_round(config: LiveWorkflowConfig) -> int:
    response = fetch_cartola_json(CARTOLA_STATUS_ENDPOINT, config.timeout_seconds)
    target_round = _int_payload_field(response.payload, "rodada_atual")
    if target_round <= 0:
        raise ValueError("rodada_atual must be a positive integer")
    status_mercado = _int_payload_field(response.payload, "status_mercado")
    if status_mercado != 1:
        raise ValueError(f"Cartola market is not open: rodada_atual={target_round} status_mercado {status_mercado}")
    return target_round


def _capture_missing(config: LiveWorkflowConfig) -> tuple[int, LiveCaptureMetadata]:
    target_round = _active_open_round(config)
    try:
        capture = load_valid_live_capture(project_root=config.project_root, season=config.season, target_round=target_round)
        return target_round, capture
    except FileNotFoundError:
        result = capture_market_round(
            MarketCaptureConfig(
                season=config.season,
                auto=True,
                force=False,
                current_year=config.current_year,
                project_root=config.project_root,
                timeout_seconds=config.timeout_seconds,
            )
        )
        capture = load_valid_live_capture(
            project_root=config.project_root,
            season=config.season,
            target_round=result.target_round,
        )
        return result.target_round, capture


def _capture_skip(config: LiveWorkflowConfig) -> tuple[int, LiveCaptureMetadata]:
    target_round = _active_open_round(config)
    capture = load_valid_live_capture(project_root=config.project_root, season=config.season, target_round=target_round)
    return target_round, capture


def _resolve_capture(config: LiveWorkflowConfig) -> tuple[int, LiveCaptureMetadata]:
    if config.capture_policy == "fresh":
        return _capture_fresh(config)
    if config.capture_policy == "missing":
        return _capture_missing(config)
    if config.capture_policy == "skip":
        return _capture_skip(config)
    raise ValueError(f"Unsupported capture policy: {config.capture_policy}")
```

Update `run_live_round()`:

```python
    target_round, capture = _resolve_capture(config)
```

Remove the earlier direct `fresh`-only branch.

- [ ] **Step 4: Run policy tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_live_workflow.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/live_workflow.py src/tests/backtesting/test_live_workflow.py
git commit -m "feat: support live capture policies"
```

---

### Task 5: Failure Metadata and Archive Safety

**Files:**
- Modify: `src/cartola/backtesting/live_workflow.py`
- Modify: `src/tests/backtesting/test_live_workflow.py`

- [ ] **Step 1: Write failing tests for failure metadata and path collision**

Append:

```python
def test_run_live_round_recommendation_failure_writes_failed_workflow_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)

    def fail_recommendation(config):
        raise ValueError("FootyStats recommendation missing join keys: {14: [264]}")

    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", fail_recommendation)

    with pytest.raises(ValueError, match="missing join keys"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
            now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
        )

    output_path = (
        tmp_path
        / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=20260429T123456123456Z"
    )
    workflow = json.loads((output_path / "live_workflow_metadata.json").read_text(encoding="utf-8"))
    assert workflow["status"] == "failed"
    assert workflow["error_stage"] == "recommendation"
    assert workflow["capture_csv_sha256"] == "a" * 64
    assert workflow["error_type"] == "ValueError"


def test_run_live_round_archive_collision_fails_before_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)
    output_path = (
        tmp_path
        / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=20260429T123456123456Z"
    )
    output_path.mkdir(parents=True)
    recommend_calls = []

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", lambda config: recommend_calls.append(config))

    with pytest.raises(FileExistsError, match="recommendation archive already exists"):
        run_live_round(
            LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
            now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
        )

    assert recommend_calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_live_workflow.py::test_run_live_round_recommendation_failure_writes_failed_workflow_metadata \
  src/tests/backtesting/test_live_workflow.py::test_run_live_round_archive_collision_fails_before_recommendation \
  -q
```

Expected: FAIL because recommendation failure metadata and collision checks are not implemented.

- [ ] **Step 3: Implement collision and recommendation failure handling**

In `live_workflow.py`, add:

```python
def _assert_archive_available(output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"recommendation archive already exists: {output_path}")
```

In `run_live_round()` after `recommendation_config` is built and before `run_recommendation()`:

```python
    _assert_archive_available(recommendation_config.output_path)
```

Wrap recommendation execution:

```python
    try:
        recommendation = run_recommendation(recommendation_config)
    except Exception as exc:
        metadata = _workflow_metadata(
            live_workflow=live_workflow,
            recommendation=None,
            status="failed",
            error_stage="recommendation",
            error=exc,
        )
        _write_workflow_metadata(recommendation_config.output_path, metadata)
        raise
```

Keep the success path writing `status="ok"`.

- [ ] **Step 4: Run live workflow tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_live_workflow.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/live_workflow.py src/tests/backtesting/test_live_workflow.py
git commit -m "feat: record live workflow failures"
```

---

### Task 6: CLI Wrapper

**Files:**
- Create: `scripts/run_live_round.py`
- Create: `src/tests/backtesting/test_run_live_round_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `src/tests/backtesting/test_run_live_round_cli.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cartola.backtesting.live_workflow import LiveWorkflowConfig, LiveWorkflowResult

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_live_round.py"
SPEC = importlib.util.spec_from_file_location("run_live_round", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_live_round_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_live_round_cli)
main = run_live_round_cli.main
parse_args = run_live_round_cli.parse_args


def test_parse_args_builds_live_workflow_defaults() -> None:
    args = parse_args(["--season", "2026", "--current-year", "2026"])

    assert args.season == 2026
    assert args.budget == 100.0
    assert args.footystats_mode == "ppg"
    assert args.capture_policy == "fresh"
    assert args.output_root == Path("data/08_reporting/recommendations")


def test_parse_args_rejects_target_round() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--season", "2026", "--target-round", "14", "--current-year", "2026"])


def test_main_builds_workflow_config_and_prints_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    observed: list[LiveWorkflowConfig] = []

    def fake_run_live_round(config: LiveWorkflowConfig) -> LiveWorkflowResult:
        observed.append(config)
        return LiveWorkflowResult(
            recommendation=None,
            output_path=tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=x",
            workflow_metadata={
                "status": "ok",
                "capture_policy": "fresh",
                "target_round": 14,
                "capture_captured_at_utc": "2026-04-29T12:00:00Z",
                "capture_age_seconds": 300.0,
                "selected_count": 12,
                "predicted_points": 73.25,
                "budget_used": 99.5,
                "recommendation_output_path": str(
                    tmp_path / "data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=x"
                ),
                "capture_metadata_path": str(tmp_path / "data/01_raw/2026/rodada-14.capture.json"),
                "footystats_mode": "ppg",
            },
        )

    monkeypatch.setattr(run_live_round_cli, "run_live_round", fake_run_live_round)

    exit_code = main(["--season", "2026", "--project-root", str(tmp_path), "--current-year", "2026"])

    assert exit_code == 0
    assert observed == [LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026)]
    output = capsys.readouterr().out
    assert "Live round complete" in output
    assert "Capture policy" in output
    assert "fresh" in output
    assert "Target round" in output
    assert "14" in output
    assert "Predicted points" in output
    assert "73.25" in output
    assert "FootyStats mode" in output
    assert "ppg" in output


def test_main_prints_expected_error_without_traceback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    def fake_run_live_round(config: LiveWorkflowConfig) -> LiveWorkflowResult:
        raise ValueError("live workflow requires season 2025 to equal current_year 2026")

    monkeypatch.setattr(run_live_round_cli, "run_live_round", fake_run_live_round)

    exit_code = main(["--season", "2025", "--project-root", str(tmp_path), "--current-year", "2026"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Live round failed" in captured.err
    assert "current_year 2026" in captured.err
    assert "Traceback" not in captured.err
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_live_round_cli.py -q
```

Expected: FAIL because `scripts/run_live_round.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/run_live_round.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cartola.backtesting.live_workflow import LiveWorkflowConfig, run_live_round


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the open Cartola market and generate a live squad recommendation.")
    parser.add_argument("--season", type=_positive_int, required=True)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/08_reporting/recommendations"))
    parser.add_argument("--footystats-mode", choices=("none", "ppg", "ppg_xg"), default="ppg")
    parser.add_argument("--footystats-league-slug", default="brazil-serie-a")
    parser.add_argument("--footystats-dir", type=Path, default=Path("data/footystats"))
    parser.add_argument("--current-year", type=_positive_int, default=None)
    parser.add_argument("--capture-policy", choices=("fresh", "missing", "skip"), default="fresh")
    parser.add_argument("--allow-finalized-live-data", action="store_true")
    return parser.parse_args(argv)


def _format_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _print_success(console: Console, metadata: dict[str, object]) -> None:
    console.print(
        Panel(
            f"season={metadata.get('season')}  round={metadata.get('target_round')}  status={metadata.get('status')}",
            title="Live round complete",
            border_style="green",
        )
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Capture policy", str(metadata.get("capture_policy")))
    table.add_row("Target round", str(metadata.get("target_round")))
    table.add_row("Capture timestamp", str(metadata.get("capture_captured_at_utc")))
    table.add_row("Capture age seconds", str(metadata.get("capture_age_seconds")))
    table.add_row("FootyStats mode", str(metadata.get("footystats_mode")))
    table.add_row("Selected players", str(metadata.get("selected_count")))
    table.add_row("Predicted points", _format_float(metadata.get("predicted_points")))
    table.add_row("Budget used", _format_float(metadata.get("budget_used")))
    table.add_row("Recommendation output", str(metadata.get("recommendation_output_path")))
    table.add_row("Capture metadata", str(metadata.get("capture_metadata_path")))
    console.print(table)


def _print_error(console: Console, error: ValueError) -> None:
    console.print(Panel(str(error), title="Live round failed", border_style="red"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = LiveWorkflowConfig(
        season=args.season,
        budget=args.budget,
        project_root=args.project_root,
        output_root=args.output_root,
        footystats_mode=args.footystats_mode,
        footystats_league_slug=args.footystats_league_slug,
        footystats_dir=args.footystats_dir,
        current_year=args.current_year,
        capture_policy=args.capture_policy,
        allow_finalized_live_data=args.allow_finalized_live_data,
    )
    stdout = Console()
    stderr = Console(stderr=True)
    try:
        result = run_live_round(config)
    except ValueError as error:
        _print_error(stderr, error)
        return 1
    _print_success(stdout, result.workflow_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_run_live_round_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_live_round.py src/tests/backtesting/test_run_live_round_cli.py
git commit -m "feat: add live round workflow cli"
```

---

### Task 7: Metadata Consistency and End-to-End Smoke Test

**Files:**
- Modify: `src/tests/backtesting/test_live_workflow.py`

- [ ] **Step 1: Write metadata consistency test**

Add to `src/tests/backtesting/test_live_workflow.py`:

```python
def test_workflow_metadata_matches_recommendation_live_workflow_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _capture_metadata(tmp_path, round_number=14)

    monkeypatch.setattr(
        "cartola.backtesting.live_workflow.capture_market_round",
        lambda config, **kwargs: MarketCaptureResult(
            csv_path=metadata.csv_path,
            metadata_path=metadata.metadata_path,
            target_round=14,
            athlete_count=747,
            status_mercado=1,
            deadline_timestamp=1777748340,
            deadline_parse_status="ok",
        ),
    )
    monkeypatch.setattr("cartola.backtesting.live_workflow.load_valid_live_capture", lambda **kwargs: metadata)
    monkeypatch.setattr("cartola.backtesting.live_workflow.run_recommendation", _recommendation_result)

    result = run_live_round(
        LiveWorkflowConfig(season=2026, project_root=tmp_path, current_year=2026),
        now=lambda: datetime(2026, 4, 29, 12, 34, 56, 123456, tzinfo=UTC),
    )

    workflow = json.loads((result.output_path / "live_workflow_metadata.json").read_text(encoding="utf-8"))
    recommendation_metadata = json.loads((result.output_path / "run_metadata.json").read_text(encoding="utf-8"))
    link = recommendation_metadata["live_workflow"]

    for key in (
        "capture_policy",
        "target_round",
        "capture_csv_path",
        "capture_metadata_path",
        "capture_csv_sha256",
        "recommendation_output_path",
    ):
        assert workflow[key] == link[key]
```

- [ ] **Step 2: Run consistency test**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_live_workflow.py::test_workflow_metadata_matches_recommendation_live_workflow_link -q
```

Expected: PASS, proving the same `live_workflow` object feeds both outputs.

- [ ] **Step 3: Run all changed tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_market_capture.py \
  src/tests/backtesting/test_recommendation.py \
  src/tests/backtesting/test_live_workflow.py \
  src/tests/backtesting/test_run_live_round_cli.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/tests/backtesting/test_live_workflow.py
git commit -m "test: verify live workflow metadata consistency"
```

---

### Task 8: Documentation and Roadmap

**Files:**
- Modify: `README.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Update README with one-command workflow**

In `README.md`, add this command near the live recommendation section:

````markdown
One-command live workflow:

```bash
uv run --frozen python scripts/run_live_round.py \
  --season 2026 \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026
```

By default this uses `--capture-policy fresh`: it refreshes the open market through
the safe live-capture primitive, then recommends the squad for the captured round.
Use `--capture-policy missing` to reuse a valid existing live capture, or
`--capture-policy skip` to require one without fetching `atletas/mercado`.
```
````

- [ ] **Step 2: Update roadmap**

In `roadmap.md`:

- Add one-command live workflow to **Delivered**.
- Remove or revise the roadmap item that says "Add a higher-level live round workflow around the capture and recommendation commands".
- Keep DNP modeling and odds/goal-environment ablation as upcoming feature bets.

- [ ] **Step 3: Commit docs**

```bash
git add README.md roadmap.md
git commit -m "docs: document one-command live workflow"
```

---

### Task 9: Full Verification and Real Smoke

**Files:**
- No planned source edits unless verification exposes a bug.

- [ ] **Step 1: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected:

- Ruff passes.
- ty passes.
- Bandit reports no issues.
- pytest passes.

- [ ] **Step 2: Run local smoke with `skip` if a valid capture exists**

If `data/01_raw/2026/rodada-14.csv` and `.capture.json` exist from the current open round, run:

```bash
uv run --frozen python scripts/run_live_round.py \
  --season 2026 \
  --budget 100 \
  --footystats-mode ppg \
  --current-year 2026 \
  --capture-policy skip
```

Expected: command exits `0`, prints `Live round complete`, and writes an archived output under:

```text
data/08_reporting/recommendations/2026/round-14/live/runs/run_started_at=...
```

If the capture does not exist or the market has moved to a different open round, run the same command with `--capture-policy fresh` instead.

- [ ] **Step 3: Inspect generated metadata**

Run:

```bash
export latest_dir="$(ls -td data/08_reporting/recommendations/2026/round-*/live/runs/run_started_at=* | head -1)"
python - <<'PY'
import json
import os
from pathlib import Path

latest = Path(os.environ["latest_dir"])
workflow = json.loads((latest / "live_workflow_metadata.json").read_text())
metadata = json.loads((latest / "run_metadata.json").read_text())
link = metadata["live_workflow"]
for key in [
    "capture_policy",
    "target_round",
    "capture_csv_path",
    "capture_metadata_path",
    "capture_csv_sha256",
    "recommendation_output_path",
]:
    assert workflow[key] == link[key], key
print(latest)
print(workflow["status"], workflow["predicted_points"], workflow["budget_used"])
PY
```

Expected: assertions pass and output prints the latest archive directory plus summary values.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short --untracked-files=all
```

Expected: no tracked source/doc changes remain after the previous commits. Generated live data and recommendation artifacts must be ignored or absent from tracked output.
