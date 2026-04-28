# FootyStats PPG Ablation Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-fixture FootyStats PPG ablation runner that compares `footystats_mode=none` vs `footystats_mode=ppg` for historical candidate seasons and writes auditable CSV/JSON reports.

**Architecture:** Add one focused orchestration module, `src/cartola/backtesting/footystats_ablation.py`, that owns CLI parsing, path safety, eligibility, paired backtest execution, metric extraction, aggregation, and report writing. Add a thin script, `scripts/run_footystats_ppg_ablation.py`, that delegates to the module. Keep the normal backtest runner unchanged and reuse it through `BacktestConfig`.

**Tech Stack:** Python 3.13, pandas, dataclasses, pathlib, pytest, existing Cartola backtesting modules.

---

## File Structure

- Create `src/cartola/backtesting/footystats_ablation.py`
  - Dataclasses for ablation config, errors, run metrics, season rows, and result.
  - `parse_args`, `main`, `run_footystats_ppg_ablation`.
  - Path validation and safe `--force` deletion.
  - Runtime eligibility via `load_footystats_ppg_rows`.
  - Paired control/treatment `BacktestConfig` construction.
  - Metric extraction from `summary.csv` and `diagnostics.csv`.
  - Aggregate calculation.
  - Atomic CSV/JSON writes.
- Create `scripts/run_footystats_ppg_ablation.py`
  - Thin executable entrypoint that imports `main`.
- Create `src/tests/backtesting/test_footystats_ablation.py`
  - Focused unit tests using temp project roots and monkeypatched `run_backtest`.
- Modify `README.md`
  - Add the ablation command and output paths.
- Modify `roadmap.md`
  - Mark the ablation report as the current next executable milestone once implemented.

---

## Task 1: CLI Parsing And Config Types

**Files:**
- Create: `src/cartola/backtesting/footystats_ablation.py`
- Create: `src/tests/backtesting/test_footystats_ablation.py`
- Create: `scripts/run_footystats_ppg_ablation.py`

- [ ] **Step 1: Write failing tests for season parsing and CLI defaults**

Add this initial test file:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from cartola.backtesting import footystats_ablation as ablation


def test_parse_seasons_preserves_order_and_rejects_duplicates() -> None:
    assert ablation.parse_seasons("2025,2023,2024") == (2025, 2023, 2024)

    with pytest.raises(ValueError, match="duplicate season"):
        ablation.parse_seasons("2023,2024,2023")


def test_parse_seasons_rejects_empty_and_non_positive_values() -> None:
    with pytest.raises(ValueError, match="empty season"):
        ablation.parse_seasons("2023,,2025")

    with pytest.raises(ValueError, match="positive integer"):
        ablation.parse_seasons("2023,0")


def test_parse_args_defaults_to_historical_ppg_ablation() -> None:
    config = ablation.config_from_args(ablation.parse_args([]))

    assert config.seasons == (2023, 2024, 2025)
    assert config.start_round == 5
    assert config.budget == 100.0
    assert config.project_root == Path(".")
    assert config.output_root == Path("data/08_reporting/backtests/footystats_ablation")
    assert config.footystats_league_slug == "brazil-serie-a"
    assert config.force is False
```

Create the script test after the module exists:

```python
def test_script_imports_main() -> None:
    from scripts import run_footystats_ppg_ablation

    assert run_footystats_ppg_ablation.main is ablation.main
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: fail with `ImportError` or `AttributeError` because `footystats_ablation.py` does not exist yet.

- [ ] **Step 3: Implement minimal module and script**

Create `src/cartola/backtesting/footystats_ablation.py`:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

DEFAULT_SEASONS: tuple[int, ...] = (2023, 2024, 2025)
DEFAULT_OUTPUT_ROOT = Path("data/08_reporting/backtests/footystats_ablation")
DEFAULT_LEAGUE_SLUG = "brazil-serie-a"


@dataclass(frozen=True)
class FootyStatsPPGAblationConfig:
    seasons: tuple[int, ...] = DEFAULT_SEASONS
    start_round: int = 5
    budget: float = 100.0
    project_root: Path = Path(".")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    footystats_league_slug: str = DEFAULT_LEAGUE_SLUG
    current_year: int | None = None
    force: bool = False

    @property
    def resolved_current_year(self) -> int:
        return self.current_year if self.current_year is not None else datetime.now(UTC).year


def parse_seasons(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if any(part.strip() == "" for part in parts):
        raise ValueError("empty season entry in --seasons")

    seasons: list[int] = []
    seen: set[int] = set()
    for part in parts:
        try:
            season = int(part.strip())
        except ValueError as exc:
            raise ValueError(f"season must be a positive integer: {part!r}") from exc
        if season <= 0:
            raise ValueError(f"season must be a positive integer: {season}")
        if season in seen:
            raise ValueError(f"duplicate season in --seasons: {season}")
        seen.add(season)
        seasons.append(season)
    return tuple(seasons)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-fixture FootyStats PPG ablation backtests.")
    parser.add_argument("--seasons", default="2023,2024,2025")
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--footystats-league-slug", default=DEFAULT_LEAGUE_SLUG)
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FootyStatsPPGAblationConfig:
    return FootyStatsPPGAblationConfig(
        seasons=parse_seasons(args.seasons),
        start_round=args.start_round,
        budget=args.budget,
        project_root=args.project_root,
        output_root=args.output_root,
        footystats_league_slug=args.footystats_league_slug,
        current_year=args.current_year,
        force=args.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    print(f"FootyStats PPG ablation config parsed for seasons={config.seasons}")
    return 2
```

Create `scripts/run_footystats_ppg_ablation.py`:

```python
from cartola.backtesting.footystats_ablation import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py scripts/run_footystats_ppg_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "feat: add footystats ppg ablation cli"
```

---

## Task 2: Output Path Validation And Force Handling

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Modify: `src/tests/backtesting/test_footystats_ablation.py`

- [ ] **Step 1: Write failing tests for path safety**

Append:

```python
def test_resolve_output_root_rejects_paths_outside_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / "footystats_ablation"
    config = ablation.FootyStatsPPGAblationConfig(
        project_root=tmp_path,
        output_root=outside,
    )

    with pytest.raises(ValueError, match="inside project_root"):
        ablation.resolve_output_root(config)


def test_resolve_output_root_allows_absolute_paths_inside_project(tmp_path: Path) -> None:
    output_root = tmp_path / "custom" / "footystats_ablation"
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, output_root=output_root)

    assert ablation.resolve_output_root(config) == output_root.resolve()


def test_resolve_output_root_rejects_protected_backtest_paths(tmp_path: Path) -> None:
    protected = tmp_path / "data" / "08_reporting" / "backtests"
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, output_root=protected)

    with pytest.raises(ValueError, match="unsafe ablation output_root"):
        ablation.resolve_output_root(config)


def test_build_backtest_config_uses_mode_specific_output_roots(tmp_path: Path) -> None:
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, output_root=Path("reports/footystats_ablation"))
    output_root = ablation.resolve_output_root(config)

    control = ablation.build_backtest_config(config, season=2025, mode="none", resolved_output_root=output_root)
    treatment = ablation.build_backtest_config(config, season=2025, mode="ppg", resolved_output_root=output_root)

    assert control.fixture_mode == "none"
    assert treatment.fixture_mode == "none"
    assert control.footystats_mode == "none"
    assert treatment.footystats_mode == "ppg"
    assert control.output_root == output_root / "runs" / "2025" / "footystats_mode=none"
    assert treatment.output_root == output_root / "runs" / "2025" / "footystats_mode=ppg"
    assert control.output_path == output_root / "runs" / "2025" / "footystats_mode=none" / "2025"
    assert treatment.output_path == output_root / "runs" / "2025" / "footystats_mode=ppg" / "2025"
    assert control.output_path != tmp_path / "data" / "08_reporting" / "backtests" / "2025"


def test_prepare_output_root_requires_force_for_existing_root(tmp_path: Path) -> None:
    output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
    output_root.mkdir(parents=True)
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        ablation.prepare_output_root(config, output_root)


def test_prepare_output_root_force_removes_only_safe_ablation_root(tmp_path: Path) -> None:
    output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
    stale = output_root / "stale.txt"
    output_root.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, force=True)

    ablation.prepare_output_root(config, output_root)

    assert output_root.exists()
    assert not stale.exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: fail with missing `resolve_output_root`, `build_backtest_config`, and `prepare_output_root`.

- [ ] **Step 3: Implement path validation and config builder**

Add imports:

```python
import shutil

from cartola.backtesting.config import BacktestConfig, FootyStatsMode
```

Add:

```python
def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_output_root(config: FootyStatsPPGAblationConfig) -> Path:
    project_root = config.project_root.resolve()
    output_root = config.output_root if config.output_root.is_absolute() else project_root / config.output_root
    resolved = output_root.resolve()
    if resolved != project_root and not _is_relative_to(resolved, project_root):
        raise ValueError(f"output_root must resolve inside project_root: {resolved}")
    if resolved.name != "footystats_ablation":
        raise ValueError("unsafe ablation output_root: final directory must be named footystats_ablation")

    protected = {
        project_root,
        project_root / "data",
        project_root / "data" / "08_reporting",
        project_root / "data" / "08_reporting" / "backtests",
    }
    protected.update(project_root / "data" / "08_reporting" / "backtests" / str(season) for season in config.seasons)
    if resolved in {path.resolve() for path in protected}:
        raise ValueError(f"unsafe ablation output_root: {resolved}")
    return resolved


def build_backtest_config(
    config: FootyStatsPPGAblationConfig,
    *,
    season: int,
    mode: FootyStatsMode,
    resolved_output_root: Path,
) -> BacktestConfig:
    if mode not in {"none", "ppg"}:
        raise ValueError(f"unsupported ablation mode: {mode}")
    backtest_config = BacktestConfig(
        season=season,
        start_round=config.start_round,
        budget=config.budget,
        project_root=config.project_root.resolve(),
        output_root=resolved_output_root / "runs" / str(season) / f"footystats_mode={mode}",
        fixture_mode="none",
        footystats_mode=mode,
        footystats_evaluation_scope="historical_candidate",
        footystats_league_slug=config.footystats_league_slug,
        current_year=config.resolved_current_year,
    )
    normal_output = config.project_root.resolve() / "data" / "08_reporting" / "backtests" / str(season)
    if backtest_config.output_path.resolve() == normal_output.resolve():
        raise ValueError(f"ablation run would overwrite normal backtest output: {normal_output}")
    return backtest_config


def prepare_output_root(config: FootyStatsPPGAblationConfig, resolved_output_root: Path) -> None:
    if resolved_output_root.exists():
        if not config.force:
            raise FileExistsError(f"ablation output root already exists: {resolved_output_root}")
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: all current tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "feat: validate footystats ablation output paths"
```

---

## Task 3: Eligibility And Run Orchestration Skeleton

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Modify: `src/tests/backtesting/test_footystats_ablation.py`

- [ ] **Step 1: Write failing tests for eligibility and run config**

Append:

```python
class StubLoadResult:
    source_path = Path("data/footystats/brazil-serie-a-matches-2025-to-2025-stats.csv")
    source_sha256 = "abc123"


def test_eligibility_failure_skips_control_and_treatment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fail_loader(**kwargs):
        raise ValueError("not a candidate")

    def fake_run_backtest(config):
        calls.append(config.footystats_mode)
        raise AssertionError("run_backtest must not run for ineligible seasons")

    monkeypatch.setattr(ablation, "load_footystats_ppg_rows", fail_loader)
    monkeypatch.setattr(ablation, "run_backtest", fake_run_backtest)

    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path, output_root=Path("reports/footystats_ablation"))
    result = ablation.run_footystats_ppg_ablation(config)

    assert calls == []
    assert len(result.seasons) == 3
    assert result.seasons[0].control_status == "skipped"
    assert result.seasons[0].treatment_status == "skipped"
    assert result.seasons[0].error_stage == "eligibility"
    assert result.seasons[0].metrics_comparable is False


def test_orchestration_runs_control_then_treatment_for_eligible_season(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, str, str]] = []

    def ok_loader(**kwargs):
        return StubLoadResult()

    def fake_run_backtest(config):
        calls.append((config.season, config.fixture_mode, config.footystats_mode))
        _write_backtest_outputs(config.output_path)
        return object()

    monkeypatch.setattr(ablation, "load_footystats_ppg_rows", ok_loader)
    monkeypatch.setattr(ablation, "run_backtest", fake_run_backtest)

    config = ablation.FootyStatsPPGAblationConfig(
        seasons=(2025,),
        project_root=tmp_path,
        output_root=Path("reports/footystats_ablation"),
    )
    result = ablation.run_footystats_ppg_ablation(config)

    assert calls == [(2025, "none", "none"), (2025, "none", "ppg")]
    assert result.seasons[0].control_status == "ok"
    assert result.seasons[0].treatment_status == "ok"
    assert result.seasons[0].treatment_source_sha256 == "abc123"
```

Add test helper:

```python
import pandas as pd


def _write_backtest_outputs(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"strategy": "baseline", "average_actual_points": 50.0},
            {"strategy": "random_forest", "average_actual_points": 55.0},
        ]
    ).to_csv(output_path / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "section": "prediction",
                "strategy": "random_forest",
                "position": "all",
                "metric": "player_r2",
                "value": 0.1,
            },
            {
                "section": "prediction",
                "strategy": "random_forest",
                "position": "all",
                "metric": "player_correlation",
                "value": 0.2,
            },
        ]
    ).to_csv(output_path / "diagnostics.csv", index=False)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: fail with missing `run_footystats_ppg_ablation` and record classes.

- [ ] **Step 3: Implement orchestration skeleton**

Add imports:

```python
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from cartola.backtesting.footystats_features import FootyStatsPPGLoadResult, load_footystats_ppg_rows
from cartola.backtesting.runner import run_backtest
```

Add dataclasses:

```python
@dataclass(frozen=True)
class AblationError:
    stage: str
    type: str
    message: str
    traceback: str


@dataclass
class SeasonAblationRecord:
    season: int | str
    row_type: str = "season"
    season_status: str = "failed"
    metrics_comparable: bool = False
    control_status: str = "skipped"
    treatment_status: str = "skipped"
    control_output_path: str | None = None
    treatment_output_path: str | None = None
    control_baseline_avg_points: float | None = None
    treatment_baseline_avg_points: float | None = None
    baseline_avg_points: float | None = None
    baseline_avg_points_equal: bool | None = None
    control_rf_avg_points: float | None = None
    treatment_rf_avg_points: float | None = None
    rf_avg_points_delta: float | None = None
    control_player_r2: float | None = None
    treatment_player_r2: float | None = None
    player_r2_delta: float | None = None
    control_player_corr: float | None = None
    treatment_player_corr: float | None = None
    player_corr_delta: float | None = None
    rf_minus_baseline_control: float | None = None
    rf_minus_baseline_treatment: float | None = None
    error_stage: str | None = None
    error_message: str | None = None
    treatment_source_path: str | None = None
    treatment_source_sha256: str | None = None
    control_config: dict[str, Any] | None = None
    treatment_config: dict[str, Any] | None = None
    errors: list[AblationError] = field(default_factory=list)


@dataclass(frozen=True)
class FootyStatsPPGAblationResult:
    config: FootyStatsPPGAblationConfig
    resolved_output_root: Path
    seasons: list[SeasonAblationRecord]
    aggregate: SeasonAblationRecord
```

Add helpers:

```python
def _error(stage: str, exc: BaseException) -> AblationError:
    return AblationError(
        stage=stage,
        type=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


def _config_dict(config: BacktestConfig) -> dict[str, object]:
    data = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}
```

Add `run_footystats_ppg_ablation` that validates eligibility, runs both modes, and calls the metric extraction helper defined in this task:

```python
def run_footystats_ppg_ablation(config: FootyStatsPPGAblationConfig) -> FootyStatsPPGAblationResult:
    resolved_output_root = resolve_output_root(config)
    prepare_output_root(config, resolved_output_root)
    records: list[SeasonAblationRecord] = []

    for season in config.seasons:
        control_config = build_backtest_config(config, season=season, mode="none", resolved_output_root=resolved_output_root)
        treatment_config = build_backtest_config(config, season=season, mode="ppg", resolved_output_root=resolved_output_root)
        record = SeasonAblationRecord(
            season=season,
            control_output_path=str(control_config.output_path),
            treatment_output_path=str(treatment_config.output_path),
            control_config=_config_dict(control_config),
            treatment_config=_config_dict(treatment_config),
        )

        try:
            eligibility = _load_eligibility(config, season)
        except Exception as exc:
            record.error_stage = "eligibility"
            record.error_message = str(exc)
            record.errors.append(_error("eligibility", exc))
            records.append(record)
            continue

        record.season_status = "candidate"
        record.treatment_source_path = str(eligibility.source_path)
        record.treatment_source_sha256 = eligibility.source_sha256

        try:
            run_backtest(control_config)
            record.control_status = "ok"
        except Exception as exc:
            record.control_status = "failed"
            record.error_stage = "control_backtest"
            record.error_message = str(exc)
            record.errors.append(_error("control_backtest", exc))
            records.append(record)
            continue

        try:
            run_backtest(treatment_config)
            record.treatment_status = "ok"
        except Exception as exc:
            record.treatment_status = "failed"
            record.error_stage = "treatment_backtest"
            record.error_message = str(exc)
            record.errors.append(_error("treatment_backtest", exc))
            records.append(record)
            continue

        try:
            _populate_metrics(record, control_config.output_path, treatment_config.output_path)
        except Exception as exc:
            record.error_stage = "metric_extraction"
            record.error_message = str(exc)
            record.errors.append(_error("metric_extraction", exc))

        records.append(record)

    aggregate = build_aggregate_record(records)
    return FootyStatsPPGAblationResult(config=config, resolved_output_root=resolved_output_root, seasons=records, aggregate=aggregate)


def _load_eligibility(config: FootyStatsPPGAblationConfig, season: int) -> FootyStatsPPGLoadResult:
    return load_footystats_ppg_rows(
        season=season,
        project_root=config.project_root.resolve(),
        footystats_dir=Path("data/footystats"),
        league_slug=config.footystats_league_slug,
        evaluation_scope="historical_candidate",
        current_year=config.resolved_current_year,
    )
```

Add first-pass metric functions that Task 4 will refine with edge-case coverage:

```python
def _populate_metrics(record: SeasonAblationRecord, control_path: Path, treatment_path: Path) -> None:
    control = extract_run_metrics(control_path)
    treatment = extract_run_metrics(treatment_path)
    record.control_baseline_avg_points = control["baseline"]
    record.treatment_baseline_avg_points = treatment["baseline"]
    record.baseline_avg_points = control["baseline"]
    record.baseline_avg_points_equal = abs(control["baseline"] - treatment["baseline"]) <= 1e-9
    record.control_rf_avg_points = control["rf"]
    record.treatment_rf_avg_points = treatment["rf"]
    record.control_player_r2 = control["r2"]
    record.treatment_player_r2 = treatment["r2"]
    record.control_player_corr = control["corr"]
    record.treatment_player_corr = treatment["corr"]
    if not record.baseline_avg_points_equal:
        raise ValueError("baseline average points differ between control and treatment")
    record.metrics_comparable = True
    record.rf_avg_points_delta = treatment["rf"] - control["rf"]
    record.player_r2_delta = treatment["r2"] - control["r2"]
    record.player_corr_delta = treatment["corr"] - control["corr"]
    record.rf_minus_baseline_control = control["rf"] - control["baseline"]
    record.rf_minus_baseline_treatment = treatment["rf"] - treatment["baseline"]


def extract_run_metrics(output_path: Path) -> dict[str, float]:
    summary = pd.read_csv(output_path / "summary.csv")
    diagnostics = pd.read_csv(output_path / "diagnostics.csv")
    return {
        "baseline": _summary_value(summary, "baseline"),
        "rf": _summary_value(summary, "random_forest"),
        "r2": _diagnostic_value(diagnostics, "player_r2"),
        "corr": _diagnostic_value(diagnostics, "player_correlation"),
    }


def _summary_value(summary: pd.DataFrame, strategy: str) -> float:
    rows = summary[summary["strategy"].eq(strategy)]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one summary row for strategy {strategy!r}, found {len(rows)}")
    return float(rows["average_actual_points"].iloc[0])


def _diagnostic_value(diagnostics: pd.DataFrame, metric: str) -> float:
    rows = diagnostics[
        diagnostics["section"].eq("prediction")
        & diagnostics["strategy"].eq("random_forest")
        & diagnostics["position"].eq("all")
        & diagnostics["metric"].eq(metric)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one diagnostics row for metric {metric!r}, found {len(rows)}")
    return float(rows["value"].iloc[0])


def build_aggregate_record(records: list[SeasonAblationRecord]) -> SeasonAblationRecord:
    included = [record for record in records if record.metrics_comparable and record.control_status == "ok" and record.treatment_status == "ok"]
    aggregate = SeasonAblationRecord(
        season="aggregate",
        row_type="aggregate",
        season_status="aggregate",
        control_status="not_applicable",
        treatment_status="not_applicable",
        metrics_comparable=bool(included),
    )
    if not included:
        return aggregate
    aggregate.control_rf_avg_points = sum(record.control_rf_avg_points or 0.0 for record in included) / len(included)
    aggregate.treatment_rf_avg_points = sum(record.treatment_rf_avg_points or 0.0 for record in included) / len(included)
    aggregate.rf_avg_points_delta = sum(record.rf_avg_points_delta or 0.0 for record in included) / len(included)
    aggregate.control_player_r2 = sum(record.control_player_r2 or 0.0 for record in included) / len(included)
    aggregate.treatment_player_r2 = sum(record.treatment_player_r2 or 0.0 for record in included) / len(included)
    aggregate.player_r2_delta = sum(record.player_r2_delta or 0.0 for record in included) / len(included)
    aggregate.control_player_corr = sum(record.control_player_corr or 0.0 for record in included) / len(included)
    aggregate.treatment_player_corr = sum(record.treatment_player_corr or 0.0 for record in included) / len(included)
    aggregate.player_corr_delta = sum(record.player_corr_delta or 0.0 for record in included) / len(included)
    return aggregate
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "feat: orchestrate footystats ppg ablation runs"
```

---

## Task 4: Metric Extraction And Aggregate Semantics

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Modify: `src/tests/backtesting/test_footystats_ablation.py`

- [ ] **Step 1: Write failing tests for extraction edge cases**

Append:

```python
def test_extract_run_metrics_requires_prediction_section(tmp_path: Path) -> None:
    output_path = tmp_path / "run"
    output_path.mkdir()
    pd.DataFrame(
        [
            {"strategy": "baseline", "average_actual_points": 50.0},
            {"strategy": "random_forest", "average_actual_points": 55.0},
        ]
    ).to_csv(output_path / "summary.csv", index=False)
    pd.DataFrame(
        [
            {"section": "rounds", "strategy": "random_forest", "position": "all", "metric": "player_r2", "value": 0.99},
            {
                "section": "prediction",
                "strategy": "random_forest",
                "position": "all",
                "metric": "player_r2",
                "value": 0.1,
            },
            {
                "section": "prediction",
                "strategy": "random_forest",
                "position": "all",
                "metric": "player_correlation",
                "value": 0.2,
            },
        ]
    ).to_csv(output_path / "diagnostics.csv", index=False)

    metrics = ablation.extract_run_metrics(output_path)

    assert metrics["r2"] == 0.1


def test_extract_run_metrics_fails_on_missing_or_duplicate_required_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "run"
    output_path.mkdir()
    pd.DataFrame([{"strategy": "baseline", "average_actual_points": 50.0}]).to_csv(
        output_path / "summary.csv", index=False
    )
    pd.DataFrame(columns=["section", "strategy", "position", "metric", "value"]).to_csv(
        output_path / "diagnostics.csv", index=False
    )

    with pytest.raises(ValueError, match="random_forest"):
        ablation.extract_run_metrics(output_path)

    pd.DataFrame(
        [
            {"strategy": "baseline", "average_actual_points": 50.0},
            {"strategy": "baseline", "average_actual_points": 50.0},
            {"strategy": "random_forest", "average_actual_points": 55.0},
        ]
    ).to_csv(output_path / "summary.csv", index=False)

    with pytest.raises(ValueError, match="baseline"):
        ablation.extract_run_metrics(output_path)


def test_baseline_mismatch_marks_record_not_comparable(tmp_path: Path) -> None:
    control_path = tmp_path / "control"
    treatment_path = tmp_path / "treatment"
    _write_backtest_outputs(control_path, baseline=50.0, rf=55.0, r2=0.1, corr=0.2)
    _write_backtest_outputs(treatment_path, baseline=51.0, rf=56.0, r2=0.2, corr=0.3)
    record = ablation.SeasonAblationRecord(season=2025, control_status="ok", treatment_status="ok")

    with pytest.raises(ValueError, match="baseline average points differ"):
        ablation.populate_metrics(record, control_path, treatment_path)

    assert record.baseline_avg_points_equal is False
    assert record.metrics_comparable is False


def test_aggregate_uses_unweighted_mean_of_successful_comparable_rows() -> None:
    good_1 = ablation.SeasonAblationRecord(
        season=2023,
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        rf_avg_points_delta=1.0,
        player_r2_delta=0.02,
        player_corr_delta=0.03,
        control_rf_avg_points=50.0,
        treatment_rf_avg_points=51.0,
        control_player_r2=0.1,
        treatment_player_r2=0.12,
        control_player_corr=0.2,
        treatment_player_corr=0.23,
    )
    good_2 = ablation.SeasonAblationRecord(
        season=2024,
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        rf_avg_points_delta=3.0,
        player_r2_delta=0.04,
        player_corr_delta=0.05,
        control_rf_avg_points=60.0,
        treatment_rf_avg_points=63.0,
        control_player_r2=0.2,
        treatment_player_r2=0.24,
        control_player_corr=0.3,
        treatment_player_corr=0.35,
    )
    failed = ablation.SeasonAblationRecord(season=2025, metrics_comparable=False)

    aggregate = ablation.build_aggregate_record([good_1, failed, good_2])

    assert aggregate.rf_avg_points_delta == 2.0
    assert aggregate.player_r2_delta == pytest.approx(0.03)
    assert aggregate.player_corr_delta == pytest.approx(0.04)
```

Update helper signature:

```python
def _write_backtest_outputs(
    output_path: Path,
    *,
    baseline: float = 50.0,
    rf: float = 55.0,
    r2: float = 0.1,
    corr: float = 0.2,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"strategy": "baseline", "average_actual_points": baseline},
            {"strategy": "random_forest", "average_actual_points": rf},
        ]
    ).to_csv(output_path / "summary.csv", index=False)
    pd.DataFrame(
        [
            {"section": "prediction", "strategy": "random_forest", "position": "all", "metric": "player_r2", "value": r2},
            {
                "section": "prediction",
                "strategy": "random_forest",
                "position": "all",
                "metric": "player_correlation",
                "value": corr,
            },
        ]
    ).to_csv(output_path / "diagnostics.csv", index=False)
```

- [ ] **Step 2: Run tests and verify they fail where expected**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: fail if helper names are private (`_populate_metrics`) or baseline mismatch does not preserve fields.

- [ ] **Step 3: Refine metric extraction API**

Rename `_populate_metrics` to public `populate_metrics` and adjust caller.

Update baseline mismatch branch:

```python
def populate_metrics(record: SeasonAblationRecord, control_path: Path, treatment_path: Path) -> None:
    control = extract_run_metrics(control_path)
    treatment = extract_run_metrics(treatment_path)
    record.control_baseline_avg_points = control["baseline"]
    record.treatment_baseline_avg_points = treatment["baseline"]
    record.control_rf_avg_points = control["rf"]
    record.treatment_rf_avg_points = treatment["rf"]
    record.control_player_r2 = control["r2"]
    record.treatment_player_r2 = treatment["r2"]
    record.control_player_corr = control["corr"]
    record.treatment_player_corr = treatment["corr"]
    record.baseline_avg_points_equal = abs(control["baseline"] - treatment["baseline"]) <= 1e-9
    if not record.baseline_avg_points_equal:
        record.baseline_avg_points = None
        record.metrics_comparable = False
        raise ValueError("baseline average points differ between control and treatment")
    record.baseline_avg_points = control["baseline"]
    record.metrics_comparable = True
    record.rf_avg_points_delta = treatment["rf"] - control["rf"]
    record.player_r2_delta = treatment["r2"] - control["r2"]
    record.player_corr_delta = treatment["corr"] - control["corr"]
    record.rf_minus_baseline_control = control["rf"] - control["baseline"]
    record.rf_minus_baseline_treatment = treatment["rf"] - treatment["baseline"]
```

Change orchestration call to:

```python
populate_metrics(record, control_config.output_path, treatment_config.output_path)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "feat: extract footystats ablation metrics"
```

---

## Task 5: CSV And JSON Report Writing

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Modify: `src/tests/backtesting/test_footystats_ablation.py`

- [ ] **Step 1: Write failing tests for report artifacts**

Append:

```python
def test_write_reports_creates_csv_and_authoritative_json(tmp_path: Path) -> None:
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path)
    output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
    output_root.mkdir(parents=True)
    record = ablation.SeasonAblationRecord(
        season=2025,
        season_status="candidate",
        metrics_comparable=True,
        control_status="ok",
        treatment_status="ok",
        control_output_path=str(output_root / "runs/2025/footystats_mode=none/2025"),
        treatment_output_path=str(output_root / "runs/2025/footystats_mode=ppg/2025"),
        control_baseline_avg_points=50.0,
        treatment_baseline_avg_points=50.0,
        baseline_avg_points=50.0,
        baseline_avg_points_equal=True,
        control_rf_avg_points=55.0,
        treatment_rf_avg_points=56.0,
        rf_avg_points_delta=1.0,
        control_player_r2=0.1,
        treatment_player_r2=0.2,
        player_r2_delta=0.1,
        control_player_corr=0.3,
        treatment_player_corr=0.4,
        player_corr_delta=0.1,
        rf_minus_baseline_control=5.0,
        rf_minus_baseline_treatment=6.0,
        treatment_source_path="data/footystats/source.csv",
        treatment_source_sha256="abc",
    )
    aggregate = ablation.build_aggregate_record([record])
    result = ablation.FootyStatsPPGAblationResult(config=config, resolved_output_root=output_root, seasons=[record], aggregate=aggregate)

    ablation.write_reports(result)

    csv_path = output_root / "ppg_ablation.csv"
    json_path = output_root / "ppg_ablation.json"
    assert csv_path.exists()
    assert json_path.exists()

    csv_frame = pd.read_csv(csv_path)
    assert csv_frame["season"].astype(str).tolist() == ["2025", "aggregate"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config"]["fixture_mode"] == "none"
    assert payload["seasons"][0]["treatment_source_sha256"] == "abc"
    assert payload["aggregate"]["included_seasons"] == [2025]
    assert payload["generated_at_utc"].endswith("Z")


def test_csv_truncates_error_message_and_uses_empty_null_cells(tmp_path: Path) -> None:
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path)
    output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
    output_root.mkdir(parents=True)
    record = ablation.SeasonAblationRecord(
        season=2025,
        error_stage="eligibility",
        error_message="x" * 600,
    )
    result = ablation.FootyStatsPPGAblationResult(
        config=config,
        resolved_output_root=output_root,
        seasons=[record],
        aggregate=ablation.build_aggregate_record([record]),
    )

    ablation.write_reports(result)

    csv_frame = pd.read_csv(output_root / "ppg_ablation.csv", keep_default_na=False)
    assert csv_frame.loc[0, "error_message"] == "x" * 500
    assert csv_frame.loc[0, "control_rf_avg_points"] == ""


def test_atomic_report_write_does_not_leave_final_file_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ablation.FootyStatsPPGAblationConfig(project_root=tmp_path)
    output_root = tmp_path / "data" / "08_reporting" / "backtests" / "footystats_ablation"
    output_root.mkdir(parents=True)
    result = ablation.FootyStatsPPGAblationResult(
        config=config,
        resolved_output_root=output_root,
        seasons=[],
        aggregate=ablation.build_aggregate_record([]),
    )

    def fail_replace(self, target):
        raise OSError("rename failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="rename failed"):
        ablation.write_reports(result)

    assert not (output_root / "ppg_ablation.csv").exists()
    assert not (output_root / "ppg_ablation.json").exists()
```

Add imports at top:

```python
import json
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: fail with missing `write_reports`.

- [ ] **Step 3: Implement report writing**

Add constants:

```python
CSV_COLUMNS = [
    "season",
    "row_type",
    "season_status",
    "metrics_comparable",
    "control_status",
    "treatment_status",
    "control_output_path",
    "treatment_output_path",
    "control_baseline_avg_points",
    "treatment_baseline_avg_points",
    "baseline_avg_points",
    "baseline_avg_points_equal",
    "control_rf_avg_points",
    "treatment_rf_avg_points",
    "rf_avg_points_delta",
    "control_player_r2",
    "treatment_player_r2",
    "player_r2_delta",
    "control_player_corr",
    "treatment_player_corr",
    "player_corr_delta",
    "rf_minus_baseline_control",
    "rf_minus_baseline_treatment",
    "error_stage",
    "error_message",
]
```

Add functions:

```python
def _csv_row(record: SeasonAblationRecord) -> dict[str, object]:
    row = {column: getattr(record, column) for column in CSV_COLUMNS}
    if row["error_message"] is not None:
        row["error_message"] = str(row["error_message"])[:500]
    return row


def _json_record(record: SeasonAblationRecord) -> dict[str, object]:
    data = asdict(record)
    data["errors"] = [asdict(error) for error in record.errors]
    return data


def _generated_at_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _aggregate_json(records: list[SeasonAblationRecord], aggregate: SeasonAblationRecord) -> dict[str, object]:
    included = [
        int(record.season)
        for record in records
        if isinstance(record.season, int)
        and record.metrics_comparable
        and record.control_status == "ok"
        and record.treatment_status == "ok"
    ]
    excluded = [
        {"season": record.season, "reason": record.error_stage or "not_comparable"}
        for record in records
        if record.season not in included
    ]
    return {
        "included_seasons": included,
        "excluded_seasons": excluded,
        "aggregation_method": "unweighted_mean_across_successful_comparable_seasons",
        "metrics": _csv_row(aggregate),
    }


def _config_json(config: FootyStatsPPGAblationConfig, resolved_output_root: Path) -> dict[str, object]:
    return {
        "project_root": str(config.project_root),
        "output_root": str(config.output_root),
        "resolved_project_root": str(config.project_root.resolve()),
        "resolved_output_root": str(resolved_output_root),
        "seasons": list(config.seasons),
        "start_round": config.start_round,
        "budget": config.budget,
        "current_year": config.current_year,
        "resolved_current_year": config.resolved_current_year,
        "fixture_mode": "none",
        "control_footystats_mode": "none",
        "treatment_footystats_mode": "ppg",
        "footystats_evaluation_scope": "historical_candidate",
        "footystats_league_slug": config.footystats_league_slug,
        "force": config.force,
    }


def write_reports(result: FootyStatsPPGAblationResult) -> None:
    result.resolved_output_root.mkdir(parents=True, exist_ok=True)
    rows = [*result.seasons, result.aggregate]
    csv_frame = pd.DataFrame([_csv_row(record) for record in rows], columns=pd.Index(CSV_COLUMNS))
    payload = {
        "config": _config_json(result.config, result.resolved_output_root),
        "seasons": [_json_record(record) for record in result.seasons],
        "aggregate": _aggregate_json(result.seasons, result.aggregate),
        "generated_at_utc": _generated_at_utc(),
    }

    csv_path = result.resolved_output_root / "ppg_ablation.csv"
    json_path = result.resolved_output_root / "ppg_ablation.json"
    csv_tmp = result.resolved_output_root / ".ppg_ablation.csv.tmp"
    json_tmp = result.resolved_output_root / ".ppg_ablation.json.tmp"
    csv_frame.to_csv(csv_tmp, index=False, na_rep="")
    json_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    csv_tmp.replace(csv_path)
    json_tmp.replace(json_path)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py src/tests/backtesting/test_footystats_ablation.py
git commit -m "feat: write footystats ablation reports"
```

---

## Task 6: Main Exit Codes And README

**Files:**
- Modify: `src/cartola/backtesting/footystats_ablation.py`
- Modify: `src/tests/backtesting/test_footystats_ablation.py`
- Modify: `README.md`
- Modify: `roadmap.md`

- [ ] **Step 1: Write failing tests for main behavior**

Append:

```python
def test_main_writes_reports_and_returns_zero_when_one_season_comparable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def ok_loader(**kwargs):
        return StubLoadResult()

    def fake_run_backtest(config):
        _write_backtest_outputs(config.output_path)
        return object()

    monkeypatch.setattr(ablation, "load_footystats_ppg_rows", ok_loader)
    monkeypatch.setattr(ablation, "run_backtest", fake_run_backtest)

    exit_code = ablation.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-root",
            "data/08_reporting/backtests/footystats_ablation",
            "--seasons",
            "2025",
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "data/08_reporting/backtests/footystats_ablation/ppg_ablation.csv").exists()


def test_main_returns_nonzero_when_no_season_comparable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_loader(**kwargs):
        raise ValueError("not candidate")

    monkeypatch.setattr(ablation, "load_footystats_ppg_rows", fail_loader)

    exit_code = ablation.main(
        [
            "--project-root",
            str(tmp_path),
            "--output-root",
            "data/08_reporting/backtests/footystats_ablation",
            "--seasons",
            "2025",
            "--current-year",
            "2026",
        ]
    )

    assert exit_code == 1
    assert (tmp_path / "data/08_reporting/backtests/footystats_ablation/ppg_ablation.json").exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py -q
```

Expected: fail because `main` returns `2` and does not write reports yet.

- [ ] **Step 3: Implement main and report call**

Replace `main`:

```python
def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    result = run_footystats_ppg_ablation(config)
    write_reports(result)
    comparable = [
        record
        for record in result.seasons
        if record.metrics_comparable and record.control_status == "ok" and record.treatment_status == "ok"
    ]
    print(f"FootyStats PPG ablation complete: output={result.resolved_output_root}")
    print(f"Comparable seasons: {len(comparable)}/{len(result.seasons)}")
    return 0 if comparable else 1
```

- [ ] **Step 4: Update README**

Add under FootyStats PPG ablation:

```markdown
### Multi-season FootyStats PPG ablation

Run the no-fixture PPG ablation across complete candidate seasons:

```bash
uv run --frozen python scripts/run_footystats_ppg_ablation.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --force
```

Outputs:

- `data/08_reporting/backtests/footystats_ablation/ppg_ablation.csv`
- `data/08_reporting/backtests/footystats_ablation/ppg_ablation.json`
- per-run backtest outputs under `data/08_reporting/backtests/footystats_ablation/runs/{season}/footystats_mode={none|ppg}/{season}/`
```

- [ ] **Step 5: Update roadmap**

In `roadmap.md`, after implementation, move the ablation report from “next” to delivered and make the next roadmap step “review PPG multi-season results and decide whether to keep, revise, or drop PPG before adding xG/odds.”

- [ ] **Step 6: Run focused tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_footystats_ablation.py src/tests/backtesting/test_cli.py -q
```

Expected: tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/cartola/backtesting/footystats_ablation.py src/tests/backtesting/test_footystats_ablation.py README.md roadmap.md
git commit -m "feat: add footystats ppg ablation report command"
```

---

## Task 7: Real Ablation Run And Final Verification

**Files:**
- Runtime outputs only under `data/08_reporting/backtests/footystats_ablation/`
- No source changes unless verification finds a bug

- [ ] **Step 1: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: ruff clean, ty clean, bandit clean, pytest passing.

- [ ] **Step 2: Run real ablation**

Run:

```bash
uv run --frozen python scripts/run_footystats_ppg_ablation.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --force
```

Expected: command exits `0`, prints output root and comparable season count, writes:

```text
data/08_reporting/backtests/footystats_ablation/ppg_ablation.csv
data/08_reporting/backtests/footystats_ablation/ppg_ablation.json
```

- [ ] **Step 3: Inspect report**

Run:

```bash
uv run --frozen python - <<'PY'
import pandas as pd
path = "data/08_reporting/backtests/footystats_ablation/ppg_ablation.csv"
frame = pd.read_csv(path)
print(frame[["season", "control_status", "treatment_status", "rf_avg_points_delta", "player_r2_delta", "player_corr_delta"]])
PY
```

Expected: rows for `2023`, `2024`, `2025`, and `aggregate`. Failed seasons are allowed only if the CSV/JSON records a clear stage and error.

- [ ] **Step 4: Verify normal outputs were not touched**

Run:

```bash
test ! -e data/08_reporting/backtests/2023/ppg_ablation.csv
test ! -e data/08_reporting/backtests/2024/ppg_ablation.csv
test ! -e data/08_reporting/backtests/2025/ppg_ablation.csv
```

Expected: all commands exit `0`.

- [ ] **Step 5: Check git state**

Run:

```bash
git status --short
```

Expected: source tree clean except ignored generated report files. If report files are tracked by accident, stop and update `.gitignore` or move outputs according to repo policy.

- [ ] **Step 6: Commit only if source/docs changed during verification**

If a verification bug required a source fix:

```bash
git add <changed source files>
git commit -m "fix: finalize footystats ppg ablation report"
```

If no source changed, do not create a commit for generated reports.

---

## Self-Review Checklist

- Spec coverage:
  - no-fixture isolation: Task 2 and Task 3
  - runtime eligibility: Task 3
  - exact output paths: Task 2
  - safe force semantics: Task 2
  - metric extraction and `section="prediction"`: Task 4
  - nullable CSV metrics and truncated CSV errors: Task 5
  - authoritative JSON: Task 5
  - atomic report writes: Task 5
  - CLI defaults and season parsing: Task 1
  - final real run and no normal output pollution: Task 7
- Completeness scan: every implementation detail is spelled out without repeated-work shortcuts.
- Type consistency:
  - `FootyStatsPPGAblationConfig`, `SeasonAblationRecord`, and `FootyStatsPPGAblationResult` names are consistent across tasks.
  - `build_backtest_config`, `resolve_output_root`, `prepare_output_root`, `populate_metrics`, `extract_run_metrics`, and `write_reports` are introduced before use in later tasks.
