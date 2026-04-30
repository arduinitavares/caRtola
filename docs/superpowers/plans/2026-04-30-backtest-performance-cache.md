# Backtest Performance Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 1a from the approved design: cache each per-round prediction frame once inside `run_backtest()` and reuse cached frames without changing backtest semantics.

**Architecture:** Add a private in-memory `RoundFrameStore` in `src/cartola/backtesting/runner.py`. The store calls the existing `build_prediction_frame()` once per detected post-exclusion round, returns deep copies, and assembles training frames from cached rounds `< target_round`. `run_backtest()` uses the store instead of repeatedly calling `build_training_frame()` while keeping the public feature builders available as reference behavior.

**Tech Stack:** Python 3.13, pandas, pytest, existing Cartola backtesting modules, `uv run --frozen`, existing `scripts/pyrepo-check --all`.

---

## Scope Boundary

Implement only Phase 1a.

Do not add:

- `jobs`;
- parallel execution;
- model registry;
- RF parameter changes;
- experiment runner;
- on-disk frame cache;
- public no-cache mode.

The approved spec is `docs/superpowers/specs/2026-04-30-backtest-performance-and-experiment-readiness-design.md`.

## Files

- Modify: `src/cartola/backtesting/runner.py`
  - add `time.perf_counter`;
  - extend `BacktestMetadata`;
  - add private `RoundFrameStore`;
  - add private `_detected_rounds`;
  - build the store after excluded-round filtering and FootyStats validation;
  - replace repeated `build_training_frame()` / target `build_prediction_frame()` calls inside `run_backtest()`.
- Modify: `src/tests/backtesting/test_runner.py`
  - add focused cache tests;
  - update metadata expectations.

No other files are in scope.

---

## Task 1: Add Focused RoundFrameStore Tests

**Files:**
- Modify: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Add module import for private runner helpers**

Change the imports near the top of `src/tests/backtesting/test_runner.py` from:

```python
from cartola.backtesting.runner import run_backtest
```

to:

```python
from cartola.backtesting import features as features_module
from cartola.backtesting import runner as runner_module
from cartola.backtesting.runner import run_backtest
```

- [ ] **Step 2: Add failing cache equivalence and mutation tests**

Add these tests after `_tiny_footystats_rows()`:

```python
def test_round_frame_store_training_frame_matches_public_builder(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    fixtures = _tiny_fixtures(range(1, 6))
    footystats_rows = _tiny_footystats_rows(range(1, 6))
    store = runner_module.RoundFrameStore(
        season_df=season_df,
        fixtures=fixtures,
        footystats_rows=footystats_rows,
        matchup_context_mode="cartola_matchup_v1",
    )
    store.build_all([1, 2, 3, 4, 5])

    cached = store.training_frame(
        target_round=5,
        playable_statuses=("Provavel",),
        empty_columns=[
            *runner_module.MARKET_COLUMNS,
            *runner_module.feature_columns_for_config(
                BacktestConfig(
                    project_root=tmp_path,
                    fixture_mode="exploratory",
                    footystats_mode="ppg",
                    matchup_context_mode="cartola_matchup_v1",
                )
            ),
            "target",
        ],
    )
    public = runner_module.build_training_frame(
        season_df,
        5,
        playable_statuses=("Provavel",),
        fixtures=fixtures,
        footystats_rows=footystats_rows,
        matchup_context_mode="cartola_matchup_v1",
    )

    assert_frame_equal(
        cached.sort_index(axis=1).reset_index(drop=True),
        public.sort_index(axis=1).reset_index(drop=True),
        check_dtype=False,
    )
    assert store.prediction_frames_built == 5
```

Add the mutation-safety test:

```python
def test_round_frame_store_returns_deep_copies_for_candidates_and_training():
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 4)], ignore_index=True)
    store = runner_module.RoundFrameStore(
        season_df=season_df,
        fixtures=None,
        footystats_rows=None,
        matchup_context_mode="none",
    )
    store.build_all([1, 2, 3])

    candidate = store.prediction_frame(3)
    candidate.loc[:, "baseline_score"] = 999.0
    candidate.loc[candidate.index[0], "apelido"] = "mutated"

    candidate_again = store.prediction_frame(3)
    assert "baseline_score" not in candidate_again.columns
    assert "mutated" not in set(candidate_again["apelido"])

    training = store.training_frame(
        target_round=3,
        playable_statuses=("Provavel",),
        empty_columns=[*runner_module.MARKET_COLUMNS, *runner_module.feature_columns_for_config(BacktestConfig()), "target"],
    )
    training.loc[:, "random_forest_score"] = 123.0
    training.loc[training.index[0], "apelido"] = "training-mutated"

    training_again = store.training_frame(
        target_round=3,
        playable_statuses=("Provavel",),
        empty_columns=[*runner_module.MARKET_COLUMNS, *runner_module.feature_columns_for_config(BacktestConfig()), "target"],
    )
    assert "random_forest_score" not in training_again.columns
    assert "training-mutated" not in set(training_again["apelido"])
```

Add the future-round boundary test:

```python
def test_round_frame_store_preserves_target_round_temporal_boundary():
    base = pd.concat([_tiny_round(round_number) for round_number in range(1, 5)], ignore_index=True)
    spiked = base.copy()
    spiked.loc[spiked["rodada"].eq(4), "pontuacao"] = 9999.0

    base_store = runner_module.RoundFrameStore(
        season_df=base,
        fixtures=None,
        footystats_rows=None,
        matchup_context_mode="none",
    )
    spiked_store = runner_module.RoundFrameStore(
        season_df=spiked,
        fixtures=None,
        footystats_rows=None,
        matchup_context_mode="none",
    )
    base_store.build_all([1, 2, 3, 4])
    spiked_store.build_all([1, 2, 3, 4])

    assert_frame_equal(
        base_store.prediction_frame(3).sort_index(axis=1).reset_index(drop=True),
        spiked_store.prediction_frame(3).sort_index(axis=1).reset_index(drop=True),
        check_dtype=False,
    )
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_round_frame_store_training_frame_matches_public_builder \
  src/tests/backtesting/test_runner.py::test_round_frame_store_returns_deep_copies_for_candidates_and_training \
  src/tests/backtesting/test_runner.py::test_round_frame_store_preserves_target_round_temporal_boundary \
  -q
```

Expected: fail with `AttributeError: module 'cartola.backtesting.runner' has no attribute 'RoundFrameStore'`.

---

## Task 2: Implement RoundFrameStore

**Files:**
- Modify: `src/cartola/backtesting/runner.py`

- [ ] **Step 1: Import `MARKET_COLUMNS`**

Change:

```python
from cartola.backtesting.config import MARKET_OPEN_PRICE_COLUMN, BacktestConfig
```

to:

```python
from cartola.backtesting.config import MARKET_COLUMNS, MARKET_OPEN_PRICE_COLUMN, BacktestConfig
```

- [ ] **Step 2: Add `RoundFrameStore` after `FixtureLoadForRun`**

Insert this class after the `FixtureLoadForRun` dataclass:

```python
class RoundFrameStore:
    def __init__(
        self,
        *,
        season_df: pd.DataFrame,
        fixtures: pd.DataFrame | None,
        footystats_rows: pd.DataFrame | None,
        matchup_context_mode: str,
    ) -> None:
        self._season_df = season_df
        self._fixtures = fixtures
        self._footystats_rows = footystats_rows
        self._matchup_context_mode = matchup_context_mode
        self._frames: dict[int, pd.DataFrame] = {}

    @property
    def prediction_frames_built(self) -> int:
        return len(self._frames)

    def build_all(self, rounds: list[int]) -> None:
        for round_number in rounds:
            if round_number in self._frames:
                continue
            self._frames[round_number] = build_prediction_frame(
                self._season_df,
                round_number,
                fixtures=self._fixtures,
                footystats_rows=self._footystats_rows,
                matchup_context_mode=self._matchup_context_mode,
            ).copy(deep=True)

    def prediction_frame(self, round_number: int) -> pd.DataFrame:
        try:
            frame = self._frames[round_number]
        except KeyError as exc:
            raise KeyError(f"Prediction frame for round {round_number} was not built.") from exc
        return frame.copy(deep=True)

    def training_frame(
        self,
        *,
        target_round: int,
        playable_statuses: tuple[str, ...],
        empty_columns: list[str],
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for round_number in sorted(self._frames):
            if round_number >= target_round:
                continue
            round_frame = self._frames[round_number].copy(deep=True)
            round_frame = round_frame[round_frame["status"].isin(playable_statuses)].copy(deep=True)
            round_frame["target"] = round_frame["pontuacao"]
            frames.append(round_frame)

        if not frames:
            return pd.DataFrame(columns=pd.Index(empty_columns))

        return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_round_frame_store_training_frame_matches_public_builder \
  src/tests/backtesting/test_runner.py::test_round_frame_store_returns_deep_copies_for_candidates_and_training \
  src/tests/backtesting/test_runner.py::test_round_frame_store_preserves_target_round_temporal_boundary \
  -q
```

Expected: pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "test: define backtest round frame cache contract"
```

---

## Task 3: Wire RoundFrameStore Into `run_backtest()`

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Add failing build-count integration test**

Add this test after the cache unit tests:

```python
def test_run_backtest_builds_each_detected_round_prediction_frame_once(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 7)], ignore_index=True)
    calls: list[int] = []
    original = runner_module.build_prediction_frame

    def counting_build_prediction_frame(*args: object, **kwargs: object) -> pd.DataFrame:
        target_round = int(args[1])
        calls.append(target_round)
        return original(*args, **kwargs)

    monkeypatch.setattr(features_module, "build_prediction_frame", counting_build_prediction_frame)
    monkeypatch.setattr(runner_module, "build_prediction_frame", counting_build_prediction_frame)

    run_backtest(BacktestConfig(project_root=tmp_path, start_round=5, budget=100), season_df=season_df)

    assert sorted(calls) == [1, 2, 3, 4, 5, 6]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_run_backtest_builds_each_detected_round_prediction_frame_once \
  -q
```

Expected: fail because current code builds prior rounds repeatedly. A typical failing list contains duplicate rounds such as `[1, 1, 2, 2, 3, 3, 4, 4, 5, 6]`.

- [ ] **Step 3: Add `_detected_rounds` helper**

Add this helper near `_max_round`:

```python
def _detected_rounds(data: pd.DataFrame) -> list[int]:
    if data.empty:
        return []
    return sorted(
        int(round_number)
        for round_number in pd.to_numeric(data["rodada"], errors="raise").dropna().unique()
    )
```

- [ ] **Step 4: Build the store after metadata inputs are resolved**

Inside `run_backtest()`, after `_validate_footystats_join_diagnostics(footystats_diagnostics)`, insert:

```python
    cached_rounds = _detected_rounds(data)
    round_frame_store = RoundFrameStore(
        season_df=data,
        fixtures=fixture_data,
        footystats_rows=footystats_rows,
        matchup_context_mode=config.matchup_context_mode,
    )
    round_frame_store.build_all(cached_rounds)
```

- [ ] **Step 5: Define empty training columns**

After:

```python
    model_feature_columns = feature_columns_for_config(config)
```

insert:

```python
    empty_training_columns = list(dict.fromkeys([*MARKET_COLUMNS, *model_feature_columns, "target"]))
```

- [ ] **Step 6: Replace repeated frame construction inside the target loop**

Replace:

```python
        training = build_training_frame(
            data,
            round_number,
            playable_statuses=config.playable_statuses,
            fixtures=fixture_data,
            footystats_rows=footystats_rows,
            matchup_context_mode=config.matchup_context_mode,
        )
        candidates = build_prediction_frame(
            data,
            round_number,
            fixtures=fixture_data,
            footystats_rows=footystats_rows,
            matchup_context_mode=config.matchup_context_mode,
        )
        candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy()
```

with:

```python
        training = round_frame_store.training_frame(
            target_round=round_number,
            playable_statuses=config.playable_statuses,
            empty_columns=empty_training_columns,
        )
        candidates = round_frame_store.prediction_frame(round_number)
        candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy(deep=True)
```

- [ ] **Step 7: Run focused runner tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_run_backtest_builds_each_detected_round_prediction_frame_once \
  src/tests/backtesting/test_runner.py::test_strict_alignment_policy_exclude_round_removes_invalid_round_before_training \
  src/tests/backtesting/test_runner.py::test_strict_alignment_policy_exclude_round_removes_missing_strict_fixture_round \
  -q
```

Expected: pass.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: cache backtest prediction frames"
```

---

## Task 4: Add Cache Metadata And Excluded-Round Coverage

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Add failing metadata assertions**

In `test_run_backtest_writes_metadata_for_no_fixture_mode`, after:

```python
    assert metadata["matchup_context_feature_columns"] == []
```

add:

```python
    assert metadata["cache_enabled"] is True
    assert metadata["prediction_frames_built"] == 5
    assert isinstance(metadata["wall_clock_seconds"], float)
    assert metadata["wall_clock_seconds"] >= 0.0
```

- [ ] **Step 2: Add excluded-round build-count test**

Add this test near the strict alignment tests:

```python
def test_run_backtest_does_not_build_prediction_frame_for_excluded_round(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    fixtures = _tiny_fixtures(range(1, 6))
    fixtures = fixtures[fixtures["rodada"] != 3].copy()
    calls: list[int] = []
    original = runner_module.build_prediction_frame

    def counting_build_prediction_frame(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(int(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "build_prediction_frame", counting_build_prediction_frame)
    monkeypatch.setattr(
        "cartola.backtesting.runner.load_strict_fixtures",
        lambda **kwargs: StrictFixturesLoadResult(
            fixtures=fixtures,
            manifest_paths=["data/01_raw/fixtures_strict/2025/partidas-1.manifest.json"],
            manifest_sha256={"data/01_raw/fixtures_strict/2025/partidas-1.manifest.json": "abc"},
            generator_versions=["fixture_snapshot_v1"],
        ),
    )

    result = run_backtest(
        BacktestConfig(
            project_root=tmp_path,
            start_round=3,
            budget=100,
            fixture_mode="strict",
            strict_alignment_policy="exclude_round",
        ),
        season_df=season_df,
    )

    assert result.metadata.excluded_rounds == [3]
    assert result.metadata.prediction_frames_built == 4
    assert sorted(calls) == [1, 2, 4, 5]
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_run_backtest_writes_metadata_for_no_fixture_mode \
  src/tests/backtesting/test_runner.py::test_run_backtest_does_not_build_prediction_frame_for_excluded_round \
  -q
```

Expected: fail because `BacktestMetadata` does not yet expose cache/runtime fields.

- [ ] **Step 4: Import `perf_counter`**

At the top of `src/cartola/backtesting/runner.py`, change:

```python
import json
from dataclasses import dataclass
```

to:

```python
import json
from dataclasses import dataclass
from time import perf_counter
```

- [ ] **Step 5: Extend `BacktestMetadata`**

Add fields to `BacktestMetadata` after `max_round`:

```python
    cache_enabled: bool
    prediction_frames_built: int
    wall_clock_seconds: float
```

- [ ] **Step 6: Capture start time and populate metadata**

At the start of `run_backtest()`, before loading `data`, insert:

```python
    started_at = perf_counter()
```

When creating `BacktestMetadata`, add:

```python
        cache_enabled=True,
        prediction_frames_built=round_frame_store.prediction_frames_built,
        wall_clock_seconds=0.0,
```

After diagnostics are normalized and before `_write_outputs(...)`, replace the frozen metadata with runtime populated:

```python
    metadata = BacktestMetadata(
        **{
            **metadata.__dict__,
            "wall_clock_seconds": round(perf_counter() - started_at, OUTPUT_FLOAT_PRECISION),
        }
    )
```

This keeps `BacktestMetadata` frozen while recording the final runtime.

- [ ] **Step 7: Run focused metadata tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_run_backtest_writes_metadata_for_no_fixture_mode \
  src/tests/backtesting/test_runner.py::test_run_backtest_does_not_build_prediction_frame_for_excluded_round \
  -q
```

Expected: pass.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: record backtest cache metadata"
```

---

## Task 5: Run Equivalence And Regression Tests

**Files:**
- Modify only if tests reveal an implementation bug:
  - `src/cartola/backtesting/runner.py`
  - `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Run focused backtesting tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py \
  src/tests/backtesting/test_features.py \
  src/tests/backtesting/test_metrics.py \
  src/tests/backtesting/test_models.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run report-consumer tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_footystats_ablation.py \
  src/tests/backtesting/test_compatibility_audit.py \
  src/tests/backtesting/test_cli.py \
  -q
```

Expected: pass.

- [ ] **Step 3: Fix only cache-related regressions**

If any test fails, inspect whether the failure is caused by:

- missing new metadata fields in expected JSON;
- cached frame mutation;
- excluded rounds being built;
- training frame using `<= target_round` instead of `< target_round`;
- report ordering.

Do not change model parameters, feature logic, optimizer behavior, or CLI scope.

- [ ] **Step 4: Commit test fixes if needed**

If Step 3 required edits, run:

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "test: align reports with cache metadata"
```

If no edits were required, skip this commit.

---

## Task 6: Benchmark And Quality Gate

**Files:**
- Do not modify implementation files in this task unless verification fails.

- [ ] **Step 1: Run the Phase 1a benchmark**

Run:

```bash
/usr/bin/time -p uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode ppg \
  --matchup-context-mode cartola_matchup_v1 \
  --current-year 2026 \
  --output-root data/08_reporting/backtests/perf_cache_phase1a
```

Expected:

- command exits `0`;
- output path is `data/08_reporting/backtests/perf_cache_phase1a/2025`;
- runtime is recorded in terminal output from `/usr/bin/time`;
- `run_metadata.json` contains `cache_enabled=true`, `prediction_frames_built`, and `wall_clock_seconds`.

- [ ] **Step 2: Record benchmark result in the final implementation summary**

Compare the runtime to the pre-cache branch if a pre-cache timing exists.

Acceptance:

- at least 30% faster than pre-cache; or
- if below 30%, record before/after timings and identify the observed next bottleneck before Phase 1b planning.

- [ ] **Step 3: Remove temporary benchmark output**

Run:

```bash
rm -rf data/08_reporting/backtests/perf_cache_phase1a
```

Expected: no benchmark output remains in git status.

- [ ] **Step 4: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: pass.

- [ ] **Step 5: Final status check**

Run:

```bash
git status --short --branch
```

Expected: clean except for committed branch changes.

---

## Self-Review Checklist

- Spec coverage: Phase 1a cache-only, excluded-round round set, copy/mutation rules, metadata, equivalence, benchmark, and no legacy switch are covered.
- Placeholder scan: no unfinished markers or open-ended edge-case instructions.
- Type consistency: the plan uses `RoundFrameStore`, `BacktestMetadata`, `feature_columns_for_config`, `MARKET_COLUMNS`, and existing `BacktestConfig` names consistently.
- Scope boundary: no `jobs`, no parallel execution, no model registry, no RF parameter changes, no experiment runner.
