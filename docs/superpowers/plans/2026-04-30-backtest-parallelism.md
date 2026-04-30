# Backtest Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add controlled target-round thread parallelism to `run_backtest()` while preserving the current scoring, feature, optimizer, and report semantics.

**Architecture:** Keep `RoundFrameStore` as the single in-memory frame source, make its copy access lock-protected, and evaluate target rounds through a private `RoundEvaluationResult` helper. The parent process owns target-round classification, aggregation, canonical sorting, metadata, and all report writes; worker threads only return in-memory records and frames.

**Tech Stack:** Python, pandas, scikit-learn `RandomForestRegressor`, `concurrent.futures.ThreadPoolExecutor`, pytest, uv.

---

## File Structure

- Modify `src/cartola/backtesting/config.py`
  - Add `BacktestConfig.jobs`.
  - Validate `jobs` before report output.

- Modify `src/cartola/backtesting/cli.py`
  - Add `--jobs`.
  - Pass `jobs` into `BacktestConfig`.

- Modify `src/cartola/backtesting/models.py`
  - Add `n_jobs` to `RandomForestPointPredictor`.
  - Pass it into the underlying `RandomForestRegressor`.

- Modify `src/cartola/backtesting/runner.py`
  - Add imports for `as_completed`, `ThreadPoolExecutor`, `Lock`, and `os`.
  - Extend `BacktestMetadata`.
  - Add `RoundEvaluationResult`.
  - Add `BacktestRoundEvaluationError`.
  - Add `_effective_model_n_jobs()`, `_thread_env()`, `_target_round_work()`, `_sort_outputs()`, and `_evaluate_target_round()`.
  - Lock `RoundFrameStore` copy access.
  - Add sequential and thread execution paths.

- Modify `src/tests/backtesting/test_config.py`
  - Add `jobs` validation tests.

- Modify `src/tests/backtesting/test_cli.py`
  - Add CLI parsing and config propagation tests.
  - Update `_metadata_for_config()` for new metadata fields.

- Modify `src/tests/backtesting/test_models.py`
  - Add `RandomForestPointPredictor.n_jobs` tests.

- Modify `src/tests/backtesting/test_runner.py`
  - Add store thread-safety tests.
  - Add effective model worker tests.
  - Add helper extraction tests.
  - Add parallel equivalence tests.
  - Add worker failure and metadata tests.

---

### Task 1: Add `jobs` To Config And CLI

**Files:**
- Modify: `src/cartola/backtesting/config.py`
- Modify: `src/cartola/backtesting/cli.py`
- Test: `src/tests/backtesting/test_config.py`
- Test: `src/tests/backtesting/test_cli.py`

- [ ] **Step 1: Write config validation tests**

Add these tests to `src/tests/backtesting/test_config.py`:

```python
import pytest

from cartola.backtesting.config import BacktestConfig


def test_backtest_config_accepts_default_jobs() -> None:
    config = BacktestConfig()

    assert config.jobs == 1


def test_backtest_config_accepts_positive_jobs() -> None:
    config = BacktestConfig(jobs=4)

    assert config.jobs == 4


@pytest.mark.parametrize("jobs", [0, -1])
def test_backtest_config_rejects_non_positive_jobs(jobs: int) -> None:
    with pytest.raises(ValueError, match="jobs must be >= 1"):
        BacktestConfig(jobs=jobs)


@pytest.mark.parametrize("jobs", [1.5, "2", True])
def test_backtest_config_rejects_non_integer_jobs(jobs: object) -> None:
    with pytest.raises(TypeError, match="jobs must be an integer"):
        BacktestConfig(jobs=jobs)  # type: ignore[arg-type]
```

- [ ] **Step 2: Write CLI tests**

Add these assertions/tests to `src/tests/backtesting/test_cli.py`:

```python
def test_parse_args_accepts_jobs() -> None:
    args = parse_args(["--jobs", "4"])

    assert args.jobs == 4


def test_parse_args_uses_jobs_default() -> None:
    args = parse_args([])

    assert args.jobs == 1
```

Update `test_main_builds_config_and_prints_completion()` to expect `jobs=1`:

```python
assert observed_configs == [
    BacktestConfig(
        season=2025,
        start_round=5,
        budget=100.0,
        project_root=tmp_path,
        jobs=1,
    )
]
```

Add this test to confirm propagation:

```python
def test_main_passes_jobs_to_config(monkeypatch, tmp_path) -> None:
    observed_configs: list[BacktestConfig] = []

    def fake_run_backtest(config: BacktestConfig) -> BacktestResult:
        observed_configs.append(config)
        return BacktestResult(
            round_results=pd.DataFrame(),
            selected_players=pd.DataFrame(),
            player_predictions=pd.DataFrame(),
            summary=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            metadata=_metadata_for_config(config),
        )

    monkeypatch.setattr("cartola.backtesting.cli.run_backtest", fake_run_backtest)

    exit_code = main(["--project-root", str(tmp_path), "--jobs", "3"])

    assert exit_code == 0
    assert observed_configs[0].jobs == 3
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_config.py \
  src/tests/backtesting/test_cli.py::test_parse_args_accepts_jobs \
  src/tests/backtesting/test_cli.py::test_parse_args_uses_jobs_default \
  src/tests/backtesting/test_cli.py::test_main_passes_jobs_to_config \
  -q
```

Expected: failures because `BacktestConfig.jobs` and `--jobs` are not implemented.

- [ ] **Step 4: Implement config and CLI**

In `src/cartola/backtesting/config.py`, add `jobs` and validation:

```python
@dataclass(frozen=True)
class BacktestConfig:
    season: int = 2025
    start_round: int = 5
    budget: float = 100.0
    playable_statuses: tuple[str, ...] = ("Provavel",)
    random_seed: int = 123
    project_root: Path = Path(".")
    output_root: Path = Path("data/08_reporting/backtests")
    fixture_mode: FixtureMode = "none"
    strict_alignment_policy: StrictAlignmentPolicy = "fail"
    matchup_context_mode: MatchupContextMode = "none"
    footystats_mode: FootyStatsMode = "none"
    footystats_evaluation_scope: FootyStatsEvaluationScope = "historical_candidate"
    footystats_league_slug: str = "brazil-serie-a"
    footystats_dir: Path = Path("data/footystats")
    current_year: int | None = None
    jobs: int = 1
    scout_columns: tuple[str, ...] = DEFAULT_SCOUT_COLUMNS

    def __post_init__(self) -> None:
        if type(self.jobs) is not int:
            raise TypeError("jobs must be an integer")
        if self.jobs < 1:
            raise ValueError("jobs must be >= 1")
```

In `src/cartola/backtesting/cli.py`, add the parser argument:

```python
parser.add_argument("--jobs", type=int, default=1)
```

Pass it into the config:

```python
config = BacktestConfig(
    season=args.season,
    start_round=args.start_round,
    budget=args.budget,
    project_root=args.project_root,
    output_root=args.output_root,
    fixture_mode=args.fixture_mode,
    strict_alignment_policy=args.strict_alignment_policy,
    matchup_context_mode=args.matchup_context_mode,
    footystats_mode=args.footystats_mode,
    footystats_evaluation_scope=args.footystats_evaluation_scope,
    footystats_league_slug=args.footystats_league_slug,
    footystats_dir=args.footystats_dir,
    current_year=args.current_year,
    jobs=args.jobs,
)
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_config.py \
  src/tests/backtesting/test_cli.py \
  -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/config.py src/cartola/backtesting/cli.py src/tests/backtesting/test_config.py src/tests/backtesting/test_cli.py
git commit -m "feat: add backtest jobs config"
```

---

### Task 2: Plumb RandomForest `n_jobs`

**Files:**
- Modify: `src/cartola/backtesting/models.py`
- Test: `src/tests/backtesting/test_models.py`

- [ ] **Step 1: Write model tests**

Add this test to `src/tests/backtesting/test_models.py`:

```python
def test_random_forest_point_predictor_sets_n_jobs() -> None:
    model = RandomForestPointPredictor(random_seed=7, feature_columns=FEATURE_COLUMNS, n_jobs=1)

    forest = model.pipeline.named_steps["model"]

    assert forest.n_jobs == 1
    assert model.n_jobs == 1
```

Add this assertion to `test_random_forest_point_predictor_fit_predict_smoke()`:

```python
assert model.n_jobs == -1
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_models.py -q
```

Expected: failure because `RandomForestPointPredictor` does not accept or expose `n_jobs`.

- [ ] **Step 3: Implement model plumbing**

Change `RandomForestPointPredictor.__init__()` in `src/cartola/backtesting/models.py` to:

```python
class RandomForestPointPredictor:
    def __init__(
        self,
        random_seed: int = 123,
        feature_columns: list[str] | None = None,
        n_jobs: int = -1,
    ) -> None:
        if feature_columns is None:
            raise ValueError("feature_columns must be provided")

        self.feature_columns = feature_columns
        self.n_jobs = n_jobs
        numeric_features = [column for column in self.feature_columns if column != "posicao"]
        categorical_features = ["posicao"] if "posicao" in self.feature_columns else []

        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            ("numeric", SimpleImputer(strategy="median"), numeric_features),
                            (
                                "categorical",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                                    ]
                                ),
                                categorical_features,
                            ),
                        ]
                    ),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=3,
                        random_state=random_seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_models.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/models.py src/tests/backtesting/test_models.py
git commit -m "feat: plumb random forest worker count"
```

---

### Task 3: Add Parallelism Metadata Helpers

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Modify: `src/tests/backtesting/test_cli.py`
- Test: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write helper and metadata tests**

Add these imports to `src/tests/backtesting/test_runner.py` if missing:

```python
import json
```

Add these tests:

```python
def test_effective_model_n_jobs() -> None:
    assert runner_module._effective_model_n_jobs(1) == -1
    assert runner_module._effective_model_n_jobs(2) == 1
    assert runner_module._effective_model_n_jobs(4) == 1


def test_thread_env_records_explicit_keys(monkeypatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setenv("BLIS_NUM_THREADS", "1")

    assert runner_module._thread_env() == {
        "OMP_NUM_THREADS": None,
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": None,
        "BLIS_NUM_THREADS": "1",
    }


def test_run_backtest_records_parallel_metadata_defaults(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)

    result = run_backtest(BacktestConfig(project_root=tmp_path, start_round=5, budget=100), season_df=season_df)

    assert result.metadata.backtest_jobs == 1
    assert result.metadata.backtest_workers_effective == 1
    assert result.metadata.model_n_jobs_effective == -1
    assert result.metadata.parallel_backend == "sequential"
    assert set(result.metadata.thread_env) == {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
    }

    metadata = json.loads((tmp_path / "data/08_reporting/backtests/2025/run_metadata.json").read_text())
    assert metadata["backtest_jobs"] == 1
    assert metadata["backtest_workers_effective"] == 1
    assert metadata["model_n_jobs_effective"] == -1
    assert metadata["parallel_backend"] == "sequential"
    assert set(metadata["thread_env"]) == {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
    }
```

Update `_metadata_for_config()` in `src/tests/backtesting/test_cli.py` to pass the new `BacktestMetadata` fields:

```python
backtest_jobs=config.jobs,
backtest_workers_effective=1,
model_n_jobs_effective=-1,
parallel_backend="sequential",
thread_env={
    "OMP_NUM_THREADS": None,
    "MKL_NUM_THREADS": None,
    "OPENBLAS_NUM_THREADS": None,
    "BLIS_NUM_THREADS": None,
},
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_effective_model_n_jobs \
  src/tests/backtesting/test_runner.py::test_thread_env_records_explicit_keys \
  src/tests/backtesting/test_runner.py::test_run_backtest_records_parallel_metadata_defaults \
  src/tests/backtesting/test_cli.py \
  -q
```

Expected: failures because metadata fields and helpers do not exist.

- [ ] **Step 3: Implement metadata helpers**

In `src/cartola/backtesting/runner.py`, add imports:

```python
import os
```

Extend `BacktestMetadata` after `wall_clock_seconds`:

```python
    backtest_jobs: int
    backtest_workers_effective: int
    model_n_jobs_effective: int
    parallel_backend: str
    thread_env: dict[str, str | None]
```

Add helpers near `_detected_rounds()`:

```python
THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _effective_model_n_jobs(backtest_jobs: int) -> int:
    if backtest_jobs == 1:
        return -1
    return 1


def _thread_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in THREAD_ENV_KEYS}
```

In `run_backtest()`, before metadata construction:

```python
target_rounds = list(range(config.start_round, max_round + 1))
worker_rounds = [
    round_number
    for round_number in target_rounds
    if round_number not in excluded_rounds and round_number in cached_round_set
]
model_n_jobs_effective = _effective_model_n_jobs(config.jobs)
backtest_workers_effective = 1 if config.jobs == 1 else min(config.jobs, len(worker_rounds)) if worker_rounds else 0
```

Add these fields to `BacktestMetadata(...)`:

```python
backtest_jobs=config.jobs,
backtest_workers_effective=backtest_workers_effective,
model_n_jobs_effective=model_n_jobs_effective,
parallel_backend="sequential" if config.jobs == 1 else "threads",
thread_env=_thread_env(),
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_effective_model_n_jobs \
  src/tests/backtesting/test_runner.py::test_thread_env_records_explicit_keys \
  src/tests/backtesting/test_run_backtest_writes_metadata_for_no_fixture_mode \
  src/tests/backtesting/test_runner.py::test_run_backtest_records_parallel_metadata_defaults \
  src/tests/backtesting/test_cli.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py src/tests/backtesting/test_cli.py
git commit -m "feat: record backtest parallelism metadata"
```

---

### Task 4: Lock `RoundFrameStore` Copy Access

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Test: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write store thread-safety tests**

Add these imports to `src/tests/backtesting/test_runner.py`:

```python
from concurrent.futures import ThreadPoolExecutor
```

Add these tests:

```python
def test_round_frame_store_uses_lock_for_copy_access(monkeypatch) -> None:
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 4)], ignore_index=True)
    store = runner_module.RoundFrameStore(
        season_df=season_df,
        fixtures=None,
        footystats_rows=None,
        matchup_context_mode="none",
    )
    store.build_all([1, 2, 3])

    calls: list[str] = []

    class RecordingLock:
        def __enter__(self) -> None:
            calls.append("enter")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            calls.append("exit")

    store._lock = RecordingLock()  # type: ignore[attr-defined]

    store.prediction_frame(3)
    store.training_frame(
        target_round=3,
        playable_statuses=("Provavel",),
        empty_columns=[*MARKET_COLUMNS, *feature_columns_for_config(BacktestConfig()), "target"],
    )

    assert calls == ["enter", "exit", "enter", "exit"]


def test_round_frame_store_concurrent_reads_do_not_mutate_cached_frames() -> None:
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    store = runner_module.RoundFrameStore(
        season_df=season_df,
        fixtures=None,
        footystats_rows=None,
        matchup_context_mode="none",
    )
    store.build_all([1, 2, 3, 4, 5])
    before = store.prediction_frame(5).sort_index(axis=1).reset_index(drop=True)

    def read_and_mutate(round_number: int) -> int:
        frame = store.prediction_frame(round_number)
        frame.loc[:, "worker_mutation"] = round_number
        training = store.training_frame(
            target_round=round_number,
            playable_statuses=("Provavel",),
            empty_columns=[*MARKET_COLUMNS, *feature_columns_for_config(BacktestConfig()), "target"],
        )
        if not training.empty:
            training.loc[:, "worker_mutation"] = round_number
        return len(frame)

    with ThreadPoolExecutor(max_workers=4) as executor:
        lengths = list(executor.map(read_and_mutate, [2, 3, 4, 5] * 3))

    after = store.prediction_frame(5).sort_index(axis=1).reset_index(drop=True)
    assert all(length > 0 for length in lengths)
    assert "worker_mutation" not in after.columns
    assert_frame_equal(before, after)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_round_frame_store_uses_lock_for_copy_access \
  src/tests/backtesting/test_runner.py::test_round_frame_store_concurrent_reads_do_not_mutate_cached_frames \
  -q
```

Expected: first test fails because `_lock` does not exist or is not used.

- [ ] **Step 3: Implement locked copy access**

In `src/cartola/backtesting/runner.py`, add:

```python
from threading import Lock
```

In `RoundFrameStore.__init__()`:

```python
self._lock = Lock()
```

Change `prediction_frame()`:

```python
def prediction_frame(self, round_number: int) -> pd.DataFrame:
    with self._lock:
        try:
            frame = self._frames[round_number]
        except KeyError as exc:
            raise KeyError(f"Prediction frame for round {round_number} was not built.") from exc
        return frame.copy(deep=True)
```

Change `training_frame()`:

```python
def training_frame(
    self,
    *,
    target_round: int,
    playable_statuses: tuple[str, ...],
    empty_columns: list[str],
) -> pd.DataFrame:
    with self._lock:
        copied_frames = [
            self._frames[round_number].copy(deep=True)
            for round_number in sorted(self._frames)
            if round_number < target_round
        ]

    frames: list[pd.DataFrame] = []
    for round_frame in copied_frames:
        round_frame = round_frame[round_frame["status"].isin(playable_statuses)].copy(deep=True)
        round_frame["target"] = round_frame["pontuacao"]
        frames.append(round_frame)

    if not frames:
        return pd.DataFrame(columns=pd.Index(empty_columns))

    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_round_frame_store_training_frame_matches_public_builder \
  src/tests/backtesting/test_runner.py::test_round_frame_store_returns_deep_copies_for_candidates_and_training \
  src/tests/backtesting/test_runner.py::test_round_frame_store_uses_lock_for_copy_access \
  src/tests/backtesting/test_runner.py::test_round_frame_store_concurrent_reads_do_not_mutate_cached_frames \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: lock round frame store reads"
```

---

### Task 5: Extract Target-Round Evaluation Helper

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Test: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write helper tests**

Add this test to `src/tests/backtesting/test_runner.py`:

```python
def test_evaluate_target_round_returns_round_records_and_frames(tmp_path) -> None:
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)
    store = runner_module.RoundFrameStore(
        season_df=season_df,
        fixtures=None,
        footystats_rows=None,
        matchup_context_mode="none",
    )
    store.build_all([1, 2, 3, 4, 5])
    model_feature_columns = feature_columns_for_config(config)
    empty_training_columns = list(dict.fromkeys([*MARKET_COLUMNS, *model_feature_columns, "target"]))

    result = runner_module._evaluate_target_round(
        round_number=5,
        config=config,
        round_frame_store=store,
        empty_training_columns=empty_training_columns,
        model_feature_columns=model_feature_columns,
        model_n_jobs_effective=1,
    )

    assert result.round_number == 5
    assert len(result.round_rows) == 3
    assert {row["strategy"] for row in result.round_rows} == {"baseline", "random_forest", "price"}
    assert len(result.selected_frames) == 3
    assert len(result.prediction_frames) == 1
    assert result.prediction_frames[0]["rodada"].eq(5).all()
```

Add this test to verify RF receives the effective worker count:

```python
def test_evaluate_target_round_passes_effective_model_n_jobs(tmp_path, monkeypatch) -> None:
    observed_n_jobs: list[int] = []

    class RecordingRandomForestPointPredictor:
        def __init__(
            self,
            random_seed: int = 123,
            feature_columns: list[str] | None = None,
            n_jobs: int = -1,
        ) -> None:
            observed_n_jobs.append(n_jobs)
            self.feature_columns = feature_columns

        def fit(self, frame: pd.DataFrame) -> "RecordingRandomForestPointPredictor":
            return self

        def predict(self, frame: pd.DataFrame) -> pd.Series:
            return frame["prior_points_mean"].astype(float)

    monkeypatch.setattr("cartola.backtesting.runner.RandomForestPointPredictor", RecordingRandomForestPointPredictor)
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)
    config = BacktestConfig(project_root=tmp_path, start_round=5, budget=100)
    store = runner_module.RoundFrameStore(
        season_df=season_df,
        fixtures=None,
        footystats_rows=None,
        matchup_context_mode="none",
    )
    store.build_all([1, 2, 3, 4, 5])
    model_feature_columns = feature_columns_for_config(config)

    runner_module._evaluate_target_round(
        round_number=5,
        config=config,
        round_frame_store=store,
        empty_training_columns=list(dict.fromkeys([*MARKET_COLUMNS, *model_feature_columns, "target"])),
        model_feature_columns=model_feature_columns,
        model_n_jobs_effective=1,
    )

    assert observed_n_jobs == [1]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_evaluate_target_round_returns_round_records_and_frames \
  src/tests/backtesting/test_runner.py::test_evaluate_target_round_passes_effective_model_n_jobs \
  -q
```

Expected: failures because `_evaluate_target_round()` and `RoundEvaluationResult` do not exist.

- [ ] **Step 3: Add `RoundEvaluationResult` and helper**

In `src/cartola/backtesting/runner.py`, add after `FixtureLoadForRun`:

```python
@dataclass(frozen=True)
class RoundEvaluationResult:
    round_number: int
    round_rows: list[dict[str, object]]
    selected_frames: list[pd.DataFrame]
    prediction_frames: list[pd.DataFrame]
```

Add this helper above `run_backtest()`:

```python
def _evaluate_target_round(
    *,
    round_number: int,
    config: BacktestConfig,
    round_frame_store: RoundFrameStore,
    empty_training_columns: list[str],
    model_feature_columns: list[str],
    model_n_jobs_effective: int,
) -> RoundEvaluationResult:
    round_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    training = round_frame_store.training_frame(
        target_round=round_number,
        playable_statuses=config.playable_statuses,
        empty_columns=empty_training_columns,
    )
    candidates = round_frame_store.prediction_frame(round_number)
    candidates = candidates[candidates["status"].isin(config.playable_statuses)].copy(deep=True)

    if training.empty or candidates.empty:
        _record_skipped_round(round_rows, round_number, "TrainingEmpty" if training.empty else "Empty")
        return RoundEvaluationResult(
            round_number=round_number,
            round_rows=round_rows,
            selected_frames=[],
            prediction_frames=[],
        )

    scored_candidates = candidates.copy()
    baseline_model = BaselinePredictor().fit(training)
    forest_model = RandomForestPointPredictor(
        random_seed=config.random_seed,
        feature_columns=model_feature_columns,
        n_jobs=model_n_jobs_effective,
    ).fit(training)
    scored_candidates["baseline_score"] = baseline_model.predict(scored_candidates)
    scored_candidates["random_forest_score"] = forest_model.predict(scored_candidates)
    scored_candidates["price_score"] = scored_candidates[MARKET_OPEN_PRICE_COLUMN].astype(float)
    prediction_frames.append(scored_candidates.copy())

    for strategy, score_column in _strategies().items():
        strategy_candidates = scored_candidates.copy()
        strategy_candidates["predicted_points"] = strategy_candidates[score_column]
        result = optimize_squad(strategy_candidates, score_column="predicted_points", config=config)
        actual_scores = _actual_scores_for_result(
            result.selected,
            round_number=round_number,
            strategy=strategy,
            solver_status=result.status,
        )
        policy_diagnostics = _policy_diagnostics_for_result(
            result.selected,
            round_number=round_number,
            strategy=strategy,
            solver_status=result.status,
        )
        policy_summary = _policy_round_summary(policy_diagnostics)
        actual_points_with_captain = actual_scores["actual_points_with_captain"]
        round_rows.append(
            {
                "rodada": round_number,
                "strategy": strategy,
                "solver_status": result.status,
                "formation": result.formation_name,
                "selected_count": result.selected_count,
                "budget_used": result.budget_used,
                "predicted_points": result.predicted_points_with_captain,
                "predicted_points_base": result.predicted_points_base,
                "captain_bonus_predicted": result.captain_bonus_predicted,
                "predicted_points_with_captain": result.predicted_points_with_captain,
                "actual_points": actual_points_with_captain,
                "actual_points_base": actual_scores["actual_points_base"],
                "captain_bonus_actual": actual_scores["captain_bonus_actual"],
                "actual_points_with_captain": actual_points_with_captain,
                "captain_id": result.captain_id,
                "captain_name": result.captain_name,
                **policy_summary,
            }
        )

        if not result.selected.empty:
            selected = result.selected.copy()
            apply_captain_policy_flags(selected, policy_diagnostics)
            selected["rodada"] = round_number
            selected["strategy"] = strategy
            selected_frames.append(selected)

    return RoundEvaluationResult(
        round_number=round_number,
        round_rows=round_rows,
        selected_frames=selected_frames,
        prediction_frames=prediction_frames,
    )
```

- [ ] **Step 4: Replace inline per-round body with helper call**

In `run_backtest()`, replace the existing inline target-round evaluation body with:

```python
for round_number in target_rounds:
    if round_number in excluded_rounds:
        continue
    if round_number not in cached_round_set:
        _record_skipped_round(round_rows, round_number, "Empty")
        continue

    result = _evaluate_target_round(
        round_number=round_number,
        config=config,
        round_frame_store=round_frame_store,
        empty_training_columns=empty_training_columns,
        model_feature_columns=model_feature_columns,
        model_n_jobs_effective=model_n_jobs_effective,
    )
    round_rows.extend(result.round_rows)
    selected_frames.extend(result.selected_frames)
    prediction_frames.extend(result.prediction_frames)
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "refactor: extract backtest round evaluation"
```

---

### Task 6: Add Canonical Output Sorting

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Test: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write sorting tests**

Add this test to `src/tests/backtesting/test_runner.py`:

```python
def test_sort_outputs_canonicalizes_report_order() -> None:
    round_results = pd.DataFrame(
        [
            {"rodada": 6, "strategy": "price"},
            {"rodada": 5, "strategy": "random_forest"},
            {"rodada": 5, "strategy": "baseline"},
        ]
    )
    selected_players = pd.DataFrame(
        [
            {"rodada": 5, "strategy": "baseline", "id_atleta": 3},
            {"rodada": 5, "strategy": "baseline", "id_atleta": 1},
        ]
    )
    player_predictions = pd.DataFrame(
        [
            {"rodada": 6, "id_atleta": 2},
            {"rodada": 5, "id_atleta": 3},
        ]
    )
    summary = pd.DataFrame([{"strategy": "price"}, {"strategy": "baseline"}])
    diagnostics = pd.DataFrame(
        [
            {"section": "z", "strategy": "price", "position": "all", "metric": "b"},
            {"section": "a", "strategy": "baseline", "position": "all", "metric": "a"},
        ]
    )

    sorted_outputs = runner_module._sort_outputs(
        round_results=round_results,
        selected_players=selected_players,
        player_predictions=player_predictions,
        summary=summary,
        diagnostics=diagnostics,
    )

    assert sorted_outputs["round_results"]["strategy"].tolist() == ["baseline", "random_forest", "price"]
    assert sorted_outputs["selected_players"]["id_atleta"].tolist() == [1, 3]
    assert sorted_outputs["player_predictions"]["rodada"].tolist() == [5, 6]
    assert sorted_outputs["summary"]["strategy"].tolist() == ["baseline", "price"]
    assert sorted_outputs["diagnostics"]["section"].tolist() == ["a", "z"]
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py::test_sort_outputs_canonicalizes_report_order -q
```

Expected: failure because `_sort_outputs()` does not exist.

- [ ] **Step 3: Implement sorting helpers**

Add these helpers in `src/cartola/backtesting/runner.py` near `_normalize_float_outputs()`:

```python
SORT_KEYS: dict[str, list[str]] = {
    "round_results": ["rodada", "strategy"],
    "selected_players": ["rodada", "strategy", "id_atleta"],
    "player_predictions": ["rodada", "id_atleta"],
    "summary": ["strategy"],
    "diagnostics": ["section", "strategy", "position", "metric"],
}


def _sort_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    available_keys = [key for key in keys if key in frame.columns]
    if not available_keys:
        return frame.copy()
    return frame.sort_values(available_keys, kind="mergesort").reset_index(drop=True)


def _sort_outputs(
    *,
    round_results: pd.DataFrame,
    selected_players: pd.DataFrame,
    player_predictions: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "round_results": _sort_frame(round_results, SORT_KEYS["round_results"]),
        "selected_players": _sort_frame(selected_players, SORT_KEYS["selected_players"]),
        "player_predictions": _sort_frame(player_predictions, SORT_KEYS["player_predictions"]),
        "summary": _sort_frame(summary, SORT_KEYS["summary"]),
        "diagnostics": _sort_frame(diagnostics, SORT_KEYS["diagnostics"]),
    }
```

In `run_backtest()`, sort before normalization:

```python
sorted_outputs = _sort_outputs(
    round_results=round_results,
    selected_players=selected_players,
    player_predictions=player_predictions,
    summary=summary,
    diagnostics=diagnostics,
)
round_results = sorted_outputs["round_results"]
selected_players = sorted_outputs["selected_players"]
player_predictions = sorted_outputs["player_predictions"]
summary = sorted_outputs["summary"]
diagnostics = sorted_outputs["diagnostics"]
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_sort_outputs_canonicalizes_report_order \
  src/tests/backtesting/test_runner.py::test_run_backtest_records_selected_players_and_prediction_diagnostics \
  src/tests/backtesting/test_runner.py::test_run_backtest_normalizes_tiny_float_drift_in_returned_outputs \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: canonicalize backtest report ordering"
```

---

### Task 7: Add Threaded Round Execution

**Files:**
- Modify: `src/cartola/backtesting/runner.py`
- Test: `src/tests/backtesting/test_runner.py`

- [ ] **Step 1: Write parallel tests**

Add imports to `src/cartola/backtesting/runner.py` during implementation, not in tests yet:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

Add these tests to `src/tests/backtesting/test_runner.py`:

```python
def _semantic_metadata(metadata: object) -> dict[str, object]:
    values = dict(metadata.__dict__)
    for key in [
        "wall_clock_seconds",
        "backtest_jobs",
        "backtest_workers_effective",
        "model_n_jobs_effective",
        "parallel_backend",
    ]:
        values.pop(key, None)
    return values


def test_run_backtest_jobs_2_matches_jobs_1_outputs(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 7)], ignore_index=True)

    sequential = run_backtest(
        BacktestConfig(project_root=tmp_path / "sequential", start_round=5, budget=100, jobs=1),
        season_df=season_df,
    )
    parallel = run_backtest(
        BacktestConfig(project_root=tmp_path / "parallel", start_round=5, budget=100, jobs=2),
        season_df=season_df,
    )

    assert_frame_equal(sequential.round_results, parallel.round_results, check_dtype=False, atol=1e-10, rtol=0)
    assert_frame_equal(sequential.selected_players, parallel.selected_players, check_dtype=False, atol=1e-10, rtol=0)
    assert_frame_equal(sequential.player_predictions, parallel.player_predictions, check_dtype=False, atol=1e-10, rtol=0)
    assert_frame_equal(sequential.summary, parallel.summary, check_dtype=False, atol=1e-10, rtol=0)
    assert_frame_equal(sequential.diagnostics, parallel.diagnostics, check_dtype=False, atol=1e-10, rtol=0)
    assert _semantic_metadata(sequential.metadata) == _semantic_metadata(parallel.metadata)
    assert parallel.metadata.backtest_jobs == 2
    assert parallel.metadata.backtest_workers_effective == 2
    assert parallel.metadata.model_n_jobs_effective == 1
    assert parallel.metadata.parallel_backend == "threads"


def test_run_backtest_jobs_above_worker_rounds_records_effective_workers(tmp_path):
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 6)], ignore_index=True)

    result = run_backtest(
        BacktestConfig(project_root=tmp_path, start_round=5, budget=100, jobs=4),
        season_df=season_df,
    )

    assert result.metadata.backtest_jobs == 4
    assert result.metadata.backtest_workers_effective == 1
    assert result.metadata.parallel_backend == "threads"


def test_run_backtest_missing_round_is_not_submitted_to_worker(tmp_path, monkeypatch):
    season_df = pd.concat([_tiny_round(1), _tiny_round(3)], ignore_index=True)
    called_rounds: list[int] = []
    original = runner_module._evaluate_target_round

    def recording_evaluate_target_round(**kwargs: object) -> object:
        called_rounds.append(int(kwargs["round_number"]))
        return original(**kwargs)

    monkeypatch.setattr(runner_module, "_evaluate_target_round", recording_evaluate_target_round)

    result = run_backtest(
        BacktestConfig(project_root=tmp_path, start_round=2, budget=100, jobs=2),
        season_df=season_df,
    )

    assert called_rounds == [3]
    round_2 = result.round_results[result.round_results["rodada"].eq(2)]
    assert round_2["solver_status"].eq("Empty").all()
```

Add worker failure tests:

```python
def test_parallel_worker_failure_preserves_round_context(tmp_path, monkeypatch) -> None:
    season_df = pd.concat([_tiny_round(round_number) for round_number in range(1, 7)], ignore_index=True)
    original = runner_module._evaluate_target_round

    def failing_evaluate_target_round(**kwargs: object) -> object:
        round_number = int(kwargs["round_number"])
        if round_number == 5:
            raise ValueError("boom")
        return original(**kwargs)

    monkeypatch.setattr(runner_module, "_evaluate_target_round", failing_evaluate_target_round)

    with pytest.raises(runner_module.BacktestRoundEvaluationError, match="round 5") as exc_info:
        run_backtest(
            BacktestConfig(project_root=tmp_path, start_round=5, budget=100, jobs=2),
            season_df=season_df,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert not (tmp_path / "data/08_reporting/backtests/2025/round_results.csv").exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_runner.py::test_run_backtest_jobs_2_matches_jobs_1_outputs \
  src/tests/backtesting/test_runner.py::test_run_backtest_jobs_above_worker_rounds_records_effective_workers \
  src/tests/backtesting/test_runner.py::test_run_backtest_missing_round_is_not_submitted_to_worker \
  src/tests/backtesting/test_runner.py::test_parallel_worker_failure_preserves_round_context \
  -q
```

Expected: failures because threaded execution and `BacktestRoundEvaluationError` do not exist.

- [ ] **Step 3: Implement threaded execution**

In `src/cartola/backtesting/runner.py`, add imports:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

Add exception class near `RoundEvaluationResult`:

```python
class BacktestRoundEvaluationError(RuntimeError):
    def __init__(self, round_number: int, message: str) -> None:
        super().__init__(f"Failed to evaluate round {round_number}: {message}")
        self.round_number = round_number
```

Add helper:

```python
def _target_round_work(
    *,
    target_rounds: list[int],
    excluded_rounds: list[int],
    cached_round_set: set[int],
) -> tuple[list[RoundEvaluationResult], list[int]]:
    skipped_results: list[RoundEvaluationResult] = []
    worker_rounds: list[int] = []
    excluded_round_set = set(excluded_rounds)
    for round_number in target_rounds:
        if round_number in excluded_round_set:
            continue
        if round_number not in cached_round_set:
            round_rows: list[dict[str, object]] = []
            _record_skipped_round(round_rows, round_number, "Empty")
            skipped_results.append(
                RoundEvaluationResult(
                    round_number=round_number,
                    round_rows=round_rows,
                    selected_frames=[],
                    prediction_frames=[],
                )
            )
            continue
        worker_rounds.append(round_number)
    return skipped_results, worker_rounds
```

Add helper:

```python
def _run_round_workers(
    *,
    config: BacktestConfig,
    worker_rounds: list[int],
    round_frame_store: RoundFrameStore,
    empty_training_columns: list[str],
    model_feature_columns: list[str],
    model_n_jobs_effective: int,
) -> list[RoundEvaluationResult]:
    if config.jobs == 1:
        return [
            _evaluate_target_round(
                round_number=round_number,
                config=config,
                round_frame_store=round_frame_store,
                empty_training_columns=empty_training_columns,
                model_feature_columns=model_feature_columns,
                model_n_jobs_effective=model_n_jobs_effective,
            )
            for round_number in worker_rounds
        ]

    if not worker_rounds:
        return []

    max_workers = min(config.jobs, len(worker_rounds))
    results: list[RoundEvaluationResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_round = {
            executor.submit(
                _evaluate_target_round,
                round_number=round_number,
                config=config,
                round_frame_store=round_frame_store,
                empty_training_columns=empty_training_columns,
                model_feature_columns=model_feature_columns,
                model_n_jobs_effective=model_n_jobs_effective,
            ): round_number
            for round_number in worker_rounds
        }
        for future in as_completed(future_to_round):
            round_number = future_to_round[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise BacktestRoundEvaluationError(round_number, str(exc)) from exc
    return results
```

In `run_backtest()`, replace the for-loop around `_evaluate_target_round()` with:

```python
skipped_results, worker_rounds = _target_round_work(
    target_rounds=target_rounds,
    excluded_rounds=excluded_rounds,
    cached_round_set=cached_round_set,
)
round_results_for_targets = [
    *skipped_results,
    *_run_round_workers(
        config=config,
        worker_rounds=worker_rounds,
        round_frame_store=round_frame_store,
        empty_training_columns=empty_training_columns,
        model_feature_columns=model_feature_columns,
        model_n_jobs_effective=model_n_jobs_effective,
    ),
]
for result in sorted(round_results_for_targets, key=lambda item: item.round_number):
    round_rows.extend(result.round_rows)
    selected_frames.extend(result.selected_frames)
    prediction_frames.extend(result.prediction_frames)
```

Compute `backtest_workers_effective` from the same `worker_rounds`:

```python
backtest_workers_effective = 1 if config.jobs == 1 else min(config.jobs, len(worker_rounds)) if worker_rounds else 0
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --frozen pytest src/tests/backtesting/test_runner.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/cartola/backtesting/runner.py src/tests/backtesting/test_runner.py
git commit -m "feat: parallelize backtest round evaluation"
```

---

### Task 8: Benchmark And Full Verification

**Files:**
- No planned source modifications.
- Benchmark output directories under `data/08_reporting/backtests/perf_parallel_jobs_*` must remain uncommitted.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run --frozen pytest \
  src/tests/backtesting/test_config.py \
  src/tests/backtesting/test_cli.py \
  src/tests/backtesting/test_models.py \
  src/tests/backtesting/test_runner.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run full quality gate**

Run:

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: ruff, ty, bandit, and pytest pass.

- [ ] **Step 3: Run benchmark matrix**

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
  --jobs 1 \
  --output-root data/08_reporting/backtests/perf_parallel_jobs_1
```

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
  --jobs 2 \
  --output-root data/08_reporting/backtests/perf_parallel_jobs_2
```

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
  --jobs 4 \
  --output-root data/08_reporting/backtests/perf_parallel_jobs_4
```

Expected: all commands complete and write reports.

- [ ] **Step 4: Compare benchmark outputs semantically**

Run this Python snippet from the repo root:

```bash
uv run --frozen python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

root = Path("data/08_reporting/backtests")
base = root / "perf_parallel_jobs_1" / "2025"
parallel_paths = [
    root / "perf_parallel_jobs_2" / "2025",
    root / "perf_parallel_jobs_4" / "2025",
]
csv_files = [
    "round_results.csv",
    "selected_players.csv",
    "player_predictions.csv",
    "summary.csv",
    "diagnostics.csv",
]
metadata_exclusions = {
    "wall_clock_seconds",
    "backtest_jobs",
    "backtest_workers_effective",
    "model_n_jobs_effective",
    "parallel_backend",
}

for parallel in parallel_paths:
    for filename in csv_files:
        left = pd.read_csv(base / filename)
        right = pd.read_csv(parallel / filename)
        assert_frame_equal(left, right, check_dtype=False, atol=1e-10, rtol=0)

    left_meta = json.loads((base / "run_metadata.json").read_text())
    right_meta = json.loads((parallel / "run_metadata.json").read_text())
    for key in metadata_exclusions:
        left_meta.pop(key, None)
        right_meta.pop(key, None)
    assert left_meta == right_meta

print("benchmark outputs are semantically equivalent")
PY
```

Expected: prints `benchmark outputs are semantically equivalent`.

- [ ] **Step 5: Decide success from timing**

Compare `/usr/bin/time -p` `real` values.

Success condition:

```text
min(jobs_2_real, jobs_4_real) <= jobs_1_real * 0.85
```

If the fastest parallel run is within 15% of `jobs=1`, repeat only `jobs=1` and the fastest parallel setting once to rule out machine noise.

If the repeated fastest parallel run still misses the 15% threshold, stop. Do not tune thread counts further in this implementation branch. Write the result into the final summary and propose a profiling/process-backend spike as the next design artifact.

- [ ] **Step 6: Remove benchmark output directories before commit**

Run:

```bash
rm -rf \
  data/08_reporting/backtests/perf_parallel_jobs_1 \
  data/08_reporting/backtests/perf_parallel_jobs_2 \
  data/08_reporting/backtests/perf_parallel_jobs_4
```

Expected: benchmark output roots are removed from the working tree.

- [ ] **Step 7: Commit final verification-only changes if any**

If no files changed during verification, do not create an empty commit.

If test-only or docs corrections were necessary, inspect the exact paths first:

```bash
git status --short
```

Stage only the displayed test or docs files that were changed during verification, then commit with:

```bash
git commit -m "test: verify backtest parallelism"
```

---

## Self-Review Checklist

- [ ] Spec coverage: `jobs` config, RF `n_jobs`, locked store access, worker helper, threaded execution, metadata, sorting, equivalence, failure behavior, and benchmark gate all map to tasks above.
- [ ] Red-flag scan: run a phrase search for unfinished planning language in this file and remove every match before execution.
- [ ] Type consistency: `RoundEvaluationResult`, `BacktestRoundEvaluationError`, `_effective_model_n_jobs`, `_thread_env`, `_target_round_work`, `_run_round_workers`, and `_sort_outputs` use the same names in tests and implementation snippets.
- [ ] Scope guard: no model registry, grid search, process pool, feature changes, optimizer changes, or on-disk cache work appears in any task.
