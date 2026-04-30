# Backtest Parallelism Design

## Goal

Add controlled target-round parallelism to the Cartola walk-forward backtest so matchup-context runs become fast enough for routine feature and model iteration, without changing scoring, feature, optimizer, or report semantics.

This is Phase 1b after the in-memory `RoundFrameStore` cache. It must build on the cache and must not introduce model-search scope.

## Current Baseline

The Phase 1a cache is merged locally. A 2025 matchup-context backtest on the Mac Studio measured:

```text
real 220.58
```

Command:

```bash
/usr/bin/time -p uv run --frozen python -m cartola.backtesting.cli \
  --season 2025 \
  --start-round 5 \
  --budget 100 \
  --fixture-mode exploratory \
  --footystats-mode ppg \
  --matchup-context-mode cartola_matchup_v1 \
  --current-year 2026 \
  --output-root data/08_reporting/backtests/perf_manual_cache
```

Phase 1a now builds 38 prediction frames once and records:

- `cache_enabled`;
- `prediction_frames_built`;
- `wall_clock_seconds`.

The remaining likely cost is per-target-round evaluation:

- assembling training data from cached frames;
- fitting `BaselinePredictor`;
- fitting `RandomForestPointPredictor`;
- predicting candidate scores;
- optimizing each strategy with the squad optimizer;
- computing diagnostics and selected-player outputs.

Target rounds are independent after cached frame construction because each target round consumes only immutable cached frames and returns its own in-memory records.

## Non-Goals

Do not add:

- model registry;
- grid search;
- experiment runner;
- process-pool backend;
- on-disk frame cache;
- public no-cache mode;
- RF parameter changes beyond plumbing `n_jobs`;
- feature changes;
- optimizer logic changes;
- report schema changes unrelated to parallelism metadata.

## Required Decisions

### Backend

Use `concurrent.futures.ThreadPoolExecutor` for v1.

Process pools are explicitly rejected for v1 because:

- cached pandas frames would need to be pickled or copied into workers;
- serialization can erase the speedup;
- memory use can grow sharply;
- error/debug context is harder;
- parent-owned `RoundFrameStore` is already designed for copy-returning reads in a single process.

If thread benchmarks do not improve runtime, write a separate process-pool spike/spec. Do not smuggle process-pool support into Phase 1b.

This choice is experimental, not ideological. scikit-learn documents that `n_jobs` controls joblib-managed estimator parallelism, while lower-level OpenMP/BLAS thread pools can also affect runtime. That makes a thread proof worth trying, but not guaranteed to win. If the benchmark matrix shows weak or negative scaling, Phase 1b stops and the next artifact is a profiling/process-backend spike, not more thread tuning.

### Worker Ownership

Parent owns:

- loading data;
- resolving fixtures;
- resolving FootyStats rows;
- validating joins/alignment;
- building `RoundFrameStore`;
- creating metadata;
- launching workers;
- aggregating worker outputs;
- sorting outputs;
- normalizing floats;
- writing reports.

Workers own only target-round evaluation for one `round_number`.

Workers must return in-memory data only:

- round-result row dicts;
- selected-player frames;
- player-prediction frames.

Workers must not write files.

Worker return schema:

```python
@dataclass(frozen=True)
class RoundEvaluationResult:
    round_number: int
    round_rows: list[dict[str, object]]
    selected_frames: list[pd.DataFrame]
    prediction_frames: list[pd.DataFrame]
```

Rules:

- `round_number` is the evaluated target round;
- `round_rows` contains exactly the rows that would have been appended for that round in sequential mode;
- `selected_frames` contains zero or more selected-player dataframes for that round;
- `prediction_frames` contains zero or one scored candidate dataframe for that round;
- skipped rounds handled by the parent use the same `RoundEvaluationResult` shape with empty frame lists.

Empty-frame schema rules:

- `round_rows` is materialized by the parent with `ROUND_RESULT_COLUMNS`;
- skipped rounds and infeasible strategies return empty `selected_frames` and `prediction_frames` lists, not placeholder dataframes;
- non-empty `selected_frames` use `result.selected` columns plus `rodada` and `strategy`;
- non-empty `prediction_frames` use the scored candidate columns after adding `baseline_score`, `random_forest_score`, and `price_score`;
- if the final run has no selected-player frames or no prediction frames, the parent uses `pd.DataFrame()` for that output, preserving the existing `_concat_or_empty()` behavior.

### Shared Store Access

The parent builds `RoundFrameStore` once before launching workers.

Each worker receives `round_number` and reads frames through the store's existing copy-returning methods:

```python
training = round_frame_store.training_frame(...)
candidates = round_frame_store.prediction_frame(round_number)
```

Workers must not access or mutate `round_frame_store._frames` directly.

Thread-safety contract:

- cached frames are immutable after `build_all()` completes;
- `build_all()` must complete before any worker is submitted;
- `RoundFrameStore` owns a private lock;
- `prediction_frame()` acquires the lock, copies the cached frame with `copy(deep=True)`, releases the lock, then returns the copy;
- `training_frame()` acquires the lock, copies all cached frames needed for rounds `< target_round`, releases the lock, then performs filtering, target-column assignment, and `pd.concat()` outside the lock;
- no model fitting, prediction, optimization, diagnostics, or report shaping may run while holding the store lock.

pandas documents that it is not fully thread-safe and specifically recommends locks around concurrent `DataFrame.copy()` operations. The lock is required for v1. A lock-free store is out of scope unless a later design includes a targeted pandas-copy stress test and accepts the risk explicitly.

## Config And CLI

Add `jobs` to `BacktestConfig`:

```python
jobs: int = 1
```

Validation:

- must be an integer;
- must be `>= 1`;
- invalid values fail before report output.

Add CLI flag:

```bash
--jobs 1
```

Default behavior remains `jobs=1`.

`jobs=1` is the behavior-preserving baseline. It must use the same sequential round order as today.

When `worker_rounds` is non-empty, the executor uses:

```python
max_workers = min(config.jobs, len(worker_rounds))
```

If there are no worker rounds, no executor is created.

## RandomForest Worker Policy

`RandomForestPointPredictor` currently hardcodes `RandomForestRegressor(n_jobs=-1)`.

Phase 1b must plumb `n_jobs` into `RandomForestPointPredictor`:

```python
RandomForestPointPredictor(
    random_seed=config.random_seed,
    feature_columns=model_feature_columns,
    n_jobs=model_n_jobs_effective,
)
```

Add private helper:

```python
def _effective_model_n_jobs(backtest_jobs: int) -> int:
    if backtest_jobs == 1:
        return -1
    return 1
```

Required behavior:

- `jobs=1` -> `model_n_jobs_effective=-1`;
- `jobs>1` -> `model_n_jobs_effective=1`.

This must affect the actual `RandomForestRegressor.n_jobs` parameter, not just metadata.

Reason: RandomForest fit/predict parallelizes over trees. Outer target-round parallelism plus RF `n_jobs=-1` would oversubscribe the 28-core machine and can be slower than sequential execution.

Dynamic allocation such as `cpu_count // backtest_jobs` is out of scope for v1. It may use more cores, but it also reintroduces nested parallelism risk and makes the first benchmark harder to interpret. If `jobs>1` with RF `n_jobs=1` underutilizes the machine, record that result and design a separate worker-allocation benchmark.

Metadata must also record present thread-control environment variables for benchmark debugging:

- `OMP_NUM_THREADS`;
- `MKL_NUM_THREADS`;
- `OPENBLAS_NUM_THREADS`;
- `BLIS_NUM_THREADS`.

If a variable is absent, record it as `null`. These fields are explanatory metadata only and must not affect behavior.

## Runner Architecture

Extract target-round evaluation into a private helper. Suggested shape:

```python
@dataclass(frozen=True)
class RoundEvaluationResult:
    round_number: int
    round_rows: list[dict[str, object]]
    selected_frames: list[pd.DataFrame]
    prediction_frames: list[pd.DataFrame]


def _evaluate_target_round(
    *,
    round_number: int,
    config: BacktestConfig,
    round_frame_store: RoundFrameStore,
    empty_training_columns: list[str],
    model_feature_columns: list[str],
    model_n_jobs_effective: int,
) -> RoundEvaluationResult:
    ...
```

If target-round evaluation needs a dependency not available through `config`, `round_number`, `round_frame_store`, `empty_training_columns`, `model_feature_columns`, or `model_n_jobs_effective`, pass it explicitly. Do not read mutable module-level state from workers.

The helper contains the current per-round body:

1. create training frame from cached rounds `< round_number`;
2. create candidate frame for `round_number`;
3. filter candidate statuses;
4. record `TrainingEmpty` or `Empty` skipped rows when needed;
5. fit baseline;
6. fit RF with `model_n_jobs_effective`;
7. score candidates;
8. run all strategies through optimizer;
9. compute captain-aware actuals and policy diagnostics;
10. return records/frames.

The helper must not write reports.

The helper receives only cached/evaluable target rounds. It does not decide whether a round is excluded or missing from the season data.

### Sequential Path

When `config.jobs == 1`, use a plain for loop:

```python
for round_number in target_rounds:
    result = _evaluate_target_round(...)
```

No thread executor is used in this path.

### Parallel Path

When `config.jobs > 1`, use:

```python
max_workers = min(config.jobs, len(worker_rounds))
ThreadPoolExecutor(max_workers=max_workers)
```

Submit one future per target round.

Worker exception behavior:

- one failed target round fails the whole backtest;
- do not silently record `Empty` for exceptions;
- wrap failures with round context.

Suggested exception wrapping:

```python
class BacktestRoundEvaluationError(RuntimeError):
    def __init__(self, round_number: int, message: str) -> None:
        super().__init__(f"Failed to evaluate round {round_number}: {message}")
        self.round_number = round_number
```

When a future raises, re-raise `BacktestRoundEvaluationError(round_number, str(exc))` with the original exception as `__cause__`.

Use `concurrent.futures.as_completed()` with a `future_to_round` mapping. Do not use bare `executor.map()` because it weakens round-specific error wrapping.

## Target Rounds

Build target rounds from `range(config.start_round, max_round + 1)` as today.

Rules:

- excluded rounds are skipped before worker submission;
- missing non-excluded rounds are handled by the parent before worker submission;
- parent records `Empty` skipped rows for missing non-excluded rounds;
- only rounds present in `RoundFrameStore` are submitted to workers;
- no worker is launched for excluded rounds.

The target-round list must be constructed in the parent.

A target-round worker may read only cached frames for rounds `< round_number` when assembling training data and exactly `round_number` when reading candidates.

## Output Aggregation And Sorting

Parent aggregates all worker-returned rows/frames.

Parent must sort outputs deterministically before normalization/writing.

Required sort keys:

- `round_results.csv`: `["rodada", "strategy"]`;
- `selected_players.csv`: `["rodada", "strategy", "id_atleta"]`;
- `player_predictions.csv`: `["rodada", "id_atleta"]`;
- `summary.csv`: `["strategy"]`;
- `diagnostics.csv`: `["section", "strategy", "position", "metric"]`.

Sorting must happen for both `jobs=1` and `jobs>1` so output ordering does not depend on execution order.

If a dataframe is empty, sorting must preserve an empty dataframe without raising.

Canonical sorting becomes the report row-order contract for both sequential and parallel runs. `jobs=1` must preserve current semantic behavior, but byte-for-byte row order from older unsorted reports is not a compatibility target.

## Output Equivalence Contract

`jobs=1` and `jobs=2` must produce equivalent reports for the same input config.

Compare:

- `round_results.csv`;
- `selected_players.csv`;
- `player_predictions.csv`;
- `summary.csv`;
- `diagnostics.csv`;
- `run_metadata.json` semantic fields.

Comparison rules:

- sort rows using the required keys;
- compare column sets exactly;
- numeric tolerance: absolute tolerance `1e-10`;
- non-numeric values compare exactly with null semantics preserved;
- exclude runtime-only fields from equality:
  - `wall_clock_seconds`;
- allow expected parallelism fields to differ:
  - `backtest_jobs`;
  - `backtest_workers_effective`;
  - `model_n_jobs_effective`;
  - `parallel_backend`;
- require `thread_env` to have the same explicit keys in both metadata files, with absent environment variables represented as JSON `null`;
- require all other metadata fields to match.

Float normalization applies to all float columns in `round_results`, `selected_players`, `player_predictions`, `summary`, and `diagnostics` except the identifier/count columns listed in `FLOAT_NORMALIZATION_EXCLUDED_COLUMNS`. Normalization happens before CSV writing and before in-memory equivalence assertions.

## Metadata

Add required metadata fields:

- `backtest_jobs`;
- `backtest_workers_effective`;
- `model_n_jobs_effective`;
- `parallel_backend`;
- `thread_env`.

Values:

- `backtest_jobs`: integer from config;
- `backtest_workers_effective`: `1` when `jobs == 1`; otherwise `min(config.jobs, len(worker_rounds))`, or `0` when there are no worker rounds;
- `model_n_jobs_effective`: result of `_effective_model_n_jobs(config.jobs)`;
- `parallel_backend`: `"sequential"` when `jobs == 1`, `"threads"` when `jobs > 1`;
- `thread_env`: object containing `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `BLIS_NUM_THREADS` with string values or `null`.

Existing Phase 1a metadata remains:

- `cache_enabled`;
- `prediction_frames_built`;
- `wall_clock_seconds`.

Do not add process-pool metadata in v1.

## Tests

### Config And CLI

Required tests:

1. `BacktestConfig(jobs=1)` is accepted.
2. `BacktestConfig(jobs=0)` fails validation.
3. `BacktestConfig(jobs=-1)` fails validation.
4. CLI parses `--jobs 2` and passes `jobs=2` to `BacktestConfig`.
5. CLI default remains `jobs=1`.

If `BacktestConfig` does not currently validate fields in `__post_init__`, Phase 1b may add validation only for `jobs`; do not broaden config validation beyond this feature.

### Model Worker Plumbing

Required tests:

1. `RandomForestPointPredictor(..., n_jobs=1)` sets the underlying `RandomForestRegressor.n_jobs` to `1`.
2. Default `RandomForestPointPredictor(..., n_jobs=-1)` preserves current single-backtest behavior.
3. `_effective_model_n_jobs(1) == -1`.
4. `_effective_model_n_jobs(2) == 1`.
5. `_effective_model_n_jobs(4) == 1`.

Existing tests that monkeypatch `RandomForestPointPredictor` must keep passing. If constructor monkeypatches do not accept `n_jobs`, update test doubles to accept the new parameter explicitly.

### Worker Helper

Required tests:

1. `_evaluate_target_round()` returns the same rows/frames as the previous inline per-round logic for one round.
2. Parent missing-round handling returns `Empty` skipped rows without launching a worker.
3. Training-empty round returns `TrainingEmpty` skipped rows.
4. Worker helper does not write output files.
5. Worker helper uses `model_n_jobs_effective` when constructing RF.
6. Worker helper receives only rounds present in `RoundFrameStore`.
7. Worker helper creates fresh predictor/model instances per target round and never shares estimator instances between workers.

### RoundFrameStore Thread Safety

Required tests:

1. `build_all()` freezes the cache before workers are launched.
2. `prediction_frame()` returns a deep copy and does not expose `_frames` directly.
3. `training_frame()` returns data assembled from deep-copied cached frames.
4. Concurrent `prediction_frame()` and `training_frame()` calls under `jobs=4` do not mutate cached frames.
5. Store copy/assembly operations acquire the private lock; RF, optimizer, and diagnostics work run outside the lock.

### Parallel Equivalence

Required tests:

1. `jobs=1` and `jobs=2` produce equivalent `round_results`.
2. `jobs=1` and `jobs=2` produce equivalent `selected_players`.
3. `jobs=1` and `jobs=2` produce equivalent `player_predictions`.
4. `jobs=1` and `jobs=2` produce equivalent `summary`.
5. `jobs=1` and `jobs=2` produce equivalent `diagnostics`.
6. Metadata equality excludes only `wall_clock_seconds`, `backtest_jobs`, `backtest_workers_effective`, `model_n_jobs_effective`, and `parallel_backend`; all other semantic metadata matches.
7. `jobs=2` metadata records `parallel_backend="threads"`.
8. `jobs=1` metadata records `parallel_backend="sequential"`.
9. `selected_players` row order is deterministic after sorting by `["rodada", "strategy", "id_atleta"]`.
10. `thread_env` contains all required keys with string values or explicit null values in both metadata files.

### Worker Failure

Required tests:

1. If target round `N` raises inside `_evaluate_target_round()`, `run_backtest()` raises `BacktestRoundEvaluationError`.
2. The error message includes `round N`.
3. The original exception is preserved as `__cause__`.
4. No partial report output is written after a worker failure if failure occurs before final aggregation/write.

## Benchmark Matrix

Run the same 2025 matchup-context benchmark with:

- `--jobs 1`;
- `--jobs 2`;
- `--jobs 4`.

Commands:

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

Record:

- `/usr/bin/time` `real`, `user`, `sys`;
- metadata `wall_clock_seconds`;
- metadata `backtest_jobs`;
- metadata `backtest_workers_effective`;
- metadata `model_n_jobs_effective`;
- metadata `parallel_backend`;
- metadata `thread_env`;
- whether report equivalence holds versus `jobs=1`.

Temporary benchmark output roots must not be committed.

Success criterion:

- `jobs=2` or `jobs=4` improves wall-clock time over `jobs=1` by at least 15%; and
- if neither improves, report the result honestly and stop before expanding parallelism further.

Do not declare Phase 1b successful on metadata alone. Runtime must be benchmarked.

Run each benchmark at least once. If the fastest parallel result is within 15% of `jobs=1`, repeat `jobs=1` and the fastest parallel setting once more to rule out ordinary machine noise before rejecting the thread design.

## Failure Semantics

Parallel execution must fail closed.

Allowed skipped statuses:

- `TrainingEmpty`;
- `Empty`.

These statuses are for expected data conditions only.

Unexpected worker exceptions are not skipped rounds. They fail the whole command with round context.

## Risks

### Threads May Not Help

Some of the work may remain Python/GIL-bound. If `jobs=2` and `jobs=4` do not improve runtime, do not add process pools immediately. First profile the remaining bottleneck.

Profiling must split the target-round work into at least:

- training-frame assembly;
- baseline fit/predict;
- RF fit/predict;
- optimizer calls;
- report-shaping/captain diagnostics.

### Memory Pressure

Each worker assembles a training dataframe. More workers can duplicate large frames.

Mitigation:

- default `jobs=1`;
- benchmark `2` and `4` before recommending a live default;
- do not test high worker counts as a default path in v1.

### Row Ordering

Parallel futures complete out of order.

Mitigation:

- parent aggregates and sorts every output dataframe with explicit keys.

### Hidden Nested Parallelism

If RF still uses `n_jobs=-1` when `jobs>1`, the run can slow down.

Mitigation:

- test the actual underlying RF `n_jobs`;
- record `model_n_jobs_effective`.

### Worker Mutation

Workers could mutate local frames.

Mitigation:

- `RoundFrameStore` methods return copies;
- store copy operations are protected by a private lock;
- worker helper must not access private store internals.

## Implementation Sequence

1. Add config and CLI `jobs` validation.
2. Add RF `n_jobs` plumbing and `_effective_model_n_jobs()`.
3. Add `RoundFrameStore` private lock and locked copy access.
4. Extract `_evaluate_target_round()` while keeping sequential behavior.
5. Add parent-side target-round classification for excluded, missing, and worker rounds.
6. Add deterministic sorting in parent output aggregation.
7. Add worker error wrapping.
8. Add thread executor path for `jobs>1`.
9. Add metadata fields.
10. Run equivalence tests.
11. Run benchmark matrix.
12. Run full quality gate.

## Acceptance Criteria

Phase 1b is accepted when:

- `jobs=1` preserves current semantic behavior, with canonical report row ordering allowed;
- `jobs=2` report outputs match `jobs=1` except allowed runtime/parallelism metadata differences;
- RF actually uses `n_jobs=1` when `jobs>1`;
- `RoundFrameStore` copy access is locked and cached frames remain immutable after `build_all()`;
- worker failures include round context and preserve original cause;
- parent owns all report writes;
- benchmark matrix is recorded;
- full quality gate passes;
- no model-search, process-pool, or feature changes are included.
