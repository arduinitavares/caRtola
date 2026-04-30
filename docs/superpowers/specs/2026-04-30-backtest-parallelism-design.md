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
- player-prediction frames;

Workers must not write files.

### Shared Store Access

The parent builds `RoundFrameStore` once before launching workers.

Each worker receives `round_number` and reads frames through the store's existing copy-returning methods:

```python
training = round_frame_store.training_frame(...)
candidates = round_frame_store.prediction_frame(round_number)
```

Workers must not access or mutate `round_frame_store._frames` directly.

The existing deep-copy behavior is sufficient for v1 because current backtest frames contain scalar values. If future feature code stores mutable objects inside cells, that feature must either avoid object-cell mutation or define its own defensive-copy contract.

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

## Runner Architecture

Extract target-round evaluation into a private helper. Suggested shape:

```python
@dataclass(frozen=True)
class RoundEvaluationResult:
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

### Sequential Path

When `config.jobs == 1`, use a plain for loop:

```python
for round_number in target_rounds:
    result = _evaluate_target_round(...)
```

No thread executor is used in this path.

### Parallel Path

When `config.jobs > 1`, use `ThreadPoolExecutor(max_workers=config.jobs)`.

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

## Target Rounds

Build target rounds from `range(config.start_round, max_round + 1)` as today.

Rules:

- excluded rounds are skipped before worker submission;
- missing non-excluded rounds are submitted only if the current sequential logic would evaluate them;
- the helper must preserve current behavior for missing non-excluded target rounds by returning `Empty` skipped rows;
- no worker is launched for excluded rounds.

The target-round list must be constructed in the parent.

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
  - `model_n_jobs_effective`;
- require all other metadata fields to match.

## Metadata

Add required metadata fields:

- `backtest_jobs`;
- `model_n_jobs_effective`;
- `parallel_backend`.

Values:

- `backtest_jobs`: integer from config;
- `model_n_jobs_effective`: result of `_effective_model_n_jobs(config.jobs)`;
- `parallel_backend`: `"sequential"` when `jobs == 1`, `"threads"` when `jobs > 1`.

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
3. CLI parses `--jobs 2` and passes `jobs=2` to `BacktestConfig`.
4. CLI default remains `jobs=1`.

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
2. Missing non-excluded target round returns `Empty` skipped rows.
3. Training-empty round returns `TrainingEmpty` skipped rows.
4. Worker helper does not write output files.
5. Worker helper uses `model_n_jobs_effective` when constructing RF.

### Parallel Equivalence

Required tests:

1. `jobs=1` and `jobs=2` produce equivalent `round_results`.
2. `jobs=1` and `jobs=2` produce equivalent `selected_players`.
3. `jobs=1` and `jobs=2` produce equivalent `player_predictions`.
4. `jobs=1` and `jobs=2` produce equivalent `summary`.
5. `jobs=1` and `jobs=2` produce equivalent `diagnostics`.
6. Metadata equality excludes only `wall_clock_seconds`, `backtest_jobs`, and `model_n_jobs_effective`; all other semantic metadata matches.
7. `jobs=2` metadata records `parallel_backend="threads"`.
8. `jobs=1` metadata records `parallel_backend="sequential"`.

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
- metadata `model_n_jobs_effective`;
- metadata `parallel_backend`;
- whether report equivalence holds versus `jobs=1`.

Temporary benchmark output roots must not be committed.

Success criterion:

- `jobs=2` or `jobs=4` improves wall-clock time over `jobs=1`; and
- if neither improves, report the result honestly and stop before expanding parallelism further.

Do not declare Phase 1b successful on metadata alone. Runtime must be benchmarked.

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
- worker helper must not access private store internals.

## Implementation Sequence

1. Add config and CLI `jobs` validation.
2. Add RF `n_jobs` plumbing and `_effective_model_n_jobs()`.
3. Extract `_evaluate_target_round()` while keeping sequential behavior.
4. Add deterministic sorting in parent output aggregation.
5. Add thread executor path for `jobs>1`.
6. Add metadata fields.
7. Add worker error wrapping.
8. Run equivalence tests.
9. Run benchmark matrix.
10. Run full quality gate.

## Acceptance Criteria

Phase 1b is accepted when:

- `jobs=1` preserves current behavior;
- `jobs=2` report outputs match `jobs=1` except allowed runtime/parallelism metadata differences;
- RF actually uses `n_jobs=1` when `jobs>1`;
- worker failures include round context and preserve original cause;
- parent owns all report writes;
- benchmark matrix is recorded;
- full quality gate passes;
- no model-search, process-pool, or feature changes are included.
