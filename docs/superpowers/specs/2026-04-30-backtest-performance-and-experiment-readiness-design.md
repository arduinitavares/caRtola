# Backtest Performance And Experiment Readiness Design

## Goal

Make Cartola walk-forward backtests fast enough to support model iteration, while preserving leakage boundaries, report comparability, and the current Cartola 2026 scoring contract.

This design covers three staged improvements:

1. Phase 1a: deterministic in-memory round-frame caching.
2. Phase 1b: controlled target-round parallelism.
3. Phase 2: a separate model experiment runner for model and hyperparameter evaluation.

Only Phase 1a should be implemented first. Phase 1b and Phase 2 are intentionally specified here so the first implementation does not block or contradict the later model-optimization roadmap.

## Current State

The current backtest is a walk-forward simulator:

1. For target round `N`, train only on rounds `< N`.
2. Predict candidate players for round `N`.
3. Optimize squad, formation, captain, and budget.
4. Write round, selected-player, prediction, summary, diagnostic, and metadata reports.

Current model behavior:

- the default model strategy is `RandomForestPointPredictor`;
- it wraps `sklearn.ensemble.RandomForestRegressor`;
- current RF parameters are fixed in code;
- there is no model registry;
- there is no grid search or hyperparameter optimization workflow.

Current feature/report dimensions include:

- `fixture_mode=none|exploratory|strict`;
- `footystats_mode=none|ppg|ppg_xg`;
- `matchup_context_mode=none|cartola_matchup_v1`;
- `scoring_contract_version=cartola_standard_2026_v1`;
- all official Cartola formations searched automatically;
- captain scoring enabled with the standard 1.5x multiplier.

## Problem

Backtests with `matchup_context_mode=cartola_matchup_v1` are too slow for practical iteration.

The clearest known bottleneck is repeated prediction-frame construction:

- `run_backtest()` loops target rounds.
- For each target round, `build_training_frame()` rebuilds prediction frames for every prior round.
- Then `run_backtest()` builds the target round's candidate prediction frame.

Example:

- target round 10 rebuilds rounds 1-9;
- target round 11 rebuilds rounds 1-10;
- target round 12 rebuilds rounds 1-11.

With expensive matchup feature joins and rolling context, this creates avoidable quadratic work. The same round frame is recomputed many times even though it should be deterministic for a fixed run configuration.

This also blocks model optimization. A hyperparameter or model experiment runner would multiply the current runtime problem.

## Design Principles

### No Legacy Switches

Do not add a public `cache_enabled` or compatibility toggle. Once equivalence is proven, round-frame caching is the engine behavior.

### Preserve Walk-Forward Boundaries

For target round `N`, features may only use information available before `N`. The cache must call the existing round-scoped feature builder:

```python
build_prediction_frame(season_df, target_round=N, ...)
```

The cache must not compute full-season rolling features globally and slice later unless a separate future design proves that logic leak-safe.

### Keep Report Semantics Stable

Phase 1a must not change:

- model features;
- model parameters;
- optimizer behavior;
- scoring contract;
- candidate pools;
- skipped-round logic;
- selected squads;
- summary metric meaning.

### Separate Engine Speed From Model Search

The normal backtest CLI answers: "run this exact backtest config."

The future experiment runner answers: "compare model configs under a controlled protocol."

Do not turn the backtest CLI into a grid-search tool.

## Phase 1a: Deterministic Round-Frame Cache

### Objective

Build each round's prediction frame once per `run_backtest()` call and reuse those frames for training and candidate prediction.

### Scope

Implement only sequential `jobs=1` behavior in Phase 1a.

Do not add parallelism in the same patch. Caching changes data flow; parallelism changes execution order, CPU pressure, and memory behavior. If both are changed together and results differ, root cause becomes unclear.

### Architecture

Add a private per-run round-frame store used by `run_backtest()`.

Suggested internal shape:

```python
class RoundFrameStore:
    def __init__(
        self,
        *,
        season_df: pd.DataFrame,
        fixtures: pd.DataFrame | None,
        footystats_rows: pd.DataFrame | None,
        matchup_context_mode: str,
    ) -> None: ...

    def build_all(self, rounds: Iterable[int]) -> None: ...

    def prediction_frame(self, round_number: int) -> pd.DataFrame: ...

    def training_frame(
        self,
        *,
        target_round: int,
        playable_statuses: tuple[str, ...],
        empty_columns: list[str],
    ) -> pd.DataFrame: ...
```

The exact type name can change during implementation, but the contract is fixed:

- one store per `run_backtest()` call;
- no persisted/on-disk cache in v1;
- cached frames are treated as immutable;
- callers receive copies before adding columns or filtering in-place;
- training frames concatenate only cached rounds `< target_round`;
- target candidates come from the cached `target_round` frame.

### Cache Scope

The cache is valid only within a single `run_backtest()` invocation.

It must never be reused across:

- seasons;
- fixture modes;
- FootyStats modes;
- matchup context modes;
- source files;
- feature-code versions;
- scoring/report runs.

For v1, this avoids cache invalidation complexity. On-disk Parquet caching is explicitly out of scope.

### Empty Training Frames

If no prior frames exist, the cached training-frame path must return an empty frame with the same effective columns the current backtest expects:

- normalized market/player columns;
- resolved model feature columns;
- FootyStats feature columns when active;
- matchup context feature columns when active;
- `target`.

### Metadata

Backtest metadata must gain these fields:

- `cache_enabled`: always `true`;
- `prediction_frames_built`: number of round frames built by the store;
- `wall_clock_seconds`: runtime of `run_backtest()` as a float;
- `backtest_jobs`: `1` in Phase 1a;
- `model_n_jobs_effective`: the RF worker setting used by the model.

If adding all metadata fields in Phase 1a creates too much report churn, `cache_enabled`, `prediction_frames_built`, and `wall_clock_seconds` are required first. The remaining worker fields become required in Phase 1b.

### Required Tests

Phase 1a must include tests for:

1. Each round prediction frame is built exactly once.
2. Cached training frame equals the current `build_training_frame()` result for multiple target rounds.
3. Cached candidate frame equals the current `build_prediction_frame()` result.
4. Future-round spike data does not affect a cached frame for an earlier round.
5. `matchup_context_mode=cartola_matchup_v1` cached frames match the uncached public helper output.
6. Phase 1a backtest reports match the pre-cache report semantics after stable sorting.
7. Metadata records cache/runtime fields.

The public `build_training_frame()` function should remain available and unchanged in Phase 1a. It remains useful as the reference implementation for equivalence tests and for non-runner callers.

### Acceptance Criteria

Phase 1a is accepted when:

- focused tests pass;
- full quality gate passes;
- the 2025 matchup-context command runs materially faster than before;
- outputs remain equivalent under stable sort and numeric tolerance for floating values;
- no implementation code exposes a public legacy no-cache mode.

## Phase 1b: Controlled Target-Round Parallelism

### Objective

Evaluate independent target rounds concurrently after Phase 1a proves deterministic cached behavior.

### Scope

Phase 1b must be a separate implementation after Phase 1a.

### Architecture

1. Parent process/thread builds the complete `RoundFrameStore`.
2. Each target round evaluation consumes cached frames and returns in-memory records/dataframes.
3. No worker writes report files directly.
4. Parent process sorts, normalizes, and writes all outputs.

Target-round parallelism is logically safe because each target round trains from cached prior rounds and produces independent outputs. Operational risks are CPU oversubscription, memory pressure, dataframe serialization overhead, and nondeterministic output ordering.

### `jobs` Contract

Add `jobs` to config and CLI in Phase 1b.

Rules:

- default: `jobs=1`;
- valid range: positive integer;
- `jobs=1` preserves sequential behavior;
- `jobs>1` may evaluate target rounds concurrently;
- output files must be sorted deterministically by round and strategy;
- errors from workers must fail the whole run with the original error type/message preserved in logs or exception context.

### RF Worker Policy

RandomForest already has internal parallelism through `n_jobs`.

To avoid oversubscription:

- when `backtest_jobs == 1`, default RF `n_jobs=-1`;
- when `backtest_jobs > 1`, default RF `n_jobs=1`;
- explicit override may be designed later, but it is out of scope for Phase 1b unless benchmark data proves it necessary.

On a 28-core machine, `jobs>1` with RF `n_jobs=-1` is forbidden because it can oversubscribe CPU and become slower than sequential execution.

### Required Tests

Phase 1b must include tests for:

1. `jobs=1` and `jobs=2` produce identical sorted outputs.
2. RF worker setting is `-1` for `jobs=1`.
3. RF worker setting is `1` for `jobs>1`.
4. Workers do not write report files directly.
5. Round-result ordering is stable after parent aggregation.
6. Metadata records `backtest_jobs` and `model_n_jobs_effective`.

### Acceptance Criteria

Phase 1b is accepted when:

- Phase 1a tests still pass;
- `jobs=1` remains deterministic;
- `jobs>1` outputs match `jobs=1`;
- benchmark results show a speedup or the metadata clearly explains why parallelism does not help for the tested config.

## Phase 2: Model Experiment Runner

### Objective

Create a controlled workflow for comparing model classes and model parameters without polluting the normal backtest CLI.

### Scope

Phase 2 must wait until Phase 1a is complete. Phase 1b is useful but not strictly required if Phase 1a already makes experiments practical.

### Non-Goals

Do not implement:

- random k-fold CV;
- broad blind grid search across all 2023-2025 results;
- live 2026 model changes before historical evaluation;
- report comparison that accepts missing scoring/model metadata.

### Model Registry

Add explicit model configuration objects.

Initial candidate set should be small:

- `random_forest_default`;
- `random_forest_shallow`;
- `random_forest_more_trees`;
- later `extra_trees`;
- later `hist_gradient_boosting`.

Each model config must serialize:

- model id;
- model class;
- exact hyperparameters;
- random seed;
- feature columns;
- model config hash.

### Experiment Runner

Add a separate script, for example:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --seasons 2023,2024,2025 \
  --fixture-mode exploratory \
  --footystats-mode ppg \
  --matchup-context-mode cartola_matchup_v1 \
  --budget 100 \
  --current-year 2026
```

The normal backtest CLI remains for a single exact config.

### Model Selection Protocol

With only 2023, 2024, and 2025, broad hyperparameter search can overfit the historical test set.

Preferred protocol:

1. Develop and tune on 2023.
2. Validate on 2024.
3. Final check on 2025.
4. Freeze the selected model before using it for 2026 live recommendation.

If an exploratory sweep evaluates all three historical seasons repeatedly, the report must label the result as exploratory, not as a final generalization claim.

### Predictive Sweep Before Full Squad Simulation

Full squad simulation includes repeated model fitting plus MILP optimization. Running that for every candidate in a large grid is expensive and can waste time.

For larger search spaces:

1. Run a walk-forward predictive sweep using player-level metrics such as MAE, RMSE, R², and correlation.
2. Select a small top-K set.
3. Run full squad/MILP simulation only for top-K candidates.

For a small initial registry, full backtests for every model are acceptable.

### Comparison Requirements

Experiment comparison must reject runs with mismatched:

- seasons;
- start round;
- budget;
- playable statuses;
- fixture mode;
- FootyStats mode;
- matchup context mode;
- scoring contract;
- formation search;
- captain scoring fields;
- optimizer contract;
- candidate pools;
- skipped rounds;
- source hashes/manifests where available.

Missing `run_metadata.json` or missing scoring/model contract fields is an error.

### Report Fields

Experiment reports must include:

- model id;
- model config hash;
- exact model parameters;
- feature modes;
- scoring contract fields;
- candidate counts by season/round;
- skipped rounds;
- solver statuses;
- player-level predictive metrics;
- squad-point metrics;
- runtime;
- `backtest_jobs`;
- `model_n_jobs_effective`;
- cache metadata;
- source paths/hashes/manifests where available.

### Decision Fields

The report should compute:

- aggregate RF squad-points delta;
- per-season squad-points delta;
- worst-season delta;
- number of improved seasons;
- `passes_2_of_3`;
- player R² delta;
- player correlation delta;
- runtime delta.

Recommended acceptance bar for a production candidate:

- at least 2 of 3 seasons improve;
- aggregate squad points improve by a practical margin;
- no season regresses badly;
- player R² and correlation do not degrade materially;
- candidate pools and skipped rounds match.

Exact numeric thresholds should be set in the Phase 2 plan after current baseline metrics are reviewed.

## Rejected Alternatives

### Combine Caching And Parallelism In One Patch

Rejected because data-flow changes and execution-order changes create ambiguous failures.

### Add Public No-Cache Compatibility Mode

Rejected because the project intentionally avoids legacy behavior paths. The correct guard is equivalence tests, not a permanent no-cache mode.

### Persist Round Frames To Disk In Phase 1a

Rejected for v1 because on-disk caching requires invalidation rules for feature code, source hashes, fixture manifests, FootyStats files, and config dimensions. In-memory per-run caching solves the immediate bottleneck with much lower risk.

### Add Grid Search To The Normal Backtest CLI

Rejected because the normal CLI should run one exact configuration. Model comparison needs stricter metadata, comparison gates, and reporting than a single-run command.

### Random K-Fold Cross Validation

Rejected because Cartola round data is temporal. Random splits can leak future information into training and overstate model quality.

## Risks

### Cache Mutation

If downstream code mutates cached frames, later target rounds can see contaminated data.

Mitigation: store frames as immutable by convention and return copies before adding columns such as predictions or `target`.

### Hidden Leakage Through Global Precomputation

If future work rewrites feature generation to compute rolling values globally, it can accidentally include target/future rows.

Mitigation: Phase 1a must call the existing target-round-scoped `build_prediction_frame()` once per round.

### CPU Oversubscription

Outer backtest parallelism plus RF `n_jobs=-1` can spawn too much work.

Mitigation: Phase 1b requires RF `n_jobs=1` when `backtest_jobs>1`.

### Selection Bias In Model Experiments

Repeatedly tuning on all historical seasons can make 2023-2025 become the training set for human decision-making.

Mitigation: use temporal model-selection protocol and label exploratory sweeps honestly.

### Memory Pressure

Caching all round frames and running parallel model fits can increase memory use.

Mitigation: Phase 1a is sequential and in-memory only; Phase 1b must benchmark memory/runtime before becoming the default.

## Implementation Sequence

1. Implement Phase 1a only.
2. Benchmark and verify output equivalence.
3. Review whether runtime is acceptable.
4. If still too slow, write a Phase 1b implementation plan for parallel target-round evaluation.
5. After backtests are fast enough, write a Phase 2 implementation plan for model experiments.

## Success Criteria For This Spec

This spec is successful if it prevents three mistakes:

1. shipping caching and parallelism together;
2. starting grid search before single backtests are fast;
3. producing faster reports that are not comparable or leak-safe.
