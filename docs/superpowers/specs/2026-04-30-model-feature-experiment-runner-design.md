# Model Feature Experiment Runner Design

## Goal

Build a controlled experiment runner that compares model and feature-pack combinations for Cartola backtests without changing live defaults, optimizer behavior, scoring semantics, or fixed-budget assumptions.

The immediate decision this supports is:

- whether `ppg_xg` is actually weak, or only weak under the current Random Forest model;
- whether `cartola_matchup_v1` improves fixed-budget squad selection enough to justify future strict fixture integration;
- whether another sklearn model is a better default than the current Random Forest.

This design intentionally comes before patrimonio growth. Dynamic budget simulation changes the business problem. First prove which fixed-budget model and feature pack produces better recommendations under the current Cartola 2026 scoring contract.

## Current State

Backtesting now supports:

- walk-forward target-round evaluation;
- deterministic in-memory round-frame caching;
- target-round parallelism through `--jobs`;
- Rich terminal summaries;
- Plotly HTML performance charts;
- `fixture_mode=none|exploratory|strict`;
- `footystats_mode=none|ppg|ppg_xg`;
- `matchup_context_mode=none|cartola_matchup_v1`;
- `scoring_contract_version=cartola_standard_2026_v1`;
- all official Cartola formations searched automatically;
- one non-tecnico captain with the standard `1.5x` multiplier;
- coach selection as part of the optimized 12 selected entries;
- fixed budget, normally `100`.

Current recommended live/replay feature mode is FootyStats PPG without matchup context.

Recent results show that Random Forest can outperform price and baseline in actual squad points, but prediction/objective reporting is not enough to answer feature-quality questions. A feature can look weak under one model and useful under another. Conversely, a model can improve squad points while worsening whole-pool calibration.

## Problem

Manual backtests are not enough to decide model and feature direction.

The project needs a repeatable experiment harness that:

- runs a predeclared model by feature matrix;
- enforces identical comparison contracts;
- records model, feature, fixture, scoring, and source metadata;
- reports prediction quality and optimized squad quality separately;
- fails closed when runs are not comparable;
- prevents broad, informal hyperparameter searching over only three historical seasons.

Without this, the project risks promoting a noisy feature or rejecting a useful one based on one model's behavior.

## Non-Goals

Do not add:

- patrimonio growth;
- live default changes;
- strict fixture integration into live recommendations;
- new scoring contracts;
- legacy raw-scoring compatibility modes;
- public fixed-formation modes;
- external model dependencies such as XGBoost, LightGBM, or CatBoost;
- broad hyperparameter grid search;
- random k-fold cross-validation;
- model search inside the normal backtest CLI.

The normal backtest CLI remains the command for "run this exact config."

The experiment runner answers "compare these predeclared configs under one controlled protocol."

## Experiment Matrix

### Models

The v1 runner supports exactly these model ids:

- `random_forest`: current incumbent model, current fixed parameters.
- `extra_trees`: nearby tree-ensemble comparison.
- `hist_gradient_boosting`: sklearn boosting model for nonlinear tabular signal.
- `ridge`: linear calibration baseline.

All model parameters are fixed in v1. No parameter grid is exposed.

The Ridge model must be implemented as a deterministic sklearn pipeline that can handle missing numeric values and feature scaling. It is included as a calibration sanity check, not because it is expected to win optimized squad points.

### Fixed Model Parameters

The v1 model registry is fixed. These parameters are part of the experiment contract and must be written to metadata.

`random_forest`:

- preprocessing: numeric median imputation; `posicao` most-frequent imputation plus one-hot encoding;
- estimator: `RandomForestRegressor`;
- `n_estimators=200`;
- `min_samples_leaf=3`;
- `random_state=config.random_seed`;
- `n_jobs=model_n_jobs_effective`.

`extra_trees`:

- preprocessing: same as `random_forest`;
- estimator: `ExtraTreesRegressor`;
- `n_estimators=200`;
- `min_samples_leaf=3`;
- `random_state=config.random_seed`;
- `n_jobs=model_n_jobs_effective`.

`hist_gradient_boosting`:

- preprocessing: numeric median imputation; `posicao` most-frequent imputation plus dense one-hot encoding with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`;
- estimator: `HistGradientBoostingRegressor`;
- `max_iter=200`;
- `learning_rate=0.05`;
- `min_samples_leaf=20`;
- `l2_regularization=0.0`;
- `random_state=config.random_seed`.

`ridge`:

- preprocessing: numeric median imputation plus standard scaling; `posicao` most-frequent imputation plus one-hot encoding with `OneHotEncoder(handle_unknown="ignore")`;
- estimator: `Ridge`;
- `alpha=1.0`;
- no model-level parallelism.

Changing any listed parameter requires a new spec revision or a later explicitly scoped hyperparameter experiment design.

### Model Integration Boundary

The normal backtest CLI must not expose model selection in v1.

Add a private predictor factory below the backtest layer:

```python
create_point_predictor(
    *,
    model_id: str,
    random_seed: int,
    feature_columns: list[str],
    n_jobs: int,
) -> PointPredictor
```

The predictor interface is:

```python
class PointPredictor(Protocol):
    feature_columns: list[str]

    def fit(self, frame: pd.DataFrame) -> Self: ...

    def predict(self, frame: pd.DataFrame) -> pd.Series: ...
```

The current public `run_backtest(config)` behavior remains equivalent to running with `primary_model_id="random_forest"`.

The experiment runner uses a private/internal backtest entry point or override to set one `primary_model_id` per child run. Do not add `--model-id` to `python -m cartola.backtesting.cli` in v1.

The implementation target is a private wrapper:

```python
run_backtest_for_experiment(
    config: BacktestConfig,
    *,
    primary_model_id: str,
) -> BacktestResult
```

This wrapper may call shared internal runner helpers, but public `run_backtest(config)` remains the normal CLI path and hardcodes `primary_model_id="random_forest"`.

Each child run has exactly three strategy rows:

- `baseline`;
- `<primary_model_id>`;
- `price`.

For example, an Extra Trees child run writes `strategy=extra_trees`, not `strategy=random_forest`.

`player_predictions.csv` must contain:

- `baseline_score`;
- `<primary_model_id>_score`;
- `price_score`.

The current normal CLI continues to write `random_forest_score` because its primary model id remains `random_forest`.

Experiment reports must not infer model identity from output directory names. They must read model identity from metadata and strategy rows.

### Feature Packs

The v1 feature-pack ids are:

- `ppg`: `footystats_mode=ppg`, `matchup_context_mode=none`.
- `ppg_xg`: `footystats_mode=ppg_xg`, `matchup_context_mode=none`.
- `ppg_matchup`: `footystats_mode=ppg`, `matchup_context_mode=cartola_matchup_v1`.
- `ppg_xg_matchup`: `footystats_mode=ppg_xg`, `matchup_context_mode=cartola_matchup_v1`.

Do not overload `footystats_mode` to mean matchup context. Matchup remains controlled by `matchup_context_mode`.

### Seasons

The v1 runner evaluates:

- `2023`;
- `2024`;
- `2025`.

The runner must reject `2026` for model/feature evaluation. The live season is not a tuning dataset.

### Fixture Groups

The runner has two comparison groups.

Each group has its own baseline:

- production-parity baseline: `random_forest` + `ppg` + `fixture_mode=none`;
- matchup-research baseline: `random_forest` + `ppg` + `fixture_mode=exploratory`.

The matchup-research baseline is a research control, not the current live default.

#### Production-Parity Group

Purpose: compare non-fixture feature packs under the current live-compatible no-fixture path.

Required fixture mode:

```text
fixture_mode=none
```

Allowed feature packs:

- `ppg`;
- `ppg_xg`.

This group answers whether xG helps the current no-fixture production path.

#### Matchup Research Group

Purpose: compare matchup context while keeping every arm on the same fixture contract.

Required fixture mode:

```text
fixture_mode=exploratory
```

Allowed feature packs:

- `ppg`;
- `ppg_xg`;
- `ppg_matchup`;
- `ppg_xg_matchup`.

Historical `exploratory` fixtures are reconstructed fixture evidence. They are acceptable for research comparisons only when all arms use the same fixture mode and the fixture coverage audit passes.

Matchup research results must not be described as strict live proof. To promote matchup context into live recommendations later, a separate strict fixture integration design is required.

The runner must reject any comparison group that mixes fixture modes.

## Evaluation Protocol

### Walk-Forward Boundary

For target round `N`, training uses only rows from rounds `< N`.

Feature builders must preserve the existing round-scoped contract:

```python
build_prediction_frame(..., target_round=N, ...)
```

The experiment runner must not introduce global full-season feature computation and slicing.

### Holdout Discipline

The v1 matrix is predeclared. Results across 2023-2025 are robustness evidence, not an invitation to iterate until one config wins.

Before each full experiment run, the matrix must be frozen into metadata. Any later change to model ids, feature packs, parameters, fixture mode, start round, budget, playable filters, or scoring contract creates a new experiment id and a new comparison.

Because 2025 has already influenced project thinking, it is not a pristine untouched holdout. However, the runner must still report 2025 separately and must not hide or average away a 2025 regression.

Future feature ideas must use 2023-2024 for exploration and treat 2025 as the confirmation season before live promotion.

### Prediction Metrics

Prediction metrics use raw per-athlete actual Cartola points, not captain-adjusted squad totals.

Do not use price strategy objective values as predicted Cartola points.

For each model and feature pack, compute prediction metrics for:

- whole candidate pool;
- selected players;
- top-K candidates by predicted score, with `K=25` and `K=50`.

Required metrics:

- MAE;
- RMSE;
- R2;
- Pearson correlation;
- Spearman correlation;
- calibration intercept from `actual ~ predicted`;
- calibration slope from `actual ~ predicted`;
- predicted-vs-actual by prediction decile;
- residuals by position.

If any metric is undefined because the input is empty or constant, record `null` plus a warning. Do not coerce undefined metrics to zero.

Prediction metrics are computed from playable candidate rows with valid raw actual points.

Top-K metrics are computed per season and target round, then aggregated. For each evaluated round, sort playable candidates by the primary model's predicted score and take `K=25` and `K=50`. If fewer than `K` candidates exist, use all available candidates and record the observed count.

Selected-player metrics include tecnico rows because tecnico is part of the optimized selected squad.

Calibration deciles are computed per season, model id, and feature-pack id over candidate rows. The aggregate report may also include all-season deciles, but it cannot replace the per-season decile output.

### Squad Metrics

Squad metrics use the current optimizer and the current scoring contract.

Required metrics:

- total actual points;
- average actual points per optimized round;
- total predicted active-contract points;
- average predicted active-contract points;
- actual delta vs group baseline;
- average actual delta per round;
- per-season actual total and average;
- worst-season delta vs baseline;
- oracle gap and capture rate fields;
- captain contribution;
- formation distribution;
- skipped rounds and solver statuses.

The primary promotion metric is optimized squad performance. Selected-player and top-K calibration are mandatory guardrails.

Current backtest round results do not produce oracle actual points or oracle capture rate. Until those columns exist, experiment reports must include nullable oracle fields with `null` values and reason `not_produced_by_backtest`. Do not silently omit the fields.

### Promotion Rule

A model/feature pack is eligible for recommendation only if all are true:

- aggregate actual points improve over the group baseline;
- at least two of three seasons improve;
- worst-season regression is no worse than `-1.5` average points per evaluated round;
- selected-player calibration slope stays within `[0.75, 1.25]`;
- top-50 Spearman correlation does not degrade by more than `0.03`;
- candidate pools match the comparison contract;
- skipped rounds and solver-status distributions match the comparison contract;
- scoring contract version is exactly `cartola_standard_2026_v1`.

If a promotion guardrail metric is `null`, the config is not eligible and must be marked `insufficient_metric_data`. Nullable oracle fields with reason `not_produced_by_backtest` are informational and are not promotion guardrails in v1.

These thresholds are v1 decision gates. If they prove too strict or too loose, revise the spec before changing code.

## Comparability Contract

Each comparison group must fail closed if any run differs unexpectedly in:

- seasons;
- start round;
- budget;
- fixture mode;
- FootyStats source identity;
- matchup context mode rules;
- scoring contract version;
- optimizer contract;
- playable/status filters;
- candidate counts by season and round;
- skipped target rounds;
- solver statuses by season, round, and strategy role;
- source data hashes, using `null` only when the source is not file-backed.

Expected feature-column differences are allowed. Candidate-pool differences are not allowed unless a future spec explicitly defines why a feature pack may change eligibility.

The runner must raise a `ComparabilityError` before ranking results when the contract fails.

Comparability mismatch is an operational failure. The command exits non-zero after writing `comparability_report.json` when the output directory is available.

### Signature Schemas

Candidate-pool signatures are independent of strategy and model id.

For each season and target round, construct a canonical candidate-pool signature from the scored candidate frame before strategy-specific optimization. It must include one sorted record per candidate:

- `id_atleta`;
- `posicao`;
- `id_clube`;
- `status`;
- `preco_pre_rodada`;
- `rodada`.

Records are sorted by `id_atleta`, encoded as canonical JSON with sorted keys, and hashed with SHA-256. Floating prices use the same normalized CSV precision as backtest reports.

Skipped-round signatures include:

- season;
- target round;
- skip status or reason;
- whether the round was excluded by fixture alignment;
- whether training data was empty.

Solver-status signatures are role based:

- `baseline` role maps only from literal `strategy=baseline`;
- `price` role maps only from literal `strategy=price`;
- `primary_model` role maps from the child run's primary model strategy, such as `random_forest`, `extra_trees`, `hist_gradient_boosting`, or `ridge`.

Common reference roles `baseline` and `price` must match literally across compared runs. Primary model statuses are compared by `primary_model` role, not by literal strategy name.

Any unexpected strategy row outside `baseline`, `price`, and the declared primary model id is a comparability error.

## Metadata

Each child backtest run must keep its normal `run_metadata.json`.

The experiment runner must also write top-level metadata containing:

- `experiment_id`;
- `experiment_started_at_utc`;
- git commit hash;
- project root;
- command arguments;
- frozen matrix of model ids and feature-pack ids;
- primary model id per child run;
- strategy role mapping per child run;
- model parameters;
- feature modes;
- seasons;
- start round;
- budget;
- fixture mode;
- source paths and hashes, using the source hash rules below;
- scoring contract version;
- optimizer contract identifier, using `cartola_standard_2026_v1` when no separate optimizer contract field exists;
- random seed;
- backtest jobs;
- model worker setting per child run;
- runtime per child run;
- candidate pool signatures;
- skipped-round signatures;
- solver-status signatures;
- comparison group id.

The config hash must include every field that can change predictions, eligibility, optimization, or scoring.

Missing metadata required for comparability is an error.

### Source Hash Rules

Top-level experiment metadata must record these source identities:

- raw Cartola season inputs: SHA-256 over the sorted list of files under `data/01_raw/<season>/`, excluding generated `*.capture.json` live-market metadata unless a child run explicitly consumes it;
- FootyStats source: path plus SHA-256 of the loaded FootyStats file when `footystats_mode != none`, otherwise `null`;
- fixture source: fixture mode plus path and SHA-256 for exploratory fixture CSVs or strict fixture manifests, otherwise `null`;
- matchup source: `matchup_context_mode` plus fixture source identity when matchup is enabled, otherwise `null`.

When a source is made of multiple files, hash each file as bytes, sort by project-root-relative path, and hash canonical JSON records containing `path`, `sha256`, and `size_bytes`.

### Model Worker Metadata

`model_n_jobs_effective` is recorded per child run.

- `random_forest`: the integer passed to `RandomForestRegressor(n_jobs=...)`;
- `extra_trees`: the integer passed to `ExtraTreesRegressor(n_jobs=...)`;
- `hist_gradient_boosting`: `null`;
- `ridge`: `null`.

The absence of model-level parallelism for HGB and Ridge is represented by `null`, not `1`.

## Execution Model

V1 uses sequential child backtest execution.

The experiment runner loops through model, feature-pack, and season child runs one at a time. It does not run child backtests concurrently.

The `--jobs` flag belongs to each child backtest and is passed through to `run_backtest()` target-round parallelism. It is not experiment-level parallelism.

Do not add `--experiment-jobs` in v1. Nested parallelism would combine experiment-level workers, child backtest target-round workers, and model-level workers, which can oversubscribe CPU and memory.

If experiment-level parallelism is needed later, write a separate design with explicit worker allocation and benchmark rules.

Child backtest failure behavior:

- any child backtest operational failure aborts the whole experiment before ranking;
- if the output directory has already been resolved and created, the runner writes `experiment_metadata.json` and `comparability_report.json` with `status="failed"` and the failed child run id;
- if failure happens before the output directory is resolved, the runner prints the error to stderr and writes no partial reports;
- `ranked_summary.csv` is not written after a child backtest failure;
- completed but incomparable child runs are recorded in `comparability_report.json` and are excluded from ranking.

## Outputs

The runner writes under:

```text
data/08_reporting/experiments/model_feature/<experiment_id>/
```

Default `experiment_id` format:

```text
group=<group>__started_at=<YYYYMMDDTHHMMSSffffffZ>__matrix=<12-char-config-hash>
```

If the target output directory already exists, the runner fails. V1 does not include `--force`.

Child run directories are deterministic under the top-level experiment path:

```text
runs/season=<season>/model=<model_id>/feature_pack=<feature_pack>/
```

Each child run sets `BacktestConfig.output_root` so the child backtest writes into that exact directory. The child season is not appended a second time below this path.

Required files:

- `experiment_metadata.json`;
- `ranked_summary.csv`;
- `per_season_summary.csv`;
- `prediction_metrics.csv`;
- `calibration_deciles.csv`;
- `comparability_report.json`;
- `comparison_report.md`;

Required HTML reports:

- `calibration_plots.html`;
- `squad_performance_comparison.html`;

The ranked summary must rank only comparable runs. Completed but incomparable runs appear in the comparability report, not in the winner table. Failed child runs abort the experiment before ranking.

HTML report generation failures are fatal for the experiment runner because chart outputs are part of the experiment artifact, not auxiliary terminal display.

## CLI

Add a dedicated script:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group production-parity \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12
```

For matchup research:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group matchup-research \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12
```

The script chooses the allowed fixture mode and feature packs from the group. It must not accept arbitrary fixture-mode mixing inside one comparison group.

`--jobs` is passed to each child backtest. It does not control experiment-level concurrency.

## Error Handling

Expected operational failures must print a concise error and return non-zero:

- unknown experiment group;
- unknown model id;
- unknown feature-pack id;
- invalid season list;
- season includes current/live year;
- fixture coverage audit failure for matchup research;
- child backtest failure;
- missing required metadata;
- comparability mismatch;
- output directory collision.

Implementation bugs should not be swallowed as clean operational failures.

## Tests

Required tests before implementation is considered complete:

- matrix generation includes exactly the allowed model and feature-pack ids;
- model registry lists exact fixed parameters for all v1 model ids;
- unknown model id is rejected by the predictor factory;
- normal backtest CLI still emits `strategy=random_forest` and does not expose `--model-id`;
- experiment child run for `extra_trees` emits `strategy=extra_trees`, not `strategy=random_forest`;
- each child run emits exactly `baseline`, `<primary_model_id>`, and `price` strategies;
- `2026` is rejected as an experiment season;
- production-parity group uses `fixture_mode=none`;
- matchup-research group uses `fixture_mode=exploratory`;
- fixture-mode mixing inside one comparison group is rejected;
- config hash changes when any model parameter, feature pack, fixture mode, budget, start round, scoring contract, or source hash changes;
- candidate-pool signature is independent of model id and strategy rows;
- candidate-pool signature changes when candidate athlete ids, statuses, positions, clubs, prices, or rounds change;
- candidate-pool mismatch raises `ComparabilityError`;
- skipped-round mismatch raises `ComparabilityError`;
- solver-status mismatch raises `ComparabilityError` by strategy role;
- primary model solver statuses compare by `primary_model` role across different model ids;
- unexpected strategy rows raise `ComparabilityError`;
- missing scoring contract metadata raises `ComparabilityError`;
- null promotion guardrail metric marks a config `insufficient_metric_data` and ineligible;
- raw Cartola source hash changes when any consumed raw season file content changes;
- HGB preprocessing uses dense one-hot output;
- `model_n_jobs_effective` is `null` for HGB and Ridge;
- baseline and price outputs are identical across model ids within the same season and feature pack, after excluding runtime/path metadata;
- top-K metrics are computed per season and target round before aggregation;
- selected-player metrics include tecnico rows;
- prediction metrics do not consume price objective totals as predictions;
- calibration slope/intercept handle empty and constant inputs as `null` plus warnings;
- ranked summary excludes incomparable runs;
- child backtest failure aborts before `ranked_summary.csv` is written;
- `--jobs` is passed to child backtests and does not launch experiment-level workers;
- promotion rule fails on aggregate-only wins with two losing seasons;
- promotion rule fails on worst-season regression beyond the v1 threshold;
- matchup feature experiment rejects `fixture_mode=none`;
- rerunning the same frozen matrix with the same seed produces equivalent aggregate reports after excluding runtime fields.

## Acceptance Criteria

The feature is complete when:

- the dedicated experiment runner can run the production-parity group for 2023-2025;
- the dedicated experiment runner can run the matchup-research group for 2023-2025 when fixture coverage passes;
- all required metadata, CSV outputs, Markdown outputs, and HTML outputs are written;
- non-comparable runs fail before ranking;
- prediction and squad metrics are separated in reports;
- the group baseline is identifiable in every comparison group;
- no live recommendation default changes are made.

## Next Decision After This

After the runner produces the first trusted experiment reports, decide one of:

1. Keep current baseline: Random Forest + PPG, no matchup.
2. Promote a new no-fixture feature/model combination.
3. Continue matchup research and write a strict fixture integration spec.
4. Reject matchup context for now and focus on calibration/model diagnostics.
5. Only after one fixed-budget direction is proven, design patrimonio growth.
