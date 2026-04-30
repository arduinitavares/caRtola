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

### Squad Metrics

Squad metrics use the current optimizer and the current scoring contract.

Required metrics:

- total actual points;
- average actual points per optimized round;
- total predicted active-contract points;
- average predicted active-contract points;
- actual delta vs current recommended baseline;
- average actual delta per round;
- per-season actual total and average;
- worst-season delta vs baseline;
- oracle gap and capture rate where existing report data supports it;
- captain contribution;
- formation distribution;
- skipped rounds and solver statuses.

The primary promotion metric is optimized squad performance. Selected-player and top-K calibration are mandatory guardrails.

### Promotion Rule

A model/feature pack is eligible for recommendation only if all are true:

- aggregate actual points improve over the current recommended baseline;
- at least two of three seasons improve;
- worst-season regression is no worse than `-1.5` average points per evaluated round;
- selected-player calibration slope stays within `[0.75, 1.25]`;
- top-50 Spearman correlation does not degrade by more than `0.03`;
- candidate pools match the comparison contract;
- skipped rounds and solver-status distributions match the comparison contract;
- scoring contract version is exactly `cartola_standard_2026_v1`.

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
- solver statuses by season, round, and strategy;
- source data hashes, using `null` only when the source is not file-backed.

Expected feature-column differences are allowed. Candidate-pool differences are not allowed unless a future spec explicitly defines why a feature pack may change eligibility.

The runner must raise a `ComparabilityError` before ranking results when the contract fails.

## Metadata

Each child backtest run must keep its normal `run_metadata.json`.

The experiment runner must also write top-level metadata containing:

- `experiment_id`;
- `experiment_started_at_utc`;
- git commit hash;
- project root;
- command arguments;
- frozen matrix of model ids and feature-pack ids;
- model parameters;
- feature modes;
- seasons;
- start round;
- budget;
- fixture mode;
- source paths and hashes, using `null` only when the data source is not file-backed;
- scoring contract version;
- optimizer contract identifier, using `cartola_standard_2026_v1` when no separate optimizer contract field exists;
- random seed;
- backtest jobs;
- model worker setting;
- runtime per child run;
- candidate pool signatures;
- skipped-round signatures;
- solver-status signatures;
- comparison group id.

The config hash must include every field that can change predictions, eligibility, optimization, or scoring.

Missing metadata required for comparability is an error.

## Outputs

The runner writes under:

```text
data/08_reporting/experiments/model_feature/<experiment_id>/
```

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

The ranked summary must rank only comparable runs. Failed or incomparable runs appear in the comparability report, not in the winner table.

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
- `2026` is rejected as an experiment season;
- production-parity group uses `fixture_mode=none`;
- matchup-research group uses `fixture_mode=exploratory`;
- fixture-mode mixing inside one comparison group is rejected;
- config hash changes when any model parameter, feature pack, fixture mode, budget, start round, scoring contract, or source hash changes;
- candidate-pool mismatch raises `ComparabilityError`;
- skipped-round mismatch raises `ComparabilityError`;
- solver-status mismatch raises `ComparabilityError`;
- missing scoring contract metadata raises `ComparabilityError`;
- prediction metrics do not consume price objective totals as predictions;
- calibration slope/intercept handle empty and constant inputs as `null` plus warnings;
- ranked summary excludes incomparable runs;
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
