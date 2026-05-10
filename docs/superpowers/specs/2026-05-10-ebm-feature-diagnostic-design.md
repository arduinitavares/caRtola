# EBM Feature Diagnostic Design

## Summary

Build an artifact-backed diagnostic runner that uses InterpretML's
Explainable Boosting Machine (EBM) to inspect whether the current Cartola
feature set contains stable, football-sensible nonlinear shapes or pairwise
interactions that our trusted Ridge/XGBoost models may not be using well.

This is not a broad AutoML platform. It is not FLAML, AutoGluon, TPOT,
Featuretools, tsfresh, Optuna, or symbolic regression. External review
converged on the same risk: the effective validation sample is closer to
season/round decisions than player-row count, so broad automated search would
mostly amplify multiple-comparison risk.

The EBM runner is a discovery-only tool. It may nominate feature-shape or
interaction hypotheses. It may not update live defaults, mark a model
promotion-eligible, write experiment-index promotion fields, or bypass the
existing sequential moving-budget backtest.

## Motivation

Manual hypothesis generation has been useful but slow. H001-H004 showed that
reasonable football ideas can fail once they are tested through moving-budget,
captain-aware backtests:

- H001 opponent-overlap policy: rejected.
- H002 goalkeeper-conflict policy: rejected.
- H003 clean-sheet defensive-stack policy: rejected.
- H004 attack-vs-defense feature pack: rejected after improving only some
  seasons and badly regressing 2025.

The next question is not "can an overnight AutoML search find a winner?" The
better question is:

```text
Do existing pre-round features show stable nonlinear shapes or low-order
interactions that are interpretable enough to become the next frozen hypothesis?
```

EBM is a good first diagnostic because it is a glassbox generalized additive
model with optional pairwise interactions. It can show contribution curves for
individual features and selected interactions, which maps directly to our
need: discover candidate football rules without pretending they are already
validated production policies.

## External Review Decision

Three outside assessments agreed on the main correction:

- Do not build the proposed three-lane AutoML discovery platform now.
- Do not let FLAML or AutoGluon own validation.
- Do not use random row-level cross-validation.
- Do not optimize final MILP squad points.
- Do not run feature factories such as tsfresh/Featuretools before proving
  there is stable unexploited signal.
- Use EBM first as a constrained diagnostic for feature shapes and
  interactions.

This spec accepts that correction. FLAML remains a possible later bounded
candidate generator, but only after the EBM diagnostic and only inside our
temporal validation harness.

## Goals

- Read completed experiment artifacts without rerunning backtests.
- Validate source context before loading player rows.
- Build a cutoff-safe diagnostic dataset from persisted `player_predictions.csv`
  artifacts.
- Fit one or more EBMs on pre-round features and raw realized player points.
- Produce feature importance, shape, interaction, and stability artifacts.
- Compare EBM predictive diagnostics against the source model's persisted
  predictions.
- Make it easy to identify 1-2 stable interactions that could become H005/H006.
- Label every output `discovery_only=true`.

## Non-Goals

- No direct live default changes.
- No promotion decisions.
- No model experiment index updates.
- No MILP squad-point optimization inside the EBM search.
- No broad hyperparameter search.
- No AutoGluon, FLAML, H2O, TPOT, Featuretools, tsfresh, Prophet, PySR, or
  imodels in V1.
- No automated rule-to-policy conversion.
- No free-form generated hypotheses without deterministic support/stability
  fields.

## Source Experiment

The initial source is the current best completed matchup experiment
artifact, for example:

```bash
data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260505T211914592073Z__matrix=d95948374d75
```

The runner must also support a targeted source experiment such as:

```bash
data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d
```

The expected primary starting context is:

```text
model_id=xgboost_depth2_l2_heavy
feature_pack=ppg_xg_matchup
fixture_mode=exploratory
budget_policy=moving
seasons=2021,2022,2023,2024,2025
```

If the user chooses `xgboost_depth2_slow` instead, the same runner must work
as long as source context validates.

## Source Artifact Contract

The runner must read persisted artifacts only:

- parent `experiment_metadata.json`;
- parent `ranked_summary.csv`;
- child `run_metadata.json`;
- child `player_predictions.csv`;
- child `round_results.csv` for source-run sanity checks;
- optionally child `selected_players.csv` for selected-player diagnostics.

The runner must not reconstruct candidate frames from current code unless a
future explicit `source_mode=reconstructed` is added. V1 supports only
artifact-backed mode.

### Required Context

For each analyzed child run, the runner must validate:

- `season`;
- `model_id`;
- `feature_pack`;
- `fixture_mode`;
- `matchup_context_mode`;
- `footystats_mode`;
- `budget_policy`;
- `scoring_contract_version`;
- primary score column, exactly `"{model_id}_score"`;
- source artifact paths;
- source row counts.

If `season` is absent from `player_predictions.csv`, derive it from validated
child context. Do not require the CSV itself to carry `season`.

### Required Player Prediction Columns

`player_predictions.csv` must contain:

- identity:
  - `rodada`;
  - `id_atleta`;
  - `apelido`;
  - `id_clube`;
  - `posicao`;
- eligibility:
  - `status`;
- objective and outcome:
  - primary score column, for example `xgboost_depth2_l2_heavy_score`;
  - `pontuacao`;
  - `entrou_em_campo`;
- price:
  - `preco_pre_rodada`;
- every model feature listed in source `run_metadata.json.feature_columns`.

If any required column is missing, write an invalid report with the missing
column names and do not fit an EBM.

## Outcome Policy

The diagnostic target is raw realized player points, not captain-adjusted squad
points:

```text
target_actual_points = pontuacao after DNP/null policy
```

DNP/null policy:

- if `pontuacao` is finite, use it;
- if `pontuacao` is null and `entrou_em_campo == false`, map to `0.0`;
- if `pontuacao` is null and `entrou_em_campo` is missing, null, or true, mark
  the row invalid;
- invalid rows are written to `invalid_ebm_rows.csv`;
- if invalid rows exceed `0.5%` of loaded diagnostic rows for a child, mark the
  child invalid.

Coaches (`posicao == "tec"`) are excluded from V1 EBM fitting. Coach scoring is
structurally different from player scoring and would distort player feature
shape diagnostics. The runner must report excluded coach row counts.

## Validation Protocol

No random row-level CV is allowed.

V1 uses season-expanding folds:

```text
Fold A: train 2021,2022 -> validate 2023
Fold B: train 2021,2022,2023 -> validate 2024
Fold C: train 2021,2022,2023,2024 -> validate 2025
```

The fold assignment is by whole season. All player rows from a validation
season stay in the same validation fold. This deliberately treats season as the
primary evidence unit and avoids player-row shuffling.

The runner may compute additional within-season descriptive metrics by round,
but these are diagnostics only. They are not extra validation folds.

### Holdout Semantics

The 2025 fold is a diagnostic holdout within this runner. Once viewed, it is no
longer a pristine production holdout. Any future feature inspired by this
diagnostic must be frozen as a new hypothesis and rerun through the existing
full moving-budget experiment workflow.

## EBM Fitting

Use `interpret.glassbox.ExplainableBoostingRegressor`.

Required default fitting parameters:

```text
interactions=0
outer_bags=8
inner_bags=0
max_rounds=20000
early_stopping_rounds=100
validation_size=0.15
random_state=123
n_jobs=-1
objective="rmse"
```

Main-effect EBMs are the default. Interaction EBMs are a second pass only after
main effects fit successfully.

Interaction pass parameters:

```text
interactions=10
max_interaction_bins=64
outer_bags=8
inner_bags=0
random_state=123
n_jobs=-1
```

Do not tune EBM hyperparameters in V1. Changing these constants creates a new
diagnostic generation.

## Feature Set

Use the source model's persisted feature columns from `run_metadata.json`.

Exclude these columns even if present:

- target/outcome columns;
- captain columns;
- actual scout result columns for the target round;
- IDs that create memorization risk:
  - `id_atleta`;
  - raw `apelido`;
  - raw club/opponent names;
- direct round labels that can create season/round memorization:
  - `season`;
  - `rodada`.

Allow numeric encoded fields already used by current models, such as:

- `preco_pre_rodada`;
- `id_clube` only if the source model already used it;
- `posicao`;
- prior rolling point/scout features;
- FootyStats PPG/xG fields;
- matchup context fields.

The manifest must record the final EBM feature list and excluded columns.

## Metrics

Compute metrics for both source model predictions and EBM predictions.

Prediction metrics:

- MAE;
- RMSE;
- Spearman correlation by season;
- top-50 Spearman by season and round where enough rows exist;
- calibration slope by season;
- mean prediction bias by season.

Ranking metrics:

- actual top-5 position scorers' mean predicted rank;
- top-10 actual scorer recall inside top-20 predicted by position;
- top-50 candidate Spearman averaged by round, then season.

Stability metrics:

- feature-importance rank by fold;
- pairwise interaction rank by fold;
- sign consistency for feature shape summaries;
- number of seasons where each top feature appears in top 10.

Metrics must aggregate by season first. Do not present row-weighted aggregates
as the primary result.

## Feature Shape Summaries

The HTML report must show EBM feature curves, but CSV artifacts must also
capture deterministic summaries:

- `feature_name`;
- `fold_id`;
- `validation_season`;
- `importance_rank`;
- `importance_score`;
- `effect_min`;
- `effect_max`;
- `effect_range`;
- `largest_positive_bin_lower`;
- `largest_positive_bin_upper`;
- `largest_negative_bin_lower`;
- `largest_negative_bin_upper`;
- `monotonicity_hint`:
  - `increasing`;
  - `decreasing`;
  - `u_shaped`;
  - `inverted_u`;
  - `mixed`;
  - `unstable`;
- `row_support`;
- `season_support`;
- `candidate_hypothesis_flag`.

The `candidate_hypothesis_flag` is true only when:

- the term is in the top 10 by importance in at least 2 validation folds;
- the term has at least `500` validation rows total;
- its effect range is at least `0.50` points in at least 2 folds;
- its shape direction is not `unstable`.

These flags nominate human review candidates only.

## Pairwise Interaction Summaries

For interaction EBMs, write:

- `interaction_name`;
- `feature_a`;
- `feature_b`;
- `fold_id`;
- `validation_season`;
- `importance_rank`;
- `importance_score`;
- `effect_range`;
- `row_support`;
- `season_support`;
- `candidate_hypothesis_flag`.

The interaction flag is true only when:

- the interaction appears in the top 10 in at least 2 validation folds;
- both features are pre-round features;
- total validation row support is at least `500`;
- effect range is at least `0.50` points in at least 2 folds.

Do not convert an interaction flag into a feature pack automatically.

## CLI

Add:

```bash
uv run --frozen python scripts/run_ebm_feature_diagnostic.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260505T211914592073Z__matrix=d95948374d75 \
  --model-id xgboost_depth2_l2_heavy \
  --feature-pack ppg_xg_matchup \
  --seasons 2021,2022,2023,2024,2025 \
  --current-year 2026
```

Optional arguments:

- `--fixture-mode exploratory`;
- `--output-root data/08_reporting/ebm_diagnostics`;
- `--max-interactions 10`;
- `--min-validation-rows 500`;
- `--random-seed 123`;
- `--profile-runtime`.

The CLI must print:

- start timestamp;
- source experiment path;
- selected child count;
- fold progress;
- output path;
- final diagnostic status.

No silent long-running command is acceptable.

## Output Directory

Write to:

```text
data/08_reporting/ebm_diagnostics/ebm_diagnostic_started_at=<timestamp>/
```

Required artifacts:

- `ebm_diagnostic_manifest.json`;
- `source_context.csv`;
- `fold_assignments.csv`;
- `predictive_metrics.csv`;
- `feature_importance_by_fold.csv`;
- `feature_shape_summary.csv`;
- `pairwise_interactions.csv`;
- `invalid_ebm_rows.csv`;
- `ebm_diagnostic_decision.json`;
- `ebm_feature_diagnostic.html`.

If the diagnostic is invalid, still write:

- manifest;
- source context when available;
- invalid row/schema report;
- decision JSON;
- incomplete HTML page explaining why no model was fit.

## Decision Status

`ebm_diagnostic_decision.json` must use one of:

- `invalid`;
- `diagnostic_complete`;
- `candidate_hypotheses_found`;
- `insufficient_signal`;

Rules:

- `invalid`: source context, schema, dependency, or row validity checks failed.
- `candidate_hypotheses_found`: at least one main-effect or interaction
  candidate flag is true and predictive metrics do not materially regress the
  source model on 2025.
- `diagnostic_complete`: model fit and report succeeded, but candidate flags
  are absent or mixed.
- `insufficient_signal`: EBM underperforms the source model in at least 2 of 3
  validation seasons by MAE and no candidate flags are true.

Material 2025 regression for the diagnostic is:

```text
EBM MAE_2025 - source_model MAE_2025 > 0.15
```

This status is not a promotion status. It only tells us whether to write a
future H005/H006 hypothesis spec.

## Dependency Handling

V1 may add `interpret` as a development dependency if it installs cleanly under
the project Python version. If `interpret` cannot be installed on Python
`3.13.12`, do not downgrade Python and do not force the dependency into the main
runtime. Instead, keep the runner behind a clear optional dependency error:

```text
InterpretML is required for EBM diagnostics. Install the optional diagnostic
dependencies or run this workflow in a compatible Python environment.
```

The dependency decision must be recorded in the implementation plan after a
local `uv add --dev interpret` or dependency-resolution check.

## HTML Report

The HTML report must be simple and offline-readable. It must include:

- source experiment summary;
- warnings and `discovery_only=true`;
- fold-level predictive comparison table;
- top feature importance table;
- top interaction table;
- feature shape plots for the top terms;
- candidate hypothesis table;
- invalid row/schema section when applicable;
- explicit next-step text:
  - freeze a new hypothesis if stable candidates exist;
  - stop if signal is insufficient;
  - fix artifacts if invalid.

Plotly is acceptable for charts. If InterpretML can emit useful HTML snippets
without adding fragile dependencies, the implementation may embed or link them,
but the report must still include deterministic CSV-backed summaries.

## Acceptance Criteria

- The runner rejects missing required source columns with explicit names.
- The runner rejects unsupported broad AutoML libraries in V1; only EBM is
  implemented.
- The runner never uses random row-level CV.
- Fold assignments are whole-season and persisted.
- DNP/null target handling is tested.
- Coach rows are excluded and counted.
- Feature exclusion rules are tested.
- Main-effect EBM output writes feature importance and shape summaries.
- Interaction EBM output writes pairwise interaction summaries.
- Invalid diagnostics write incomplete-but-readable artifacts.
- CLI prints progress and output path.
- All artifacts include `discovery_only=true`.
- No existing experiment promotion/index fields are modified.

## Next Step After This Spec

Write a TDD implementation plan for `scripts/run_ebm_feature_diagnostic.py` and
`src/cartola/backtesting/ebm_feature_diagnostic.py`.

The implementation must start with validators and synthetic fixture tests,
then add dependency handling, main-effect fitting, interaction summaries, and
HTML output. Do not start by rendering HTML.
