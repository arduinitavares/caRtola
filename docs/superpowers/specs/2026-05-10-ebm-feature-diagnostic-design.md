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

A later implementation review adds four binding constraints:

- EBM may not use random internal validation for early stopping.
- Raw identity-like fields must be excluded unless categorical handling is
  explicitly implemented and tested.
- Residual diagnostics are required because raw-point EBMs mostly rediscover
  existing predictive structure.
- Candidate flags must be aggregated across folds and must include bin/cell
  support checks, not only total row support.

## Goals

- Read completed experiment artifacts without rerunning backtests.
- Validate source context before loading player rows.
- Build a cutoff-safe diagnostic dataset from persisted `player_predictions.csv`
  artifacts.
- Fit EBMs on pre-round features for both raw realized points and source-model
  residuals.
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

### Child Run Resolution

The CLI context must resolve to exactly one child run per requested season.
The matching key is:

```text
season
model_id
feature_pack
fixture_mode
matchup_context_mode
footystats_mode
budget_policy
scoring_contract_version
primary_score_column
```

Resolution may use parent `experiment_metadata.json`, parent index fields, and
child `run_metadata.json`, but it must not infer model identity from directory
names alone.

Missing or duplicate matches invalidate the diagnostic before fitting any EBM.
`source_context.csv` must still be written and must list:

- requested context values;
- matched child path when exactly one match exists;
- conflicting child paths when duplicates exist;
- missing required metadata fields;
- selected primary score column;
- `source_prediction_provenance_status`.

`source_prediction_provenance_status` is `verified` only when:

- the parent `experiment_metadata.json` has exactly one `child_runs` entry for
  the matching key;
- that child entry's `output_path` resolves to the analyzed child directory;
- child `run_metadata.json` agrees with the parent child entry for the matching
  key fields;
- `player_predictions.csv`, `round_results.csv`, and `run_metadata.json` live
  under that resolved child directory;
- the primary score column exists in `player_predictions.csv`.

If provenance cannot be verified from those artifact relationships, the
diagnostic is invalid.

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

The diagnostic uses two supervised targets. The raw-point target is a sanity
check. The residual target is the primary discovery target because it asks what
the source model missed. Neither target uses captain-adjusted squad points.

```text
target_actual_points = pontuacao after DNP/null policy
target_source_residual = target_actual_points - source_model_score
```

For predictive comparisons:

- actual-point EBM predictions are compared directly against
  `target_actual_points`;
- residual EBM predictions are converted to corrected point predictions:
  `source_model_score + predicted_source_residual`;
- raw predicted residuals are still persisted for diagnostics, but MAE/RMSE
  comparisons against the source model use the corrected point prediction.

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

### EBM Inner Validation

InterpretML's internal validation must not randomly split player rows.

V1 allows exactly two non-random EBM validation modes. The implementation plan
must choose one after checking the installed API and fold row counts.

Mode 1, `temporal_inner_validation`, uses the latest training season as the
EBM validation set:

```text
Outer Fold A:
  train: 2021
  EBM inner validation: 2022
  external validation: 2023

Outer Fold B:
  train: 2021,2022
  EBM inner validation: 2023
  external validation: 2024

Outer Fold C:
  train: 2021,2022,2023
  EBM inner validation: 2024
  external validation: 2025
```

Use this mode only when every fold has at least `1` training season, `1000`
valid training rows, `500` valid inner-validation rows, and `500` valid
external-validation rows after DNP/coach/feature filtering. The implementation
must call EBM fitting with explicit validation arrays when the installed
InterpretML API supports `fit(X, y, X_val=..., y_val=...)`.

Mode 2, `disabled_full_outer_train`, trains on every outer training season and
sets internal validation off or equivalent:

```text
Fold A:
  train: 2021,2022
  external validation: 2023

Fold B:
  train: 2021,2022,2023
  external validation: 2024

Fold C:
  train: 2021,2022,2023,2024
  external validation: 2025
```

Use this mode when explicit temporal validation is unsupported or would leave
too little training evidence. It preserves the season-expanding train sets
while still preventing random row validation.

This mode requires every fold to have at least `2` outer-training seasons,
`1500` valid training rows, and `500` valid external-validation rows after
DNP/coach/feature filtering.

The selected mode must be recorded:

```text
inner_validation_mode=temporal_inner_validation | disabled_full_outer_train
early_stopping_mode=explicit_temporal_validation | disabled_or_no_random_split
```

The implementation plan must verify the installed API with `inspect.signature`
before coding the adapter. Tests must prove that no external validation-season
rows and no future inner-validation rows enter the training set.

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
validation_size_or_validation_fraction=0.0 when explicit X_val/y_val is unavailable
random_state=123
n_jobs=-1
objective="rmse"
```

InterpretML parameter names have changed across versions. The implementation
must inspect the installed `ExplainableBoostingRegressor` constructor and
`fit()` signatures, then map these semantic settings to the installed API.
The manifest must record the installed InterpretML version plus the constructor
and fit signatures used for the run. A missing or incompatible parameter must
produce an invalid diagnostic rather than silently changing the validation
contract.

Before any model fit, a compatibility adapter test must assert the resolved
constructor parameters, fit keyword arguments, and validation mode for the
installed InterpretML version.

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

Use only the source model's persisted feature columns from
`run_metadata.json`, after applying the leakage and identity exclusion list.
Fail if any retained feature is non-numeric except explicitly generated
position one-hot columns.

Exclude these columns even if present:

- target/outcome columns;
- captain columns;
- actual scout result columns for the target round;
- IDs and identity-like fields that create memorization risk:
  - `id_atleta`;
  - `id_clube`;
  - raw `apelido`;
  - raw club/opponent names;
  - raw club/opponent IDs;
- direct round labels that can create season/round memorization:
  - `season`;
  - `rodada`.

Allow numeric fields whose distance is football-meaningful, such as:

- `preco_pre_rodada`;
- prior rolling point/scout features;
- FootyStats PPG/xG fields;
- matchup context fields.

`posicao` must not be treated as an arbitrary ordinal number. V1 must either
one-hot encode position or run separate position-specific diagnostics. The
manifest must record `position_handling`.

Raw categorical identifiers may be admitted only in a later diagnostic
generation with explicit categorical feature handling, a stable category
contract, and tests proving that the resulting shapes are not numeric-ID
artifacts.

The manifest must record the final EBM feature list, excluded columns, and
excluded identity columns.

## Metrics

Compute metrics for both source model predictions and EBM predictions on the
exact same valid, non-coach external-validation rows. Each fold must write
`shared_evaluation_row_count`; a row-set mismatch between source and EBM metrics
invalidates that fold.

Prediction metrics:

- MAE;
- RMSE;
- Spearman correlation by season;
- top-50 Spearman by season and round when the round has at least `50` valid
  non-coach candidates;
- calibration slope by season;
- mean prediction bias by season.

Ranking metrics:

- actual top-5 position scorers' mean predicted rank;
- top-10 actual scorer recall inside top-20 predicted by position;
- top-50 candidate Spearman averaged by round, then season.

Ranking uses descending ranks with average tie handling. Spearman metrics are
null when either side has fewer than two distinct values.

Stability metrics:

- feature-importance rank by fold and target type;
- pairwise interaction rank by fold and target type;
- sign consistency for feature shape summaries;
- number of seasons where each top feature appears in top 10.

Metrics must aggregate by season first. Do not present row-weighted aggregates
as the primary result. Primary hypothesis nomination uses the residual target;
raw-point target metrics are sanity checks and context.

## Term Support Extraction Contract

Candidate flags depend on EBM term support, so support must be computed from
the fitted EBM's learned term/bin definitions, not from ad hoc raw-value ranges.

For each fitted EBM, the runner must persist these manifest fields:

- `term_support_extraction_method`;
- `term_support_extraction_status`;
- EBM metadata attributes used for bin assignment;
- fallback reason when exact assignment is unavailable.

Accepted support extraction:

- Continuous features: map each external-validation row to the exact learned
  continuous bin used by the fitted EBM, including explicit missing-value bins
  if the fitted EBM exposes them.
- Position one-hot features: use the generated one-hot column value as the
  term bin. The support for an active position bin is the rows where that
  one-hot column is `1`; the support for the inactive/reference bin is rows
  where it is `0`.
- Missing values: count missing-value rows in the fitted EBM's missing bin when
  exposed. If missing-bin assignment cannot be reproduced exactly, support is
  unavailable for that term.
- Pairwise interactions: map each external-validation row to the ordered pair
  of learned bin IDs for `feature_a` and `feature_b`. Cell support is counted
  on those learned bin ID pairs, not on raw feature ranges.

If exact bin/cell assignment cannot be reproduced for a term, write the term
with `term_support_extraction_status=unavailable`, set all support-dependent
candidate fields to false, and record the reason. Importance and effect values
may still be reported, but unavailable support must block
`fold_candidate_signal` and aggregated `candidate_hypothesis_flag`.

The implementation plan must include a synthetic end-to-end test with known
bins, known missing-value rows, one-hot position rows, and known interaction
cells. The test must assert the row and round support values used for candidate
gating.

## Feature Shape Summaries

The HTML report must show EBM feature curves, but CSV artifacts must also
capture deterministic summaries:

- `target_type`:
  - `actual_points`;
  - `source_residual`;
- `feature_name`;
- `fold_id`;
- `validation_season`;
- `importance_rank`;
- `importance_score`;
- `effect_min`;
- `effect_max`;
- `effect_range`;
- `term_support_extraction_status`;
- `largest_positive_bin_lower`;
- `largest_positive_bin_upper`;
- `largest_positive_bin_row_support`;
- `largest_positive_bin_round_support`;
- `largest_positive_bin_season_support`;
- `largest_negative_bin_lower`;
- `largest_negative_bin_upper`;
- `largest_negative_bin_row_support`;
- `largest_negative_bin_round_support`;
- `largest_negative_bin_season_support`;
- `monotonicity_hint`:
  - `increasing`;
  - `decreasing`;
  - `u_shaped`;
  - `inverted_u`;
  - `mixed`;
  - `unstable`;
- `row_support`;
- `season_support`;
- `fold_candidate_signal`.

The per-fold `fold_candidate_signal` is true only when:

- the term is in the top 10 by importance for that target type in that fold;
- the term has at least `500` validation rows in that fold;
- the bins defining `effect_min` and `effect_max` each have at least `50`
  rows;
- the bins defining `effect_min` and `effect_max` each span at least `5`
  distinct rounds;
- its effect range is at least `0.50` points;
- its shape direction is not `unstable`.

These per-fold signals do not nominate a hypothesis by themselves.

## Aggregated Candidate Hypotheses

Write `candidate_hypotheses.csv` after all folds and target types complete.
This is the only artifact allowed to contain cross-fold
`candidate_hypothesis_flag`.

Required columns:

- `target_type`;
- `candidate_type`:
  - `main_effect`;
  - `interaction`;
- `term_name`;
- `feature_a`;
- `feature_b`;
- `fold_signal_count`;
- `validation_seasons_with_signal`;
- `total_row_support`;
- `min_bin_or_cell_row_support`;
- `min_bin_or_cell_round_support`;
- `effect_range_median`;
- `direction_summary`;
- `failed_validation_seasons`;
- `candidate_hypothesis_flag`;
- `candidate_scope`.

The aggregated `candidate_hypothesis_flag` is true only when:

- `target_type == "source_residual"`;
- `fold_signal_count >= 2`;
- total validation row support is at least `1000`;
- every effect-defining bin or cell used by the signal has at least `50` rows;
- every effect-defining bin or cell used by the signal spans at least `5`
  distinct rounds;
- directions are not contradictory across signaled folds.

Direction compatibility is strict:

- `increasing` and `decreasing` contradict each other;
- `u_shaped` and `inverted_u` contradict each other and contradict monotone
  directions;
- `mixed` and `unstable` cannot produce an aggregated candidate;
- interaction terms must use `direction_summary=interaction_mixed` unless a
  later generation defines a tested interaction direction classifier, and may
  become candidates only through repeated high-support effect ranges.

`candidate_scope` must be `human_review_only`. A flagged candidate is not a
feature, policy, model candidate, promotion signal, or experiment-index value.

## Pairwise Interaction Summaries

For interaction EBMs, write:

- `target_type`:
  - `actual_points`;
  - `source_residual`;
- `interaction_name`;
- `feature_a`;
- `feature_b`;
- `fold_id`;
- `validation_season`;
- `importance_rank`;
- `importance_score`;
- `effect_range`;
- `term_support_extraction_status`;
- `max_effect_cell_row_support`;
- `max_effect_cell_round_support`;
- `min_effect_cell_row_support`;
- `min_effect_cell_round_support`;
- `row_support`;
- `season_support`;
- `fold_candidate_signal`.

The per-fold interaction signal is true only when:

- the interaction appears in the top 10 for that target type in that fold;
- both features are pre-round features;
- total validation row support is at least `500`;
- the cells defining the effect range each have at least `50` rows;
- the cells defining the effect range each span at least `5` distinct rounds;
- effect range is at least `0.50` points.

Do not convert an interaction signal or aggregated flag into a feature pack
automatically.

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
- `candidate_hypotheses.csv`;
- `invalid_ebm_rows.csv`;
- `ebm_diagnostic_decision.json`;
- `ebm_feature_diagnostic.html`.

All JSON artifacts must include `discovery_only=true`.

The manifest must include:

- `discovery_only=true`;
- `holdout_usage_ledger`, including `2025=diagnostic_exposed` when Fold C is
  run or inspected;
- installed InterpretML version;
- inspected EBM constructor signature;
- inspected EBM `fit()` signature;
- `inner_validation_mode`;
- `early_stopping_mode`;
- `position_handling`;
- final feature list;
- excluded feature list;
- excluded identity-column list.
- child-run matching key;
- resolved child paths by season;
- `source_prediction_provenance_status`;
- `term_support_extraction_method`;
- `term_support_extraction_status`;

Every CSV artifact must include a `discovery_only` column with value `true`.

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
- `candidate_hypotheses_found`: at least one aggregated
  `candidate_hypothesis_flag` is true for `target_type=source_residual` and
  residual-corrected EBM predictive metrics do not materially regress the
  source model on 2025.
- `diagnostic_complete`: model fit and report succeeded, but aggregated
  candidate flags are absent or mixed.
- `insufficient_signal`: residual-corrected EBM predictions underperform the
  source model in at least 2 of 3 validation seasons by MAE and no aggregated
  residual-target candidate flags are true.

Candidate-related decisions use residual-corrected point predictions only:

```text
residual_corrected_prediction = source_model_score + predicted_source_residual
```

Raw-point EBM metrics are reported for context but do not produce
`candidate_hypotheses_found` or `insufficient_signal`.

Material 2025 regression for the diagnostic is:

```text
residual_corrected_EBM MAE_2025 - source_model MAE_2025 > 0.15
or
residual_corrected_EBM top50_spearman_2025 - source_model top50_spearman_2025 < -0.02
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
- holdout usage ledger;
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

Numeric CSV/JSON artifacts are the primary output and must be written before
HTML rendering starts. HTML rendering failure must not invalidate completed
numeric artifacts; it should write/report an HTML warning when possible.

Plotly is acceptable for charts. If InterpretML can emit useful HTML snippets
without adding fragile dependencies, the implementation may embed or link them,
but the report must still include deterministic CSV-backed summaries.

## Acceptance Criteria

- The runner rejects missing required source columns with explicit names.
- The runner requires exactly one validated child run per requested
  season/context and writes duplicate/missing matches to `source_context.csv`.
- Source prediction provenance is validated from metadata before fitting.
- The runner rejects unsupported broad AutoML libraries in V1; only EBM is
  implemented.
- The runner never uses random row-level CV.
- EBM inner validation uses explicit temporal validation arrays or disables
  internal validation/early stopping; this is tested.
- Fold assignments are whole-season and persisted.
- DNP/null target handling is tested.
- Coach rows are excluded and counted.
- Raw identity columns are excluded and this is tested.
- Position handling is categorical or position-specific, never arbitrary
  ordinal numeric.
- Feature exclusion rules are tested.
- Source and EBM metrics use identical shared evaluation rows.
- Raw-point and residual-target artifacts are both written.
- Main-effect EBM output writes feature importance and shape summaries.
- Interaction EBM output writes pairwise interaction summaries.
- Term-support extraction uses fitted EBM learned bins/cells or marks support
  unavailable and blocks candidate flags.
- Bin/cell-level support thresholds are enforced before any candidate flag.
- Cross-fold `candidate_hypothesis_flag` appears only in
  `candidate_hypotheses.csv`.
- Decision-status metrics use residual-corrected predictions for candidate
  decisions.
- Invalid diagnostics write incomplete-but-readable artifacts.
- CLI prints progress and output path.
- All artifacts include `discovery_only=true`.
- CSV artifacts include a `discovery_only` column.
- InterpretML version and inspected signatures are recorded.
- The holdout ledger records 2025 diagnostic exposure.
- Synthetic tests cover bin support, missing-bin support, one-hot position
  support, interaction cell support, duplicate child detection, and
  residual-corrected metric selection.
- No existing experiment promotion/index fields are modified.

## Next Step After This Spec

Write a TDD implementation plan for `scripts/run_ebm_feature_diagnostic.py` and
`src/cartola/backtesting/ebm_feature_diagnostic.py`.

The implementation must start with validators and synthetic fixture tests,
then add dependency handling, main-effect fitting, interaction summaries, and
HTML output. Do not start by rendering HTML.
