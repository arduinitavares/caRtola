# Constrained Ridge Tuning Design

## Goal

Add a narrow, auditable tuning experiment for the current best no-fixture Cartola model profile:

```text
ridge + ppg_xg + fixture_mode=none
```

The purpose is not broad model search. The purpose is to answer one controlled question:

> Does a different Ridge regularization strength improve fixed-budget squad recommendations enough to replace the current `alpha=1.0` live profile?

The result must be useful for production. If a tuned variant wins, the project must be able to represent that variant as a deployable model profile in the same predictor factory used by live recommendations.

## Current State

The production-parity experiment completed for seasons `2023`, `2024`, and `2025` with `start_round=5`, `budget=100`, `fixture_mode=none`, and `current_year=2026`.

Best result:

- model: `ridge`;
- feature pack: `ppg_xg`;
- total actual points: `6515.24`;
- average actual points: `63.8749`;
- baseline total actual points for `random_forest + ppg`: `6029.86`;
- aggregate lift over that baseline: `+485.38`;
- average lift over that baseline: `+4.7586` points per optimized round;
- improved seasons: `3/3`;
- selected-player calibration slope: `0.7565`;
- promotion reason: `passes_v1_guardrails`.

Second result:

- model: `ridge`;
- feature pack: `ppg`;
- total actual points: `6382.02`;
- aggregate lift over `random_forest + ppg`: `+352.16`;
- improved seasons: `3/3`;
- selected-player calibration slope: `0.7966`;
- promotion reason: `passes_v1_guardrails`.

Tree ensembles were materially less calibrated:

- Random Forest and Extra Trees overpredicted actual optimized squad totals by roughly `29%` to `32%`;
- HGB overpredicted more and was much slower;
- HGB child runs are too expensive for speculative tuning.

The current Ridge implementation already uses numeric median imputation plus `StandardScaler`, and categorical `posicao` one-hot encoding. That makes Ridge alpha tuning interpretable enough for a first constrained pass.

## Problem

The project now has a plausible live model, but the default `Ridge(alpha=1.0)` was not tuned.

Naive tuning is dangerous here because:

- only three comparable historical seasons are available;
- the incumbent was selected using those same seasons;
- repeatedly changing a matrix after seeing results becomes manual hyperparameter search;
- squad points are a discontinuous objective because MILP formation selection, budget constraints, and captain choice can flip on small score changes;
- tuning broad model families would multiply comparisons before the project has enough independent seasons.

The next step should therefore be a frozen Ridge-only tuning generation, not Optuna, external libraries, broad tree grids, or HGB tuning.

## Non-Goals

Do not add:

- Optuna;
- XGBoost, LightGBM, CatBoost, or other external model libraries;
- HistGradientBoosting tuning;
- broad RandomForest or ExtraTrees hyperparameter search;
- tree output calibration wrappers;
- sklearn random k-fold CV;
- `RidgeCV` as the primary selector;
- predictive-only winner selection;
- fixture or matchup context;
- patrimonio simulation;
- live default changes;
- 2026 model tuning.

`2026` is the live production season. It may receive recommendations, but it must not be used as a tuning or promotion dataset.

## Design Summary

Create **Constrained Ridge Tuning v1** as a separate experiment type.

The runner:

1. freezes a predeclared Ridge alpha by feature-pack candidate matrix;
2. reruns the current incumbent inside the same experiment;
3. evaluates every candidate with the same walk-forward backtest and optimizer contract;
4. reports squad performance, prediction metrics, calibration, comparability, and tuning-specific deltas;
5. applies stricter promotion gates against the freshly rerun incumbent;
6. reruns the incumbent and top challengers in a final stage before any recommendation to promote.

The final result may be:

- promote a tuned Ridge profile;
- keep `ridge + ppg_xg + alpha=1.0`;
- conclude the result is inconclusive and gather more evidence.

All three outcomes are valid.

## Experiment Scope

### Seasons And Protocol

Required defaults:

- seasons: `2023,2024,2025`;
- start round: `5`;
- budget: `100`;
- current year: `2026`;
- fixture mode: `none`;
- matchup context mode: `none`;
- footystats modes represented by feature packs: `ppg`, `ppg_xg`;
- scoring contract: `cartola_standard_2026_v1`;
- all official formations;
- one non-tecnico captain with `1.5x` multiplier;
- tecnico included in optimized squad and selected-player metrics;
- child backtests remain sequential;
- `--jobs` applies only inside each child backtest.

The runner rejects:

- season `2026`;
- fixture modes other than `none`;
- matchup modes other than `none`;
- arbitrary CLI-supplied alpha lists in v1;
- unknown model parameters;
- output collisions unless an explicit `--force` behavior is implemented.

### Candidate Matrix

The v1 matrix is fixed.

Feature packs:

```text
ppg
ppg_xg
```

Ridge alpha values:

```text
0.01
0.03
0.1
0.3
1.0
3.0
10.0
30.0
100.0
300.0
```

Candidate ids are deterministic:

```text
ridge_alpha_0_01__ppg
ridge_alpha_0_03__ppg
ridge_alpha_0_1__ppg
ridge_alpha_0_3__ppg
ridge_alpha_1_0__ppg
ridge_alpha_3_0__ppg
ridge_alpha_10_0__ppg
ridge_alpha_30_0__ppg
ridge_alpha_100_0__ppg
ridge_alpha_300_0__ppg
ridge_alpha_0_01__ppg_xg
ridge_alpha_0_03__ppg_xg
ridge_alpha_0_1__ppg_xg
ridge_alpha_0_3__ppg_xg
ridge_alpha_1_0__ppg_xg
ridge_alpha_3_0__ppg_xg
ridge_alpha_10_0__ppg_xg
ridge_alpha_30_0__ppg_xg
ridge_alpha_100_0__ppg_xg
ridge_alpha_300_0__ppg_xg
```

Primary incumbent:

```text
ridge_alpha_1_0__ppg_xg
```

Secondary control:

```text
ridge_alpha_1_0__ppg
```

Every aggregate report must include:

- `candidate_id`;
- `model_id`;
- `feature_pack`;
- `model_params_json`;
- `model_params_hash`;
- `tuning_generation_hash`.

Reports must never group tuned Ridge variants by only `model_id` and `feature_pack`, because that would collapse all alpha values into the same row.

## Model Integration Contract

The current live and experiment code uses `model_id="ridge"` for Ridge. Tuning must not invent public model ids such as `ridge_alpha_3`.

Use this separation:

- `model_id`: model family, always `ridge` for this feature;
- `model_params`: exact estimator parameters, such as `{"alpha": 3.0}`;
- `candidate_id`: stable experiment identity combining model family, params, and feature pack.

The primary strategy row inside child backtest outputs remains:

```text
strategy=ridge
```

The prediction score column remains:

```text
ridge_score
```

The tuning runner is responsible for carrying `candidate_id` and `model_params_hash` in child metadata and top-level reports.

The predictor factory must support a private/internal parameter override path for experiment use:

```python
create_point_predictor(
    *,
    model_id: str,
    random_seed: int,
    feature_columns: list[str],
    n_jobs: int,
    model_params: Mapping[str, object] | None = None,
) -> PointPredictor
```

Rules:

- normal backtest CLI behavior remains unchanged;
- live recommendation behavior remains unchanged until a separate promotion change;
- `model_params=None` preserves current registry defaults;
- for Ridge, the only v1 override key is `alpha`;
- unknown override keys raise `ValueError`;
- non-positive Ridge alpha values raise `ValueError`;
- all effective model parameters are written to metadata.

If a tuned alpha wins, a later production change can promote it by adding a named deployable profile or changing the Ridge registry default. Raw arbitrary model params should not be exposed as a live CLI surface in this feature.

## Execution Stages

### Stage 0: Matrix Freeze

Before any child run starts, write a tuning generation manifest containing:

- exact candidate list;
- exact alpha list;
- feature packs;
- seasons;
- start round;
- budget;
- scoring contract;
- fixture mode;
- matchup mode;
- source data hashes;
- git commit and dirty state;
- Python and dependency versions;
- `uv.lock` hash;
- runner version;
- tuning generation hash.

Changing any candidate, alpha, source, season, budget, scoring contract, or feature-pack definition creates a new tuning generation.

### Stage 1: Screen Run

Run every `(season, candidate_id)` child through the full walk-forward backtest and MILP optimizer.

This is intentionally full simulation, not predictive-only screening. Ridge is cheap enough that bypassing the optimizer would save little while optimizing the wrong objective.

For each child, write the same authoritative report artifacts as model-feature experiments, plus tuning metadata:

- `candidate_id`;
- `model_params_json`;
- `model_params_hash`;
- `tuning_stage=screen`;
- `incumbent_candidate_id`;
- `feature_pack_control_candidate_id`.

### Stage 2: Promotion Screen

Build a ranked screen summary with two delta families:

1. **Primary incumbent deltas**
   - compare every candidate against `ridge_alpha_1_0__ppg_xg`.
   - this decides whether anything should replace the current live candidate.

2. **Same-feature-pack deltas**
   - compare `ppg` candidates against `ridge_alpha_1_0__ppg`;
   - compare `ppg_xg` candidates against `ridge_alpha_1_0__ppg_xg`;
   - this shows whether alpha tuning helped each feature pack independently.

Screen eligibility requires all tuning guardrails to pass.

### Stage 3: Final Rerun

If no challenger passes the screen gates, the experiment ends with `promotion_recommendation=keep_incumbent`.

If one or more challengers pass, rerun:

- `ridge_alpha_1_0__ppg_xg`;
- `ridge_alpha_1_0__ppg`;
- the top two passing challengers by primary-incumbent aggregate delta.

Final reruns must be marked:

```text
tuning_stage=final
```

Only final-stage results may be used for a production recommendation. Screen-stage winners are candidates, not promoted profiles.

Because Ridge and the optimizer should be deterministic, final reruns should match screen results after excluding runtime, output path, and timestamp fields. If a final rerun changes actual squad totals by more than `0.01`, mark the candidate `non_reproducible` and ineligible.

## Promotion Rules

Promotion is from the current live candidate:

```text
ridge_alpha_1_0__ppg_xg
```

A challenger is eligible only if all conditions pass on the final rerun.

### Comparability Gates

Fail closed on:

- candidate-pool signature mismatch;
- skipped-round mismatch;
- solver-status signature mismatch by strategy role;
- scoring-contract mismatch;
- fixture/matchup mode mismatch;
- source hash mismatch;
- dependency or model-param hash mismatch between screen and final rerun;
- missing or null required promotion metrics.

Null required metrics are ineligible with reason:

```text
insufficient_metric_data
```

### Practical Lift Gates

Compared with the primary incumbent final rerun:

- aggregate actual points must improve by at least `0.5` point per evaluated optimized round;
- improved seasons must be at least `2` out of `3`;
- no season may regress by more than `0.5` average point per optimized round;
- total evaluated rounds must match the incumbent exactly.

The practical threshold is computed from the actual final-run round count:

```text
required_aggregate_lift = 0.5 * total_rounds
```

For the current `102`-round protocol, this is `51.0` aggregate points.

If a candidate wins by less than this threshold, report:

```text
promotion_recommendation=keep_incumbent
promotion_reason=lift_below_practical_threshold
```

### Prediction And Calibration Gates

Compared with the primary incumbent final rerun:

- selected-player calibration slope must remain within `[0.75, 1.25]`;
- top-50 candidate Spearman must not regress by more than `0.03`;
- candidate-pool MAE must not worsen by more than `5%`;
- selected-player MAE must not worsen by more than `5%`.

These are guardrails, not the primary ranking objective. The primary objective remains final-run optimized squad actual points after all gates pass.

## Reports And Output Layout

Write outputs under:

```text
data/08_reporting/experiments/model_tuning/<experiment_id>/
```

Recommended experiment id shape:

```text
group=ridge-alpha-tuning__started_at=<timestamp>__matrix=<hash>
```

Child paths:

```text
runs/stage=<screen|final>/season=<season>/candidate=<candidate_id>/
```

Top-level reports:

- `tuning_generation_manifest.json`;
- `ranked_summary.csv`;
- `per_season_summary.csv`;
- `prediction_metrics.csv`;
- `calibration_deciles.csv`;
- `comparability_report.json`;
- `promotion_report.json`;
- `comparison_report.md`;
- `squad_performance_comparison.html`;
- `calibration_plots.html`.

`ranked_summary.csv` must include:

- rank;
- candidate id;
- model id;
- feature pack;
- alpha;
- tuning stage;
- total rounds;
- total actual points;
- average actual points;
- total predicted points;
- average predicted points;
- primary incumbent total actual points;
- aggregate delta vs primary incumbent;
- average delta per round vs primary incumbent;
- same-feature-pack incumbent total actual points;
- aggregate delta vs same-feature-pack incumbent;
- improved seasons vs primary incumbent;
- worst season average delta vs primary incumbent;
- selected calibration slope;
- top-50 Spearman delta vs primary incumbent;
- candidate-pool MAE delta percent vs primary incumbent;
- selected-player MAE delta percent vs primary incumbent;
- promotion eligible;
- promotion reason.

`promotion_report.json` must state one of:

- `promote_candidate`;
- `keep_incumbent`;
- `inconclusive`;
- `experiment_failed`.

It must also include the exact candidate id and model params for any recommended promoted profile.

## CLI Shape

Add a separate command rather than widening the existing model-feature runner into a general hyperparameter tool:

```bash
uv run --frozen python scripts/run_ridge_tuning.py \
  --seasons 2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 20
```

The command does not accept `--alphas` in v1. The matrix is fixed in code and written to the generation manifest.

Optional flags may mirror existing experiment behavior:

- `--force` only if output collision behavior exists;
- tracker/index flags from Experiment Observability v1, if that feature is available;
- `--skip-final-rerun` only for local diagnostics, and any such run must be marked ineligible for promotion.

## Why Not Predictive-Only Selection

Prediction metrics matter, but they are not the production objective.

A predictive-only stage can reject broken candidates, but it cannot choose a live profile by itself because the optimizer uses:

- formation constraints;
- positional quotas;
- budget;
- tecnico selection;
- captain multiplier.

For Ridge v1, full simulation is cheap enough to keep the business objective in the loop. Predictive-only screening is deferred until the project has a large search space, such as Optuna or external libraries.

## Why Not RidgeCV

`RidgeCV` optimizes row-level predictive error. That is useful for generic regression, but it is not the objective here.

The project needs a single deployable alpha value chosen under the same walk-forward, candidate-pool, optimizer, and scoring contract used by live recommendation. A fixed alpha matrix gives a clearer audit trail than internal cross-validation and avoids accidental leakage or fold semantics that do not match Cartola rounds.

## Why Not Tree Tuning

Random Forest and Extra Trees did not merely underperform Ridge. They also showed large overprediction and weaker selected-player calibration.

Tuning `min_samples_leaf`, `max_features`, or `max_depth` may reduce variance, but it is unlikely to solve a structural calibration problem by itself. Tree calibration wrappers may be useful later, but they are a separate model-design feature.

## Deferred Work

Defer:

- RandomForest and ExtraTrees calibration wrappers;
- small calibrated-tree candidate menu;
- HGB profiling and tuning;
- predictive-only screening infrastructure;
- Optuna with fixed seeded trials;
- XGBoost, LightGBM, and CatBoost;
- strict fixture/matchup tuning;
- live default/profile promotion after a tuned winner is identified.

## Required Tests

Add tests that prove:

1. The Ridge alpha matrix is exactly the fixed v1 list.
2. Both `ppg` and `ppg_xg` are included for every alpha.
3. Season `2026` is rejected.
4. Fixture modes other than `none` are rejected.
5. Matchup modes other than `none` are rejected.
6. `ridge_alpha_1_0__ppg_xg` is present and marked as the primary incumbent.
7. `ridge_alpha_1_0__ppg` is present and marked as the secondary control.
8. Every candidate has a stable `candidate_id`.
9. Every candidate has a stable `model_params_hash`.
10. Changing alpha changes `model_params_hash` and `tuning_generation_hash`.
11. `create_point_predictor(..., model_params={"alpha": value})` creates a Ridge predictor with that alpha.
12. Unknown Ridge override keys raise `ValueError`.
13. Non-positive Ridge alpha values raise `ValueError`.
14. Normal backtest CLI behavior still uses Ridge alpha `1.0` only when Ridge is selected through existing registry defaults.
15. Live recommendation commands are unchanged by this feature.
16. Screen-stage winners are not promotion eligible without final rerun.
17. Final rerun mismatch larger than `0.01` marks the candidate ineligible.
18. Null required guardrail metrics mark the candidate ineligible.
19. Practical lift threshold is `0.5 * total_rounds`.
20. A candidate below the practical lift threshold cannot be promoted.
21. Candidate-pool, skipped-round, solver-status, scoring-contract, and source-hash mismatches fail comparability.
22. Reports group by `candidate_id`, not only by `model_id` and `feature_pack`.
23. The command writes `promotion_report.json`.
24. The command records exact model params in metadata and reports.

## Acceptance Criteria

The feature is accepted when:

- a single command runs the fixed Ridge alpha matrix over `2023`, `2024`, and `2025`;
- the incumbent and secondary control are rerun inside the same tuning generation;
- the output reports rank candidates by final-run optimized squad actual points after guardrails pass;
- every result can be traced to exact source hashes, model params, scoring contract, and git state;
- no candidate is promoted from screen-stage results alone;
- the final recommendation is one of `promote_candidate`, `keep_incumbent`, `inconclusive`, or `experiment_failed`;
- no public backtest, live recommendation, or normal model-feature experiment behavior changes unless explicitly requested in a later promotion task.

## Implementation Order

1. Add model parameter override plumbing for Ridge only.
2. Add candidate identity and fixed matrix generation.
3. Add the separate `scripts/run_ridge_tuning.py` orchestration command.
4. Reuse existing walk-forward backtest and experiment report writers where possible.
5. Add tuning-specific aggregate deltas and promotion gates.
6. Add final rerun support.
7. Add tests.
8. Run the quality gate.

Do not implement external model libraries, Optuna, tree calibration, or live default promotion in this feature.
