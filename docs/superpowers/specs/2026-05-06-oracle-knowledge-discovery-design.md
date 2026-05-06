# Oracle Knowledge Discovery Design

## Goal

Build a discovery-only report that creates hindsight-best historical squads and profiles them against pre-round-known features.

The report should answer questions like:

- Are hindsight-best attackers more often home or away?
- Are hindsight-best defenders and goalkeepers concentrated in specific matchup contexts?
- What pre-round price rank, predicted rank, recent-form rank, and opponent profile do oracle players usually have?
- Which traits appear in oracle captains versus normal selected captains?
- Where does the model fail to see or rank players who later become round winners?

The output is a research map for future frozen hypotheses. It is not promotion evidence and must not change live defaults directly.

## Core Rule

Use actual post-round points only to define hindsight-best squads.

Use only pre-round-known fields to describe, profile, and explain those squads.

No feature, policy, or model default may be promoted from this report without a later frozen walk-forward experiment.

## Scope

V1 creates a standalone oracle discovery runner and report.

Recommended entrypoint:

```text
scripts/run_oracle_knowledge_discovery.py
```

Recommended implementation module:

```text
src/cartola/backtesting/oracle_discovery.py
```

The runner should read historical backtest/model experiment artifacts in default artifact mode. It may reproduce target-round candidate frames only in explicit reconstructed mode. It should not run model training as part of normal experiment ranking, and it must not be called from `scripts/run_model_experiments.py`.

Default source mode is:

```text
source_mode=artifact
```

In artifact mode, model-vs-oracle analysis must read persisted child-run artifacts from the source backtest or experiment run. It must not rebuild candidate frames with newer code or newer data and then compare them to old model decisions.

An opt-in reconstructed mode may be added:

```text
source_mode=reconstructed
```

Rows from reconstructed mode must carry `source_mode=reconstructed`, `code_sha`, and data/source hashes. Reconstructed mode is useful for fresh analysis, but it is not an exact audit of an older experiment.

Reconstructed mode is deferred from the first implementation pass.

## Non-Goals

Do not build:

- an oracle-trained production model;
- an automatic policy generator;
- an Optuna objective based on oracle traits;
- a full reinforcement learning agent;
- broad trait mining that claims causality;
- additive regret accounting unless formulas enforce a true identity;
- a user-facing recommendation mode based on hindsight oracles.

Do not treat “oracle players overindex on trait X” as a validated rule. Treat it as a hypothesis candidate.

## Source Artifact Contract

Artifact mode requires exact source artifacts.

For each analyzed child run, require:

- `round_results.csv`;
- `selected_players.csv`;
- `player_predictions.csv`;
- `summary.csv`;
- `run_metadata.json`.

### Source Run Context

Artifact mode also requires validated source-run context. Do not infer model identity, feature pack, or score columns silently from path names.

Required `source_run_context` fields:

- `source_experiment_id`;
- `source_child_id`;
- `source_child_path`;
- `season`;
- `model_id`;
- `feature_pack`;
- `fixture_mode`;
- `matchup_context_mode`;
- `budget_policy`;
- `primary_strategy`;
- `strategy_score_columns`;
- `analyzed_strategies`.

`strategy_score_columns` must map every analyzed strategy to exactly one score column:

```json
{
  "baseline": "baseline_score",
  "price": "price_score",
  "xgboost_depth2_l2_heavy": "xgboost_depth2_l2_heavy_score"
}
```

Allowed sources for `source_run_context`:

1. Parent experiment `experiment_metadata.json` child records.
2. Experiment index rows when they contain enough child context.
3. Explicit CLI arguments.
4. A validated sidecar JSON file.

If multiple sources disagree, fail the run. If no source provides the required context, fail the run. Path parsing may be used only as a consistency check after a trusted source has provided the values.

If parent experiment metadata provides `strategy_roles` but not explicit score columns, v1 may construct score columns with this deterministic rule:

- `baseline` uses `baseline_score`;
- `price` uses `price_score`;
- the primary model strategy uses `{model_id}_score`.

Every constructed score column must exist in `player_predictions.csv` or the child run is invalid. Any non-standard strategy requires explicit CLI or sidecar mapping.

Minimum `round_results.csv` columns:

- `rodada`;
- `strategy`;
- `solver_status`;
- `budget_before_round`;
- `budget_after_round`;
- `budget_delta`;
- `budget_used`;
- `actual_points_with_captain`;
- `captain_id`;
- `scoring_contract_version` if present, otherwise recover from metadata.

Minimum `selected_players.csv` columns:

- `rodada`;
- `strategy`;
- `id_atleta`;
- `apelido`;
- `posicao`;
- `id_clube`;
- `nome_clube`;
- `entrou_em_campo`;
- `preco_pre_rodada`;
- `pontuacao`;
- `variacao`;
- `is_captain`.

Minimum `player_predictions.csv` columns:

- `rodada`;
- `id_atleta`;
- `apelido`;
- `posicao`;
- `id_clube`;
- `nome_clube`;
- `status`;
- `entrou_em_campo`;
- `preco_pre_rodada`;
- `pontuacao`;
- `variacao`;
- the score column named by validated source-run context for the analyzed strategy;
- all profile columns used by enabled report sections.

Each report section must declare required columns before execution. In strict mode, missing required columns fail the run. In permissive mode, only the affected section is disabled and marked unavailable.

Minimum `run_metadata.json` fields:

- `season`;
- `start_round`;
- `initial_budget`;
- `budget_policy`;
- `fixture_mode`;
- `matchup_context_mode`;
- `footystats_mode`;
- `playable_statuses` if present, otherwise default to `("Provavel",)`;
- `scoring_contract_version`;
- `fixture_source_directory`;
- `fixture_manifest_sha256`;
- footystats source path/hash fields when `footystats_mode != "none"`.

The runner must fail fast with a missing-file or missing-column error unless explicitly invoked with a permissive `--allow-incomplete` report mode. In permissive mode, incomplete sections must render as incomplete rather than silently dropping rows.

Canonical join keys:

```text
season, rodada, model_id, feature_pack, fixture_mode, matchup_context_mode,
budget_policy, source_mode, oracle_type, candidate_universe, budget_path,
id_atleta
```

## Candidate Universe Contract

Every oracle row must declare `candidate_universe`.

| Universe | Required filters |
| --- | --- |
| `model_candidate` | Use exactly the persisted `player_predictions.csv` rows for the model child and round. Preserve the score column named by `source_run_context` and all filters already applied by the source run. |
| `full_market` | Use all rows present in the target-round market snapshot before lock, after applying the declared status policy, valid position, valid club, finite `preco_pre_rodada`, duplicate handling by `id_atleta`, and objective values that are finite after the declared DNP/null policy. |
| `full_market_feature_available` | Same as `full_market`, additionally requiring all profile/feature columns used by the report section. |
| `selected_squad` | Use exactly the persisted `selected_players.csv` rows for a model strategy and round. |

V1 must implement `model_candidate` and `selected_squad`.

`full_market` is allowed only if the runner can construct the market universe from the same data source identity as the source run. Otherwise the report must label it `source_mode=reconstructed` and keep it separate from exact model-candidate diagnostics.

Without a same-source `full_market` or `full_market_feature_available` universe, V1 must not claim candidate-generation failure or “not visible/eligible to the model.” It may only report:

```text
absent_from_model_candidate_artifact
```

When full-market reference is unavailable, fields that depend on it must be null with:

```text
full_market_status=not_available
```

“Pre-round eligible” means:

```text
Rows present in the target-round market snapshot before lock, after applying
the configured status policy and required optimizer columns, before using
target-round outcome fields as an oracle objective.
```

Do not use post-round availability to exclude players from the oracle universe.

## DNP And Actual-Points Policy

Oracle objective values must be finite after applying a declared DNP/null-outcome policy.

Policy:

- If `pontuacao` is finite, use it.
- If `pontuacao` is missing and `entrou_em_campo` is explicitly false, treat the row as a known DNP/no-score outcome and set `oracle_actual_points = 0.0`.
- If `pontuacao` is missing and `entrou_em_campo` is absent from the artifact, mark the row invalid.
- If `pontuacao` is missing and `entrou_em_campo` is missing or ambiguous, mark the row invalid.
- If `pontuacao` is infinite or non-numeric, mark the row invalid.
- Invalid rows are written to an invalid-row artifact and invalidate oracle optimization for that round unless the user explicitly chooses a permissive report mode.

This prevents hindsight exclusion bias where the oracle avoids no-shows by silently dropping them.

## Oracle Result Adapter

The optimizer can be reused, but its result field names are prediction-oriented.

V1 must wrap `SquadOptimizationResult` in an oracle adapter before writing artifacts.

Map:

```text
predicted_points_base          -> oracle_actual_points_base
captain_bonus_predicted        -> oracle_captain_bonus_actual
predicted_points_with_captain  -> oracle_actual_points_with_captain
predicted_points               -> oracle_objective_points
```

Preserve raw optimizer status fields separately:

- `optimizer_status`;
- `optimizer_formation`;
- `optimizer_budget_used`;
- `optimizer_selected_count`;
- `optimizer_captain_id`.

Do not write oracle objective values under `predicted_*` column names.

## Oracle Variants

### 1. Budget-Constrained Oracle

Primary reference.

For each season and target round:

- use the pre-round eligible universe;
- use the same scoring contract as backtests;
- use actual realized points as the optimizer objective;
- keep formation constraints;
- keep captain rules;
- keep tecnico selectable but never captainable;
- enforce the round budget.

This answers:

```text
Given real constraints and perfect hindsight, what was the best squad available?
```

Budget source must be explicit.

V1 required budget path:

- `model_budget_path`: use the model/strategy `budget_before_round`;

Optional later budget paths:

- `initial_budget_path`: replay an oracle from the configured initial budget and update by selected oracle `variacao`;
- `common_reference_budget_path`: use a single frozen budget path for cross-model ceiling comparisons;
- `fixed_reference_budget`: optional analysis-only appendix if needed later.

For cross-model comparison, `model_budget_path` is useful for regret against what that model could afford. It must not be used to rank models by regret because weaker prior budget paths lower the attainable ceiling.

`initial_budget_path` is deferred to v1.1 unless there is a concrete analysis question that requires an independently compounding oracle.

### 2. Model-Candidate Oracle

Primary diagnostic companion.

Use the exact candidate pool available to a model round, but replace the score column with actual realized points.

This separates:

```text
The player was absent from the model candidate artifact.
```

from:

```text
The player was visible but ranked or optimized poorly.
```

### 3. Full-Market Oracle

Secondary ceiling.

Use all pre-round market players with valid position, price, club, and objective values that are finite after the DNP/null policy. This shows whether the model candidate artifact excludes important players.

This is not a fair primary benchmark if the model intentionally filters candidates for safety or data availability.

V1 may omit this if exact source identity cannot be guaranteed from artifacts.

### 4. Unlimited-Budget Oracle

Appendix only.

Ignore the budget constraint while preserving squad size, official formations, club/position constraints, captain rules, and valid pre-round membership.

This is useful for scarcity context:

- which positions dominate pure points;
- how far price constraints pull squads away from the pure-points ceiling;
- which high-scoring profiles are usually unaffordable.

It is not a model benchmark. It must be visually and structurally separated from benchmark charts and placed in an appendix section.

## Existing Optimizer Reuse

The existing `optimize_squad(candidates, score_column, config, budget=...)` should be reused.

Oracle optimization should add a finite numeric score column such as:

```text
oracle_actual_points
```

Then call:

```python
optimize_squad(candidates, score_column="oracle_actual_points", config=config, budget=budget_before_round)
```

This preserves:

- official formation search;
- budget constraint;
- captain multiplier;
- tecnico captain exclusion;
- deterministic tie-break behavior;
- optimizer status reporting.

No separate MILP implementation should be introduced in v1.

After optimization, write adapted oracle columns from the oracle result adapter. Do not expose `predicted_*` optimizer fields as report metrics.

## Pre-Round Feature Profiles

For every oracle-selected player and oracle captain, profile only fields that would have been known before the round:

- season;
- target round;
- position;
- club;
- opponent club;
- home/away;
- fixture mode and matchup context mode;
- pre-round price;
- price percentile/rank by position and overall;
- model predicted points;
- predicted rank by position and overall;
- recent points rolling features;
- prior average/weighted/rolling form;
- club pre-match PPG/xG features;
- opponent pre-match PPG/xG features;
- matchup features;
- status/availability field as captured pre-round;
- selected squad stack counts by club and fixture;
- whether selected players include both teams in the same fixture.

Forbidden profile fields:

- goals/assists/clean sheets from the target round;
- target-round actual points except as the oracle objective and later outcome metric;
- target-round `variacao` except budget-path reporting after selection;
- post-lock lineup information unavailable before the round;
- any field reconstructed from future rounds.

## Comparison Baselines

Raw oracle profile rates are not enough. Every profile statistic should compare against at least one baseline:

1. Full pre-round market.
2. Model candidate pool.
3. Top predicted players by position.
4. Model-selected squad.
5. Budget-feasible profile subsets only if a named, tested computation exists.

Example output:

```text
Oracle attackers home share: 63%
Full candidate attackers home share: 50%
Top-predicted attackers home share: 57%
Model-selected attackers home share: 58%
```

This avoids mistaking common market structure for a meaningful oracle pattern.

## Knowledge Discovery Sections

### Oracle Squad Profile

For each oracle type:

- player count by position;
- home/away share by position;
- price percentile distribution;
- predicted rank distribution;
- recent-form percentile distribution;
- team strength and opponent strength buckets;
- fixture concentration and same-match exposure;
- club stack sizes;
- actual point contribution by position.

### Oracle Captain Profile

For oracle captains:

- position distribution;
- home/away distribution;
- price percentile;
- predicted rank;
- recent form rank;
- team/opponent profile;
- model captain overlap;
- captain actual point gap versus model captain.

### Model Coverage And Recall

For actual top scorers and oracle-selected players:

- whether they were in the full market, only when a same-source full-market universe is available;
- whether they were in the model candidate pool;
- whether they were absent from the model candidate artifact;
- predicted rank overall;
- predicted rank by position;
- whether the optimizer selected them;
- whether they were individually affordable under `budget_before_round`;
- whether a named squad-level counterfactual proves budget blocked them;
- whether candidate data was missing.

### Profile Gap Tables

Compare oracle versus model-selected squads:

- by position;
- by home/away;
- by price band;
- by predicted rank band;
- by recent form band;
- by matchup context bucket;
- by fixture concentration bucket.

### Season Stability

Every highlighted pattern must be reported by season.

A pattern is only “stable” if the direction is consistent in at least `ceil(0.8 * season_count)` seasons. For a five-season run, that means at least four seasons. If a pattern appears only because of one season, the report should label it as unstable.

## Hypothesis Candidates

V1 should not generate free-form hypotheses automatically.

It may emit deterministic `hypothesis_candidates.csv` only if thresholds are predeclared in metadata. Otherwise, the HTML may include a “patterns to inspect” section without assigning hypothesis IDs.

If deterministic hypothesis candidates are enabled, each row must include:

- `hypothesis_id`;
- plain-language statement;
- source metric;
- seasons where observed;
- effect direction;
- stability label;
- leakage risk label;
- proposed future test;
- allowed experiment group;
- triggering threshold;
- sample size;
- effect-size floor;
- season stability count.

Examples:

```text
H001: Home attackers are overrepresented among budget-constrained oracle forwards.
H002: Away defenders are underrepresented except when opponent xG is low.
H003: Oracle captains are usually top-10 predicted players, but model captain misses occur when recent-form volatility is high.
```

Hypotheses are not recommendations. They become candidates for later frozen policy/model experiments.

## Regret Diagnostics

V1 should include non-additive diagnostics, not an additive decomposition.

Allowed diagnostics:

- selected-squad captain oracle;
- model-candidate oracle gap;
- full-market oracle gap;
- model selected points versus oracle selected points;
- candidate-pool recall for oracle players;
- prediction-rank recall for oracle players.

Selected-squad captain oracle formula:

```text
captain_regret =
  0.5 * (best_actual_non_tecnico_in_selected_squad - selected_captain_actual)
```

This is valid because the squad is fixed and only the captain changes.

Do not claim that captain regret, selection regret, budget regret, and formation regret sum to total regret unless a later implementation defines an ordered counterfactual identity and tests it.

## Leakage Guardrails

The report must clearly label itself:

```text
Discovery-only hindsight analysis. Not promotion evidence.
```

Required rules:

- output cannot mark a model or policy as promotable;
- report cannot write to experiment `ranked_summary.csv`;
- report cannot change experiment index promotion fields;
- report cannot run inside `scripts/run_model_experiments.py`;
- generated hypotheses must be frozen before being tested;
- future testing must use a separate experiment generation and predeclared policy/model changes.

Recommended validation workflow for later policy work:

```text
Discovery on 2021-2023.
Freeze hypothesis and policy.
Tune/check on 2024 only if needed.
Validate once on 2025.
Then rerun all 2021-2025 as context, not as sole proof.
```

For very small-sample decisions, use leave-one-season-out reporting as supplementary evidence.

## Artifacts

Recommended output root:

```text
data/08_reporting/oracle_discovery/<run_id>/
```

CSV artifacts:

- `oracle_round_results.csv`
- `oracle_selected_players.csv`
- `oracle_captain_profiles.csv`
- `oracle_player_profiles.csv`
- `model_vs_oracle_recall.csv`
- `profile_gap_summary.csv`
- `hypothesis_candidates.csv` when deterministic thresholds are configured
- `invalid_oracle_rows.csv`

### Required Output Schemas

Shared identity columns:

- `source_mode`;
- `source_experiment_id`;
- `source_child_id`;
- `season`;
- `rodada`;
- `model_id`;
- `feature_pack`;
- `fixture_mode`;
- `matchup_context_mode`;
- `budget_policy`;
- `oracle_type`;
- `candidate_universe`;
- `budget_path`.

`oracle_round_results.csv` required columns:

- shared identity columns;
- `optimizer_status`;
- `optimizer_formation`;
- `optimizer_budget_used`;
- `budget_before_round`;
- `oracle_actual_points_base`;
- `oracle_captain_bonus_actual`;
- `oracle_actual_points_with_captain`;
- `optimizer_captain_id`;
- `optimizer_selected_count`;
- `full_market_status`.

`oracle_selected_players.csv` required columns:

- shared identity columns;
- `id_atleta`;
- `apelido`;
- `posicao`;
- `id_clube`;
- `nome_clube`;
- `preco_pre_rodada`;
- `oracle_actual_points`;
- `is_oracle_captain`;
- `model_score_column`;
- `model_score`;
- `model_predicted_rank_overall`;
- `model_predicted_rank_position`;
- `entrou_em_campo`;
- `status`.

`oracle_captain_profiles.csv` required columns:

- shared identity columns;
- `captain_id`;
- `captain_name`;
- `captain_position`;
- `captain_club`;
- `captain_status`;
- `captain_is_home`;
- `captain_price_percentile_position`;
- `captain_price_rank_position`;
- `captain_model_score`;
- `captain_model_predicted_rank_overall`;
- `captain_model_predicted_rank_position`;
- `captain_recent_form_percentile_position`;
- `captain_oracle_actual_points`;
- `model_captain_id`;
- `model_captain_actual_points`;
- `selected_squad_captain_regret`;

`oracle_player_profiles.csv` required columns:

- shared identity columns;
- `id_atleta`;
- `posicao`;
- `profile_section`;
- `profile_metric`;
- `profile_value`;
- `baseline_name`;
- `baseline_value`;
- `sample_size`;
- `full_market_status`;

`model_vs_oracle_recall.csv` required columns:

- shared identity columns;
- `id_atleta`;
- `posicao`;
- `in_selected_squad`;
- `in_model_candidate_artifact`;
- `absent_from_model_candidate_artifact`;
- `in_full_market`;
- `full_market_status`;
- `model_predicted_rank_overall`;
- `model_predicted_rank_position`;
- `individually_affordable`;
- `squad_budget_blocked_by_counterfactual`;
- `recall_bucket`.

`profile_gap_summary.csv` required columns:

- `source_mode`;
- `season`;
- `model_id`;
- `feature_pack`;
- `oracle_type`;
- `candidate_universe`;
- `budget_path`;
- `profile_section`;
- `profile_metric`;
- `oracle_value`;
- `baseline_name`;
- `baseline_value`;
- `absolute_gap`;
- `relative_gap`;
- `sample_size`;
- `season_stability_count`;
- `stability_label`;
- `full_market_status`.

HTML artifacts:

- `oracle_knowledge_discovery.html`

Metadata:

- `oracle_discovery_metadata.json`

Required metadata fields:

- seasons;
- start round;
- current year;
- initial budget;
- budget policy;
- oracle variants enabled;
- candidate universe definitions;
- scoring contract version;
- fixture mode;
- matchup context mode;
- source experiment IDs, if reading existing experiment artifacts;
- source mode;
- source artifact paths;
- source hashes or metadata references;
- generated timestamp;
- git SHA if available;
- dependency versions if available.

## Acceptance Criteria

Before trusting v1:

1. Perfect prediction test:
   - if `predicted_points == actual_points` on a synthetic round, the model-candidate oracle and model optimizer produce equal objective value.

2. Captain test:
   - selected-squad captain oracle never chooses `tec`.

3. Budget isolation test:
   - budget-constrained oracle uses the specified `budget_before_round`, not an internally mutated budget unless explicitly running `initial_budget_path`.

4. Universe separation test:
   - full-market oracle and model-candidate oracle can produce different ceilings when the model candidate pool excludes a high scorer.

5. Missing actual points test:
   - oracle objective values are finite after applying the DNP/null policy; explicit DNP nulls become zero, while ambiguous missing values produce invalid rows or strict-mode errors.

6. Metadata test:
   - every output row includes season, round, oracle type, candidate universe, budget policy, fixture mode, matchup context mode, scoring contract version, and optimizer status.

7. Isolation test:
   - normal model experiments do not import or execute oracle discovery logic.

8. Season-stability test:
   - highlighted profile patterns include per-season rows and an instability label when driven by one season.

9. Source artifact schema test:
   - missing required files or columns fail in strict mode and render explicit incomplete sections in permissive mode.

10. Source mode labeling test:
   - artifact rows carry `source_mode=artifact`; reconstructed rows carry `source_mode=reconstructed`, code SHA, and source hashes.

11. DNP/null test:
   - explicit DNP rows with missing `pontuacao` become `oracle_actual_points=0.0`; ambiguous null rows are invalid.

12. Oracle result adapter test:
   - oracle outputs contain `oracle_actual_points_*` columns and do not write objective values under `predicted_*` names.

13. Budget path continuity test:
   - when `initial_budget_path` is later enabled, missing selected-player `variacao` invalidates the oracle path instead of silently using zero.

14. Source context test:
   - a child run containing `baseline_score`, `price_score`, and `{model_id}_score` must use only the score column declared in `source_run_context`; no implicit score-column guessing is allowed.

15. Moving-budget compatibility test:
   - old artifacts missing `budget_before_round` or `budget_policy=moving` fail with an explicit “not moving-budget compatible” error.

16. Full-market availability test:
   - when no same-source full-market universe is available, full-market-dependent fields are null with `full_market_status=not_available`, and the report does not use candidate-generation-failure language.

17. Permissive-mode section test:
   - missing profile columns disable the affected section and mark it unavailable instead of rendering zero-row metrics as if they were real.

## Final Design Decision

Implement a narrowed discovery-only oracle report.

Primary value comes from profiling hindsight-best squads against pre-round-known context and comparing those profiles against candidate/model baselines.

V1 focuses on exact artifact-backed counterfactual diagnostics and conservative profile tables. Do not implement policy learning, neural models, or trait-driven optimizer penalties in this feature. Those belong in later frozen experiments derived from reviewed hypothesis candidates.
