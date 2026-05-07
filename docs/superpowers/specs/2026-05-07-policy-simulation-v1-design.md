# Policy Simulation V1 Design

## Goal

Build a frozen policy-simulation workflow for hypothesis H001: opponent-overlap exposure.

The workflow should answer one narrow question:

```text
If we reduce selected-player exposure to both sides of the same match, do historical moving-budget results improve when predictions and candidate pools are held constant?
```

This is a hypothesis test for optimizer policy behavior. It is not a model-training workflow and not production promotion evidence by itself.

## Source Hypothesis

H001 is defined in:

```text
docs/research/policy_hypotheses.md
```

The frozen hypothesis says:

- model-selected squads average more selected assets in opponent-overlap situations than oracle squads;
- reducing opponent-overlap exposure may improve realized points or reduce downside risk;
- V1 should compare `no_policy`, two soft penalties, and two hard caps.

The policy simulation must not revise H001 acceptance criteria after seeing results. If thresholds or variants need changing, create a new hypothesis generation.

## Scope

V1 creates a standalone artifact-backed policy simulation runner.

Recommended entrypoint:

```text
scripts/run_policy_simulation.py
```

Recommended implementation modules:

```text
src/cartola/backtesting/policy_simulation.py
src/cartola/backtesting/optimizer_policies.py
```

V1 reads completed model experiment artifacts and re-optimizes squads under explicit policy variants.

V1 must support selecting a narrow source slice:

```text
--experiment-path data/08_reporting/experiments/model_feature/<experiment_id>
--hypothesis-id H001
--models xgboost_depth2_slow
--feature-packs ppg_xg_matchup
--policy-set opponent-overlap-v1
```

The recommended first run is:

```text
model_id=xgboost_depth2_slow
feature_pack=ppg_xg_matchup
seasons=2021,2022,2023,2024,2025
```

Optional controls may include:

```text
model_id=ridge
feature_pack=ppg_xg_matchup
```

The runner should not default to replaying every child in a large experiment. Full-matrix replay is too slow and makes policy conclusions harder to interpret.

## Non-Goals

Do not build:

- a new predictive model;
- an Optuna or adaptive policy tuner;
- reinforcement learning;
- automatic production-policy promotion;
- a broad policy-mining engine;
- a live recommendation change;
- full-market oracle analysis;
- new matchup features.

V1 tests one optimizer-policy axis only: opponent-overlap exposure.

## Research Framework

The framework is hypothesis-driven policy research:

1. Discovery: oracle diagnostics reveal a pattern.
2. Hypothesis: H001 freezes the claim and criteria.
3. Translation: policy variants encode the claim as optimizer behavior.
4. Simulation: historical predictions are replayed under each policy.
5. Decision: results mark H001 as rejected, candidate policy, or needs more evidence.

The runner should write a decision artifact, but it must not edit `docs/research/policy_hypotheses.md` automatically.

## Source Artifact Contract

V1 is artifact-backed. It must not retrain models or rebuild candidate frames.

For each selected child run, require:

- `player_predictions.csv`;
- `round_results.csv`;
- `selected_players.csv`;
- `run_metadata.json`;
- parent experiment metadata sufficient to identify `model_id`, `feature_pack`, `fixture_mode`, `matchup_context_mode`, `budget_policy`, and score-column mapping.

Required `player_predictions.csv` columns:

- `rodada`;
- `id_atleta`;
- `apelido`;
- `posicao`;
- `id_clube`;
- `nome_clube`;
- `preco_pre_rodada`;
- `pontuacao`;
- `entrou_em_campo`;
- `variacao`;
- primary score column for the selected model strategy.

Required `round_results.csv` columns:

- `rodada`;
- `strategy`;
- `solver_status`;
- `budget_before_round`;
- `budget_after_round`;
- `budget_delta`;
- `budget_used`;
- `actual_points_with_captain`;
- `predicted_points_with_captain`;
- `captain_id`;
- `scoring_contract_version` if present.

Required `selected_players.csv` columns:

- `rodada`;
- `strategy`;
- `id_atleta`;
- `id_clube`;
- `posicao`;
- `preco_pre_rodada`;
- `pontuacao`;
- `entrou_em_campo`;
- `variacao`;
- `is_captain`.

Required source metadata:

- `season`;
- `start_round`;
- `initial_budget`;
- `budget_policy=moving`;
- `fixture_mode`;
- `matchup_context_mode`;
- `scoring_contract_version=cartola_standard_2026_v1`;
- primary model id;
- feature pack;
- strategy roles or explicit score-column mapping;
- fixture identity metadata for every replayed round.

If parent experiment metadata provides strategy roles but not explicit score columns, V1 uses this deterministic score-column mapping:

```text
baseline -> baseline_score
price -> price_score
primary model strategy -> {model_id}_score
```

Every mapped score column must exist in `player_predictions.csv`.

Candidate-pool signatures must be computed from persisted `player_predictions.csv` rows after duplicate normalization, using sorted rows and stable JSON hashing over:

```text
rodada, id_atleta, id_clube, posicao, preco_pre_rodada, primary_score_column
```

If parent experiment metadata already stores candidate-pool signatures, the computed signatures must match those stored values. If stored signatures are missing, the simulation may compute signatures for internal policy comparability, but the manifest must record `source_candidate_signature_status=computed_from_artifact`.

If any required artifact or column is missing, fail before running simulations.

Old fixed-budget artifacts are ineligible.

## Fixture Contract

Opponent overlap must be computed from fixture rows proven to match the source experiment fixture identity.

For exploratory fixture runs, load fixtures with the existing historical fixture loader:

```text
cartola.backtesting.data.load_fixtures(season, project_root)
```

For strict fixture runs, load strict fixtures from the strict fixture artifact path and validate strict manifests before replay. The implementation should use the existing strict fixture loading/validation boundary rather than `load_fixtures`.

Required fixture columns:

- `rodada`;
- `id_clube_home`;
- `id_clube_away`.

### Fixture Identity

For each replayed season and round, compute a fixture signature from sorted fixture rows using:

```text
rodada, id_clube_home, id_clube_away
```

The hash algorithm is SHA-256 over canonical JSON records with integer club and round IDs.

The manifest and comparability report must store:

- `fixture_identity_status`;
- computed fixture signatures by season and round;
- source fixture signatures when available;
- strict manifest hashes when fixture mode is `strict`;
- fixture coverage status by season and round.

For strict fixture runs, the source strict manifest hash must match the current strict manifest hash. If not, fail.

For exploratory fixture runs, the source experiment should contain fixture source path/hash metadata. If the source experiment has enough fixture hashes, current fixture signatures must match the stored source signatures.

If the source experiment lacks fixture hashes, mark:

```text
fixture_identity_status=unverified
```

In that case the policy simulation may still run as diagnostic evidence, but:

- `candidate_policy` decisions are suppressed;
- the ranked summary uses `decision_status=diagnostic_only`;
- the HTML report must show that fixture identity is unverified;
- results must not be used as H001 acceptance evidence.

If the source child used `fixture_mode=strict`, the policy simulation must require strict fixture metadata and fail if strict evidence is missing.

The policy simulation must not silently fall back from strict to exploratory.

### Fixture Coverage

For H001, missing fixture coverage is not the same as a verified no-fixture state.

A candidate club contributes zero overlap only when either:

- the validated round fixture source is marked complete and the club is absent from both `id_clube_home` and `id_clube_away`; or
- the source run used an explicitly declared neutral/no-fixture policy for that round and the report is marked non-decision evidence.

If fixture coverage is missing for any candidate or selected club in a replayed round, mark the round invalid for H001.

The policy simulation must write missing fixture coverage into `policy_comparability_report.json` and suppress policy decisions for affected child runs.

Fixture coverage validation must also fail when a club appears in more than one fixture in the same round. H001 assumes at most one fixture per club per round.

## Opponent-Overlap Definition

For each target round, a selected asset is in opponent overlap when:

1. its selected `id_clube` appears in a fixture for that round;
2. the opposing club in that same fixture also has at least one selected asset.

V1 counts all selected assets with a club identity, including `tec`.

Rationale:

- moving-budget and scoring semantics include tecnico;
- coach points are team-outcome exposure;
- excluding tecnico would make policy accounting differ from selected-squad accounting.

The report may also include a non-tecnico diagnostic later, but V1 policy constraints use all selected assets.

Example:

```text
Selected clubs in one round:
  Flamengo: 2 assets
  Palmeiras: 1 asset
  Gremio: 2 assets

Fixtures:
  Flamengo vs Palmeiras
  Gremio vs Santos

Opponent-overlap assets:
  Flamengo assets: 2
  Palmeiras assets: 1
  Gremio assets: 0

avg_players_in_opponent_overlap contribution:
  3
```

## Policy Set

V1 policy set:

```text
opponent-overlap-v1
```

It includes five variants:

```text
no_policy
soft_overlap_penalty_low
soft_overlap_penalty_medium
hard_max_overlap_3
hard_max_overlap_2
```

### no_policy

The current optimizer behavior.

Acceptance requirement:

For every selected source child, `no_policy` should reproduce the original source selected squad and round result for optimal rounds according to the reproduction contract in the Comparability section.

If `no_policy` does not reproduce the source run, mark the child invalid and stop. This protects the simulation from source-context drift.

### soft_overlap_penalty_low

Objective:

```text
maximize predicted_points_with_captain - 0.15 * opponent_overlap_asset_count
```

### soft_overlap_penalty_medium

Objective:

```text
maximize predicted_points_with_captain - 0.35 * opponent_overlap_asset_count
```

### hard_max_overlap_3

Constraint:

```text
opponent_overlap_asset_count <= 3
```

### hard_max_overlap_2

Constraint:

```text
opponent_overlap_asset_count <= 2
```

## Optimizer Design

The existing `optimize_squad()` function should remain the default public optimizer.

V1 should add a policy-aware optimizer wrapper or optional internal policy argument without changing normal backtest behavior.

Recommended interface:

```text
optimize_squad_with_policy(
    candidates,
    score_column,
    config,
    budget,
    policy,
    fixtures_for_round,
)
```

The default policy must be equivalent to `no_policy`.

The policy-aware optimizer should:

- search the same official formations;
- keep the same budget constraint;
- keep the same captain rule and 1.5x multiplier;
- use the same primary predicted score column;
- apply policy penalties or constraints inside each formation solve;
- keep deterministic tie-break behavior after the policy-adjusted primary objective is fixed.

The policy implementation must not mutate candidate rows in place.

### Linearization Requirement

Opponent overlap must be modeled inside the MILP, not approximated after the fact.

For every fixture in the target round:

- create a binary variable meaning at least one selected asset from the home club;
- create a binary variable meaning at least one selected asset from the away club;
- create a binary variable meaning both sides are selected;
- create selected-overlap variables for assets from either club when both sides are selected.

`opponent_overlap_asset_count` is the sum of selected-overlap variables across target-round fixtures.

Soft policies subtract a fixed weight times this count from the objective.

Hard policies constrain this count.

If a candidate club has a verified no-fixture state for the round, it contributes no opponent-overlap variables and is not penalized.

If fixture coverage is missing or unverified for the club, the round is invalid for H001 policy decisions.

### Required MILP Constraints

For each target-round fixture `f` with home club `H` and away club `A`:

```text
home_count_f = sum(selected_i for candidate i where id_clube == H)
away_count_f = sum(selected_i for candidate i where id_clube == A)
home_present_f binary
away_present_f binary
both_sides_selected_f binary
```

Let `M` be the maximum possible selected assets from one club in a squad. V1 can use the squad size from the formation, including tecnico.

Presence constraints:

```text
home_count_f >= home_present_f
home_count_f <= M * home_present_f
away_count_f >= away_present_f
away_count_f <= M * away_present_f
```

These force `home_present_f` and `away_present_f` to be `1` exactly when at least one asset from that side is selected.

Both-sides constraints:

```text
both_sides_selected_f <= home_present_f
both_sides_selected_f <= away_present_f
both_sides_selected_f >= home_present_f + away_present_f - 1
```

These force `both_sides_selected_f` to be `1` exactly when both fixture sides have selected assets.

For each candidate `i` whose club is `H` or `A`, create:

```text
overlap_i_f binary
```

Overlap constraints:

```text
overlap_i_f <= selected_i
overlap_i_f <= both_sides_selected_f
overlap_i_f >= selected_i + both_sides_selected_f - 1
```

These force `overlap_i_f` to be `1` exactly when candidate `i` is selected and both sides of fixture `f` are selected.

Then:

```text
opponent_overlap_asset_count = sum(overlap_i_f)
opponent_overlap_match_count = sum(both_sides_selected_f)
```

The implementation must include a synthetic hard-cap test where the unconstrained optimum selects players from both fixture sides, `hard_max_overlap_2` forces a different selected set, and the solver cannot satisfy the cap by leaving overlap variables at zero.

## Moving-Budget Semantics

Each policy variant has its own budget path.

For every selected source child and policy variant:

```text
current_budget starts at source initial_budget.

For each target round in order:
  read persisted player_predictions.csv rows for that round;
  optimize under current_budget and the policy variant;
  score actual captain-aware points from selected historical pontuacao;
  budget_delta = sum(selected variacao);
  current_budget += budget_delta;
```

Budget update happens only after selection and scoring.

Every selected asset in an optimal round must have finite numeric `variacao`. If not, mark the policy run invalid.

If a policy variant is infeasible for a round:

- write the round with `solver_status` from the optimizer;
- carry the budget unchanged;
- count it as a non-optimal policy round;
- make that policy variant ineligible unless the same round was also non-optimal under `no_policy`.

## Scoring Semantics

Use the same scoring contract as the source run:

```text
cartola_standard_2026_v1
```

Round totals must use captain-aware actual points:

```text
actual_points_with_captain =
  sum(selected pontuacao) + 0.5 * selected_captain_pontuacao
```

The tecnico cannot be captain.

Missing or non-finite `pontuacao` in selected assets must follow the existing historical scoring semantics. If the artifact cannot distinguish DNP from corrupt missing values, mark the round invalid rather than silently dropping rows.

V1 requires `entrou_em_campo` in `player_predictions.csv`. If a policy variant selects a row with missing `pontuacao`:

- when `entrou_em_campo` explicitly indicates a true no-score DNP under the source scoring contract, score it as the contract defines;
- when `entrou_em_campo` is missing or ambiguous, invalidate the round.

## Comparability

Policy variants are comparable only when:

- source experiment id matches;
- source child id matches;
- model id matches;
- feature pack matches;
- fixture mode matches;
- matchup mode matches;
- budget policy is `moving`;
- scoring contract matches;
- target rounds match;
- candidate-pool signature matches for every replayed round;
- prediction score column matches;
- fixture signature matches;
- fixture coverage is complete for replayed candidate and selected clubs;
- `no_policy` reproduces the source run.

`no_policy` reproduction means:

- same selected `id_atleta` set;
- same captain id;
- same formation;
- same budget used within tolerance;
- same predicted captain-aware objective within tolerance;
- same actual captain-aware points within tolerance.

Numeric tolerances:

```text
budget_used tolerance: 1e-6
predicted captain-aware objective tolerance: 1e-6
actual captain-aware points tolerance: 1e-6
```

The policy simulation should write a fail-closed comparability report. Rankings should be suppressed when comparability fails.

## CLI Design

Recommended command:

```bash
uv run --frozen python scripts/run_policy_simulation.py \
  --experiment-path data/08_reporting/experiments/model_feature/<experiment_id> \
  --hypothesis-id H001 \
  --policy-set opponent-overlap-v1 \
  --models xgboost_depth2_slow \
  --feature-packs ppg_xg_matchup \
  --current-year 2026
```

Optional arguments:

```text
--seasons 2021,2022,2023,2024,2025
--fixture-mode exploratory
--matchup-context-mode cartola_matchup_v1
--output-root data/08_reporting/policy_simulations
--allow-incomplete-report
```

The command should show Rich progress:

- total selected child runs;
- current child;
- current policy variant;
- current season and round;
- elapsed time;
- output path;
- failure reason.

No silent long-running command is acceptable.

## Output Directory

Default output root:

```text
data/08_reporting/policy_simulations/
```

Run directory:

```text
data/08_reporting/policy_simulations/policy_simulation_started_at=<timestamp>__hypothesis=H001__policy_set=opponent-overlap-v1/
```

## Output Artifacts

Required artifacts:

```text
policy_simulation_manifest.json
policy_ranked_summary.csv
policy_per_season_summary.csv
policy_round_results.csv
policy_selected_players.csv
policy_profile_summary.csv
policy_comparability_report.json
policy_simulation_report.html
```

### policy_simulation_manifest.json

Required fields:

- `simulation_id`;
- `hypothesis_id`;
- `hypothesis_registry_path`;
- `policy_set`;
- `policy_variants`;
- `experiment_path`;
- `source_experiment_id`;
- `selected_models`;
- `selected_feature_packs`;
- `selected_seasons`;
- `scoring_contract_version`;
- `budget_policy`;
- `initial_budget`;
- `fixture_mode`;
- `matchup_context_mode`;
- `source_hashes`;
- `policy_definitions`;
- `created_at`.

### policy_round_results.csv

Required columns:

- identity columns:
  - `simulation_id`;
  - `hypothesis_id`;
  - `source_experiment_id`;
  - `source_child_id`;
  - `season`;
  - `rodada`;
  - `model_id`;
  - `feature_pack`;
  - `fixture_mode`;
  - `matchup_context_mode`;
  - `policy_variant`;
- optimizer columns:
  - `solver_status`;
  - `formation`;
  - `selected_count`;
  - `captain_id`;
  - `budget_before_round`;
  - `budget_used`;
  - `budget_remaining`;
  - `budget_delta`;
  - `budget_after_round`;
  - `predicted_points_with_captain`;
  - `actual_points_with_captain`;
  - `opponent_overlap_asset_count`;
  - `opponent_overlap_match_count`;
  - `same_club_max_selected_count`;
  - `infeasibility_reason`;

### policy_selected_players.csv

Required columns:

- all identity columns from `policy_round_results.csv`;
- `id_atleta`;
- `apelido`;
- `posicao`;
- `id_clube`;
- `nome_clube`;
- `preco_pre_rodada`;
- `pontuacao`;
- `variacao`;
- `is_captain`;
- `predicted_points`;
- `actual_points_with_captain`;
- `opponent_overlap_in_lineup`;
- `same_club_selected_count`.

### policy_ranked_summary.csv

Required columns:

- `policy_variant`;
- `model_id`;
- `feature_pack`;
- `total_actual_points`;
- `delta_vs_no_policy`;
- `avg_delta_per_round_vs_no_policy`;
- `improved_seasons`;
- `regressed_seasons`;
- `season_count`;
- `optimal_round_count`;
- `non_optimal_round_count`;
- `initial_budget`;
- `final_budget`;
- `min_budget`;
- `max_budget_drawdown`;
- `budget_constrained_rounds`;
- `avg_opponent_overlap_asset_count`;
- `avg_opponent_overlap_match_count`;
- `decision_status`;
- `decision_reason`.

### policy_per_season_summary.csv

Required columns:

- `season`;
- `policy_variant`;
- `model_id`;
- `feature_pack`;
- `total_actual_points`;
- `delta_vs_no_policy`;
- `optimal_round_count`;
- `non_optimal_round_count`;
- `initial_budget`;
- `final_budget`;
- `min_budget`;
- `max_budget_drawdown`;
- `avg_opponent_overlap_asset_count`;
- `avg_opponent_overlap_match_count`.

### policy_profile_summary.csv

Required metrics:

- `opponent_overlap_round_rate`;
- `avg_players_in_opponent_overlap`;
- `avg_same_club_selected_count`;
- `home_player_share` if fixture home/away context is available;
- `captain_opponent_overlap_rate`;
- `captain_actual_points_share`;
- `single_round_delta_concentration`.

### policy_comparability_report.json

Required fields:

- `status`;
- `failure_reasons`;
- `candidate_pool_signatures`;
- `prediction_score_column_signatures`;
- `fixture_signatures`;
- `source_reproduction_status`;
- `solver_status_signatures`;
- `budget_policy`;
- `scoring_contract_version`.

### policy_simulation_report.html

The HTML report should include:

- ranked policy summary;
- per-season bars for actual points and delta versus `no_policy`;
- budget-path chart per policy variant;
- opponent-overlap exposure chart;
- infeasible/non-optimal rounds table;
- captain contribution diagnostic;
- decision table for H001 acceptance/rejection criteria.

The report must show:

- `hypothesis_id`;
- `policy_set`;
- source experiment id;
- model/feature slice;
- fixture mode;
- budget policy;
- warning that this is research evidence only.

## Decision Logic

The runner should compute a decision per policy variant:

```text
candidate_policy
rejected
needs_more_data
diagnostic_only
ineligible
```

A policy simulation may emit `candidate_policy` or `rejected` only when the selected seasons exactly match H001 generation 1:

```text
2021,2022,2023,2024,2025
```

Any other season slice, including smoke tests, must emit `diagnostic_only` or `needs_more_data`.

A variant is `ineligible` if:

- comparability fails;
- `no_policy` source reproduction fails;
- fixture coverage is missing for replayed candidate or selected clubs;
- required metrics are missing;
- non-optimal rounds increase versus `no_policy`;
- selected assets have missing `variacao` in optimal rounds.

When `fixture_identity_status=unverified` but all other validation checks pass, variants are `diagnostic_only`, not `ineligible`.

A variant is `candidate_policy` only if it satisfies all H001 acceptance criteria:

- total actual points improve versus `no_policy`;
- at least `3` of `5` seasons improve;
- 2025 does not materially regress;
- non-optimal rounds do not increase;
- final budget, min budget, and max drawdown do not materially worsen;
- no one or two rounds explain most of the total lift.

For V1, define "materially regress 2025" as:

```text
2025 delta_vs_no_policy < -25 total points
```

Define "materially worsen budget" as any of:

```text
final_budget_delta_vs_no_policy < -5
min_budget_delta_vs_no_policy < -5
max_drawdown_delta_vs_no_policy > 5
```

Define "one or two rounds explain most lift" as:

```text
top_2_round_delta_sum / total_positive_delta > 0.50
```

Where:

```text
round_delta = policy actual_points_with_captain - no_policy actual_points_with_captain
positive_round_delta = max(round_delta, 0)
total_positive_delta = sum(positive_round_delta across all replayed seasons and rounds)
top_2_round_delta_sum = sum(two largest positive_round_delta values across all replayed seasons and rounds)
```

If `total_positive_delta <= 0`, the concentration metric is `NA` and the variant cannot be `candidate_policy` because the total lift criterion already fails.

These thresholds are part of H001 generation 1. Changing them after results requires H001 generation 2.

## Testing Strategy

Unit tests:

- policy definitions are frozen and serialized into the manifest;
- opponent-overlap count matches hand-built fixture examples;
- soft penalty changes objective without changing feasibility constraints;
- hard caps can make an otherwise feasible squad infeasible;
- tecnico is included in overlap counting but excluded from captain eligibility;
- verified no-fixture clubs produce zero overlap penalty without invalidating the round;
- fixture-drift between source metadata and current fixture rows suppresses policy decisions;
- missing fixture coverage invalidates H001 rounds rather than applying zero penalty;
- missing required artifact columns fail before simulation;
- fixed-budget source artifacts are rejected;
- `no_policy` reproduction mismatch fails comparability;
- missing selected `variacao` invalidates optimal rounds;
- decision logic rejects variants that improve aggregate but fail 2025 threshold;
- decision logic rejects variants whose lift is dominated by top two rounds.
- non-H001 season slices emit `diagnostic_only` instead of `candidate_policy` or `rejected`.

Integration smoke test:

- build a tiny synthetic source experiment with two seasons, two rounds, and one fixture per round;
- run `no_policy`, one soft policy, and one hard policy;
- assert all required artifacts exist;
- assert HTML contains policy labels and H001 warning text;
- assert moving budget updates independently per policy variant.

Real artifact acceptance:

- run against the completed `xgboost-sensitivity-v2` experiment slice for `xgboost_depth2_slow + ppg_xg_matchup`;
- confirm `no_policy` reproduces source selected squads;
- confirm reports write without invalid rows;
- inspect `policy_ranked_summary.csv` and `policy_simulation_report.html`.

## Performance Notes

The oracle discovery full run showed repeated CBC solves can take about an hour on a large experiment.

Policy simulation V1 should keep runtime bounded by:

- requiring explicit model/feature selectors;
- defaulting to one model/feature slice;
- showing progress for every policy/season/round;
- avoiding full experiment matrices unless explicitly requested.

Future optimization options:

- cache per-round fixture-policy variables;
- stream artifact rows instead of buffering all rows;
- support multiprocessing across independent child runs only after sequential moving-budget paths are preserved per policy variant.

## Success Criteria

The design is successful when the implementation can answer:

```text
For H001 generation 1, did any opponent-overlap policy variant beat no_policy under moving-budget replay, without unacceptable 2025, feasibility, or budget regressions?
```

If yes, the policy becomes a candidate policy for a fresh validation experiment.

If no, H001 is rejected or marked needs-more-data.

No live default changes as part of V1.
