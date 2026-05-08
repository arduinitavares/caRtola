# H004 Attack-vs-Defense Mismatch Design

## Summary

H004 tests whether the current best matchup model is missing predictive signal in
pre-match attack-vs-defense context.

This is not an optimizer-policy experiment. H001, H002, and H003 showed that
simple optimizer constraints or bonuses are too blunt. H004 moves the question
back into the model: can a small, frozen set of prior-only interaction features
improve expected player-point prediction and then improve optimized squads under
moving-budget replay?

The work has two phases:

1. Run a residual diagnostic against the current control artifacts.
2. Implement one frozen H004 feature pack only if the diagnostic shows residual
   signal worth testing.

No H004 output may change live defaults, optimizer policy, experiment promotion
fields, or recommendation defaults.

## Motivation

The user observed selected squads containing players from opposing teams and
wanted a data-driven way to understand whether football context should influence
selection. Three fixture-verified policy simulations were then tested:

- H001 opponent-overlap exposure: rejected.
- H002 goalkeeper-vs-opponent-attack exposure: rejected.
- H003 clean-sheet defensive-stack exposure: rejected.

These results do not prove football context is useless. They show that simple
selection-level rules are unstable. H004 tests a narrower claim: the model may
need better continuous context features, especially around team attacking
strength, opponent defensive weakness, home/away context, and player position.

## Source Control And Evidence Boundary

The control must be rerun side-by-side with H004 in the same experiment matrix.
Do not compare H004 against old experiment artifacts.

Required shared conditions:

- seasons: `2021,2022,2023,2024,2025`;
- start round: `5`;
- budget policy: `moving`;
- initial budget: `100`;
- scoring contract: `cartola_standard_2026_v1`;
- fixture mode: `exploratory` for historical research unless a strict historical
  source exists;
- model control: `xgboost_depth2_slow + ppg_xg_matchup`;
- candidate-pool signatures must match between control and H004;
- raw Cartola source identity, FootyStats source identity, fixture source paths,
  and fixture hashes must be recorded.

Exploratory fixture evidence is research evidence. It is not a live-default
promotion on its own.

## Phase 1: Residual Diagnostic

### Goal

Before adding features, test whether the existing control model misses in a way
that correlates with already-available matchup and FootyStats context.

### Input

Use the completed source experiment for the current best control when available:

```bash
data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d
```

The diagnostic reads only persisted artifacts:

- `player_predictions.csv`;
- `selected_players.csv`;
- `round_results.csv`;
- child `run_metadata.json`;
- parent experiment metadata/index rows when needed.

If required context columns are missing, the diagnostic fails explicitly.

### Required Columns

`player_predictions.csv` must contain:

- identity: `season`, `rodada`, `id_atleta`, `posicao`, `id_clube`;
- prediction: primary model score column from validated source context;
- outcome: `pontuacao`, `entrou_em_campo`;
- existing context:
  - `matchup_is_home`;
  - `footystats_xg_diff`;
  - `footystats_ppg_diff`;
  - `matchup_opponent_allowed_points_roll5`;
  - `matchup_opponent_allowed_position_points_roll5`;
  - `matchup_club_position_points_roll5`;
  - `matchup_opponent_allowed_position_count`;
  - `matchup_club_position_count`.

### Diagnostic Metrics

For each season and position, compute residuals:

```text
prediction_residual = actual_pontuacao - predicted_points
```

Use only rows where the player actually entered the field. Explicit DNP rows are
excluded from residual-correlation statistics but counted in a separate DNP
context table.

Compute:

- Spearman correlation between residual and each existing context column;
- mean residual by quintile for each context column;
- top-actual-player recall:
  - actual top 5 by position per round;
  - their median predicted rank;
  - their median context values;
- selected-player residual profile versus all model candidates.

The diagnostic is descriptive only. It cannot mark a model or policy as
promotable.

### Diagnostic Decision Gate

H004 feature implementation proceeds only if at least one of these is true:

- at least three of five seasons show same-direction residual correlation for
  `footystats_xg_diff` or `matchup_opponent_allowed_position_points_roll5` in
  attacker/midfielder positions;
- at least three of five seasons show same-direction residual correlation for
  defensive positions with home/xG context;
- actual top-5 position scorers are consistently ranked worse by prediction than
  their context profile suggests, in at least three seasons.

If the diagnostic does not pass, stop H004 and record it as a null discovery
result. Do not implement the feature pack.

## Phase 2: Frozen Feature Pack

If Phase 1 passes, add exactly one new feature pack:

```text
feature_pack = ppg_xg_matchup_h004
```

This is a model feature pack, not a policy set. It must not change candidate
filtering, optimizer constraints, score contracts, or live defaults.

### H004 Feature Columns

H004 v1 adds exactly four model columns:

```text
h004_position_softness_delta
h004_position_mismatch_score
h004_home_xg_edge
h004_role_xg_mismatch
```

No additional variants, thresholds, or alternative transforms are part of H004
v1.

### Formula Definitions

Let:

```text
position_prior = prior-only mean points for this position before target round
club_position = matchup_club_position_points_roll5
opponent_position_allowed = matchup_opponent_allowed_position_points_roll5
opponent_all_allowed = matchup_opponent_allowed_points_roll5
xg_diff = footystats_xg_diff
is_home = matchup_is_home
```

Then:

```text
h004_position_softness_delta =
  opponent_position_allowed - opponent_all_allowed

h004_position_mismatch_score =
  (club_position - position_prior) + (opponent_position_allowed - position_prior)

h004_home_xg_edge =
  is_home * xg_diff

h004_role_xg_mismatch =
  xg_diff * h004_position_softness_delta
```

For `tec`, set all four H004 columns to `0.0`.

### Fallback And Missingness Rules

H004 must not introduce NaN or infinite feature values.

Fallbacks reuse existing matchup v1 semantics:

- missing opponent allowed position points fall back to opponent allowed all
  points, then position prior, then global prior;
- missing club position points fall back to position prior, then global prior;
- missing count columns become `0`;
- if FootyStats xG context is missing for a candidate club-round, the run fails
  instead of silently filling H004 values.

No clipping is applied in H004 v1. If extreme values are found, the diagnostic
must report them but the feature formulas remain frozen for that generation.

## Experiment Design

Add a side-by-side research group:

```text
group = h004-attack-defense-mismatch
```

The matrix contains exactly:

- control: `xgboost_depth2_slow + ppg_xg_matchup`;
- challenger: `xgboost_depth2_slow + ppg_xg_matchup_h004`.

Run:

```bash
uv run --frozen python scripts/run_model_experiments.py \
  --group h004-attack-defense-mismatch \
  --seasons 2021,2022,2023,2024,2025 \
  --start-round 5 \
  --budget 100 \
  --current-year 2026 \
  --jobs 12 \
  --profile-runtime
```

The experiment runner must write normal model-feature artifacts and comparability
reports. H004 does not need a separate custom runner unless the existing
experiment runner cannot express the two-row matrix cleanly.

## Acceptance Gates

H004 becomes a candidate research profile only if all gates pass:

- `comparability_report.status == "ok"`;
- candidate-pool signatures match control for every season and target round;
- no additional non-optimal, infeasible, skipped, or budget-constrained rounds
  versus control;
- aggregate total actual point delta is at least `+85`;
- at least `4 / 5` seasons improve;
- worst season delta is no worse than `-20`;
- 2025 delta is no worse than `-10`;
- final budget delta is nonnegative in aggregate;
- no season final-budget delta is worse than `-2`;
- top-50 Spearman delta is nonnegative in at least `4 / 5` seasons;
- selected-player calibration slope remains between `0.8` and `1.2` where
  sample size is sufficient;
- top two seasons contribute less than `70%` of total positive lift.

If any gate fails, H004 remains rejected or diagnostic-only.

## Tests

### Residual Diagnostic Tests

- Reject old or incomplete artifacts with clear missing-column messages.
- Use explicit source context for the primary score column; no guessing from path
  names or first score column.
- Exclude explicit DNP rows from residual correlations and count them in a DNP
  context table.
- Produce deterministic season/position output ordering.

### Feature Tests

- Feature-column registry includes `ppg_xg_matchup_h004` only when selected.
- Base `ppg_xg_matchup` columns remain unchanged.
- H004 columns are finite for early rounds and low-sample positions.
- `tec` rows receive zero H004 values.
- Raw opponent IDs and names do not enter the model feature list.
- Mutating target/future `pontuacao`, scouts, or result fields does not change
  H004 columns for the same target round.

### Experiment Tests

- The H004 group builds exactly two child specs per season.
- Control and challenger use identical fixture mode, FootyStats mode, matchup
  mode, model ID, initial budget, scoring contract, and start round.
- Comparability fails closed if candidate-pool signatures differ.
- Missing H004 columns fail loudly rather than silently producing empty reports.

## Risks

- The feature family can become p-hacking if more variants are added after
  looking at results.
- FootyStats pre-match fields are trusted source data; if they contain revised
  post-match information, H004 can amplify leakage.
- Shallow XGBoost may already infer some interactions, making H004 redundant.
- A feature can improve optimized squad points while worsening calibration; the
  acceptance gates therefore require both realized squad lift and predictive
  guardrails.

## Non-Goals

- No optimizer policy changes.
- No live-default changes.
- No Optuna or hyperparameter tuning.
- No neural models.
- No broad feature search.
- No raw opponent IDs as model features.
- No promotion claim from exploratory fixture evidence alone.

## Final Decision Rule

H004 proceeds in this order:

1. Residual diagnostic.
2. If diagnostic passes, implement exactly one frozen H004 feature pack.
3. Run side-by-side control versus H004 in the same experiment matrix.
4. Apply the gates above without post-run threshold changes.

If H004 fails, the next research direction should shift away from hand-built
fixture interactions and toward either data-quality expansion or candidate
availability/confirmed-lineup risk, not another optimizer policy.
