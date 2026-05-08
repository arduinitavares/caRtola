# Clean-Sheet Defensive Stack Policy Design

## Goal

Build and test H003: a narrow optimizer policy that gives a small bonus for
selecting a same-team defensive pair from a strong home defensive-context proxy.

The policy answers one question:

```text
If the source model already likes individual players, does a small same-team
defensive-pair bonus capture shared clean-sheet upside that independent player
predictions miss?
```

This is a frozen policy-simulation hypothesis. It is not a model retrain, not a
new feature pack, and not a live-default change.

## Source Hypothesis

H003 is recorded in:

```text
docs/research/policy_hypotheses.md
```

H001 and H002 were rejected:

- broad opponent-overlap penalties were unstable and regressed 2025;
- goalkeeper-vs-opponent-attack conflict penalties improved only one season.

The next policy must therefore use stronger football context than simple
opponent overlap. The outside critique recommended a clean-sheet defensive stack
because Cartola defensive scoring has a shared event: `SG` benefits goalkeeper,
center backs, and fullbacks together.

## Non-Goals

Do not build:

- a new predictive model;
- new FootyStats loaders;
- new xGA or clean-sheet probability data;
- attack stacking or favorite-team player boosts;
- captain-specific favorite-team bonuses;
- adaptive policy tuning;
- a broad policy-mining engine;
- production default changes.

H003 must use only columns already persisted in source experiment artifacts.

## Source Experiment

Primary source experiment:

```text
data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d
```

Primary slice:

```text
model_id=xgboost_depth2_slow
feature_pack=ppg_xg_matchup
seasons=2021,2022,2023,2024,2025
budget_policy=moving
fixture_identity_status=verified
```

The referenced source experiment is eligible only if every selected child
`run_metadata.json` contains:

```text
fixture_source_directory
fixture_source_paths
fixture_source_sha256
```

For this source experiment, each selected child should have 38
`fixture_source_sha256` entries, one per `partidas-*.csv` fixture source file.
The policy simulation must recompute SHA-256 for those files and emit
`fixture_identity_status=verified` before any `candidate_policy` decision is
allowed. If hashes are missing or mismatched, H003 output is
`diagnostic_only` or invalid, never `candidate_policy`.

The policy simulation must remain artifact-backed: it reads the persisted
`player_predictions.csv`, `round_results.csv`, `selected_players.csv`, and
`run_metadata.json` files from each selected child run. It must not rebuild
candidate frames or retrain models.

## Required Artifact Columns

In addition to the existing policy-simulation artifact contract, H003 requires
these columns in `player_predictions.csv`:

```text
matchup_is_home
footystats_ppg_diff
footystats_xg_diff
```

These columns already exist in the `ppg_xg_matchup` source artifacts. They are
pre-match context columns produced before the target-round result is known.

If any selected child lacks these columns, the policy simulation must fail for
H003 unless the caller explicitly allows incomplete reports. In incomplete mode,
the affected rows must be written to `policy_invalid_rows.csv`; they must not be
silently treated as neutral.

H003-specific invalid rows use the existing `policy_invalid_rows.csv` schema:

```text
season
model_id
feature_pack
child_path
error_type
error_message
```

Any incomplete-mode run with H003 invalid rows must suppress
`candidate_policy`; affected variants are `ineligible` or the whole report is
`diagnostic_only`, depending on the comparability failure.

## Eligibility Rule

H003 is a home favorite defensive-pair proxy. It is not a direct clean-sheet
probability model.

A club-round is eligible for the H003 defensive-pair bonus when all are true:

```text
matchup_is_home == 1
footystats_ppg_diff >= 0.75
footystats_xg_diff >= 0.20
```

Rationale:

- home-only keeps the first test narrow;
- `footystats_ppg_diff >= 0.75` means the team has a substantial pre-match
  strength edge;
- `footystats_xg_diff >= 0.20` prevents pure PPG table position from being the
  only signal;
- the thresholds have broad enough coverage across 2021-2025 and are frozen
  before implementation.

The thresholds must not be changed after seeing H003 simulation results. Any
threshold change creates a new hypothesis generation.

### Context Agreement

For each `rodada, id_clube`, all candidate rows must agree on the context used
for eligibility:

```text
matchup_is_home
footystats_ppg_diff
footystats_xg_diff
```

Agreement rules:

- `matchup_is_home` must coerce to a finite whole number and have exactly one
  unique value;
- `footystats_ppg_diff` and `footystats_xg_diff` must coerce to finite numeric
  values;
- floating context values agree only when `max(value) - min(value) <= 1e-6`.

Disagreement invalidates the affected child replay for H003. Do not choose one
candidate row as canonical when club-round context conflicts.

## Policy Mechanic

Define a qualifying defensive pair as:

```text
one selected GOL from an eligible club
plus
one selected LAT or ZAG from the same eligible club
```

The policy adds a small positive objective bonus when such a pair is selected.

Rules:

- use selected-player binary variables only;
- apply at most one qualifying-pair bonus per eligible club;
- apply at most one qualifying-pair bonus per squad;
- do not alter raw predicted points written to selected-player artifacts;
- do not alter prices, budget constraints, formations, captain rules, or
  actual-point scoring;
- record the policy variant in the existing policy artifacts.

This makes H003 a small correlation bonus, not a hard stack requirement.

## Policy Variants

The frozen policy set is:

```text
policy_set_id=clean-sheet-stack-v1

no_policy
home_cs_pair_bonus_025
home_cs_pair_bonus_050
home_cs_pair_bonus_075
home_cs_pair_bonus_100
```

Variant definitions:

```text
home_cs_pair_bonus_025: +0.25 optimizer points
home_cs_pair_bonus_050: +0.50 optimizer points
home_cs_pair_bonus_075: +0.75 optimizer points
home_cs_pair_bonus_100: +1.00 optimizer points
```

If multiple variants pass all gates, the smallest passing bonus wins. Do not
choose the highest total-points variant just because it has the highest total.

## Optimizer Design

Extend `OptimizerPolicy` with a small defensive-stack policy surface:

```text
clean_sheet_pair_bonus: float
clean_sheet_pair_anchor_position: "gol"
clean_sheet_pair_partner_positions: ("lat", "zag")
clean_sheet_pair_min_ppg_diff: float
clean_sheet_pair_min_xg_diff: float
clean_sheet_pair_home_only: bool
max_clean_sheet_pair_bonuses: int
```

Implementation details:

- derive eligible club IDs from the current round's candidate frame, not from
  external files;
- eligibility is per `rodada, id_clube`;
- a club is eligible only if all candidate rows for that club-round agree on
  the required context columns after duplicate normalization;
- conflicting context values for the same club-round invalidate that policy
  replay row;
- create one binary variable per eligible club indicating whether a qualifying
  `GOL + LAT/ZAG` pair is selected;
- linearize the pair variable so it cannot be set to `1` unless both sides of
  the pair exist in the selected squad;
- cap the sum of pair variables at `max_clean_sheet_pair_bonuses=1`;
- add `clean_sheet_pair_bonus * clean_sheet_pair_count` to the optimizer
  objective.

H003 is a candidate-context policy. It does not need fixture rows inside the
optimizer because eligible clubs are derived from the persisted candidate
columns. The policy simulation still requires verified fixture identity for
decision evidence because those persisted candidate columns were created from
fixture-backed matchup context.

The original `no_policy` tie-breaking behavior must remain reproducible.

### Pair Linearization

For each eligible club `c`, define:

```text
gk_count_c = sum(selected_i for candidate i where id_clube == c and posicao == "gol")
partner_count_c = sum(selected_i for candidate i where id_clube == c and posicao in {"lat", "zag"})
gk_present_c binary
partner_present_c binary
clean_sheet_pair_c binary
```

Let `M_partner` be the maximum number of `LAT/ZAG` assets that can be selected
in the current formation. Use `M_partner >= 1`.

Presence constraints:

```text
gk_count_c >= gk_present_c
gk_count_c <= gk_present_c
partner_count_c >= partner_present_c
partner_count_c <= M_partner * partner_present_c
```

Pair constraints:

```text
clean_sheet_pair_c <= gk_present_c
clean_sheet_pair_c <= partner_present_c
clean_sheet_pair_c >= gk_present_c + partner_present_c - 1
```

Global cap:

```text
sum(clean_sheet_pair_c for eligible clubs c) <= max_clean_sheet_pair_bonuses
```

Objective addition:

```text
maximize predicted_points_with_captain + clean_sheet_pair_bonus * sum(clean_sheet_pair_c)
```

This forces the bonus to apply exactly when an eligible club contributes both a
selected goalkeeper and at least one selected fullback/center back.

## Reporting And Decision Metrics

Add these columns to `policy_round_results.csv`:

```text
clean_sheet_pair_count
clean_sheet_pair_bonus_applied
selected_ids_changed_vs_no_policy
```

Add these columns to `policy_ranked_summary.csv`:

```text
changed_rounds
changed_round_rate
```

`selected_ids_changed_vs_no_policy` compares the selected `id_atleta` set for a
policy variant against `no_policy` for the same season, model, feature pack,
strategy, and round. Captain-only changes do not count as selected-ID changes.

`changed_rounds` is the count of completed rounds where
`selected_ids_changed_vs_no_policy=true`. `changed_round_rate` is
`changed_rounds / comparable_completed_rounds`.

The report must state:

- policy set ID;
- thresholds used;
- fixture identity status;
- source experiment path;
- invalid row count;
- decision status per variant.

## Acceptance Criteria

H003 becomes a `candidate_policy` only if a policy variant:

- improves total actual captain-aware points by at least `+75` versus
  `no_policy` across 2021-2025;
- improves at least `3` of `5` seasons;
- has median season delta greater than `0`;
- does not regress 2025 by more than `-15` points;
- does not regress any season by more than `-25` points;
- does not increase non-optimal solver rounds;
- does not fail the budget guardrails defined below;
- changes selections in at least `15` evaluated rounds;
- changes selections in no more than `40%` of evaluated rounds;
- does not fail the top-two-round concentration guardrail defined below.

If several variants pass, choose the smallest passing bonus.

Define budget guardrails exactly as:

```text
final_budget_delta_vs_no_policy >= -5
min_budget_delta_vs_no_policy >= -5
max_drawdown_delta_vs_no_policy <= 5
```

Define top-two-round concentration exactly as:

```text
round_delta = policy actual_points_with_captain - no_policy actual_points_with_captain
positive_round_delta = max(round_delta, 0)
total_positive_delta = sum(positive_round_delta across all replayed seasons and rounds)
top_2_round_delta_sum = sum(two largest positive_round_delta values)
top_two_positive_delta_concentration = top_2_round_delta_sum / total_positive_delta
```

If `total_positive_delta <= 0`, the concentration metric is `NA` and the
variant cannot be `candidate_policy` because the total-lift criterion already
fails. H003 fails the concentration gate when:

```text
top_two_positive_delta_concentration > 0.50
```

## Rejection Criteria

Reject H003 if:

- no variant beats `no_policy` by the practical `+75` point threshold;
- positive aggregate lift comes from fewer than `3` seasons;
- 2025 regresses beyond the frozen threshold;
- one old season explains most of the gain;
- the policy creates additional infeasible or non-optimal rounds;
- selection changes do not meet the minimum or maximum changed-round gates;
- any required pre-match context column is missing or silently neutralized.

## Tests

Add tests before implementation:

1. Policy registry test:
   - `clean-sheet-stack-v1` returns exactly the frozen variants.

2. Optimizer synthetic test:
   - with an eligible home club, a small pair bonus can select `GOL + ZAG/LAT`
     over an otherwise slightly better independent defender;
   - the same bonus does not apply when `matchup_is_home=0`;
   - the same bonus does not apply when thresholds are not met;
   - the cap prevents more than one pair bonus per squad.

3. Linearization test:
   - the pair bonus variable cannot be set unless both selected players exist.

4. Policy replay test:
   - H003 requires `matchup_is_home`, `footystats_ppg_diff`, and
     `footystats_xg_diff`;
   - missing columns produce invalid rows or strict failure, not zero bonuses.

5. CLI smoke test:
   - `--policy-set clean-sheet-stack-v1` writes the normal policy artifacts.

6. Real-run acceptance:
   - run H003 against the verified source experiment and inspect
     `policy_ranked_summary.csv`, `policy_per_season_summary.csv`,
     `policy_invalid_rows.csv`, and `policy_comparability_report.json`.

## First Run Command

```text
uv run --frozen python scripts/run_policy_simulation.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d \
  --hypothesis-id H003 \
  --policy-set clean-sheet-stack-v1 \
  --models xgboost_depth2_slow \
  --feature-packs ppg_xg_matchup \
  --seasons 2021,2022,2023,2024,2025 \
  --current-year 2026
```

## Decision Handling

The policy simulation may emit `candidate_policy` only when:

- selected seasons are exactly `2021,2022,2023,2024,2025`;
- fixture identity is verified;
- comparability status is `ok`;
- invalid row count is `0`;
- all H003 acceptance criteria pass.

Otherwise the result is `rejected`, `ineligible`, or `diagnostic_only`.

## Implementation Boundary

Expected files:

```text
src/cartola/backtesting/optimizer_policies.py
src/cartola/backtesting/optimizer.py
src/cartola/backtesting/policy_simulation.py
src/tests/backtesting/test_optimizer_policies.py
src/tests/backtesting/test_optimizer.py
src/tests/backtesting/test_policy_simulation.py
src/tests/backtesting/test_run_policy_simulation_cli.py
docs/research/policy_hypotheses.md
roadmap.md
```

Do not change model training, feature generation, live recommendation defaults,
or experiment-ranking logic.
