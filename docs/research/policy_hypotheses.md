# Policy Hypotheses

This registry keeps optimizer-policy research explicit and auditable.

The workflow is:

1. Observe a pattern in oracle/profile diagnostics.
2. Freeze a hypothesis before changing optimizer behavior.
3. Translate the hypothesis into one or more policy variants.
4. Test the policy variants with the same model predictions, candidate pools, scoring contract, and moving-budget semantics.
5. Decide whether the hypothesis is rejected, kept for more evidence, or promoted to a policy candidate.

Rules:

- A hypothesis is not production evidence.
- Oracle diagnostics are hindsight research only.
- Do not change live defaults from a hypothesis without a frozen validation experiment.
- Do not revise acceptance criteria after seeing policy-simulation results; create a new hypothesis generation instead.
- Keep each hypothesis narrow enough that one simulation run can reject or support it.

## H001: Opponent-Overlap Exposure

Status: `rejected`

### Observation

The model-selected squads contain more players involved in selected-player-vs-selected-player opponent overlap than oracle squads.

Current oracle discovery reference:

```text
source: data/08_reporting/oracle_discovery/oracle_discovery_started_at=20260507T161147870260Z/profile_gap_summary.csv
experiment: xgboost-sensitivity-v2, 2021-2025
```

Observed aggregate profile gaps:

```text
avg_players_in_opponent_overlap:
  oracle:         2.525
  model-selected: 3.617
  gap:           -1.091

opponent_overlap_round_rate:
  oracle:         0.698
  model-selected: 0.724
  gap:           -0.025
```

Season-level caveat:

The count gap is stable across 2021-2025, but the yes/no overlap-rate gap is mixed by season. This argues for testing soft penalties before enforcing a hard ban.

### Hypothesis

Reducing opponent-overlap exposure improves realized squad performance or reduces downside risk.

### Proposed Mechanism

Selecting too many players from both sides of the same match may create contradictory upside assumptions, especially when defenders or goalkeepers are selected against attackers from the opposing club.

This may increase lineup fragility: one match event can benefit one selected player while directly harming another selected player.

### Policy Variants

The first simulation should compare:

```text
no_policy
soft_overlap_penalty_low
soft_overlap_penalty_medium
hard_max_overlap_3
hard_max_overlap_2
```

Policy definitions must be frozen in the implementation plan before running the experiment.

### Primary Test

Run a policy simulation where only optimizer policy changes.

Constants:

- same completed source experiment or same model-prediction generation;
- same seasons: `2021,2022,2023,2024,2025`;
- same start round;
- same scoring contract: `cartola_standard_2026_v1`;
- same moving-budget semantics;
- same candidate pools;
- same prediction score columns;
- same fixture source used to identify opponent relationships.

### Acceptance Criteria

H001 becomes a `candidate_policy` only if a policy variant:

- improves total actual captain-aware points versus `no_policy`;
- improves at least `3` of `5` seasons;
- does not materially regress 2025;
- does not increase infeasible or non-optimal solver rounds;
- does not materially worsen final budget, minimum budget, or max drawdown;
- does not win only through one or two extreme rounds.

### Rejection Criteria

Reject H001 if:

- no policy variant beats `no_policy` on total actual points;
- gains appear only in old seasons while 2025 regresses meaningfully;
- the best policy increases infeasible/non-optimal solver rounds;
- the policy reduces opponent overlap but does not improve squad outcomes;
- the result depends on post-hoc changes to thresholds or policy definitions.

### Required Outputs

The policy simulation should write:

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

### Decision

`rejected`

Simulation output:

```text
data/08_reporting/policy_simulations/policy_simulation_started_at=20260507T233158737698Z
```

Evidence quality:

```text
fixture_identity_status: verified
comparability_status: ok
invalid_row_count: 0
```

Best variant:

```text
policy_variant: soft_overlap_penalty_low
total_delta_vs_no_policy: +187.91
improved_seasons: 3 / 5
season_2025_delta: -57.38
```

Reason:

The best opponent-overlap policy improved aggregate points, but it materially
regressed the most recent completed season. Under the frozen H001 criteria, that
is not stable enough to become a policy candidate.

## H002: Goalkeeper Against Opponent Attack Exposure

Status: `rejected`

### Observation

H001 tested broad selected-player opponent overlap and was too blunt. The
actionable version of the original concern is narrower: selecting a goalkeeper
while also selecting attackers, or an attacking captain, from the opposing team
creates directly contradictory upside assumptions.

Before implementing H002, an outside critique supported a narrow goalkeeper
conflict test but warned against broad midfielder/defender bans, broad
opponent-overlap penalties, or production promotion from one season-specific
gain.

### Hypothesis

Penalizing goalkeeper-vs-opponent-attacker conflicts improves realized squad
performance or reduces downside risk without the instability of broad
opponent-overlap penalties.

### Proposed Mechanism

A goalkeeper benefits from a clean sheet and saves, while opposing attackers and
attacking midfielders benefit from shots and goals. The strongest contradiction
is therefore:

- selected goalkeeper vs selected opposing attackers;
- selected goalkeeper vs selected opposing captain when the captain is an
  attacker or attacking midfielder.

### Policy Variants

The frozen H002 simulation compares:

```text
no_policy
gk_vs_selected_ata_soft_low
gk_vs_selected_ata_soft_medium
gk_vs_opponent_captain_soft
gk_vs_opponent_attack_hard
```

### Primary Test

Run a policy simulation where only optimizer policy changes.

Constants:

- same completed source experiment or same model-prediction generation;
- same seasons: `2021,2022,2023,2024,2025`;
- same scoring contract: `cartola_standard_2026_v1`;
- same moving-budget semantics;
- same candidate pools;
- same prediction score columns;
- same verified fixture source used to identify opponent relationships.

Source experiment:

```text
data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d
```

Command:

```text
uv run --frozen python scripts/run_policy_simulation.py \
  --experiment-path data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d \
  --hypothesis-id H002 \
  --policy-set gk-conflict-v1 \
  --models xgboost_depth2_slow \
  --feature-packs ppg_xg_matchup \
  --seasons 2021,2022,2023,2024,2025 \
  --current-year 2026
```

### Acceptance Criteria

H002 becomes a `candidate_policy` only if a policy variant:

- improves total actual captain-aware points versus `no_policy`;
- improves at least `3` of `5` seasons;
- does not materially regress 2025;
- does not increase infeasible or non-optimal solver rounds;
- does not materially worsen final budget, minimum budget, or max drawdown;
- does not win only through one or two extreme rounds.

### Decision

`rejected`

Simulation output:

```text
data/08_reporting/policy_simulations/policy_simulation_started_at=20260508T004719246886Z
```

Evidence quality:

```text
fixture_identity_status: verified
comparability_status: ok
invalid_row_count: 0
```

Best variant:

```text
policy_variant: gk_vs_opponent_captain_soft
total_delta_vs_no_policy: +46.19
improved_seasons: 1 / 5
season_2025_delta: 0.00
```

Per-season signal:

```text
2021:  0.00
2022: +46.19
2023:  0.00
2024:  0.00
2025:  0.00
```

Reason:

The best goalkeeper-conflict policy produced a positive aggregate result, but
the gain came from one season only. Under the frozen H002 criteria, that is not
stable enough to become a policy candidate.

## H003: Clean-Sheet Defensive Stack

Status: `pending_policy_simulation`

### Observation

H001 and H002 showed that simple opponent-overlap penalties are not stable
enough to become optimizer policies. The next policy should use stronger
football context.

The narrowest mechanism is defensive correlation: Cartola clean-sheet scoring
(`SG`) creates shared upside for `GOL`, `LAT`, and `ZAG` from the same club.
If a team has strong pre-match home context, a small same-team defensive-pair
bonus may capture shared clean-sheet upside that independent player predictions
do not fully model.

Before implementation, an outside critique recommended testing clean-sheet
defensive stacking first and avoiding broad strong-team boosts, attack stacking,
captain favorite-team boosts, or any threshold tuning after seeing results.

H003 is a clean-sheet proxy policy, not a direct clean-sheet probability model.
It uses persisted pre-match PPG and xG differential columns because no direct
clean-sheet probability source is currently part of the artifact contract.

### Hypothesis

In fixtures with strong pre-match home defensive-context proxy values, a small
optimizer bonus for selecting `GOL + LAT/ZAG` from the same team improves
moving-budget historical squad performance versus `no_policy`.

### Eligibility Rule

A club-round is eligible only when:

```text
matchup_is_home == 1
footystats_ppg_diff >= 0.75
footystats_xg_diff >= 0.20
```

These columns must come from persisted `player_predictions.csv` artifacts. Do
not use actual clean sheets, goals conceded, final scores, post-round standings,
or full-season ranks.

### Policy Variants

The frozen H003 simulation compares:

```text
no_policy
home_cs_pair_bonus_025
home_cs_pair_bonus_050
home_cs_pair_bonus_075
home_cs_pair_bonus_100
```

Each bonus applies at most once per squad for a selected eligible same-club
`GOL + LAT/ZAG` pair.

If multiple variants pass all gates, choose the smallest passing bonus.

### Primary Test

Run a policy simulation where only optimizer policy changes.

Constants:

- same completed source experiment;
- same model and feature pack: `xgboost_depth2_slow + ppg_xg_matchup`;
- same seasons: `2021,2022,2023,2024,2025`;
- same scoring contract: `cartola_standard_2026_v1`;
- same moving-budget semantics;
- same candidate pools;
- same prediction score columns;
- same verified fixture source.

Source experiment:

```text
data/08_reporting/experiments/model_feature/group=xgboost-sensitivity-v2__started_at=20260507T231127138806Z__matrix=f019652c883d
```

Fixture evidence precondition:

```text
each selected child run_metadata.json must contain fixture_source_sha256
fixture_identity_status must be verified
invalid_row_count must be 0
```

Command:

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

### Acceptance Criteria

H003 becomes a `candidate_policy` only if a policy variant:

- improves total actual captain-aware points by at least `+75` versus
  `no_policy`;
- improves at least `3` of `5` seasons;
- has median season delta greater than `0`;
- does not regress 2025 by more than `-15` points;
- does not regress any season by more than `-25` points;
- does not increase infeasible or non-optimal solver rounds;
- satisfies budget guardrails:
  `final_budget_delta >= -5`, `min_budget_delta >= -5`, and
  `max_drawdown_delta <= 5`;
- changes selections in at least `15` evaluated rounds;
- changes selections in no more than `40%` of evaluated rounds;
- has `top_two_positive_delta_concentration <= 0.50`.

### Rejection Criteria

Reject H003 if:

- no policy variant beats `no_policy` by the practical `+75` point threshold;
- positive aggregate lift comes from fewer than `3` seasons;
- 2025 regresses beyond the frozen threshold;
- one old season explains most of the gain;
- the policy creates additional infeasible or non-optimal rounds;
- fixture identity is unverified or any H003 rows are invalid;
- changed-round or budget guardrails fail;
- the result depends on post-hoc changes to thresholds or policy definitions.

### Decision

`pending`
