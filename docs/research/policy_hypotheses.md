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

Status: `pending_policy_simulation`

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

`pending`

