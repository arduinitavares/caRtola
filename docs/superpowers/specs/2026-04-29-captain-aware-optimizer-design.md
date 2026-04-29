# Captain-Aware Optimizer And Captain Policy Diagnostics

Date: 2026-04-29

## Purpose

Cartola lineups are scored as 11 players plus 1 tecnico, with one selected non-tecnico player marked as captain. In standard Cartola, the captain receives a 1.5x score multiplier. The current optimizer selects a legal squad under budget and a fixed formation, but it does not choose a captain and the default formation set only contains `4-3-3`.

This design upgrades the recommendation and backtest optimizer so live squads match the real scoring contract more closely:

- search all 7 official Cartola formations;
- select the captain inside the same optimizer that selects the squad;
- default captain scoring to enabled for live recommendations;
- report safe/upside captain diagnostics on the same optimized squad;
- evaluate replay/backtest actuals with the captain multiplier.

No new data source, captain model, fixture data, or captain compatibility audit is required. Existing compatibility, ablation, and summary readers must still record or respect the scoring contract so mixed-contract metrics are not compared silently.

## Current Code Context

Current relevant contracts:

- `BacktestConfig.DEFAULT_FORMATIONS` contains only `4-3-3`.
- `optimize_squad()` in `optimizer.py` solves one formation using PuLP binary selected-player variables.
- `SquadOptimizationResult` currently has no captain fields.
- `runner.py` records one row per round/strategy with `predicted_points` and `actual_points`.
- `recommendation.py` uses the random forest score, calls `optimize_squad()`, and writes `recommended_squad.csv`, `candidate_predictions.csv`, `recommendation_summary.json`, and `run_metadata.json`.
- Replay/oracle scoring currently sums `pontuacao`; it does not apply captain scoring.

## Official Rule Contract

The v1 optimizer uses these rules:

- lineup has 11 players plus 1 tecnico;
- captain is one of the 11 selected non-tecnico players;
- tecnico cannot be captain;
- captain is not an extra slot;
- standard captain multiplier is `1.5`;
- some leagues can disable captain scoring, so `captain_mode` must exist;
- Reserva de Luxo and bench substitution are out of scope for v1.

The 7 official formation counts are:

| Formation | gol | lat | zag | mei | ata | tec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-4-3 | 1 | 0 | 3 | 4 | 3 | 1 |
| 3-5-2 | 1 | 0 | 3 | 5 | 2 | 1 |
| 4-3-3 | 1 | 2 | 2 | 3 | 3 | 1 |
| 4-4-2 | 1 | 2 | 2 | 4 | 2 | 1 |
| 4-5-1 | 1 | 2 | 2 | 5 | 1 | 1 |
| 5-3-2 | 1 | 2 | 3 | 3 | 2 | 1 |
| 5-4-1 | 1 | 2 | 3 | 4 | 1 | 1 |

## Scope

In scope:

- add all 7 formations;
- add `captain_mode: "enabled" | "disabled"`;
- add `captain_multiplier: float = 1.5`;
- add `formation_mode: "auto" | "fixed"`;
- add one-solve-per-formation search;
- add captain variables to the MILP when captain mode is enabled;
- compute captain policy diagnostics on the same selected squad;
- update recommendation, backtest, replay, oracle, metadata, and terminal/report output.

Out of scope:

- a separate captain prediction model;
- risk-aware squad re-optimization;
- Reserva de Luxo;
- bench selection;
- tecnico rule reconstruction from club average plus win bonus;
- strict fixture integration into live recommendation.

## Optimization Formulation

Config types:

```python
CaptainMode = Literal["enabled", "disabled"]
FormationMode = Literal["auto", "fixed"]
```

`captain_multiplier` must be finite and `>= 1.0`. Reject NaN, infinity, and values below `1.0`.

For each candidate `i`:

- `x_i` is binary: player is selected;
- `c_i` is binary: player is captain;
- `e_i` is the pre-round expected Cartola score from the active score column;
- `price_i` is `preco_pre_rodada`.

For captain-enabled mode:

```text
maximize sum(x_i * e_i) + (captain_multiplier - 1.0) * sum(c_i * e_i)
```

Subject to:

```text
sum(price_i * x_i) <= budget
position counts match the selected formation
sum(c_i) == 1
c_i <= x_i for every candidate
c_i == 0 when posicao == "tec"
```

For captain-disabled mode:

```text
maximize sum(x_i * e_i)
sum(c_i) == 0
```

The budget constraint must use only `x_i`; captaincy does not consume budget.

The captain term is a bonus, not a full extra score. With the default multiplier:

```text
predicted_points_with_captain = predicted_points_base + 0.5 * captain_predicted_points
```

Actual scoring must be computed outside the optimizer from explicit actual-score inputs. Live mode must not invent actual zeros when `pontuacao` is absent.

## Formation Search

V1 should solve one MILP per allowed formation and choose the best optimal result by `predicted_points_with_captain` when captain is enabled, or by `predicted_points_base` when captain is disabled.

This is preferred over a single MILP with formation variables because:

- there are only 7 formations;
- separate solves are simpler to test and debug;
- infeasible formations can be reported directly;
- runtime should remain small relative to model fitting.

The selected formation in config should support:

- `formation_mode="auto"`: search all available formations;
- `formation_mode="fixed"` with `formation_name`: solve only that formation.

Live recommendation should default to `formation_mode="auto"`.

Tie-breaking across formations and equivalent squads must be deterministic:

1. highest active objective;
2. lowest formation name;
3. lowest sorted selected `id_atleta` tuple;
4. lowest captain `id_atleta`.

If all searched formations are infeasible, return one `Infeasible` result containing per-formation statuses. Do not return an empty `Optimal` result.

## Captain Policies

V1 builds exactly one squad: the joint EV-optimized squad. The optimizer-selected captain is the `ev` captain.

Safe and upside policies are diagnostics only. They must choose alternative captains from the same selected non-tecnico players. They must not re-optimize the squad.

This preserves attribution:

```text
policy_delta = base_actual + captain_policy_bonus - ev_actual_total
```

Any difference is caused only by captain relabeling, not by a different squad.

### Policy Definitions

All policies operate on selected non-tecnico players only.

- `ev`: optimizer-selected captain. Equivalent to maximizing expected total under the MILP.
- `safe`: deterministic low-volatility alternative. Use `prior_points_std` when available:
  ```text
  safe_score = predicted_points - prior_points_std_filled
  ```
  Missing or non-finite `prior_points_std` is treated as `0`.
- `upside`: deterministic volatility-upside alternative:
  ```text
  upside_score = predicted_points + prior_points_std_filled
  ```

Tie-breaking must be deterministic:

1. higher policy score;
2. higher predicted points;
3. lower `id_atleta`.

If all `prior_points_std` values are missing or zero, `safe` and `upside` may equal `ev`. That is acceptable and should be visible in diagnostics.

The recommendation output should also include the top 3 selected non-tecnico captain alternatives sorted by predicted points.

`captain_policy_diagnostics` should be a JSON list of records:

```json
[
  {
    "policy": "ev",
    "captain_id": 123,
    "captain_name": "Player",
    "captain_position": "ata",
    "captain_club": "Club",
    "captain_predicted_points": 8.2,
    "predicted_captain_bonus": 4.1,
    "predicted_points_with_policy": 76.5,
    "actual_captain_points": null,
    "actual_captain_bonus": null,
    "actual_points_with_policy": null,
    "actual_delta_vs_ev": null
  }
]
```

CSV outputs should not serialize this nested object into a single opaque column unless the CSV is explicitly a metadata summary. Per-policy detail belongs in JSON or in a separate normalized `captain_policy_diagnostics.csv`.

## Compatibility And Defaults

Defaults are entry-point-specific:

| Entry point | `captain_mode` default | `formation_mode` default | Reason |
| --- | --- | --- | --- |
| live recommendation | `enabled` | `auto` | match standard Cartola gameplay |
| replay recommendation | `enabled` | `auto` | evaluate the lineup a user would now play |
| normal backtest CLI | `disabled` unless explicitly requested for a new scoring contract | `fixed` unless explicitly requested | avoid silently changing historical reports |
| FootyStats ablation | explicit, recorded config required | explicit, recorded config required | prevent mixed-contract comparisons |
| compatibility audit | `disabled` and `fixed` unless audit is expanded to record scoring contract | `fixed` | keep legacy compatibility metrics stable |

When backtests or ablations run with `captain_mode="enabled"` or `formation_mode="auto"`, outputs must include:

- `scoring_contract_version`;
- `captain_mode`;
- `captain_multiplier`;
- `formation_mode`;
- `allowed_formations`.

Ablation and audit report readers must either:

- include these fields in their CSV/JSON reports and refuse to aggregate mixed contracts; or
- run with `captain_mode="disabled"` and `formation_mode="fixed"` to preserve legacy comparability.

This is not a data compatibility issue. It is a metric comparability issue: captain-aware totals and fixed-formation raw totals are different scoring contracts.

## Result Schema

`SquadOptimizationResult` should add:

- `predicted_points_base`;
- `captain_bonus_predicted`;
- `predicted_points_with_captain`;
- `actual_points_base`;
- `captain_bonus_actual`;
- `actual_points_with_captain`;
- `captain_mode`;
- `captain_multiplier`;
- `captain_id`;
- `captain_name`;
- `captain_position`;
- `captain_club`;
- `captain_predicted_points`;
- `formation_scores`;
- `captain_policy_diagnostics`.

Empty or infeasible results should set:

- point totals to `0.0` for predicted base/bonus/total;
- actual totals to `None` unless explicit actual scoring was computed;
- captain fields to `None`;
- `formation_scores` to per-formation status records when available;
- `captain_policy_diagnostics` to an empty list.

`formation_scores` should be a JSON list of records:

```json
[
  {
    "formation": "4-3-3",
    "solver_status": "Optimal",
    "predicted_points_base": 71.2,
    "captain_bonus_predicted": 4.5,
    "predicted_points_with_captain": 75.7,
    "captain_id": 123,
    "captain_name": "Player",
    "infeasibility_reason": null
  }
]
```

The selected dataframe should include:

- `is_captain`;
- `captain_policy_ev`;
- `captain_policy_safe`;
- `captain_policy_upside`;

where policy columns are boolean markers for the same selected squad.

Selected-player point columns keep per-athlete semantics. `selected_players["predicted_points"]` and recommendation `recommended_squad["predicted_points"]` remain raw per-athlete model scores. Do not multiply the captain row by `1.5` in selected-player CSVs.

If useful for display, add a separate selected-player column:

```text
captain_adjusted_points = predicted_points + captain_bonus_predicted_for_this_row
```

where the bonus is non-zero only for the captain.

## Recommendation Output

`recommendation_summary.json` should include:

- `formation`;
- `formation_mode`;
- `captain_mode`;
- `captain_multiplier`;
- `captain_id`;
- `captain_name`;
- `captain_position`;
- `captain_club`;
- `predicted_points_base`;
- `captain_bonus_predicted`;
- `predicted_points_with_captain`;
- `actual_points_base` in replay mode when available;
- `captain_bonus_actual` in replay mode when available;
- `actual_points_with_captain` in replay mode when available;
- `captain_policy_diagnostics`;
- `captain_alternatives`.

`recommended_squad.csv` should include:

- `is_captain`;
- `captain_policy_ev`;
- `captain_policy_safe`;
- `captain_policy_upside`;
- `predicted_points`;
- existing identity, price, club, position, and replay columns.

`run_metadata.json` should include:

- `scoring_contract_version`;
- `formation_mode`;
- `allowed_formations`;
- `captain_mode`;
- `captain_multiplier`;
- `captain_policy_definitions`;
- existing data source and feature metadata.

Terminal output for live recommendations should show:

- formation;
- captain;
- predicted base;
- captain bonus;
- predicted total;
- budget used;
- top captain alternatives.

## Backtest Output

Backtest round rows should record explicit base and captain-adjusted fields:

- `predicted_points_base`;
- `captain_bonus_predicted`;
- `predicted_points_with_captain`;
- `actual_points_base`;
- `captain_bonus_actual`;
- `actual_points_with_captain`;
- `captain_id`;
- `captain_name`;
- `captain_policy_ev_id`;
- `captain_policy_safe_id`;
- `captain_policy_upside_id`;
- `actual_points_with_ev_captain`;
- `actual_points_with_safe_captain`;
- `actual_points_with_upside_captain`.

Summary and diagnostics should use `actual_points_with_captain` as the primary real Cartola score when `captain_mode="enabled"`.

If existing round-level output columns named `predicted_points` and `actual_points` remain in `round_results.csv`, `summary.csv`, diagnostics, and recommendation summaries, they must be aliases for the active scoring contract:

- captain enabled: total with captain;
- captain disabled: base score.

This alias rule does not apply to selected-player rows. Selected-player `predicted_points` remains a per-athlete raw prediction and selected-player `pontuacao` remains a per-athlete actual score.

Metadata must record the scoring contract so old raw-squad reports are not compared blindly with captain-aware reports.

## Metrics And Random Baselines

`metrics.py` must apply one scoring contract consistently.

When `captain_mode="enabled"`:

- `round_results.actual_points` should be captain-adjusted active-contract totals;
- random valid squad diagnostics must also choose a valid captain for each random squad and apply the same multiplier;
- random squad captain can be selected by actual `pontuacao` for oracle-like random actual diagnostics, or by the same random-squad policy used by the metric. The selected policy must be named in diagnostics metadata.

If random selection diagnostics are not made captain-aware in the first implementation, suppress them or rename them so they are not compared against captain-adjusted strategy totals.

Prediction diagnostics remain player-level and continue to compare raw per-athlete predicted score columns against raw per-athlete `pontuacao`.

## Oracle Replay

Replay oracle scoring must also be captain-aware.

For an oracle based on actual `pontuacao`, the optimizer should use `pontuacao` as the score column and select both squad and captain under the same constraints. Oracle actual total is:

```text
oracle_actual_points_with_captain =
  oracle_actual_points_base + (captain_multiplier - 1.0) * oracle_captain_actual_points
```

If any required selected actual score is missing or non-finite, the actual/captain/oracle metrics must be null with a warning, not coerced to zero.

## Data Requirements

No new data is required.

Required candidate columns:

- `id_atleta`;
- `apelido`;
- `posicao`;
- `preco_pre_rodada`;
- score column used for optimization;
- `pontuacao` only when actual scoring is requested;
- `prior_points_std` for safe/upside diagnostics, optional with fallback `0`.

The optimizer must reject or fail cleanly on:

- missing required score column;
- missing price column;
- non-finite score values used in optimization;
- non-finite price values;
- no feasible formation.

Duplicate `id_atleta` handling should remain deterministic: deduplicate before creating selected/captain variables, preserving the current first-row behavior unless a separate dedupe policy is specified.

## Defaults

Recommended live recommendation defaults:

```text
captain_mode = "enabled"
captain_multiplier = 1.5
formation_mode = "auto"
```

Recommended legacy report defaults:

```text
captain_mode = "disabled"
formation_mode = "fixed"
formation_name = "4-3-3"
```

Backtest, audit, and ablation commands may opt into captain-aware/auto-formation mode, but then they must record the scoring contract and avoid aggregating mixed contracts.

For leagues without captain scoring:

```text
captain_mode = "disabled"
```

In disabled mode, the system may still suggest a non-scored captain for UI completion, but it must be labeled `captain_scoring_enabled=false` and must not affect the objective or metrics.

## Testing Requirements

Optimizer tests:

- all 7 formation counts are available and exact;
- `3-4-3` and `3-5-2` select zero `lat`;
- captain must be selected: a high-score unaffordable phantom player cannot become captain;
- tecnico cannot be captain even if highest projected;
- captain budget is not counted separately;
- predicted total equals base plus `(multiplier - 1) * captain_score`;
- negative expected points select the least-negative captain;
- disabled mode matches legacy base objective;
- some formations infeasible but at least one feasible returns best feasible;
- all formations infeasible fails loudly;
- tie-breaking is deterministic.

Recommendation tests:

- live summary includes captain fields;
- `recommended_squad.csv` marks exactly one `is_captain` when captain enabled;
- safe/upside policies are selected from the same squad and do not change selected players;
- captain alternatives exclude tecnico;
- terminal output prints base, bonus, total, and captain.

Backtest/replay tests:

- actual totals include captain multiplier exactly once;
- policy diagnostics compare EV/safe/upside on the same squad;
- missing actuals produce null actual/captain/oracle metrics with warnings;
- oracle optimizer applies captain rules;
- metadata records `captain_mode`, `captain_multiplier`, and scoring contract version.

Regression tests:

- existing `captain_mode="disabled"` behavior matches the old optimizer objective;
- selected-player `predicted_points` remains raw per-athlete score and is not captain-adjusted;
- compatibility audit output either records scoring contract or uses disabled/fixed defaults;
- FootyStats ablation output records and validates scoring contract before comparing runs;
- random selection diagnostics are captain-aware or suppressed under captain-enabled mode;
- no FootyStats, fixture, strict snapshot, or live market capture path changes are required.

## Rollout

Suggested implementation order:

1. Add all 7 formations and formation tests.
2. Add captain config fields and result fields.
3. Add captain-aware single-formation optimization.
4. Add auto-formation search.
5. Add captain policy diagnostics on selected squads.
6. Update recommendation outputs and terminal display.
7. Update backtest/replay/oracle scoring.
8. Add full tests and run the quality gate.

## Acceptance Criteria

- Live recommendation selects a legal squad, a legal captain, and reports captain-adjusted predicted points.
- Replay/backtest actual points include the selected captain multiplier.
- Safe/upside policy diagnostics use the same selected squad as EV.
- Formation auto-search can choose any of the 7 official formations.
- Captain mode can be disabled and then reverts to the legacy base objective.
- Metadata makes the scoring contract explicit.
