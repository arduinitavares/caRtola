# Moving Budget Backtesting Design

## Goal

Hard-replace fixed per-round budget backtesting with Cartola-style moving-budget semantics.

`BacktestConfig.budget` and CLI `--budget` now mean **initial budget** for historical backtests, historical replay paths that evaluate multiple rounds, model experiments, tuning runs, reports, and promotion decisions. These paths must evaluate each strategy with its own evolving patrimonio path.

Live open-round recommendations and single-round recommendation replay are separate one-round workflows: `--budget` means the caller's current available patrimonio for that round, and no post-selection budget update is possible until finalized post-round `variacao` exists in historical replay data.

The system should punish models that destroy purchasing power through selected-player devaluation, because that is how Cartola works in practice.

## Decision

Fixed-budget backtesting is no longer a normal supported evaluation mode.

Required semantics:

```text
current_budget[strategy] starts at initial_budget.

For each target round:
  train/predict using only prior-round data
  optimize each strategy under current_budget[strategy]
  score actual captain-aware points
  budget_delta = sum(selected squad variacao)
  current_budget[strategy] = current_budget[strategy] + budget_delta

Next round for that strategy uses the updated budget.
```

Each strategy has an independent budget path. The baseline, price strategy, and primary model do not share patrimonio.

## Motivation

The existing fixed-budget model resets every round to `100`. That is useful for isolated lineup comparison, but it is not faithful to Cartola. Poor picks can reduce team patrimonio and make future high-value squads infeasible. Strong point models that repeatedly select overvalued players should be penalized if they lose purchasing power.

External research and Cartola guidance support this:

- GE/Gato Mestre describes Cartola as a points and cartoletas game where more cartoletas increase later buying power, and explains that valuation depends on "mínimo para valorizar" and season-specific rules.
- Cartola FAQ-style documentation describes team patrimonio as changing according to selected players' per-round price variation.
- Finance and fantasy-sports backtesting analogies point to path-dependent portfolio evaluation: final points alone hides drawdown and purchasing-power risk.

The valuation formula is proprietary and has changed across seasons. The backtest must not reverse-engineer or predict valuation in v1. It should use the historical `variacao` value only after the squad has already been selected.

Data contract: historical `variacao` is the same-round post-round price change for that asset. The repo already uses `preco_pre_rodada = preco - variacao`; moving-budget replay depends on this interpretation. Candidate frames, optimizer-selected rows, and selected-player artifacts must preserve `variacao`.

## Non-Goals

Do not add:

- a user-facing fixed-budget mode;
- a hidden compatibility switch for normal backtests;
- valuation prediction;
- a points-plus-expected-valuation optimizer objective;
- live open-round budget-path updates without finalized `variacao`;
- transfer costs or roster continuity constraints;
- reserves/bench modeling;
- a broad report redesign;
- cross-run migration of old fixed-budget report artifacts.

Old fixed-budget reports remain historical artifacts, but they are non-comparable with new moving-budget reports.

## Current System

Today:

- `BacktestConfig.budget` is passed directly into `optimize_squad()`;
- every target round uses the same budget;
- target rounds can run independently and in parallel;
- `summary.csv` ranks strategies by total actual captain-aware points;
- `round_results.csv` records `budget_used` but not the budget available before the round;
- experiment comparability checks candidate pools and solver statuses, but not budget policy;
- experiment index rows store `budget`, but not whether it was fixed or moving.

Moving budget changes the execution model. Target-round optimization is now path-dependent because round `t + 1` depends on selected players' `variacao` from round `t`.

## Budget Semantics

### Initial Budget

`BacktestConfig.budget` and CLI `--budget` are renamed semantically to initial budget. The field name may remain `budget` for compatibility with configuration construction, but all UI/report labels should say "Initial budget" where practical.

### Budget Before Round

For a strategy and target round, `budget_before_round` is that strategy's current patrimonio before selecting the squad.

### Optimization Constraint

The optimizer must constrain squad cost using `budget_before_round`.

Implementation may pass a per-round `BacktestConfig` copy into `optimize_squad()`:

```python
round_config = replace(config, budget=budget_before_round)
```

The optimizer does not need to own moving-budget state. It should remain a single-round optimizer.

### Budget Delta

`budget_delta` is the sum of selected rows' `variacao` values.

Rules:

- include every selected squad asset;
- include `tec` rows if selected and if `variacao` exists;
- use historical `variacao` only after selection and scoring;
- do not use `variacao` as a feature or optimizer input unless it is already part of prior-round feature engineering;
- every selected asset row in a completed historical target round must contain finite numeric `variacao`;
- empty, skipped, or infeasible selections do not require selected-row variation because no assets were selected;
- if a selected row is missing `variacao` or contains non-finite variation, fail the run or mark it invalid. Do not silently treat missing variation as zero.

### Budget After Round

```text
budget_after_round = budget_before_round + budget_delta
```

Unspent budget carries forward automatically because the update applies to patrimonio, not just spent money.

If `budget_after_round` reaches zero or becomes negative, preserve that value in the budget path. The next round should run through the normal optimizer path and will usually become infeasible. Do not clamp the budget to zero and do not silently reset it.

### Budget Remaining

```text
budget_remaining = budget_before_round - budget_used
```

Budget remaining is reported for diagnostics. It is not lost.

### Drawdown

Each strategy tracks a patrimonio path:

```text
initial_budget, after_round_5, after_round_6, ...
```

For each round:

```text
budget_peak = max(previous peaks, budget_after_round)
budget_drawdown = budget_peak - budget_after_round
```

`max_budget_drawdown` in summary is the maximum `budget_drawdown` across evaluated rounds.

### Infeasible Rounds

If no squad satisfies the current budget and formation constraints:

- preserve existing optimizer status behavior (`Infeasible`, `Empty`, etc.);
- selected squad is empty;
- `budget_delta = 0`;
- `budget_after_round = budget_before_round`;
- `budget_remaining = budget_before_round`;
- round is visible in outputs and should make promotion/comparability stricter through solver-status signatures.

## Execution Model

V1 prioritizes correctness over target-round parallelism.

The implementation is sequential and owns every target round inside one state machine:

```text
build/cache prediction frames for all detected rounds
initialize budget state per strategy
for target_round in ascending order:
  build training frame using rounds < target_round
  fit models
  score candidate frame
  for strategy in baseline, primary model, price:
    optimize under that strategy's current budget
    score selected squad
    update that strategy's budget state
```

Skipped, empty, and infeasible strategy rows must be emitted inside this sequential loop with unchanged budget:

```text
budget_before_round = current_budget[strategy]
budget_delta = 0
budget_after_round = current_budget[strategy]
budget_remaining = current_budget[strategy] - budget_used
```

For empty/skipped/infeasible rows, `budget_used` should remain the existing solver/result value, usually `0`, and the unchanged state must carry into the next round. Budget fields must not be `NA` for a strategy row in the moving-budget replay.

If a target round is excluded because required evidence is unavailable, budget still carries unchanged through that round, but the run becomes ineligible for promotion. Carrying the budget path forward does not make a missing-evidence run comparable.

Target-round workers must be disabled for moving-budget backtests. The CLI can keep accepting `--jobs`, but metadata must show:

- requested jobs;
- effective target-round workers = `1`;
- parallel backend = `sequential_moving_budget`;
- a warning when `--jobs > 1` says target-round parallelism is disabled by moving-budget semantics.

Model-internal parallelism can still be used because target rounds are no longer evaluated concurrently. Native thread caps remain an operational performance concern, but they do not change budget correctness.

Future optimization can split scoring from replay, but only after the sequential implementation is correct and covered by tests.

## Artifact Changes

### Metadata

Add to `run_metadata.json`:

```json
{
  "budget_policy": "moving",
  "initial_budget": 100.0
}
```

`budget_policy` is required. New runs must always write `"moving"`.

### round_results.csv

Keep existing `budget_used`.

Add:

```text
budget_before_round
budget_after_round
budget_delta
budget_remaining
```

Recommended internal diagnostics, if low-cost:

```text
budget_peak
budget_drawdown
```

The four required columns are part of the public contract. `budget_peak` and `budget_drawdown` can be included if they simplify summary computation.

### summary.csv

Add:

```text
initial_budget
final_budget
total_budget_delta
min_budget
max_budget_drawdown
budget_constrained_rounds
```

Definitions:

- `initial_budget`: first `budget_before_round` for the strategy;
- `final_budget`: last `budget_after_round` for the strategy;
- `total_budget_delta`: sum of `budget_delta`;
- `min_budget`: minimum over all `budget_before_round` and `budget_after_round` values;
- `max_budget_drawdown`: maximum peak-to-trough patrimonio loss;
- `budget_constrained_rounds`: count of optimal rounds where `budget_remaining <= 1e-6`.

Budget path metrics use every target-round row for the strategy, including skipped, empty, and infeasible rows with unchanged budget. Point totals continue to follow the existing scoring convention for completed/optimal rows.

Ranking remains by total actual captain-aware points. Budget metrics are guardrails and diagnostics, not the primary score in v1.

Benchmark deltas such as `actual_points_delta_vs_price` continue to compare each strategy's total actual points over that strategy's optimal rows. They do not restrict to common optimal rounds. Solver-status and skipped-round comparability remain promotion gates. If the benchmark strategy has zero optimal rows, benchmark deltas must be `NA`; an all-infeasible benchmark total of `0` is not a valid baseline.

### Experiments

Model-feature experiment artifacts must carry moving-budget metadata:

- `experiment_metadata.json`: `budget_policy="moving"`, `initial_budget`;
- child `run_metadata.json`: same fields;
- `per_season_summary.csv`: include child summary budget columns;
- `ranked_summary.csv`: aggregate budget diagnostics across seasons;
- SQLite experiment index: persist `budget_policy` at experiment and child-run level.

Existing fixed-budget experiments that lack `budget_policy` are fixed-budget artifacts and must not be compared against moving-budget outputs for promotion decisions.

`ranked_summary.csv` should aggregate budget diagnostics explicitly:

- `total_budget_delta`: sum across seasons;
- `average_final_budget`: simple mean of per-season final budgets;
- `worst_min_budget`: minimum per-season `min_budget`;
- `worst_max_budget_drawdown`: maximum per-season drawdown;
- `budget_constrained_rounds`: sum across seasons.

Any artifact or index row with missing `budget_policy` is interpreted as `fixed` and excluded from moving-budget promotion comparisons unless the caller explicitly asks for historical fixed-budget archaeology.

Artifact and index readers must normalize missing `budget_policy` to `fixed` before any promotion or aggregation filter runs. This normalization should be central, not repeated ad hoc in reports.

## Comparability

Promotion decisions must compare only runs with:

- same `budget_policy`;
- same `initial_budget`;
- same scoring contract;
- same season set;
- same start round;
- same candidate-pool signatures;
- same solver-status signature rules;
- same fixture/FootyStats/matchup modes unless explicitly grouped.

Old fixed-budget reports are non-comparable. Incumbents and challengers must be rerun under moving-budget semantics on the approved comparable season set before any promotion decision. For now, that means `2023,2024,2025`; `2021` and `2022` must pass the same compatibility/comparability audits before joining the promotion leaderboard.

## Failure Semantics

The run must fail fast if selected assets needed for budget update have invalid variation:

- missing `variacao` column in selected players;
- null `variacao` in a selected row;
- non-numeric `variacao`;
- non-finite `variacao`.

The run should not fail merely because a strategy becomes infeasible under its own budget path. Infeasibility is valid model evidence and should be surfaced in solver status, round results, summary, experiments, and promotion gates.

## CLI And UX

Keep CLI argument names for now:

```bash
--budget 100
```

But display labels should say "Initial budget" in Rich output and reports.

`--jobs` remains accepted, but target-round parallelism is not effective in v1. The CLI should report this clearly through run metadata and, if already displayed, warnings.

## Testing Requirements

Required tests:

- budget state starts at initial budget;
- budget update sums selected `variacao`, including `tec`;
- missing selected `variacao` fails;
- non-finite selected `variacao` fails;
- next round uses updated budget;
- next round proves changed feasible selection with selected-player IDs, not only lower `budget_used`;
- each strategy has an independent budget path;
- budget remaining carries forward as cash;
- infeasible round leaves budget unchanged and records solver status;
- fixture/evidence-excluded rounds carry budget unchanged but make the run promotion-ineligible;
- benchmark deltas are `NA` when the benchmark has zero optimal rows;
- moving budget disables target-round parallelism;
- `round_results.csv` has required budget columns;
- `summary.csv` has required budget columns;
- metadata has `budget_policy="moving"` and `initial_budget`;
- experiment artifacts/index include budget policy;
- artifact/index readers normalize missing `budget_policy` to `fixed`;
- missing `budget_policy` is interpreted as `fixed` and excluded from moving-budget promotion/report aggregation.

## Acceptance Criteria

The feature is complete when:

- `uv run --frozen scripts/pyrepo-check --all` passes;
- a small synthetic backtest proves round `t + 1` selection changes after round `t` devaluation;
- a real historical smoke backtest writes moving-budget round and summary columns;
- `scripts/run_model_experiments.py --group production-parity ...` produces moving-budget artifacts and index rows;
- documentation/roadmap states that old fixed-budget evidence is superseded and incumbents must be rerun.
